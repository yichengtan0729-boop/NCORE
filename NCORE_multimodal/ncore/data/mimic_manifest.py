from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def _read(path):
    if path is None or str(path).strip() == "":
        return None
    return pd.read_csv(path)


def build_manifest(
    cxr_metadata_csv: str,
    report_index_csv: str,
    ehr_index_csv: str,
    admissions_csv: str,
    ventilation_csv: str,
    output_csv: str,
    cxr_split_csv: Optional[str] = None,
    horizon_hours: int = 24,
    image_root: Optional[str] = None,
):
    """Build a strict study-indexed cohort manifest.

    Required prepared indexes:
      report_index: subject_id, study_id, report_path, report_time
      ehr_index:    subject_id, study_id, ehr_path (+ optional hadm_id/stay_id)

    The function deliberately requires an explicit report_time because using a full
    report with an earlier image acquisition anchor can leak future text.
    """
    cxr = pd.read_csv(cxr_metadata_csv)
    rep = pd.read_csv(report_index_csv)
    ehr = pd.read_csv(ehr_index_csv)
    adm = pd.read_csv(admissions_csv)
    vent = pd.read_csv(ventilation_csv)

    for df, cols, name in [
        (cxr, ["subject_id", "study_id", "dicom_id"], "cxr"),
        (rep, ["subject_id", "study_id", "report_path", "report_time"], "report_index"),
        (ehr, ["subject_id", "study_id", "ehr_path"], "ehr_index"),
    ]:
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise ValueError(f"{name} missing columns: {miss}")

    # Prefer AP/PA frontal image if ViewPosition exists; one image per study for v1.
    if "ViewPosition" in cxr.columns:
        frontal = cxr[cxr["ViewPosition"].astype(str).isin(["AP", "PA"])].copy()
        if len(frontal):
            cxr = frontal
    cxr = cxr.sort_values(["subject_id", "study_id"]).groupby(["subject_id", "study_id"], as_index=False).first()

    if image_root is not None:
        root = Path(image_root)
        if "image_path" not in cxr.columns:
            # MIMIC-CXR-JPG standard hierarchy: files/pXX/pXXXXXXXX/sXXXXXXXX/dicom.jpg
            def make_path(r):
                sid = int(r.subject_id); st = int(r.study_id); d = str(r.dicom_id)
                p2 = str(sid)[:2]
                return str(root / "files" / f"p{p2}" / f"p{sid}" / f"s{st}" / f"{d}.jpg")
            cxr["image_path"] = cxr.apply(make_path, axis=1)

    df = cxr.merge(rep, on=["subject_id", "study_id"], how="inner", suffixes=("", "_rep"))
    df = df.merge(ehr, on=["subject_id", "study_id"], how="inner", suffixes=("", "_ehr"))
    df["index_time"] = pd.to_datetime(df["report_time"])

    # Mortality label.
    adm_cols = [c for c in ["subject_id", "hadm_id", "hospital_expire_flag", "deathtime", "admittime", "dischtime"] if c in adm.columns]
    if "hadm_id" in df.columns and "hadm_id" in adm.columns:
        df = df.merge(adm[adm_cols], on=["subject_id", "hadm_id"], how="left")
    else:
        # If hadm_id not pre-associated, use admission interval.
        adm["admittime"] = pd.to_datetime(adm["admittime"])
        adm["dischtime"] = pd.to_datetime(adm["dischtime"])
        rows = []
        for _, r in df.iterrows():
            cand = adm[(adm.subject_id == r.subject_id) & (adm.admittime <= r.index_time) & (adm.dischtime >= r.index_time)]
            rr = r.to_dict()
            if len(cand):
                a = cand.iloc[0]
                rr["hadm_id"] = a.hadm_id
                rr["hospital_expire_flag"] = a.get("hospital_expire_flag", np.nan)
            rows.append(rr)
        df = pd.DataFrame(rows)

    df["y_mortality"] = df.get("hospital_expire_flag", 0).fillna(0).astype(float)
    df["mask_mortality"] = (~df.get("hospital_expire_flag", pd.Series(np.nan, index=df.index)).isna()).astype(float)

    # Ventilation label: positive if FIRST invasive start after anchor and within horizon.
    v = vent.copy()
    for c in ["starttime", "endtime"]:
        if c in v.columns:
            v[c] = pd.to_datetime(v[c])
    status_col = "ventilation_status" if "ventilation_status" in v.columns else ("ventilation" if "ventilation" in v.columns else None)
    if status_col:
        v = v[v[status_col].astype(str).str.lower().str.contains("invasive")]

    yv, mv = [], []
    for _, r in df.iterrows():
        q = v[v.subject_id == r.subject_id]
        if "hadm_id" in q.columns and pd.notna(r.get("hadm_id", np.nan)):
            q = q[q.hadm_id == r.hadm_id]
        t0 = r.index_time
        # already ventilated at t0 -> not in risk set
        already = ((q.starttime <= t0) & (q.endtime.isna() | (q.endtime >= t0))).any() if len(q) else False
        if already:
            yv.append(0.0); mv.append(0.0); continue
        future = q[(q.starttime > t0) & (q.starttime <= t0 + pd.Timedelta(hours=horizon_hours))]
        yv.append(float(len(future) > 0)); mv.append(1.0)
    df["y_vent24"] = yv
    df["mask_vent24"] = mv

    if cxr_split_csv:
        sp = pd.read_csv(cxr_split_csv)
        if "split" in sp.columns:
            sp = sp[["subject_id", "split"]].drop_duplicates("subject_id")
            df = df.merge(sp, on="subject_id", how="left")
    if "split" not in df.columns:
        # Deterministic patient split fallback.
        subjects = sorted(df.subject_id.unique())
        rng = np.random.RandomState(7); rng.shuffle(subjects)
        n = len(subjects); ntr = int(.8*n); nv = int(.1*n)
        mp = {s: ("train" if i < ntr else "val" if i < ntr+nv else "test") for i,s in enumerate(subjects)}
        df["split"] = df.subject_id.map(mp)

    # Keep one row per study and ensure subject-disjoint split.
    df = df.drop_duplicates(["subject_id", "study_id"]).reset_index(drop=True)
    split_counts = df.groupby("subject_id")["split"].nunique()
    if (split_counts > 1).any():
        raise RuntimeError("subject_id appears in multiple splits; fix split mapping")

    cols = [c for c in [
        "subject_id", "hadm_id", "stay_id", "study_id", "dicom_id", "split", "index_time",
        "image_path", "report_path", "report_embedding_path", "ehr_path",
        "y_vent24", "mask_vent24", "y_mortality", "mask_mortality"
    ] if c in df.columns]
    out = df[cols].copy()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
