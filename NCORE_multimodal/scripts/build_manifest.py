import argparse
from ncore.config import load_config
from ncore.data.mimic_manifest import build_manifest

ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--horizon",type=int,default=24)
a=ap.parse_args(); cfg=load_config(a.config); p=cfg["paths"]
out=build_manifest(p["cxr_metadata_csv"],p["report_index_csv"],p["ehr_index_csv"],p["admissions_csv"],p["ventilation_csv"],p["manifest"],p.get("cxr_split_csv"),a.horizon,p.get("mimic_cxr_root"))
print(out.head()); print(out.split.value_counts()); print("saved",p["manifest"])
