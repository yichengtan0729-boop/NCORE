"""Create mask-patterned CXR, ECG, and temporal EHR data for five-stage smoke tests."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    root = Path("dummy_strong_data").resolve()
    root.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    split_sizes = {"train": 32, "val": 16, "test": 16}
    split_offsets = {"train": 1000, "val": 2000, "test": 3000}
    availability_patterns = [
        (False, True),
        (False, False),
        (True, True),
        (True, False),
    ]
    for split, count in split_sizes.items():
        for index in range(count):
            subject_id = split_offsets[split] + index
            vent24 = float(index % 2)
            mortality = float((index // 2) % 2)
            cxr = rng.normal(size=512).astype("float32")
            ecg = rng.normal(size=768).astype("float32")
            ehr = rng.normal(size=(8 + index % 5, 76)).astype("float32")
            cxr[:8] += 0.8 * vent24 + 0.4 * mortality
            ecg[:8] += 0.5 * vent24 + 0.8 * mortality
            ehr[:, :4] += 0.7 * vent24 + 0.7 * mortality
            cxr_path = root / f"{split}_{subject_id}_cxr.npy"
            ecg_path = root / f"{split}_{subject_id}_ecg.npy"
            ehr_path = root / f"{split}_{subject_id}_ehr.npy"
            np.save(cxr_path, cxr)
            np.save(ecg_path, ecg)
            np.save(ehr_path, ehr)
            has_cxr, has_ecg = availability_patterns[index % len(availability_patterns)]
            rows.append(
                {
                    "subject_id": subject_id,
                    "study_id": subject_id * 10,
                    "split": split,
                    "index_time": "2026-01-01",
                    "cxr_embedding_path": str(cxr_path) if has_cxr else "",
                    "ecg_embedding_path": str(ecg_path) if has_ecg else "",
                    "ehr_path": str(ehr_path),
                    "y_vent24": vent24,
                    "mask_vent24": 1,
                    "y_mortality": mortality,
                    "mask_mortality": 1,
                }
            )
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(manifest)


if __name__ == "__main__":
    main()
