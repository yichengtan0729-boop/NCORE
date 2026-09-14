# Data preparation for NCORE

The core model is generic. For the default medical experiment, create **one study-indexed row per prediction anchor**.

## A. Files you point to in `configs/paths.yaml`

1. `cxr_metadata_csv`: MIMIC-CXR-JPG metadata.
2. `cxr_split_csv`: official CXR split if you want to inherit it.
3. `report_index_csv`: your explicit image-study/report crosswalk with report availability time.
4. `ehr_index_csv`: one pre-index EHR window file per study.
5. `admissions_csv`: MIMIC-IV admissions.
6. `ventilation_csv`: derived ventilation intervals.
7. `manifest`: output final CSV.

## B. Report index

Required columns:

```text
subject_id,study_id,report_path,report_time
```

If you use precomputed report embeddings, first run:

```bash
python scripts/encode_reports.py \
  --index /path/to/report_index.csv \
  --model /path/to/local/clinical_text_encoder \
  --outdir /path/to/report_embeddings \
  --local_files_only
```

This creates an index with `report_embedding_path`. Point `paths.report_index_csv` to that new CSV before building the final manifest.

**Important:** `report_time` must mean the report is already available. If you cannot establish that timing, do not describe the task as pre-report early warning. Treat it as post-report risk stratification or remove the report modality.

## C. EHR index

Required columns:

```text
subject_id,study_id,ehr_path
```

Recommended optional columns:

```text
hadm_id,stay_id,index_time
```

Each `ehr_path` should be a NumPy array `[T,F]`, using only events available before the report anchor. With the MedFuse-style feature set, `F=76`; if your extractor differs, change `model.encoders.ehr.input_dim` in YAML.

## D. Build final manifest

```bash
python scripts/build_manifest.py --config configs/ncore_tri.yaml --horizon 24
```

The builder:

- keeps one frontal image per study for the first implementation;
- anchors on `report_time`;
- adds in-hospital mortality;
- marks future invasive ventilation within 24h;
- masks samples already invasively ventilated at the anchor;
- checks subject-disjoint splitting.

## E. Final manifest columns

```text
subject_id
hadm_id
stay_id
study_id
dicom_id
split
index_time
image_path
report_path
report_embedding_path
ehr_path
y_vent24
mask_vent24
y_mortality
mask_mortality
```

## F. Changing or adding modalities

Every modality needs two YAML entries:

```yaml
model:
  modalities: [image, report, ehr, my_modality]
  encoders:
    my_modality:
      backend: precomputed
      input_dim: 256
      output_dim: 256

data:
  modality_fields:
    my_modality:
      kind: vector
      column: my_modality_path
```

The action space automatically becomes `(number of modalities * K) + STOP`.
