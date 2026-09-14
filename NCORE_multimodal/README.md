# NCORE: Patient-Specific Non-Commutative Operator Reasoning for Multimodal Clinical Prediction

This repository is a self-contained research implementation that combines:

1. **MedFuse-style clinical multimodal data/encoder compatibility** for MIMIC-IV + MIMIC-CXR.
2. **GtNN-style non-commuting graph/operator words** over a shared concept set.
3. **NCORE extensions**: a strong mask-aware direct predictor, patient-specific identity-mixed residual operators, gated state updates, discrete RL path selection, direct-relative prediction reward, reverse-order reward, optional evidence/faithfulness rewards, and GRPO-style policy updates.

The recommended strong-backbone configuration uses precomputed BioMedCLIP CXR embeddings, precomputed ECG-FM embeddings, and a temporal EHR Transformer. The core remains generic over the number and names of modalities: add a modality to YAML and a matching encoder/data field.

## 0. What you change first

All paths are centralized in:

```text
configs/paths.yaml
```

Copy from:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

Then edit only that file for MIMIC, processed EHR windows, report files/embeddings, checkpoints, and optional MedFuse checkout.

## 1. Main mathematical pipeline

For modality `m` and patient `n`:

```text
raw modality x_m
  -> encoder h_m
  -> concept projector Z_m in R^{K x d}
  -> patient operator T_m in R^{K x K}
  -> concept-local operator T_{m,k}
  -> policy selects action a_t=(m,k)
  -> state update S_{t+1}=O_{m,k}(S_t)
  -> STOP
  -> operator delta logits

encoder vectors + availability mask
  -> mask-aware direct fusion
  -> direct logits

final logits = direct logits + alpha * operator delta logits
```

An action update is implemented as:

```text
propagated = T_{m,k} @ S_t
evidence   = G_k @ Z_m
S_{t+1}    = LayerNorm(S_t + beta * phi_m(propagated + evidence))
```

Each available operator is mixed with identity before localization, while every missing-modality operator is exactly identity. The learnable gates `alpha` and `beta` start small, so the initial residual model recovers the direct predictor while retaining patient-specific noncommutative reasoning.

## 2. Repository layout

```text
configs/
  paths.example.yaml        # all filesystem paths in one place
  ncore_tri.yaml            # default 3-modality experiment
  ncore_four_modal.yaml     # example extension to a 4th precomputed modality
  ncore_strong.yaml         # legacy operator-only strong-backbone configuration
  ncore_strong_residual.yaml
  ncore_strong_residual_mortality.yaml
ncore/
  config.py
  data/
    dataset.py              # generic manifest dataset
    mimic_manifest.py       # cohort/label manifest builder
  models/
    encoders.py             # image, EHR, text, precomputed, MedFuse adapters
    concepts.py             # shared concept projection
    operators.py            # patient-specific + concept-local operators
    policy.py               # RL policy
    noncommutative.py       # GtNN-like fixed/global operator-word baseline
    baselines.py            # concat and sequential fusion baselines
    model.py                # NCORE model + rollouts
  rl/grpo.py
  training.py
scripts/
  build_manifest.py
  encode_reports.py
train.py
evaluate.py
tests/
```

## 3. Manifest format

The training CSV/Parquet should contain at least:

```text
subject_id,study_id,split,index_time,image_path,report_path,ehr_path,
y_vent24,y_mortality,mask_vent24,mask_mortality
```

Optional extra modality columns can be added, e.g. `lab_embedding_path`.

`image_path` may be a single image path. `report_path` may be a text file path or a `.npy` embedding path depending on config. `ehr_path` is expected to be `.npy` shaped `[T,F]` for the native LSTM encoder or `[D]` for precomputed mode.

## 4. Recommended training sequence

### NCORE-v5 performance-first mortality pipeline

NCORE-v5 keeps the base Direct predictor and adds bounded pairwise and gated
pooling residuals. Non-commutative reasoning is split into a label-free
ReasonGate (whether to reason) and a no-STOP PathPolicy (how to reason). Model,
threshold, correction-bound, temperature, and ensemble choices are fitted on
validation only.

The formal config is fixed to 30/10/16/10/12 epochs and does not early-stop:

```bash
bash scripts/run_v5.sh configs/ncore_performance_v5_mortality.yaml 42
```

Run all three formal seeds:

```bash
bash scripts/run_v5_multiseed.sh configs/ncore_performance_v5_mortality.yaml
```

Evaluate the validation-selected single checkpoint (use
`best_unconfirmed.pt` only when the performance guard reports that no candidate
improved Direct):

```bash
python evaluate.py \
  --config configs/ncore_performance_v5_mortality.yaml \
  --checkpoint strong_residual_outputs/ncore_performance_v5/mortality/seed_42/best_final_v5.pt \
  --stage grpo \
  --split test \
  --oracle-diagnostics \
  --output-json seed_42_single.json
```

`--oracle-diagnostics` is a label-aware test diagnostic for measuring the
operator ceiling and routing gap. It is never used for gradients, checkpoint
selection, threshold fitting, temperature fitting, or ensemble fitting.

Evaluate the validation-selected three-checkpoint logit ensemble:

```bash
python evaluate_ensemble.py \
  --config configs/ncore_performance_v5_mortality.yaml \
  --checkpoint-dir strong_residual_outputs/ncore_performance_v5/mortality/seed_42 \
  --split test \
  --output-json seed_42_ensemble.json
```

Summarize per-seed JSON files as mean±standard deviation:

```bash
python scripts/summarize_v5_seeds.py seed_13.json seed_42.json seed_2026.json
```

Validation selections are recorded in `correction_bound_v5.json`,
`reason_threshold_v5.json`, `top5_checkpoints_v5.json`, and
`ensemble_weights_v5.json`. The independent Direct and Final temperatures are
also validation-fitted. A complete synthetic smoke run uses only the separate
2/2/2/2/2 config:

```bash
bash scripts/run_v5_smoke.sh
```

### Strong residual pipeline

Generate the synthetic strong-backbone inputs when running the self-contained smoke test:

```bash
python scripts/make_dummy_strong_data.py
```

Run the five stages in order. Each stage automatically loads the best checkpoint from its predecessor:

```bash
python train.py --config configs/ncore_strong_residual.yaml --stage direct
python train.py --config configs/ncore_strong_residual.yaml --stage operator_warmup
python train.py --config configs/ncore_strong_residual.yaml --stage supervised
python train.py --config configs/ncore_strong_residual.yaml --stage policy_warmup
python train.py --config configs/ncore_strong_residual.yaml --stage grpo
```

The direct stage trains the direct projections/fusion/task head and the temporal EHR encoder. Operator warmup freezes that branch and trains the residual path with fixed routing. Joint supervised fine-tuning uses separate direct, EHR, and NCORE learning rates. Policy warmup includes STOP among its candidates and ranks paths by improvement over the direct prediction. GRPO freezes the direct branch by default and uses the same training-derived class weights as supervised learning.

For mortality, use the task-specific override:

```bash
python train.py --config configs/ncore_strong_residual_mortality.yaml --stage direct
```

### Evaluation

```bash
python evaluate.py \
  --config configs/ncore_strong_residual.yaml \
  --checkpoint strong_residual_outputs/ncore_strong_residual/vent24/best.pt \
  --split test
```

Evaluation reports sample counts/prevalence, direct and final AUROC/AUPRC, operator/state/identity gates, rollout length, STOP rate, and modality action frequencies.

### Legacy pipeline

Existing configurations and commands remain supported:

```bash
python train.py --config configs/ncore_tri.yaml --stage supervised
python train.py --config configs/ncore_tri.yaml --stage policy_warmup
python train.py --config configs/ncore_tri.yaml --stage grpo
```

## 5. Baselines included

- `concat`: generic MedFuse-style projected feature concatenation.
- `sequential_lstm`: fixed-order modality-sequence fusion.
- `gtnn_global`: global trainable mixture over non-commutative modality words.
- `ncore`: patient-specific concept operators + RL path selection.

For exact MedFuse encoder reuse, set `encoders.image.backend: medfuse_cxr` and/or `encoders.ehr.backend: medfuse_ehr`, set `paths.medfuse_repo`, and point to your pretrained weights.

## 6. Adding a fourth modality

Copy `configs/ncore_four_modal.yaml`. The default example adds `lab_embedding` as a precomputed vector. The NCORE core dynamically builds:

```text
num_actions = num_modalities * K + 1
```

No core model code needs to change.

## 7. Important cohort rules

- If the radiology report is an input, anchor prediction at a time when the report is available.
- EHR windows must contain only information available at or before the anchor.
- For future invasive ventilation, exclude samples already invasively ventilated at the anchor.
- Split by `subject_id`, not by study/image.
- Do not use report-derived CheXpert labels as the main target when the full report is also an input.

## 8. Quick smoke test

```bash
pytest -q
```

The tests use synthetic precomputed features and do not require MIMIC. They cover direct recovery at zero operator gate, exact identity for missing modalities, action masking, dynamic action count, direct-relative reward signs, STOP behavior, task-dependent shapes, and training-only class weighting.
