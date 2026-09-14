# Third-party method/code notes

## MedFuse

This project does **not** copy the MedFuse source tree. It provides optional adapters that import a local MedFuse checkout and reuse the public feature APIs of its EHR LSTM and CXR model. The baseline repository uses a projected CXR feature concatenated with EHR features for its `joint/early/unified` fusion and also provides an LSTM fusion variant.

## Graph neural networks and non-commuting operators (GtNN)

NCORE adopts the mathematical idea of multiple operators on a shared vertex/concept set and ordered, non-commuting operator words. The included `GlobalNonCommutativeFilter` is a compact GtNN-like baseline with global coefficients over ordered modality words. NCORE changes the setting by generating operators per patient and selecting concept-local operator words with a patient-conditioned policy.

## NCORE-specific additions

- Shared clinical concept nodes across arbitrary modalities.
- Patient-specific operator construction from modality concept embeddings.
- Learnable concept-local gates.
- RL action = `(modality, concept)` plus STOP.
- Reverse-order reward.
- Evidence-support reward.
- Optional counterfactual selected-evidence ablation reward.
- GRPO-style group-relative policy optimization.
