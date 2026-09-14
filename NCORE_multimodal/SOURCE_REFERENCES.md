# Method provenance and implementation mapping

## Clinical baseline

**MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images**  
Nasir Hayat, Krzysztof J. Geras, Farah E. Shamout. MLHC 2022.

Implementation mapping in this repo:

- `ncore/models/encoders.py`: optional adapters for upstream MedFuse EHR/CXR feature APIs.
- `ncore/models/baselines.py`: generic concatenation and fixed-order sequential-fusion controls.
- `ncore/data/mimic_manifest.py`: study-indexed MIMIC-CXR + MIMIC-IV cohort construction, adapted for future-outcome prediction and a third report modality.

## General mathematical method

**Graph neural networks and non-commuting operators**  
Mauricio Velasco, Kaiying O'Hare, Bernardo Rychtenberg, Soledad Villar. NeurIPS 2024.

Implementation mapping:

- `ncore/models/noncommutative.py`: ordered words and a global trainable non-commutative filter baseline.
- `ncore/models/operators.py`: multiple non-expansive operators on the same concept vertex set.

## NCORE modifications

- Modality-specific observations are mapped to a shared clinical concept set.
- Operators are generated per patient instead of being a fixed graph tuple.
- Each RL action selects a `(modality, concept)` localized operator.
- The state transition includes both operator propagation and current modality evidence injection.
- The policy learns patient-specific operator words and STOP decisions.
- Rewards include prediction quality, original-vs-reverse order advantage, evidence support, optional selected-evidence counterfactual faithfulness, and path length.
- GRPO-style group-relative advantages are computed among multiple paths sampled for the same patient.
