import numpy as np
import pandas as pd
import torch

from ncore.training import (
    make_loader,
    compute_train_pos_weight,
    dataset_label_statistics,
)


def test_pos_weight_is_derived_only_from_training_split(tmp_path):
    feature_path = tmp_path / "feature.npy"
    np.save(feature_path, np.zeros(4, dtype=np.float32))
    rows = []
    for split, labels in [("train", [1, 0, 0, 0]), ("val", [1, 1, 1, 0])]:
        for index, label in enumerate(labels):
            rows.append(
                {
                    "split": split,
                    "feature_path": str(feature_path),
                    "target": label,
                    "target_mask": 1,
                }
            )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    cfg = {
        "paths": {"manifest": str(manifest)},
        "experiment": {"task_names": ["target"]},
        "model": {
            "modalities": ["feature"],
            "encoders": {"feature": {"backend": "precomputed", "input_dim": 4}},
        },
        "training": {"batch_size": 2, "num_workers": 0},
        "data": {
            "format": "csv",
            "labels": {
                "target": {"column": "target", "mask_column": "target_mask"}
            },
            "modality_fields": {
                "feature": {"kind": "vector", "column": "feature_path"}
            },
        },
    }
    train_loader = make_loader(cfg, "train", False)
    val_loader = make_loader(cfg, "val", False)
    pos_weight, train_stats = compute_train_pos_weight(
        train_loader, torch.device("cpu")
    )
    val_stats = dataset_label_statistics(val_loader.dataset)
    assert pos_weight.item() == 3.0
    assert train_stats["target"]["prevalence"] == 0.25
    assert val_stats["target"]["prevalence"] == 0.75
