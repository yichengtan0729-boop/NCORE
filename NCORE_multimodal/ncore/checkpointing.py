from __future__ import annotations


def load_compatible_model_state(model, checkpoint):
    """Load only matching tensors and report all compatibility decisions."""
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    compatible = {}
    skipped_incompatible = []
    unexpected = []
    for key, value in source.items():
        if key not in target:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(target[key].shape):
            skipped_incompatible.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(target[key].shape),
                }
            )
        else:
            compatible[key] = value
    result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded_keys": sorted(compatible),
        "missing_keys": sorted(result.missing_keys),
        "skipped_incompatible_keys": skipped_incompatible,
        "unexpected_keys": sorted(unexpected),
    }
