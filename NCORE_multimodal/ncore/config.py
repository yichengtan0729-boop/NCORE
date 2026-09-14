from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import copy
import warnings
import yaml


V5_MINIMUM_EPOCHS = {
    "epochs_direct": 30,
    "epochs_operator_warmup": 10,
    "epochs_supervised": 16,
    "epochs_policy_warmup": 10,
    "epochs_grpo": 12,
}


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_config_file(path: Path, stack=None) -> Dict[str, Any]:
    path = path.resolve()
    stack = [] if stack is None else stack
    if path in stack:
        chain = " -> ".join(str(item) for item in stack + [path])
        raise ValueError(f"Circular config include: {chain}")
    cfg = yaml.safe_load(path.read_text()) or {}
    include = cfg.pop("include_paths", None)
    if include:
        includes = include if isinstance(include, list) else [include]
        base = {}
        for item in includes:
            inc_path = Path(item)
            if not inc_path.is_absolute():
                candidates = [
                    Path.cwd() / inc_path,
                    path.parent / inc_path,
                    path.parent / inc_path.name,
                ]
                inc_path = next(
                    (candidate for candidate in candidates if candidate.exists()),
                    candidates[-1],
                )
            base = _deep_merge(
                base, _load_config_file(inc_path, stack + [path])
            )
        cfg = _deep_merge(base, cfg)
    return cfg


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg = _load_config_file(Path(path))
    task_name = cfg.get("task", {}).get("name")
    if task_name:
        available = list(cfg.get("data", {}).get("labels", {}))
        if task_name == "multitask":
            task_names = available
        elif task_name in available:
            task_names = [task_name]
        else:
            raise ValueError(
                f"Unknown task '{task_name}'. Expected one of {available + ['multitask']}"
            )
        cfg.setdefault("experiment", {})["task_names"] = task_names
        if task_name != "multitask":
            if not cfg.get("model", {}).get("performance_v5", {}).get("enabled", False):
                cfg["experiment"]["primary_metric"] = f"final_auprc_{task_name}"
    validate_v5_training_plan(cfg)
    return cfg


def validate_v5_training_plan(cfg: Dict[str, Any]):
    v5 = bool(cfg.get("model", {}).get("performance_v5", {}).get("enabled", False))
    v6 = bool(cfg.get("model", {}).get("performance_v6", {}).get("enabled", False))
    formal = bool(
        cfg.get("experiment", {}).get("formal_v5", False)
        or cfg.get("experiment", {}).get("formal_v6", False)
    )
    if not ((v5 or v6) and formal):
        return []
    training = cfg.get("training", {})
    below = []
    for name, minimum in V5_MINIMUM_EPOCHS.items():
        actual = int(training.get(name, 0))
        if actual < minimum:
            below.append((name, actual, minimum))
    if below:
        details = ", ".join(
            f"{name}={actual}<{minimum}" for name, actual, minimum in below
        )
        warnings.warn(f"formal NCORE training plan is below minimum: {details}")
    return below


def training_plan_line(cfg: Dict[str, Any]) -> str:
    training = cfg.get("training", {})
    return (
        "[training-plan] "
        f"direct={int(training.get('epochs_direct', 0))} "
        f"operator_warmup={int(training.get('epochs_operator_warmup', 0))} "
        f"supervised={int(training.get('epochs_supervised', 0))} "
        f"policy_warmup={int(training.get('epochs_policy_warmup', 0))} "
        f"grpo={int(training.get('epochs_grpo', 0))}"
    )


def ensure_output_dir(cfg: Dict[str, Any]) -> Path:
    root = Path(cfg["paths"]["output_root"])
    name = cfg["experiment"]["name"]
    out = root / name
    task_name = cfg.get("task", {}).get("name")
    if task_name and task_name != "multitask":
        out = out / task_name
    if bool(cfg.get("experiment", {}).get("seed_subdirectory", False)):
        out = out / f"seed_{int(cfg.get('seed', 42))}"
    out.mkdir(parents=True, exist_ok=True)
    return out
