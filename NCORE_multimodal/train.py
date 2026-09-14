from pathlib import Path
import argparse
import json
import torch

from ncore.config import load_config, ensure_output_dir, training_plan_line
from ncore.utils import seed_everything
from ncore.models.model import NCORE
from ncore.checkpointing import load_compatible_model_state
from ncore.training import (
    make_loader,
    compute_train_pos_weight,
    log_data_statistics,
    direct_initialization_error,
    evaluate_model,
    load_saved_direct_temperature,
    load_saved_correction_bound,
    load_saved_final_temperature,
    load_saved_reason_threshold,
    select_policy_checkpoint,
    train_direct,
    train_operator_warmup,
    train_supervised,
    train_policy_warmup,
    train_grpo,
)


STAGES = ["direct", "operator_warmup", "supervised", "policy_warmup", "grpo"]


def _load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _automatic_checkpoint(stage, output_dir, resume):
    if resume:
        last_names = {
            "direct": "last_direct.pt",
            "operator_warmup": "last_operator_warmup.pt",
            "supervised": "last_supervised.pt",
            "policy_warmup": "last_policy_warmup.pt",
            "grpo": "last.pt",
        }
        return output_dir / last_names[stage]
    predecessors = {
        "operator_warmup": ["best_direct.pt"],
        "supervised": ["best_operator_warmup.pt", "best_direct.pt"],
        "policy_warmup": [
            "best_supervised.pt",
            "last_supervised.pt",
            "best_operator_warmup.pt",
            "best_direct.pt",
        ],
        "grpo": ["best_policy_warmup.pt", "last_policy_warmup.pt", "best_supervised.pt"],
    }
    for name in predecessors.get(stage, []):
        candidate = output_dir / name
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize matching model tensors from a v1/v2 checkpoint.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer and epoch from the current stage's last checkpoint.",
    )
    args = parser.parse_args()

    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint cannot be used together")

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    print(training_plan_line(cfg))
    seed_everything(int(cfg.get("seed", 7)))
    requested_device = cfg.get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    output_dir = ensure_output_dir(cfg)
    model = NCORE(cfg).to(device)

    no_rl = args.stage == "grpo" and not bool(
        cfg.get("grpo", {}).get("enabled", True)
    )
    if no_rl and (args.checkpoint or args.resume or args.init_checkpoint):
        parser.error(
            "The no-RL ablation always uses best_supervised.pt; do not pass "
            "--checkpoint, --resume, or --init-checkpoint."
        )
    if (
        args.stage == "grpo"
        and not no_rl
        and not args.checkpoint
        and not args.resume
    ):
        select_policy_checkpoint(output_dir, cfg)
    checkpoint_path = (
        output_dir / "best_supervised.pt"
        if no_rl
        else Path(args.checkpoint) if args.checkpoint
        else _automatic_checkpoint(args.stage, output_dir, args.resume)
    )
    initialization_only = False
    configured_init = cfg["training"].get("init_checkpoint")
    init_path = args.init_checkpoint or configured_init
    if checkpoint_path is None and init_path:
        checkpoint_path = Path(init_path)
        initialization_only = True
    checkpoint_state = None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_state = _load(checkpoint_path, device)
        report = load_compatible_model_state(model, checkpoint_state)
        counts = {
            "loaded": len(report["loaded_keys"]),
            "missing": len(report["missing_keys"]),
            "skipped_incompatible": len(report["skipped_incompatible_keys"]),
            "unexpected": len(report["unexpected_keys"]),
        }
        print(
            f"[checkpoint] path={checkpoint_path} init_only={initialization_only} "
            f"counts={json.dumps(counts, sort_keys=True)}"
        )
        print(f"[checkpoint] loaded_keys={report['loaded_keys']}")
        print(f"[checkpoint] missing_keys={report['missing_keys']}")
        print(
            "[checkpoint] skipped_incompatible_keys="
            f"{report['skipped_incompatible_keys']}"
        )
    if args.stage != "direct":
        load_saved_direct_temperature(model, output_dir)
        load_saved_correction_bound(model, output_dir)
        load_saved_final_temperature(model, output_dir)
        load_saved_reason_threshold(model, output_dir)

    train_loader = make_loader(cfg, "train", True)
    val_loader = make_loader(cfg, "val", False)
    loaders = {"train": train_loader, "val": val_loader}
    try:
        loaders["test"] = make_loader(cfg, "test", False)
    except (FileNotFoundError, ValueError):
        pass
    pos_weight, _ = compute_train_pos_weight(train_loader, device)
    log_data_statistics(loaders, pos_weight)

    resume_state = checkpoint_state if args.resume and not initialization_only else None
    if (
        args.stage == "operator_warmup"
        and checkpoint_path is not None
        and checkpoint_path.name == "best_direct.pt"
    ):
        error = direct_initialization_error(model, val_loader, device)
        print(f"[sanity] max_abs_final_minus_direct={error:.8g}")
        if error > 1e-6:
            raise RuntimeError(
                "Residual initialization does not recover the direct predictor"
            )

    common = (model, train_loader, val_loader, cfg, device, output_dir, pos_weight)
    if args.stage == "direct":
        train_direct(*common, resume_state=resume_state)
    elif args.stage == "operator_warmup":
        train_operator_warmup(*common, resume_state=resume_state)
    elif args.stage == "supervised":
        train_supervised(*common, resume_state=resume_state)
    elif args.stage == "policy_warmup":
        train_policy_warmup(*common, resume_state=resume_state)
    elif not no_rl:
        train_grpo(*common, resume_state=resume_state)
    else:
        metrics = evaluate_model(
            model,
            val_loader,
            cfg,
            device,
            deterministic=True,
            split="val",
            pos_weight=pos_weight,
            stage="supervised",
        )
        print(
            f"[grpo:no_rl] checkpoint={checkpoint_path} metrics={metrics}"
        )


if __name__ == "__main__":
    main()
