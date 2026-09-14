import argparse
import json
import torch

from pathlib import Path

from ncore.config import load_config
from ncore.models.model import NCORE
from ncore.checkpointing import load_compatible_model_state
from ncore.training import (
    compute_train_pos_weight,
    evaluate_model,
    load_saved_direct_temperature,
    load_saved_correction_bound,
    load_saved_final_temperature,
    load_saved_reason_threshold,
    make_loader,
    print_target_report,
)


def _load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--oracle-diagnostics",
        action="store_true",
        help=(
            "Report label-aware oracle ceiling diagnostics only; never use "
            "these values for test-time selection."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["direct", "operator_warmup", "supervised", "policy_warmup", "grpo"],
        default=None,
        help="Evaluation path; defaults to the checkpoint stage.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    requested_device = cfg.get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    model = NCORE(cfg).to(device)
    state = _load(args.checkpoint, device)
    report = load_compatible_model_state(model, state)
    print(
        "[checkpoint] "
        f"loaded={len(report['loaded_keys'])} "
        f"missing={len(report['missing_keys'])} "
        f"skipped_incompatible={len(report['skipped_incompatible_keys'])} "
        f"unexpected={len(report['unexpected_keys'])}"
    )
    load_saved_direct_temperature(model, Path(args.checkpoint).resolve().parent)
    load_saved_correction_bound(model, Path(args.checkpoint).resolve().parent)
    load_saved_final_temperature(model, Path(args.checkpoint).resolve().parent)
    load_saved_reason_threshold(model, Path(args.checkpoint).resolve().parent)
    loader = make_loader(cfg, args.split, False)
    train_loader = make_loader(cfg, "train", False)
    pos_weight, _ = compute_train_pos_weight(train_loader, device)
    metrics = evaluate_model(
        model,
        loader,
        cfg,
        device,
        deterministic=True,
        split=args.split,
        pos_weight=pos_weight,
        stage=args.stage or state.get("stage", "grpo"),
        epoch=state.get("epoch"),
        include_oracle=args.oracle_diagnostics,
    )
    rendered = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n")
    if getattr(model, "performance_v5", False):
        print_target_report(metrics, cfg)


if __name__ == "__main__":
    main()
