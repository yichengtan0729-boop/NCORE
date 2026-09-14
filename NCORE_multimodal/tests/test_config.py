from ncore.config import load_config, ensure_output_dir


def test_mortality_override_selects_one_task_and_separate_output(tmp_path):
    cfg = load_config("configs/ncore_strong_residual_mortality.yaml")
    cfg["paths"]["output_root"] = str(tmp_path)
    assert cfg["experiment"]["task_names"] == ["mortality"]
    assert cfg["experiment"]["primary_metric"] == "final_auprc_mortality"
    assert ensure_output_dir(cfg) == tmp_path / "ncore_strong_residual" / "mortality"
