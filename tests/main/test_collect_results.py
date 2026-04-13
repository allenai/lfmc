import json
from pathlib import Path

from lfmc.main.collect_results import (
    ExperimentConfig,
    _normalize_excluded_bands,
    collect_results,
    generate_table_1,
    generate_table_5,
    generate_table_6,
    load_experiments,
)

SAMPLE_RESULTS = {
    "all": {"rmse": 18.91, "mae": 12.58, "r2_score": 0.72},
    "baseline": {"rmse": 33.66, "mae": 25.38, "r2_score": 0.11},
    "Winter": {"rmse": 15.31, "mae": 10.74, "r2_score": 0.77},
    "Spring": {"rmse": 22.85, "mae": 15.35, "r2_score": 0.69},
    "Summer": {"rmse": 19.70, "mae": 13.05, "r2_score": 0.67},
    "Autumn": {"rmse": 12.70, "mae": 9.27, "r2_score": 0.75},
    "Tree cover": {"rmse": 18.00, "mae": 11.97, "r2_score": 0.68},
    "Grassland": {"rmse": 20.09, "mae": 13.62, "r2_score": 0.73},
    "Shrubland": {"rmse": 19.53, "mae": 12.28, "r2_score": 0.74},
    "Built-up": {"rmse": 16.79, "mae": 11.78, "r2_score": 0.77},
    "Bare / sparse vegetation": {"rmse": 20.52, "mae": 15.67, "r2_score": 0.79},
    "elevation_0_500": {"rmse": 18.34, "mae": 11.59, "r2_score": 0.73},
    "elevation_500_1000": {"rmse": 17.93, "mae": 11.98, "r2_score": 0.77},
    "elevation_1000_1500": {"rmse": 21.65, "mae": 14.54, "r2_score": 0.73},
    "elevation_1500_2000": {"rmse": 18.91, "mae": 13.56, "r2_score": 0.75},
    "elevation_2000_2500": {"rmse": 19.35, "mae": 12.44, "r2_score": 0.61},
    "elevation_2500_3000": {"rmse": 15.25, "mae": 10.00, "r2_score": 0.54},
    "elevation_3000_3500": {"rmse": 14.54, "mae": 10.41, "r2_score": 0.32},
    "high_fire_danger": {"rmse": 19.0, "mae": 12.6, "r2_score": 0.70},
    "non_high_fire_danger": {"rmse": 18.5, "mae": 12.3, "r2_score": 0.74},
}


def _write_experiment(folder: Path, config: dict, results: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "experiment_config.json", "w") as f:
        json.dump(config, f)
    with open(folder / "results.json", "w") as f:
        json.dump(results, f)


def _default_config(**overrides) -> dict:
    config = {
        "split_type": "random",
        "load_weights": True,
        "excluded_bands": "",
        "output_hw": 32,
        "output_timesteps": 12,
        "patch_size": 16,
        "completed_at": "2026-03-24T00:00:00+00:00",
    }
    config.update(overrides)
    return config


def test_normalize_excluded_bands():
    assert _normalize_excluded_bands("") == ""
    assert _normalize_excluded_bands("S1") == "S1"
    assert _normalize_excluded_bands("NDVI,S2_RGB,S2_SWIR") == "NDVI,S2_RGB,S2_SWIR"
    assert _normalize_excluded_bands("S2_SWIR,NDVI,S2_RGB") == "NDVI,S2_RGB,S2_SWIR"


def test_load_experiments(tmp_path: Path):
    _write_experiment(tmp_path / "exp1", _default_config(), SAMPLE_RESULTS)
    _write_experiment(tmp_path / "exp2", _default_config(load_weights=False), SAMPLE_RESULTS)

    experiments = load_experiments(tmp_path)
    assert len(experiments) == 2
    configs = {exp.config.load_weights for exp in experiments}
    assert configs == {True, False}


def test_load_experiments_skips_missing_results(tmp_path: Path):
    folder = tmp_path / "incomplete"
    folder.mkdir()
    with open(folder / "experiment_config.json", "w") as f:
        json.dump(_default_config(), f)

    experiments = load_experiments(tmp_path)
    assert len(experiments) == 0


def test_experiment_config_properties():
    default = ExperimentConfig(
        split_type="random", load_weights=True, excluded_bands="", output_hw=32, output_timesteps=12, patch_size=16
    )
    assert default.is_default_shape
    assert not default.has_excluded_bands

    non_default = ExperimentConfig(
        split_type="spatial", load_weights=False, excluded_bands="S1", output_hw=16, output_timesteps=6, patch_size=8
    )
    assert not non_default.is_default_shape
    assert non_default.has_excluded_bands


def test_generate_table_1(tmp_path: Path):
    _write_experiment(tmp_path / "pretrained", _default_config(), SAMPLE_RESULTS)
    random_results = {**SAMPLE_RESULTS, "all": {"rmse": 23.61, "mae": 16.33, "r2_score": 0.57}}
    _write_experiment(tmp_path / "random", _default_config(load_weights=False), random_results)

    experiments = load_experiments(tmp_path)
    table = generate_table_1(experiments, "random")

    assert "Pretrained" in table
    assert "Random initialized" in table
    assert "Monthly predictions" in table
    assert "18.91" in table
    assert "23.61" in table
    assert "33.66" in table


def test_generate_table_5(tmp_path: Path):
    _write_experiment(tmp_path / "default", _default_config(), SAMPLE_RESULTS)
    small_results = {**SAMPLE_RESULTS, "all": {"rmse": 20.25, "mae": 13.46, "r2_score": 0.68}}
    _write_experiment(
        tmp_path / "small",
        _default_config(output_hw=1, output_timesteps=12, patch_size=1),
        small_results,
    )

    experiments = load_experiments(tmp_path)
    table = generate_table_5(experiments, "random")

    assert "| 32 | 12 | 16 |" in table
    assert "| 1 | 12 | 1 |" in table
    assert "20.25" in table


def test_generate_table_6(tmp_path: Path):
    _write_experiment(tmp_path / "none_pretrained", _default_config(), SAMPLE_RESULTS)
    _write_experiment(tmp_path / "none_random", _default_config(load_weights=False), SAMPLE_RESULTS)
    s1_results = {**SAMPLE_RESULTS, "all": {"rmse": 18.82, "mae": 13.10, "r2_score": 0.72}}
    _write_experiment(tmp_path / "s1_pretrained", _default_config(excluded_bands="S1"), s1_results)

    experiments = load_experiments(tmp_path)
    table = generate_table_6(experiments, "random")

    assert "None" in table
    assert "S1" in table
    assert "18.82" in table


def test_collect_results_both_splits(tmp_path: Path):
    _write_experiment(tmp_path / "random_pretrained", _default_config(), SAMPLE_RESULTS)
    _write_experiment(tmp_path / "spatial_pretrained", _default_config(split_type="spatial"), SAMPLE_RESULTS)

    output = collect_results(tmp_path)

    assert "# Results (random vs spatial)" in output
    assert "## Table 1: Overall Results" in output
    assert "## Table 2: Season Breakdown" in output
    assert "## Table 3: Land Cover Breakdown" in output
    assert "## Table 4: Elevation Breakdown" in output
    assert "## Table 5: Shape Ablations" in output
    assert "## Table 6: Data Ablations" in output
    assert "Random RMSE" in output
    assert "Spatial RMSE" in output


def test_load_experiments_deduplicates_by_completed_at(tmp_path: Path):
    old_results = {**SAMPLE_RESULTS, "all": {"rmse": 99.0, "mae": 99.0, "r2_score": 0.01}}
    _write_experiment(
        tmp_path / "old_run",
        _default_config(completed_at="2026-03-20T00:00:00+00:00"),
        old_results,
    )
    _write_experiment(
        tmp_path / "new_run",
        _default_config(completed_at="2026-03-24T12:00:00+00:00"),
        SAMPLE_RESULTS,
    )

    experiments = load_experiments(tmp_path)
    assert len(experiments) == 1
    assert experiments[0].results["all"]["rmse"] == 18.91
    assert experiments[0].path == tmp_path / "new_run"


def test_load_experiments_dedup_without_completed_at(tmp_path: Path):
    """Without completed_at, first experiment found wins (both have empty string)."""
    old_results = {**SAMPLE_RESULTS, "all": {"rmse": 99.0, "mae": 99.0, "r2_score": 0.01}}
    old_config = _default_config()
    del old_config["completed_at"]
    new_config = _default_config()
    del new_config["completed_at"]

    _write_experiment(tmp_path / "aaa", old_config, old_results)
    _write_experiment(tmp_path / "zzz", new_config, SAMPLE_RESULTS)

    experiments = load_experiments(tmp_path)
    assert len(experiments) == 1


def test_collect_results_empty_dir(tmp_path: Path):
    output = collect_results(tmp_path)
    assert output == "No experiments found."
