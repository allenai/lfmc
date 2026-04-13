import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_excluded_bands(bands: str) -> str:
    if not bands:
        return ""
    normalized = {band.strip() for band in bands.split(",") if band.strip()}
    return ",".join(sorted(normalized))


@dataclass(frozen=True)
class ExperimentConfig:
    split_type: str
    load_weights: bool
    excluded_bands: str
    output_hw: int
    output_timesteps: int
    patch_size: int

    @property
    def is_default_shape(self) -> bool:
        return self.output_hw == 32 and self.output_timesteps == 12 and self.patch_size == 16

    @property
    def has_excluded_bands(self) -> bool:
        return self.excluded_bands != ""


ResultsDict = dict[str, dict[str, float]]


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ResultsDict
    path: Path
    completed_at: str


EXCLUDED_BANDS_DISPLAY: dict[str, str] = {
    _normalize_excluded_bands(""): "None",
    _normalize_excluded_bands("S1"): "S1",
    _normalize_excluded_bands("S2_RGB,S2_Red_Edge,S2_NIR_10m,S2_NIR_20m,S2_SWIR,NDVI"): "S2",
    _normalize_excluded_bands("ERA5"): "ERA5",
    _normalize_excluded_bands("TC"): "TC",
    _normalize_excluded_bands("SRTM"): "SRTM",
    _normalize_excluded_bands("location"): "loc.",
}


def _metrics_row(results: ResultsDict, key: str) -> str:
    if key not in results:
        return "- | - | -"
    m = results[key]
    return f"{m['rmse']:.2f} | {m['mae']:.2f} | {m['r2_score']:.2f}"


def _find_experiment(
    experiments: list[Experiment],
    split_type: str,
    load_weights: bool,
    excluded_bands: str = "",
    output_hw: int = 32,
    output_timesteps: int = 12,
    patch_size: int = 16,
) -> Experiment | None:
    for exp in experiments:
        if (
            exp.config.split_type == split_type
            and exp.config.load_weights == load_weights
            and exp.config.excluded_bands == excluded_bands
            and exp.config.output_hw == output_hw
            and exp.config.output_timesteps == output_timesteps
            and exp.config.patch_size == patch_size
        ):
            return exp
    return None


def load_experiments(results_dir: Path) -> list[Experiment]:
    by_config: dict[ExperimentConfig, Experiment] = {}
    for config_path in sorted(results_dir.rglob("experiment_config.json")):
        folder = config_path.parent
        results_path = folder / "results.json"
        if not results_path.exists():
            logger.warning("Skipping %s: no results.json found", folder)
            continue
        with open(config_path) as f:
            config_data = json.load(f)
        with open(results_path) as f:
            results_data = json.load(f)
        config = ExperimentConfig(
            split_type=config_data["split_type"],
            load_weights=config_data["load_weights"],
            excluded_bands=_normalize_excluded_bands(config_data.get("excluded_bands", "")),
            output_hw=config_data["output_hw"],
            output_timesteps=config_data["output_timesteps"],
            patch_size=config_data["patch_size"],
        )
        completed_at = config_data.get("completed_at", "")
        experiment = Experiment(config=config, results=results_data, path=folder, completed_at=completed_at)
        if config in by_config:
            existing = by_config[config]
            if experiment.completed_at > existing.completed_at:
                logger.info("Duplicate config, keeping newer: %s over %s", folder, existing.path)
                by_config[config] = experiment
            else:
                logger.info("Duplicate config, keeping newer: %s over %s", existing.path, folder)
        else:
            by_config[config] = experiment
    experiments = list(by_config.values())
    logger.info("Loaded %d experiments from %s", len(experiments), results_dir)
    return experiments


def generate_table_1(experiments: list[Experiment], split_type: str) -> str:
    """Pretrained vs Random initialized vs Monthly predictions baseline."""
    pretrained = _find_experiment(experiments, split_type, load_weights=True)
    random_init = _find_experiment(experiments, split_type, load_weights=False)

    lines = [
        "| Category | RMSE | MAE | R2 |",
        "| --- | --- | --- | --- |",
    ]
    if pretrained:
        lines.append(f"| Pretrained | {_metrics_row(pretrained.results, 'all')} |")
    if random_init:
        lines.append(f"| Random initialized | {_metrics_row(random_init.results, 'all')} |")
    baseline_exp = pretrained or random_init
    if baseline_exp:
        lines.append(f"| Monthly predictions | {_metrics_row(baseline_exp.results, 'baseline')} |")

    return "\n".join(lines)


def generate_table_2(experiments: list[Experiment], split_type: str) -> str:
    """Season breakdown from the default pretrained experiment."""
    pretrained = _find_experiment(experiments, split_type, load_weights=True)
    if not pretrained:
        return "*No default pretrained experiment found.*"

    keys_display = [
        ("all", "Overall"),
        ("Winter", "Winter season"),
        ("Spring", "Spring season"),
        ("Summer", "Summer season"),
        ("Autumn", "Autumn season"),
    ]
    lines = [
        "| Season | RMSE | MAE | R2 |",
        "| --- | --- | --- | --- |",
    ]
    for key, display in keys_display:
        lines.append(f"| {display} | {_metrics_row(pretrained.results, key)} |")
    return "\n".join(lines)


def generate_table_3(experiments: list[Experiment], split_type: str) -> str:
    """Land cover breakdown from the default pretrained experiment."""
    pretrained = _find_experiment(experiments, split_type, load_weights=True)
    if not pretrained:
        return "*No default pretrained experiment found.*"

    keys_display = [
        ("all", "Overall"),
        ("Tree cover", "Trees"),
        ("Grassland", "Grass"),
        ("Shrubland", "Shrub"),
        ("Built-up", "Built-up"),
        ("Bare / sparse vegetation", "Bare / Sparse"),
    ]
    lines = [
        "| Land Cover Class | RMSE | MAE | R2 |",
        "| --- | --- | --- | --- |",
    ]
    for key, display in keys_display:
        lines.append(f"| {display} | {_metrics_row(pretrained.results, key)} |")
    return "\n".join(lines)


def generate_table_4(experiments: list[Experiment], split_type: str) -> str:
    """Elevation breakdown from the default pretrained experiment."""
    pretrained = _find_experiment(experiments, split_type, load_weights=True)
    if not pretrained:
        return "*No default pretrained experiment found.*"

    keys_display = [
        ("all", "Overall"),
        ("elevation_0_500", "Elevation: 0-500m"),
        ("elevation_500_1000", "Elevation: 500-1000m"),
        ("elevation_1000_1500", "Elevation: 1000-1500m"),
        ("elevation_1500_2000", "Elevation: 1500-2000m"),
        ("elevation_2000_2500", "Elevation: 2000-2500m"),
        ("elevation_2500_3000", "Elevation: 2500-3000m"),
        ("elevation_3000_3500", "Elevation: 3000-3500m"),
    ]
    lines = [
        "| Category | RMSE | MAE | R2 |",
        "| --- | --- | --- | --- |",
    ]
    for key, display in keys_display:
        lines.append(f"| {display} | {_metrics_row(pretrained.results, key)} |")
    return "\n".join(lines)


def generate_table_5(experiments: list[Experiment], split_type: str) -> str:
    """Shape ablations -- pretrained only, varying output_hw/output_timesteps/patch_size."""
    shape_experiments = [
        exp
        for exp in experiments
        if exp.config.split_type == split_type and exp.config.load_weights and not exp.config.has_excluded_bands
    ]

    def sort_key(exp: Experiment) -> tuple[int, int, int, int]:
        c = exp.config
        is_default = int(not c.is_default_shape)
        return (is_default, -c.output_hw, -c.output_timesteps, -c.patch_size)

    shape_experiments.sort(key=sort_key)

    if not shape_experiments:
        return "*No shape experiments found.*"

    lines = [
        "| H, W | T | P | RMSE | MAE | R2 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for exp in shape_experiments:
        c = exp.config
        lines.append(f"| {c.output_hw} | {c.output_timesteps} | {c.patch_size} | {_metrics_row(exp.results, 'all')} |")
    return "\n".join(lines)


def generate_table_6(experiments: list[Experiment], split_type: str) -> str:
    """Data ablations -- pretrained and random, each with one input removed."""
    data_experiments: dict[str, dict[bool, Experiment]] = {}
    for exp in experiments:
        if exp.config.split_type == split_type and exp.config.is_default_shape:
            excluded = exp.config.excluded_bands
            if excluded not in data_experiments:
                data_experiments[excluded] = {}
            data_experiments[excluded][exp.config.load_weights] = exp

    ordered_keys = [""]
    for key in sorted(data_experiments.keys()):
        if key != "":
            ordered_keys.append(key)

    lines = [
        "| Excluded | PT RMSE | PT MAE | PT R2 | RI RMSE | RI MAE | RI R2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for excluded_key in ordered_keys:
        if excluded_key not in data_experiments:
            continue
        display_name = EXCLUDED_BANDS_DISPLAY.get(excluded_key, excluded_key)
        exps = data_experiments[excluded_key]
        pretrained = exps.get(True)
        random_init = exps.get(False)
        pretrained_metrics = _metrics_row(pretrained.results, "all") if pretrained else "- | - | -"
        random_metrics = _metrics_row(random_init.results, "all") if random_init else "- | - | -"
        lines.append(f"| {display_name} | {pretrained_metrics} | {random_metrics} |")

    return "\n".join(lines)


TABLE_GENERATORS: list[tuple[str, Callable[[list[Experiment], str], str]]] = [
    ("Table 1: Overall Results", generate_table_1),
    ("Table 2: Season Breakdown", generate_table_2),
    ("Table 3: Land Cover Breakdown", generate_table_3),
    ("Table 4: Elevation Breakdown", generate_table_4),
    ("Table 5: Shape Ablations", generate_table_5),
    ("Table 6: Data Ablations", generate_table_6),
]


def _combined_breakdown(
    experiments: list[Experiment],
    keys_display: list[tuple[str, str]],
    header_label: str,
) -> str:
    """Side-by-side random vs spatial for a single pretrained breakdown."""
    random_exp = _find_experiment(experiments, "random", load_weights=True)
    spatial_exp = _find_experiment(experiments, "spatial", load_weights=True)
    if not random_exp and not spatial_exp:
        return "*No experiments found.*"

    lines = [
        f"| {header_label} | Random RMSE | Random MAE | Random R2 | Spatial RMSE | Spatial MAE | Spatial R2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, display in keys_display:
        rand = _metrics_row(random_exp.results, key) if random_exp else "- | - | -"
        spat = _metrics_row(spatial_exp.results, key) if spatial_exp else "- | - | -"
        lines.append(f"| {display} | {rand} | {spat} |")
    return "\n".join(lines)


def generate_combined_table_1(experiments: list[Experiment]) -> str:
    random_pt = _find_experiment(experiments, "random", load_weights=True)
    random_ri = _find_experiment(experiments, "random", load_weights=False)
    spatial_pt = _find_experiment(experiments, "spatial", load_weights=True)
    spatial_ri = _find_experiment(experiments, "spatial", load_weights=False)

    lines = [
        "| Category | Random RMSE | Random MAE | Random R2 | Spatial RMSE | Spatial MAE | Spatial R2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    rand_pt = _metrics_row(random_pt.results, "all") if random_pt else "- | - | -"
    spat_pt = _metrics_row(spatial_pt.results, "all") if spatial_pt else "- | - | -"
    lines.append(f"| Pretrained | {rand_pt} | {spat_pt} |")

    rand_ri = _metrics_row(random_ri.results, "all") if random_ri else "- | - | -"
    spat_ri = _metrics_row(spatial_ri.results, "all") if spatial_ri else "- | - | -"
    lines.append(f"| Random init | {rand_ri} | {spat_ri} |")

    baseline_exp_r = random_pt or random_ri
    baseline_exp_s = spatial_pt or spatial_ri
    rand_bl = _metrics_row(baseline_exp_r.results, "baseline") if baseline_exp_r else "- | - | -"
    spat_bl = _metrics_row(baseline_exp_s.results, "baseline") if baseline_exp_s else "- | - | -"
    lines.append(f"| Monthly pred | {rand_bl} | {spat_bl} |")

    return "\n".join(lines)


def generate_combined_table_2(experiments: list[Experiment]) -> str:
    return _combined_breakdown(
        experiments,
        [
            ("all", "Overall"),
            ("Winter", "Winter"),
            ("Spring", "Spring"),
            ("Summer", "Summer"),
            ("Autumn", "Autumn"),
        ],
        "Season",
    )


def generate_combined_table_3(experiments: list[Experiment]) -> str:
    table = _combined_breakdown(
        experiments,
        [
            ("all", "Overall"),
            ("Tree cover", "Trees"),
            ("Grassland", "Grass"),
            ("Shrubland", "Shrub"),
            ("Built-up", "Built-up"),
            ("Bare / sparse vegetation", "Bare / Sparse"),
        ],
        "Land Cover",
    )
    table += (
        "\n\nBuilt-up and Bare/Sparse have limited site diversity"
        " (21 and 15 sites respectively), which may reduce the"
        " reliability of spatial split metrics for these classes."
    )
    return table


def generate_combined_table_4(experiments: list[Experiment]) -> str:
    return _combined_breakdown(
        experiments,
        [
            ("all", "Overall"),
            ("elevation_0_500", "0-500m"),
            ("elevation_500_1000", "500-1000m"),
            ("elevation_1000_1500", "1000-1500m"),
            ("elevation_1500_2000", "1500-2000m"),
            ("elevation_2000_2500", "2000-2500m"),
            ("elevation_2500_3000", "2500-3000m"),
            ("elevation_3000_3500", "3000-3500m"),
        ],
        "Elevation",
    )


def generate_combined_table_5(experiments: list[Experiment]) -> str:
    shape_configs: list[tuple[int, int, int]] = []
    for exp in experiments:
        c = exp.config
        if c.load_weights and not c.has_excluded_bands:
            key = (c.output_hw, c.output_timesteps, c.patch_size)
            if key not in shape_configs:
                shape_configs.append(key)

    def sort_key(k: tuple[int, int, int]) -> tuple[int, int, int, int]:
        is_default = int(not (k[0] == 32 and k[1] == 12 and k[2] == 16))
        return (is_default, -k[0], -k[1], -k[2])

    shape_configs.sort(key=sort_key)

    if not shape_configs:
        return "*No shape experiments found.*"

    lines = [
        "| H, W | T | P | Random RMSE | Random MAE | Random R2 | Spatial RMSE | Spatial MAE | Spatial R2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for hw, ts, ps in shape_configs:
        rand = _find_experiment(
            experiments, "random", load_weights=True, output_hw=hw, output_timesteps=ts, patch_size=ps
        )
        spat = _find_experiment(
            experiments, "spatial", load_weights=True, output_hw=hw, output_timesteps=ts, patch_size=ps
        )
        r = _metrics_row(rand.results, "all") if rand else "- | - | -"
        s = _metrics_row(spat.results, "all") if spat else "- | - | -"
        lines.append(f"| {hw} | {ts} | {ps} | {r} | {s} |")
    return "\n".join(lines)


def generate_combined_table_6(experiments: list[Experiment]) -> str:
    excluded_keys: list[str] = [""]
    seen = {""}
    for exp in experiments:
        c = exp.config
        if c.is_default_shape and c.excluded_bands not in seen:
            seen.add(c.excluded_bands)
            excluded_keys.append(c.excluded_bands)

    lines = [
        "| Excluded | Rnd PT RMSE | Rnd PT MAE | Rnd PT R2 | Rnd RI RMSE | Rnd RI MAE | Rnd RI R2 | Spt PT RMSE | Spt PT MAE | Spt PT R2 | Spt RI RMSE | Spt RI MAE | Spt RI R2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for excluded_key in excluded_keys:
        display = EXCLUDED_BANDS_DISPLAY.get(excluded_key, excluded_key)
        rpt = _find_experiment(experiments, "random", True, excluded_bands=excluded_key)
        rri = _find_experiment(experiments, "random", False, excluded_bands=excluded_key)
        spt = _find_experiment(experiments, "spatial", True, excluded_bands=excluded_key)
        sri = _find_experiment(experiments, "spatial", False, excluded_bands=excluded_key)
        lines.append(
            f"| {display} | {_metrics_row(rpt.results, 'all') if rpt else '- | - | -'}"
            f" | {_metrics_row(rri.results, 'all') if rri else '- | - | -'}"
            f" | {_metrics_row(spt.results, 'all') if spt else '- | - | -'}"
            f" | {_metrics_row(sri.results, 'all') if sri else '- | - | -'} |"
        )
    lines.append("")
    lines.append("Rnd = Random split, Spt = Spatial split, PT = Pretrained, RI = Random Initialized")
    return "\n".join(lines)


COMBINED_TABLE_GENERATORS: list[tuple[str, Callable[[list[Experiment]], str]]] = [
    ("Table 1: Overall Results", generate_combined_table_1),
    ("Table 2: Season Breakdown", generate_combined_table_2),
    ("Table 3: Land Cover Breakdown", generate_combined_table_3),
    ("Table 4: Elevation Breakdown", generate_combined_table_4),
    ("Table 5: Shape Ablations", generate_combined_table_5),
    ("Table 6: Data Ablations", generate_combined_table_6),
]


def collect_results(results_dir: Path) -> str:
    experiments = load_experiments(results_dir)
    if not experiments:
        return "No experiments found."

    split_types = sorted({exp.config.split_type for exp in experiments})
    timestamps = [exp.completed_at for exp in experiments if exp.completed_at]
    output_parts: list[str] = []

    if timestamps:
        earliest = min(timestamps)[:10]
        latest = max(timestamps)[:10]
        if earliest == latest:
            output_parts.append(f"*Experiments completed: {latest}*\n")
        else:
            output_parts.append(f"*Experiments completed: {earliest} to {latest}*\n")

    if "random" in split_types:
        output_parts.append(
            "**Note:** Random split results differ slightly from the published paper"
            " ([arXiv:2506.20132](https://arxiv.org/abs/2506.20132))."
            " In particular, the random initialized baseline improved (RMSE 23.61 to 21.88),"
            " narrowing the gap with the pretrained model from ~20% to ~12%."
            " The exact cause is unknown but may include differences in random seeds,"
            " library versions, or a fix to checkpoint resume logic.\n"
        )

    if len(split_types) > 1:
        output_parts.append("# Results (random vs spatial)\n")
        for title, generator in COMBINED_TABLE_GENERATORS:
            output_parts.append(f"## {title}\n")
            output_parts.append(generator(experiments))
            output_parts.append("")
    else:
        for split_type in split_types:
            output_parts.append(f"# Results: {split_type} split\n")
            for title, table_gen in TABLE_GENERATORS:
                output_parts.append(f"## {title}\n")
                output_parts.append(table_gen(experiments, split_type))
                output_parts.append("")

    return "\n".join(output_parts)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser("Collect experiment results and generate paper tables")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing experiment result subfolders",
    )
    args = parser.parse_args()
    print(collect_results(args.results_dir))


if __name__ == "__main__":
    main()
