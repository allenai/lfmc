"""Launch all paper experiments on Beaker (both random and spatial splits)."""

import argparse
import signal
import time
from pathlib import Path

from beaker import Beaker

from lfmc.core.splits import SplitType

from .beaker_args import BeakerArgs, add_common_beaker_args, get_beaker_args
from .beaker_finetune import launch_experiment

SHAPE_ABLATIONS = [
    {"output_hw": 16, "num_timesteps": 12, "patch_size": 16},
    {"output_hw": 32, "num_timesteps": 6, "patch_size": 16},
    {"output_hw": 32, "num_timesteps": 3, "patch_size": 16},
    {"output_hw": 1, "num_timesteps": 12, "patch_size": 1},
    {"output_hw": 8, "num_timesteps": 12, "patch_size": 8},
]

DATA_ABLATIONS = [
    ["S1"],
    ["S2_RGB", "S2_Red_Edge", "S2_NIR_10m", "S2_NIR_20m", "S2_SWIR", "NDVI"],
    ["ERA5"],
    ["TC"],
    ["SRTM"],
    ["location"],
]

POLL_INTERVAL_SECONDS = 120
SIGTERM_EXIT_CODE = 128 + signal.SIGTERM


def build_experiment_configs() -> list[dict]:
    experiments: list[dict] = []

    for split_type in SplitType:
        for load_weights in [True, False]:
            experiments.append(
                {
                    "split_type": split_type,
                    "load_weights": load_weights,
                    "output_hw": 32,
                    "num_timesteps": 12,
                    "patch_size": 16,
                    "excluded_bands": frozenset(),
                }
            )

        for excluded in DATA_ABLATIONS:
            for load_weights in [True, False]:
                experiments.append(
                    {
                        "split_type": split_type,
                        "load_weights": load_weights,
                        "output_hw": 32,
                        "num_timesteps": 12,
                        "patch_size": 16,
                        "excluded_bands": frozenset(excluded),
                    }
                )

        for shape in SHAPE_ABLATIONS:
            experiments.append(
                {
                    "split_type": split_type,
                    "load_weights": True,
                    "excluded_bands": frozenset(),
                    **shape,
                }
            )

    return experiments


def experiment_label(exp: dict, index: int, total: int) -> str:
    split = exp["split_type"]
    weights = "pretrained" if exp["load_weights"] else "random"
    excluded = ",".join(sorted(exp["excluded_bands"])) or "none"
    shape = f"{exp['output_hw']}hw_{exp['num_timesteps']}ts_{exp['patch_size']}ps"
    return f"[{index}/{total}] {split} {weights} {shape} excluded={excluded}"


def wait_for_all(experiment_ids: dict[str, str], workspace: str) -> None:
    """Poll Beaker until all experiments are finalized (succeeded or failed)."""
    beaker = Beaker.from_env(default_workspace=workspace)
    pending = dict(experiment_ids)

    print(f"\nWaiting for {len(pending)} experiments to finish...")
    while pending:
        time.sleep(POLL_INTERVAL_SECONDS)
        still_pending = {}
        for exp_id, label in pending.items():
            exp = beaker.experiment.get(exp_id)
            jobs = exp.jobs
            if not jobs:
                still_pending[exp_id] = label
                continue
            latest_job = jobs[-1]
            status = latest_job.status
            if status.finalized is None:
                still_pending[exp_id] = label
            elif status.exit_code == SIGTERM_EXIT_CODE:
                print(f"  Preempted: {label} -- waiting for auto-resume...")
                still_pending[exp_id] = label
            else:
                result = "succeeded" if status.exit_code == 0 else f"failed (exit_code={status.exit_code})"
                print(f"  Finished: {label} -- {result}")
        pending = still_pending
        if pending:
            print(f"  {len(pending)} experiments still running...")

    print("\nAll experiments finished.")


def launch_all(
    beaker_args: BeakerArgs,
    model_name: str,
    data_folder: Path,
    h5py_folder: Path,
    h5pys_only: bool,
    dry_run: bool = False,
    wait: bool = False,
) -> None:
    experiments = build_experiment_configs()
    print(f"Total experiments: {len(experiments)}")

    launched: dict[str, str] = {}
    for i, exp in enumerate(experiments, 1):
        label = experiment_label(exp, i, len(experiments))

        if dry_run:
            print(f"  DRY RUN: {label}")
            continue

        print(f"  Launching: {label}")
        exp_id = launch_experiment(
            beaker_args=beaker_args,
            model_name=model_name,
            data_folder=data_folder,
            h5py_folder=h5py_folder,
            h5pys_only=h5pys_only,
            output_hw=exp["output_hw"],
            num_timesteps=exp["num_timesteps"],
            patch_size=exp["patch_size"],
            load_weights=exp["load_weights"],
            split_type=exp["split_type"],
            validation_state_regions=None,
            test_state_regions=None,
            excluded_bands=exp["excluded_bands"],
        )
        launched[exp_id] = label

    if wait and launched:
        wait_for_all(launched, beaker_args.workspace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch all paper experiments on Beaker")
    add_common_beaker_args(parser)
    parser.add_argument("--model-name", choices={"base", "nano", "tiny"}, required=True)
    parser.add_argument("--data-folder", type=Path, required=True)
    parser.add_argument("--h5py-folder", type=Path, required=True)
    parser.add_argument("--h5pys-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true", help="Print experiments without launching")
    args = parser.parse_args()
    beaker_args = get_beaker_args(args)
    wait = beaker_args.wait
    beaker_args.wait = False

    launch_all(
        beaker_args=beaker_args,
        model_name=args.model_name,
        data_folder=args.data_folder,
        h5py_folder=args.h5py_folder,
        h5pys_only=args.h5pys_only,
        dry_run=args.dry_run,
        wait=wait,
    )
