import argparse
from dataclasses import dataclass

from beaker import Priority


def add_common_beaker_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--budget",
        type=str,
        required=True,
        help="The budget to use for the experiment",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Which workspace to run the experiment in",
    )
    parser.add_argument(
        "--weka-bucket",
        type=str,
        required=True,
        help="The WEKA bucket to use for the experiment",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        required=True,
        help="The image name to use for the experiment",
    )
    parser.add_argument(
        "--clusters",
        type=str,
        nargs="+",
        help="The clusters where the experiment can be run",
    )
    parser.add_argument(
        "--priority",
        type=str,
        help="The priority to use for the experiment",
        default=Priority.normal,
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        help="Number of GPUs",
        default=1,
    )
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        help="Whether to wait for the experiment to finish before exiting",
    )


@dataclass
class BeakerArgs:
    """Common Beaker arguments.

    Args:
        budget: The budget to use for the experiment.
        workspace: The Beaker workspace to run the job in.
        weka_bucket: The WEKA bucket to use for the experiment.
        image_name: The image name to use for the experiment.
        clusters: The clusters where the experiment can be run.
        priority: The priority to use for the experiment.
        gpu_count: The number of GPUs to use.
        wait: Whether to wait for the experiment to finish before exiting.
    """

    budget: str
    workspace: str
    weka_bucket: str
    image_name: str
    clusters: list[str]
    priority: Priority
    gpu_count: int
    wait: bool


def get_beaker_args(args: argparse.Namespace) -> BeakerArgs:
    return BeakerArgs(
        budget=args.budget,
        workspace=args.workspace,
        weka_bucket=args.weka_bucket,
        image_name=args.image_name,
        clusters=args.clusters,
        priority=args.priority,
        gpu_count=args.gpu_count,
        wait=args.wait,
    )
