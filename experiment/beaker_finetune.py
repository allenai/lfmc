import argparse
import uuid
from pathlib import Path, PurePath

from beaker import (
    Beaker,
    Constraints,
    DataMount,
    DataSource,
    ExperimentSpec,
    TaskResources,
)
from beaker.services.experiment import ExperimentClient

from .beaker_args import BeakerArgs, add_common_beaker_args, get_beaker_args


def launch_experiment(
    beaker_args: BeakerArgs,
    model_name: str,
    data_folder: Path,
    h5py_folder: Path,
    h5pys_only: bool,
    output_hw: int,
    patch_size: int,
    load_weights: bool,
) -> None:
    """Launch experiment for LFMC model finetuning on Beaker.

    Args:
        beaker_args: The Beaker arguments
        model_name: The name of the pretrained model
        data_folder: The folder containing the training data
        h5py_folder: The folder containing the H5py files
        h5pys_only: Whether to only use H5pys, not TIFs
        output_hw: The output height and width
        patch_size: The patch size
        load_weights: Whether to load the weights
    """
    beaker = Beaker.from_env(default_workspace=beaker_args.workspace)
    weka_path = PurePath("/weka")

    task_name = "lfmc_finetune"
    with beaker.session():
        arguments = [
            "--output-folder=/output",
            "--config-dir=/stage/data/config",
            f"--data-folder={str(weka_path / data_folder.relative_to('/'))}",
            f"--h5py-folder={str(weka_path / h5py_folder.relative_to('/'))}",
            "--pretrained-models-folder=/stage/data/models",
            f"--pretrained-model-name={model_name}",
            f"--output-hw={output_hw}",
            f"--patch-size={patch_size}",
            "--load-weights" if load_weights else "--no-load-weights",
        ]
        if h5pys_only:
            arguments.append("--h5pys-only")

        spec = ExperimentSpec.new(
            task_name=task_name,
            beaker_image=beaker_args.image_name,
            budget=beaker_args.budget,
            priority=beaker_args.priority,
            command=["finetune-model"],
            arguments=arguments,
            constraints=Constraints(cluster=beaker_args.clusters),
            preemptible=True,
            datasets=[
                DataMount(
                    source=DataSource(weka=beaker_args.weka_bucket),
                    mount_path=str(weka_path),
                ),
            ],
            resources=TaskResources(gpu_count=beaker_args.gpu_count),
            result_path="/output",
        )
        unique_id = str(uuid.uuid4())[0:8]
        experiment = beaker.experiment.create(f"{task_name}_{unique_id}", spec)

        experiment_client = ExperimentClient(beaker)
        print(f"Experiment created: {experiment.id}: {experiment_client.url(experiment)}")
        if beaker_args.wait:
            print(f"Waiting for experiment {experiment.id} to finish")
            experiment_client.wait_for(experiment.id)
            print(f"Experiment {experiment.id} finished")
            result_dataset = experiment_client.results(experiment.id)
            if result_dataset is not None:
                print(f"Result dataset: {result_dataset.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Beaker experiment for LFMC model finetuning")
    add_common_beaker_args(parser)
    parser.add_argument(
        "--model-name",
        choices=set(["base", "nano", "tiny"]),
        help="The name of the pretrained model",
        required=True,
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        help="The folder containing the training data",
        required=True,
    )
    parser.add_argument(
        "--h5py-folder",
        type=Path,
        help="The folder containing the H5py files",
        required=True,
    )
    parser.add_argument(
        "--h5pys-only",
        action=argparse.BooleanOptionalAction,
        help="Only use H5pys, not TIFs",
        default=False,
    )
    parser.add_argument(
        "--output-hw",
        type=int,
        help="The output height and width",
        default=32,
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        help="The patch size",
        default=16,
    )
    parser.add_argument(
        "--load-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to load the weights",
    )
    args = parser.parse_args()
    beaker_args = get_beaker_args(args)

    launch_experiment(
        beaker_args=beaker_args,
        data_folder=args.data_folder,
        h5py_folder=args.h5py_folder,
        h5pys_only=args.h5pys_only,
        model_name=args.model_name,
        output_hw=args.output_hw,
        patch_size=args.patch_size,
        load_weights=args.load_weights,
    )
