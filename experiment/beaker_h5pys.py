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
    data_folder: Path,
    h5py_folder: Path,
) -> None:
    """Launch experiment for LFMC H5py creation on Beaker.

    Args:
        beaker_args: The Beaker arguments
        data_folder: The folder containing the training data
        h5py_folder: The folder containing the H5py files
    """
    beaker = Beaker.from_env(default_workspace=beaker_args.workspace)
    weka_path = PurePath("/weka")

    task_name = "lfmc_h5pys"
    with beaker.session():
        arguments = [
            "--config-dir=/stage/data/config",
            f"--data-folder={str(weka_path / data_folder.relative_to('/'))}",
            f"--h5py-folder={str(weka_path / h5py_folder.relative_to('/'))}",
        ]

        spec = ExperimentSpec.new(
            task_name=task_name,
            beaker_image=beaker_args.image_name,
            budget=beaker_args.budget,
            priority=beaker_args.priority,
            command=["create-h5pys"],
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
    parser = argparse.ArgumentParser(description="Launch Beaker experiment for LFMC H5py creation")
    add_common_beaker_args(parser)
    parser.add_argument(
        "--data-folder",
        type=Path,
        help="The folder containing the training data",
    )
    parser.add_argument(
        "--h5py-folder",
        type=Path,
        help="The folder containing the H5py files",
    )
    args = parser.parse_args()
    beaker_args = get_beaker_args(args)

    launch_experiment(
        beaker_args=beaker_args,
        data_folder=args.data_folder,
        h5py_folder=args.h5py_folder,
    )
