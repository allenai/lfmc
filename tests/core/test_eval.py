from pathlib import Path

import pandas as pd

from galileo.data.dataset import Normalizer
from galileo.galileo import Encoder
from lfmc.core.const import MeteorologicalSeason, WorldCoverClass
from lfmc.core.eval import FinetuningConfig, LFMCEval, finetune_and_evaluate


def test_finetune_and_test(
    tmp_path: Path,
    normalizer: Normalizer,
    encoder: Encoder,
    data_folder: Path,
    h5py_folder: Path,
):
    output_folder = tmp_path / "finetuned"
    output_folder.mkdir(parents=True, exist_ok=True)
    lfmc_eval = LFMCEval(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
        validation_folds=frozenset([0]),
        test_folds=frozenset([1]),
    )
    finetuning_config = FinetuningConfig(
        max_epochs=1,
        weight_decay=0.001,
        learning_rate=0.001,
        batch_size=16,
        patience=5,
    )
    finetuned_model = lfmc_eval.finetune(
        pretrained_model=encoder,
        output_folder=output_folder,
        finetuning_config=finetuning_config,
    )
    assert finetuned_model is not None
    assert finetuned_model.encoder is not None
    assert finetuned_model.head is not None

    labels, preds, _, _ = lfmc_eval.test(
        name="test",
        finetuned_model=finetuned_model,
        filter=None,
    )
    assert labels is not None
    assert preds is not None
    assert labels.shape == preds.shape
    assert 0 < labels.shape[0] <= len(list(data_folder.glob("*.tif")))
    assert 0 < preds.shape[0] <= len(list(data_folder.glob("*.tif")))


def test_finetune_and_evaluate(
    tmp_path: Path,
    normalizer: Normalizer,
    encoder: Encoder,
    data_folder: Path,
    h5py_folder: Path,
):
    output_folder = tmp_path / "results"
    output_folder.mkdir(parents=True, exist_ok=True)
    results, df = finetune_and_evaluate(
        normalizer=normalizer,
        pretrained_model=encoder,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        output_folder=output_folder,
        validation_folds=frozenset([0]),
        test_folds=frozenset([1]),
    )
    assert results is not None
    assert isinstance(results, dict)
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    filter_names = [
        "all",
        # Not all keys are present in the results due to the small size of the dataset
        MeteorologicalSeason.SUMMER,
        MeteorologicalSeason.AUTUMN,
        WorldCoverClass.TREE_COVER,
        WorldCoverClass.GRASSLAND,
        "elevation_1500_2000",
        "elevation_2000_2500",
        "high_fire_danger",
        "non_high_fire_danger",
    ]

    for filter_name in filter_names:
        assert filter_name in results
        assert "r2_score" in results[filter_name]
        assert "mae" in results[filter_name]
        assert "rmse" in results[filter_name]
        assert isinstance(results[filter_name]["r2_score"], float)
        assert isinstance(results[filter_name]["mae"], float)
        assert isinstance(results[filter_name]["rmse"], float)

    assert 0 <= df.shape[0] <= len(list(data_folder.glob("*.tif")))
    assert "latitude" in df.columns
    assert "longitude" in df.columns
    assert "label" in df.columns
    assert "prediction" in df.columns
