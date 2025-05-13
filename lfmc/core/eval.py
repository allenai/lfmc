from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from frozenlist import FrozenList
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from galileo.data.dataset import Normalizer
from galileo.galileo import Encoder
from galileo.utils import device
from lfmc.core.bands import SPACE_BANDS, SPACE_TIME_BANDS, STATIC_BANDS, TIME_BANDS
from lfmc.core.const import MAX_LFMC_VALUE, MeteorologicalSeason, WorldCoverClass
from lfmc.core.dataset import LFMCDataset
from lfmc.core.filter import Filter
from lfmc.core.finetuning import DEFAULT_FINETUNING_CONFIG, FinetuningConfig, FineTuningModel
from lfmc.core.hyperparameters import DEFAULT_HYPERPARAMETERS, HyperParameters
from lfmc.core.mode import Mode

logger = getLogger(__name__)

ResultsDict = dict[str, dict[str, float]]


@dataclass
class FinetuningState:
    epoch: int
    best_model_dict: dict | None
    best_loss: torch.Tensor | None
    epochs_since_improvement: int


class LFMCEval:
    def __init__(
        self,
        normalizer: Normalizer,
        data_folder: Path,
        h5py_folder: Path,
        h5pys_only: bool = False,
        output_hw: int = 32,
        output_timesteps: int = 12,
        patch_size: int = 16,
        validation_folds: frozenset[int] | None = None,
        test_folds: frozenset[int] | None = None,
        validation_state_regions: frozenset[str] | None = None,
        test_state_regions: frozenset[str] | None = None,
        excluded_bands: frozenset[str] = frozenset(),
    ):
        if validation_folds is None and validation_state_regions is None:
            raise ValueError("validation_folds or validation_state_regions must be provided")
        if validation_folds is not None and validation_state_regions is not None:
            raise ValueError("validation_folds and validation_state_regions cannot both be provided")
        if test_folds is None and test_state_regions is None:
            raise ValueError("test_folds or test_state_regions must be provided")
        if test_folds is not None and test_state_regions is not None:
            raise ValueError("test_folds and test_state_regions cannot both be provided")
        self.normalizer = normalizer
        self.data_folder = data_folder
        self.h5py_folder = h5py_folder
        self.h5pys_only = h5pys_only
        self.output_hw = output_hw
        self.output_timesteps = output_timesteps
        self.patch_size = patch_size
        self.validation_folds = validation_folds
        self.test_folds = test_folds
        self.validation_state_regions = validation_state_regions
        self.test_state_regions = test_state_regions
        self.space_time_bands = FrozenList(SPACE_TIME_BANDS - excluded_bands)
        self.space_bands = FrozenList(SPACE_BANDS - excluded_bands)
        self.time_bands = FrozenList(TIME_BANDS - excluded_bands)
        self.static_bands = FrozenList(STATIC_BANDS - excluded_bands)

    @classmethod
    def _new_finetuning_model(cls, model: Encoder) -> FineTuningModel:
        num_classes = 1
        head = nn.Linear(model.embedding_size, num_classes)
        finetuning_model = FineTuningModel(model, head).to(device)
        finetuning_model.train()
        return finetuning_model

    def _save_checkpoint(
        self,
        checkpoint_folder: Path,
        model: FineTuningModel,
        optimizer: torch.optim.Optimizer,
        state: FinetuningState,
    ):
        checkpoint_dict = {
            "epoch": state.epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_model_dict": state.best_model_dict,
            "best_loss": state.best_loss,
            "epochs_since_improvement": state.epochs_since_improvement,
        }
        torch.save(checkpoint_dict, checkpoint_folder / "checkpoint.pth")

    def _load_checkpoint(
        self,
        checkpoint_folder: Path,
        model: FineTuningModel,
        optimizer: torch.optim.Optimizer,
        state: FinetuningState,
    ) -> tuple[FineTuningModel, torch.optim.Optimizer, FinetuningState]:
        checkpoint_path = checkpoint_folder / "checkpoint.pth"
        if not checkpoint_path.exists():
            return model, optimizer, state
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return (
            model,
            optimizer,
            FinetuningState(
                epoch=checkpoint["epoch"],
                best_model_dict=checkpoint["best_model_dict"],
                best_loss=checkpoint["best_loss"],
                epochs_since_improvement=checkpoint["epochs_since_improvement"],
            ),
        )

    def _create_dataset(self, mode: Mode, filter: Filter | None = None) -> LFMCDataset:
        return LFMCDataset(
            normalizer=self.normalizer,
            data_folder=self.data_folder,
            h5py_folder=self.h5py_folder,
            h5pys_only=self.h5pys_only,
            output_hw=self.output_hw,
            output_timesteps=self.output_timesteps,
            mode=mode,
            validation_folds=self.validation_folds,
            test_folds=self.test_folds,
            validation_state_regions=self.validation_state_regions,
            test_state_regions=self.test_state_regions,
            filter=filter,
            space_time_bands=self.space_time_bands,
            space_bands=self.space_bands,
            time_bands=self.time_bands,
            static_bands=self.static_bands,
        )

    def finetune(
        self,
        pretrained_model: Encoder,
        output_folder: Path,
        hyperparams: HyperParameters = DEFAULT_HYPERPARAMETERS,
        finetuning_config: FinetuningConfig = DEFAULT_FINETUNING_CONFIG,
    ) -> FineTuningModel:
        finetuning_model = self._new_finetuning_model(pretrained_model)

        # Early return if the model already exists
        final_model_path = output_folder / "finetuned_model.pth"
        if final_model_path.exists():
            finetuning_model.load_state_dict(torch.load(final_model_path))
            finetuning_model.eval()
            return finetuning_model

        loss_fn = nn.MSELoss()

        train_loader = DataLoader(
            self._create_dataset(Mode.TRAIN),
            batch_size=finetuning_config.batch_size,
            shuffle=True,
            num_workers=hyperparams.num_workers,
        )
        validation_loader = DataLoader(
            self._create_dataset(Mode.VALIDATION),
            batch_size=finetuning_config.batch_size,
            shuffle=False,
            num_workers=hyperparams.num_workers,
        )

        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            finetuning_model.parameters(),
            lr=finetuning_config.learning_rate,
            weight_decay=finetuning_config.weight_decay,
        )

        state = FinetuningState(
            epoch=0,
            best_model_dict=None,
            best_loss=None,
            epochs_since_improvement=0,
        )

        # Load checkpoint if it exists
        finetuning_model, optimizer, state = self._load_checkpoint(output_folder, finetuning_model, optimizer, state)

        for epoch in (pbar := tqdm(range(finetuning_config.max_epochs), desc="Finetuning")):
            finetuning_model.train()
            epoch_train_loss = 0.0

            for masked_output, (_, _), label in tqdm(train_loader, desc="Training", leave=False):
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [x.to(device) for x in masked_output]
                optimizer.zero_grad()
                predictions = finetuning_model(
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
                    patch_size=self.patch_size,
                )[:, 0]
                loss = loss_fn(predictions, label.float().to(device))
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()

            train_loss = epoch_train_loss / len(train_loader)

            finetuning_model.eval()
            all_predictions = []
            all_labels = []
            for masked_output, (_, _), label in tqdm(validation_loader, desc="Validation", leave=False):
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [x.to(device) for x in masked_output]
                with torch.no_grad():
                    predictions = finetuning_model(
                        s_t_x,
                        sp_x,
                        t_x,
                        st_x,
                        s_t_m,
                        sp_m,
                        t_m,
                        st_m,
                        months,
                        patch_size=self.patch_size,
                    )[:, 0]
                    all_predictions.append(predictions)
                    all_labels.append(label)

            validation_loss = torch.mean(loss_fn(torch.cat(all_predictions), torch.cat(all_labels).float().to(device)))
            pbar.set_description(f"Train loss: {train_loss:.4f}, Validation loss: {validation_loss:.4f}")
            is_best = state.best_loss is None or validation_loss < state.best_loss
            if is_best:
                state.best_loss = validation_loss
                state.best_model_dict = deepcopy(finetuning_model.state_dict())
                state.epochs_since_improvement = 0
            else:
                state.epochs_since_improvement += 1
                if state.epochs_since_improvement >= finetuning_config.patience:
                    logger.info(f"Early stopping at epoch {epoch} with validation loss {validation_loss:.4f}")
                    break

            self._save_checkpoint(output_folder, finetuning_model, optimizer, state)

        if state.best_model_dict is None:
            raise ValueError("No best model found")

        finetuning_model.load_state_dict(state.best_model_dict)
        torch.save(finetuning_model.state_dict(), final_model_path)
        finetuning_model.eval()
        return finetuning_model

    def test(
        self,
        name: str,
        finetuned_model: FineTuningModel,
        filter: Filter | None = None,
        hyperparams: HyperParameters = DEFAULT_HYPERPARAMETERS,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        test_loader = DataLoader(
            self._create_dataset(Mode.TEST, filter),
            batch_size=hyperparams.batch_size,
            shuffle=False,
            num_workers=hyperparams.num_workers,
        )

        labels_list = []
        preds_list = []
        latitudes_list = []
        longitudes_list = []
        for masked_output, (latitude, longitude), label in tqdm(test_loader, desc=f"Evaluating {name}", leave=False):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [x.to(device) for x in masked_output]
            finetuned_model.eval()
            with torch.no_grad():
                predictions = finetuned_model(
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
                    patch_size=self.patch_size,
                )[:, 0]
                labels_list.append(label.cpu().numpy())
                preds_list.append(predictions.cpu().numpy())
                latitudes_list.append(latitude.cpu().numpy())
                longitudes_list.append(longitude.cpu().numpy())

        all_labels = np.concatenate(labels_list) if len(labels_list) > 0 else np.array([])
        all_preds = np.concatenate(preds_list) if len(preds_list) > 0 else np.array([])
        all_latitudes = np.concatenate(latitudes_list) if len(latitudes_list) > 0 else np.array([])
        all_longitudes = np.concatenate(longitudes_list) if len(longitudes_list) > 0 else np.array([])
        return all_labels, all_preds, all_latitudes, all_longitudes

    def baseline(self) -> tuple[np.ndarray, np.ndarray]:
        train_dataset = self._create_dataset(Mode.TRAIN)
        test_dataset = self._create_dataset(Mode.TEST)

        month_to_labels = defaultdict(list)
        for i in range(len(train_dataset)):
            output, _, label = train_dataset[i]
            end_month = int(output.months[-1])
            month_to_labels[end_month].append(label)

        month_to_mean = {month: np.mean(labels) for month, labels in month_to_labels.items()}
        preds, targets = [], []
        for i in range(len(test_dataset)):
            output, _, label = test_dataset[i]
            end_month = int(output.months[-1])
            if end_month not in month_to_mean:
                continue
            preds.append(month_to_mean[end_month])
            targets.append(label)

        return np.array(preds), np.array(targets)

    def compute_metrics(self, name: str, preds: np.ndarray, labels: np.ndarray) -> ResultsDict:
        if preds.size == 0 or labels.size == 0:
            return {}
        adjusted_preds = preds * MAX_LFMC_VALUE
        adjusted_labels = labels * MAX_LFMC_VALUE
        return {
            name: {
                "r2_score": r2_score(adjusted_labels, adjusted_preds),
                "mae": mean_absolute_error(adjusted_labels, adjusted_preds),
                "rmse": root_mean_squared_error(adjusted_labels, adjusted_preds),
            }
        }


def finetune_and_evaluate(
    normalizer: Normalizer,
    pretrained_model: Encoder,
    data_folder: Path,
    h5py_folder: Path,
    output_folder: Path,
    h5pys_only: bool = False,
    output_hw: int = 32,
    output_timesteps: int = 12,
    patch_size: int = 16,
    hyperparams: HyperParameters = DEFAULT_HYPERPARAMETERS,
    finetuning_config: FinetuningConfig = DEFAULT_FINETUNING_CONFIG,
    validation_folds: frozenset[int] | None = None,
    test_folds: frozenset[int] | None = None,
    validation_state_regions: frozenset[str] | None = None,
    test_state_regions: frozenset[str] | None = None,
    excluded_bands: frozenset[str] = frozenset(),
) -> tuple[ResultsDict, pd.DataFrame]:
    logger.info("Data folder: %s", data_folder)
    logger.info("H5py folder: %s", h5py_folder)
    logger.info("Output folder: %s", output_folder)
    logger.info("H5pys only: %s", h5pys_only)
    logger.info("Output HW: %s", output_hw)
    logger.info("Output timesteps: %s", output_timesteps)
    logger.info("Patch size: %s", patch_size)
    logger.info("Hyperparams: %s", hyperparams)
    logger.info("Finetuning config: %s", finetuning_config)
    logger.info("Validation folds: %s", validation_folds)
    logger.info("Test folds: %s", test_folds)
    logger.info("Validation state regions: %s", validation_state_regions)
    logger.info("Test state regions: %s", test_state_regions)
    logger.info("Excluded bands: %s", excluded_bands)

    filters = {
        "all": None,
        MeteorologicalSeason.WINTER: Filter(seasons={MeteorologicalSeason.WINTER}),
        MeteorologicalSeason.SPRING: Filter(seasons={MeteorologicalSeason.SPRING}),
        MeteorologicalSeason.SUMMER: Filter(seasons={MeteorologicalSeason.SUMMER}),
        MeteorologicalSeason.AUTUMN: Filter(seasons={MeteorologicalSeason.AUTUMN}),
        WorldCoverClass.TREE_COVER: Filter(landcover={WorldCoverClass.TREE_COVER}),
        WorldCoverClass.GRASSLAND: Filter(landcover={WorldCoverClass.GRASSLAND}),
        WorldCoverClass.SHRUBLAND: Filter(landcover={WorldCoverClass.SHRUBLAND}),
        WorldCoverClass.BUILT_UP: Filter(landcover={WorldCoverClass.BUILT_UP}),
        WorldCoverClass.BARE_VEGETATION: Filter(landcover={WorldCoverClass.BARE_VEGETATION}),
        "elevation_0_500": Filter(elevation=(0, 500)),
        "elevation_500_1000": Filter(elevation=(500, 1000)),
        "elevation_1000_1500": Filter(elevation=(1000, 1500)),
        "elevation_1500_2000": Filter(elevation=(1500, 2000)),
        "elevation_2000_2500": Filter(elevation=(2000, 2500)),
        "elevation_2500_3000": Filter(elevation=(2500, 3000)),
        "elevation_3000_3500": Filter(elevation=(3000, 3500)),
        "high_fire_danger": Filter(high_fire_danger=True),
        "non_high_fire_danger": Filter(high_fire_danger=False),
    }

    # "baseline" is a special key that is not a filter
    assert "baseline" not in filters.keys()

    lfmc_eval = LFMCEval(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=h5pys_only,
        output_hw=output_hw,
        output_timesteps=output_timesteps,
        patch_size=patch_size,
        validation_folds=validation_folds,
        test_folds=test_folds,
        validation_state_regions=validation_state_regions,
        test_state_regions=test_state_regions,
        excluded_bands=excluded_bands,
    )

    finetuned_model = lfmc_eval.finetune(pretrained_model, output_folder, hyperparams, finetuning_config)

    all_results: dict[str, dict[str, float]] = {}
    df = pd.DataFrame()
    for filter_name, filter in filters.items():
        labels, preds, latitudes, longitudes = lfmc_eval.test(filter_name, finetuned_model, filter=filter)
        results = lfmc_eval.compute_metrics(filter_name, preds, labels)
        all_results.update(results)
        if filter_name == "all":
            df["latitude"] = latitudes
            df["longitude"] = longitudes
            df["label"] = labels
            df["prediction"] = preds

    baseline_preds, baseline_labels = lfmc_eval.baseline()
    baseline_results = lfmc_eval.compute_metrics("baseline", baseline_preds, baseline_labels)
    all_results.update(baseline_results)

    return all_results, df
