from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from galileo.data.config import NORMALIZATION_DICT_FILENAME
from galileo.data.dataset import Dataset, Normalizer
from galileo.galileo import Encoder

ROOT_DIR = Path(__file__).parent.parent


@pytest.fixture
def config_dir() -> Path:
    return ROOT_DIR / "data" / "config"


@pytest.fixture
def normalizer(config_dir: Path) -> Normalizer:
    normalization_dicts = Dataset.load_normalization_values(config_dir / NORMALIZATION_DICT_FILENAME)
    return Normalizer(std=True, normalizing_dicts=normalization_dicts)


@pytest.fixture
def encoder() -> Encoder:
    return Encoder.load_from_folder(Path(ROOT_DIR / "data" / "models" / "nano"))


@pytest.fixture
def data_folder() -> Path:
    current_dir = Path(__file__).parent
    return Path(current_dir / "data" / "tifs")


@pytest.fixture
def h5py_folder(tmp_path: Path) -> Path:
    h5py_folder = tmp_path / "h5pys"
    h5py_folder.mkdir(parents=True, exist_ok=True)
    return h5py_folder


@pytest.fixture(autouse=True)
def set_num_splits(monkeypatch: MonkeyPatch):
    # Use fewer folds for tests due to the small size of the dataset
    monkeypatch.setenv("NUM_FOLDS", "4")
