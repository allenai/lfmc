import json
from pathlib import Path

import torch

from galileo.data.config import CONFIG_FILENAME, ENCODER_FILENAME
from galileo.galileo import Encoder
from galileo.utils import device


def load_from_folder(folder: Path, load_weights: bool = True) -> Encoder:
    """
    This is copied from galileo.galileo.Encoder.load_from_folder with an additional parameter
    to control whether or not to load the weights.
    """
    if not (folder / CONFIG_FILENAME).exists():
        all_files_in_folder = [f.name for f in folder.glob("*")]
        raise ValueError(f"Expected {CONFIG_FILENAME} in {folder}, found {all_files_in_folder}")
    if not (folder / ENCODER_FILENAME).exists():
        all_files_in_folder = [f.name for f in folder.glob("*")]
        raise ValueError(f"Expected {ENCODER_FILENAME} in {folder}, found {all_files_in_folder}")

    with (folder / CONFIG_FILENAME).open("r") as f:
        config = json.load(f)
        model_config = config["model"]
        encoder_config = model_config["encoder"]
    encoder = Encoder(**encoder_config)

    if load_weights:
        state_dict = torch.load(folder / ENCODER_FILENAME, map_location=device)
        for key in list(state_dict.keys()):
            # this cleans the state dict, which occasionally had an extra
            # ".backbone" included in the key names
            state_dict[key.replace(".backbone", "")] = state_dict.pop(key)
        encoder.load_state_dict(state_dict)
    return encoder
