from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn

from galileo.config import BASE_GSD
from galileo.galileo import Encoder


@dataclass(frozen=True)
class FinetuningConfig:
    max_epochs: int
    weight_decay: float
    learning_rate: float
    batch_size: int
    patience: int


DEFAULT_FINETUNING_CONFIG = FinetuningConfig(
    max_epochs=100,
    weight_decay=0.05,
    learning_rate=3e-4,
    batch_size=64,
    patience=5,
)


class FineTuningModel(nn.Module):
    def __init__(self, encoder: Encoder, head: nn.Module):
        super().__init__()
        self.encoder: Encoder = deepcopy(encoder)
        # Ensure the model is trainable because we can call this having called requires_grad_(False)
        self.encoder.requires_grad_(True)
        # ... but don't unfreeze the position encoder, which shouldn't be trainable
        self.encoder.pos_embed.requires_grad_(False)
        self.encoder.month_embed.requires_grad_(False)
        self.head = head

    def forward(
        self,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
        months: torch.Tensor,
        patch_size: int | None = None,
        input_resolution_m: int | None = BASE_GSD,
    ) -> torch.Tensor:
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = self.encoder(
            s_t_x,
            sp_x,
            t_x,
            st_x,
            s_t_m,
            sp_m,
            t_m,
            st_m,
            months,
            patch_size=patch_size,
            input_resolution_m=input_resolution_m,
        )

        return self.head(
            self.encoder.average_tokens(
                s_t_x,
                sp_x,
                t_x,
                st_x,
                s_t_m,
                sp_m,
                t_m,
                st_m,
            )
        )
