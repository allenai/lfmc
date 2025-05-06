from dataclasses import dataclass
from itertools import chain

import pandas as pd

from lfmc.core.const import (
    HIGH_FIRE_DANGER_THRESHOLD,
    METEOROLOGICAL_SEASON_MONTHS,
    Column,
    MeteorologicalSeason,
    WorldCoverClass,
)


@dataclass
class Filter:
    seasons: set[MeteorologicalSeason] | None = None
    """Seasons"""

    landcover: set[WorldCoverClass] | None = None
    """Landcover classes"""

    elevation: tuple[int, int] | None = None
    """Elevation range in meters [lower, upper)"""

    high_fire_danger: bool | None = None
    """Whether to filter out samples with high fire danger (LFMC value <100%)"""


def apply_filter(df: pd.DataFrame, filter: Filter | None) -> pd.DataFrame:
    if filter is None:
        return df
    if filter.seasons is not None:
        months = set(chain.from_iterable(METEOROLOGICAL_SEASON_MONTHS[season] for season in filter.seasons))
        df = df[df[Column.SAMPLING_DATE].dt.month.isin(months)]
    if filter.landcover is not None:
        df = df[df[Column.LANDCOVER].isin(filter.landcover)]
    if filter.elevation is not None:
        df = df[
            df[Column.ELEVATION].between(
                filter.elevation[0],
                filter.elevation[1],
                inclusive="left",
            )
        ]
    if filter.high_fire_danger is not None:
        df = (
            df[df[Column.LFMC_VALUE] < HIGH_FIRE_DANGER_THRESHOLD]
            if filter.high_fire_danger
            else df[df[Column.LFMC_VALUE] >= HIGH_FIRE_DANGER_THRESHOLD]
        )
    return df
