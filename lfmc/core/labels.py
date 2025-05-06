from pathlib import Path

import pandas as pd

from lfmc.core.const import Column


def read_labels(path: Path) -> pd.DataFrame:
    """Reads the LFMC labels CSV file.

    From the Globe-LFMC-2.0 paper:

    "For remote sensing applications, it is recommended to average the LFMC measurements taken on
    the same date and located within the same pixel of the product employed in the study. The
    choice of which functional type to include in the average can be guided by the land cover type
    of that pixel. For example, in open canopy forests, both trees and shrubs (or grass) could be
    included."
    """
    data = pd.read_csv(path)
    grouped = data.groupby(
        [
            Column.LATITUDE,
            Column.LONGITUDE,
            Column.SAMPLING_DATE,
        ],
        as_index=False,
    ).agg(
        {
            Column.SITE_NAME: "first",
            Column.SORTING_ID: "first",
            Column.LFMC_VALUE: "mean",
            Column.STATE_REGION: "first",
            Column.COUNTRY: "first",
            Column.LANDCOVER: "first",
            Column.ELEVATION: "first",
        }
    )
    grouped[Column.SAMPLING_DATE] = pd.to_datetime(grouped[Column.SAMPLING_DATE])
    return grouped
