from galileo.data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
)
from lfmc.core.bands import SPACE_BANDS, SPACE_TIME_BANDS, STATIC_BANDS, TIME_BANDS


def test_bands_are_subsets():
    def assert_subset(a, b):
        assert a.issubset(b), f"{a} is not a subset of {b}"

    assert_subset(SPACE_TIME_BANDS, frozenset(SPACE_TIME_BANDS_GROUPS_IDX.keys()))
    assert_subset(SPACE_BANDS, frozenset(SPACE_BAND_GROUPS_IDX.keys()))
    assert_subset(TIME_BANDS, frozenset(TIME_BAND_GROUPS_IDX.keys()))
    assert_subset(STATIC_BANDS, frozenset(STATIC_BAND_GROUPS_IDX.keys()))
