from galileo.data.dataset import SPACE_TIME_BANDS_GROUPS_IDX, TIME_BAND_GROUPS_IDX

SPACE_TIME_BANDS = frozenset(SPACE_TIME_BANDS_GROUPS_IDX.keys())
SPACE_BANDS = frozenset(["SRTM"])
TIME_BANDS = frozenset(TIME_BAND_GROUPS_IDX.keys())
STATIC_BANDS = frozenset(["location"])
