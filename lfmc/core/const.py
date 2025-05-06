from enum import StrEnum
from pathlib import Path

from frozendict import frozendict

CONUS_STATES = [
    "Alabama",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]

LABELS_PATH = Path("data/labels/lfmc_data_conus.csv")

MAX_LFMC_VALUE = 302  # 99.9th percentile of the LFMC values


class Column(StrEnum):
    SORTING_ID = "sorting_id"
    CONTACT = "contact"
    SITE_NAME = "site_name"
    COUNTRY = "country"
    STATE_REGION = "state_region"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    SAMPLING_DATE = "sampling_date"
    PROTOCOL = "protocol"
    LFMC_VALUE = "lfmc_value"
    SPECIES_COLLECTED = "species_collected"
    SPECIES_FUNCTIONAL_TYPE = "species_functional_type"
    LANDCOVER = "landcover"
    ELEVATION = "elevation"


class FileSuffix(StrEnum):
    TIF = "tif"
    H5 = "h5"


WGS84_EPSG = 4326


class WorldCoverClass(StrEnum):
    TREE_COVER = "Tree cover"
    SHRUBLAND = "Shrubland"
    GRASSLAND = "Grassland"
    CROPLAND = "Cropland"
    BUILT_UP = "Built-up"
    BARE_VEGETATION = "Bare / sparse vegetation"
    SNOW_AND_ICE = "Snow and ice"
    WATER = "Permanent water bodies"
    HERBACEOUS_WETLAND = "Herbaceous wetland"
    MANGROVES = "Mangroves"
    MOSS_AND_LICHEN = "Moss and lichen"


WORLD_COVER_CLASS_MAP = frozendict(
    {
        10: WorldCoverClass.TREE_COVER,
        20: WorldCoverClass.SHRUBLAND,
        30: WorldCoverClass.GRASSLAND,
        40: WorldCoverClass.CROPLAND,
        50: WorldCoverClass.BUILT_UP,
        60: WorldCoverClass.BARE_VEGETATION,
        70: WorldCoverClass.SNOW_AND_ICE,
        80: WorldCoverClass.WATER,
        90: WorldCoverClass.HERBACEOUS_WETLAND,
        95: WorldCoverClass.MANGROVES,
        100: WorldCoverClass.MOSS_AND_LICHEN,
    }
)


class MeteorologicalSeason(StrEnum):
    WINTER = "Winter"
    SPRING = "Spring"
    SUMMER = "Summer"
    AUTUMN = "Autumn"


METEOROLOGICAL_SEASON_MONTHS = frozendict(
    {
        MeteorologicalSeason.WINTER: frozenset({12, 1, 2}),
        MeteorologicalSeason.SPRING: frozenset({3, 4, 5}),
        MeteorologicalSeason.SUMMER: frozenset({6, 7, 8}),
        MeteorologicalSeason.AUTUMN: frozenset({9, 10, 11}),
    }
)


# Use 120% as the high fire danger threshold to match (Zhu et al. 2021)
HIGH_FIRE_DANGER_THRESHOLD = 120
