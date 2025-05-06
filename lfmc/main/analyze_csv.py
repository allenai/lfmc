import argparse
import logging
from pathlib import Path

from lfmc.core.const import HIGH_FIRE_DANGER_THRESHOLD, LABELS_PATH, METEOROLOGICAL_SEASON_MONTHS, Column
from lfmc.core.labels import read_labels


def analyze_csv(input_csv_path: Path):
    logging.info("Reading the CSV file")
    df = read_labels(input_csv_path)

    logging.info("Analyzing the CSV file")
    logging.info("Number of rows: %d", len(df))
    logging.info("Number of sites: %d", df[Column.SITE_NAME].nunique())
    logging.info("Median LFMC: %f", df[Column.LFMC_VALUE].median())
    logging.info("Min LFMC: %f", df[Column.LFMC_VALUE].min())
    logging.info("Max LFMC: %f", df[Column.LFMC_VALUE].max())
    logging.info(
        "Percentiles:\n%s",
        df[Column.LFMC_VALUE].quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]),
    )
    logging.info("State regions and counts:\n%s", df[Column.STATE_REGION].value_counts())
    logging.info("Landcover classes and counts:\n%s", df[Column.LANDCOVER].value_counts())
    logging.info("Elevation range: %d - %d", df[Column.ELEVATION].min(), df[Column.ELEVATION].max())
    for i in range(0, 3500, 500):
        logging.info(
            "Elevation count in range %d - %d: %d",
            i,
            i + 500,
            len(df[df[Column.ELEVATION].between(i, i + 500, inclusive="left")]),
        )
    for season in METEOROLOGICAL_SEASON_MONTHS.keys():
        logging.info(
            "Season %s count: %d",
            season,
            len(df[df[Column.SAMPLING_DATE].dt.month.isin(METEOROLOGICAL_SEASON_MONTHS[season])]),
        )
    logging.info("High fire danger count: %d", len(df[df[Column.LFMC_VALUE] < HIGH_FIRE_DANGER_THRESHOLD]))
    logging.info("Non-high fire danger count: %d", len(df[df[Column.LFMC_VALUE] >= HIGH_FIRE_DANGER_THRESHOLD]))


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser("Creates the LFMC CSV file")
    parser.add_argument(
        "--input-csv-path",
        type=Path,
        default=LABELS_PATH,
    )
    args = parser.parse_args()
    analyze_csv(args.input_csv_path)


if __name__ == "__main__":
    main()
