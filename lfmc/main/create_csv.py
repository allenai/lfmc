import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from lfmc.core.const import CONUS_STATES, LABELS_PATH, Column

_SHEET_NAME = "LFMC data"

_COLUMNS = {
    "Sorting ID": Column.SORTING_ID,
    "Contact": Column.CONTACT,
    "Site name": Column.SITE_NAME,
    "Country": Column.COUNTRY,
    "State/Region": Column.STATE_REGION,
    "Latitude (WGS84, EPSG:4326)": Column.LATITUDE,
    "Longitude (WGS84, EPSG:4326)": Column.LONGITUDE,
    "Sampling date (YYYYMMDD)": Column.SAMPLING_DATE,
    "Protocol": Column.PROTOCOL,
    "LFMC value (%)": Column.LFMC_VALUE,
    "Species collected": Column.SPECIES_COLLECTED,
    "Species functional type": Column.SPECIES_FUNCTIONAL_TYPE,
}

INPUT_EXCEL_URL = "https://springernature.figshare.com/ndownloader/files/45049786"
INPUT_EXCEL_PATH = Path(__file__).parent.parent.parent / "data" / "external" / "Globe-LFMC-2.0 final.xlsx"


def parse_datetime(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def download_excel(output_path: Path):
    response = requests.get(INPUT_EXCEL_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as progress_bar:
        with open(output_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")


def create_csv(
    input_excel_path: Path,
    output_csv_path: Path,
    start_date: datetime,
):
    logging.info("Reading the Excel file")
    df = pd.read_excel(input_excel_path, sheet_name=_SHEET_NAME, usecols=list(_COLUMNS.keys()))

    logging.info("Renaming the columns")
    df = df.rename(columns=_COLUMNS)

    logging.info("Filtering the DataFrame by date and location")
    df = df[df[Column.SAMPLING_DATE] >= start_date]
    logging.info("After filtering by date, the DataFrame has %d rows", len(df))
    df = df[(df[Column.COUNTRY] == "USA") & (df[Column.STATE_REGION].isin(CONUS_STATES))]
    logging.info("After filtering by location, the DataFrame has %d rows", len(df))

    logging.info("Writing the CSV file")
    df.to_csv(output_csv_path, index=False)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser("Creates the LFMC CSV file")
    parser.add_argument(
        "--output-csv-path",
        type=Path,
        default=Path(LABELS_PATH),
    )
    parser.add_argument(
        "--start-date",
        type=parse_datetime,
        default=datetime(2017, 1, 1),
    )
    parser.add_argument(
        "--force-download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if args.force_download or not INPUT_EXCEL_PATH.exists():
        download_excel(INPUT_EXCEL_PATH)

    create_csv(
        INPUT_EXCEL_PATH,
        args.output_csv_path,
        args.start_date,
    )


if __name__ == "__main__":
    main()
