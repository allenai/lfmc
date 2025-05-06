import argparse
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import rasterio
import rtree
from frozendict import frozendict
from rasterio.crs import CRS
from rasterio.warp import transform
from tqdm import tqdm

from lfmc.core.const import LABELS_PATH, WGS84_EPSG, WORLD_COVER_CLASS_MAP, Column


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class SpatialIndex:
    dir: Path
    index: rtree.index.Index
    bounds_lookup: frozendict[int, Path]


def build_spatial_index(tif_dir: Path) -> SpatialIndex:
    index = rtree.index.Index()
    bounds_lookup = {}
    for i, tif_path in enumerate(tif_dir.rglob("*.tif")):
        with rasterio.open(tif_path) as src:
            bbox = src.bounds
            index.insert(i, (bbox.left, bbox.bottom, bbox.right, bbox.top))
            bounds_lookup[i] = tif_path
    return SpatialIndex(dir=tif_dir, index=index, bounds_lookup=frozendict(bounds_lookup))


def query_spatial_index(point: Point, spatial_index: SpatialIndex) -> int | float | None:
    candidates = list(spatial_index.index.intersection((point.x, point.y, point.x, point.y)))
    wgs84_crs = CRS.from_epsg(WGS84_EPSG)

    for i in candidates:
        path = spatial_index.bounds_lookup[i]
        with rasterio.open(path) as src:
            if src.crs != wgs84_crs:
                lons, lats = transform(wgs84_crs, src.crs, [point.x], [point.y])
            else:
                lons, lats = [point.x], [point.y]

            try:
                row, col = src.index(lons[0], lats[0])
                return src.read(1)[row, col].item()
            except IndexError:
                continue

    logging.warning(f"No value found for point {point} in {spatial_index.dir}")
    return None


def augment_labels(df: pd.DataFrame, worldcover_path: Path, srtm_path: Path) -> pd.DataFrame:
    worldcover_spatial_index = build_spatial_index(worldcover_path)
    srtm_spatial_index = build_spatial_index(srtm_path)

    @lru_cache(maxsize=None)
    def worldcover_query(point: Point) -> str | None:
        value = query_spatial_index(point, worldcover_spatial_index)
        if value is None:
            return None
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"WorldCover value is not an integer: {value}")
        if value not in WORLD_COVER_CLASS_MAP:
            raise ValueError(f"WorldCover value is not in the map: {value}")
        return WORLD_COVER_CLASS_MAP[value]

    @lru_cache(maxsize=None)
    def srtm_query(point: Point) -> int | float | None:
        return query_spatial_index(point, srtm_spatial_index)

    def lookup(row: pd.Series, query_fn: Callable[[Point], Any]) -> pd.Series:
        return pd.Series(query_fn(Point(x=row[Column.LONGITUDE], y=row[Column.LATITUDE])))

    tqdm.pandas(desc="Adding WorldCover")
    df[Column.LANDCOVER] = df.progress_apply(lookup, axis=1, args=(worldcover_query,))  # type: ignore
    tqdm.pandas(desc="Adding elevation")
    df[Column.ELEVATION] = df.progress_apply(lookup, axis=1, args=(srtm_query,))  # type: ignore
    return df


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser("Augment labels with WorldCover and elevation")
    parser.add_argument(
        "--input-csv-path",
        type=Path,
        default=LABELS_PATH,
    )
    parser.add_argument(
        "--output-csv-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--worldcover-folder",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--srtm-folder",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    if not args.input_csv_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {args.input_csv_path}")
    if not args.worldcover_folder.is_dir():
        raise FileNotFoundError(f"WorldCover directory not found: {args.worldcover_folder}")
    if not args.srtm_folder.is_dir():
        raise FileNotFoundError(f"SRTM directory not found: {args.srtm_folder}")

    df = pd.read_csv(args.input_csv_path)
    df = augment_labels(df, args.worldcover_folder, args.srtm_folder)
    df.to_csv(args.output_csv_path, index=False)


if __name__ == "__main__":
    main()
