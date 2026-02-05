from __future__ import annotations

import io
import zipfile
from typing import List

import geopandas as gpd
from fiona.io import ZipMemoryFile


REQUIRED_SHP_EXTS = {".shp", ".dbf", ".shx"}


def list_zip_files(zip_bytes: bytes) -> List[str]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return [name for name in zf.namelist() if not name.endswith("/")]


def validate_isochrone_zip(zip_bytes: bytes) -> None:
    names = list_zip_files(zip_bytes)
    lower = [n.lower() for n in names]
    exts = {"." + n.split(".")[-1] for n in lower if "." in n}
    missing = REQUIRED_SHP_EXTS - exts
    if missing:
        raise ValueError(
            "ZIP is missing required shapefile parts: {missing}.".format(missing=", ".join(sorted(missing)))
        )
    if not any(n.lower().endswith(".shp") for n in names):
        raise ValueError("No .shp file found in ZIP.")


def load_isochrones_from_zip(zip_bytes: bytes) -> gpd.GeoDataFrame:
    try:
        with ZipMemoryFile(zip_bytes) as memfile:
            layers = memfile.listlayers()
            if not layers:
                raise ValueError("No layers found in the ZIP.")
            with memfile.open(layer=layers[0]) as src:
                gdf = gpd.GeoDataFrame.from_features(src, crs=src.crs)
    except Exception as exc:
        raise ValueError(
            "Unable to read shapefile from ZIP in memory. Ensure the ZIP contains a valid shapefile."
        ) from exc
    if gdf.empty:
        raise ValueError("No features found in the uploaded isochrone file.")
    return gdf
