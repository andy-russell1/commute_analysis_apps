from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
LOGO_DIR = ASSETS_DIR / "logos"
DATA_DIR = ASSETS_DIR / "data"
EUROSTAT_DIR = DATA_DIR / "eurostat"
EUROSTAT_WORKBOOK_PATH = EUROSTAT_DIR / "europe_talent_distribution.xlsx"
EUROSTAT_BOUNDARY_LOOKUP_PATH = EUROSTAT_DIR / "boundary_lookup_selected.geojson"
