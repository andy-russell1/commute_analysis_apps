from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import streamlit as st

from .config import (
    CANONICAL_GEOGRAPHY_LOOKUP_FILE,
    CONFIG,
    DEMOGRAPHICS_BY_CUSTOM_GEOGRAPHY_FILE,
    DEMOGRAPHICS_BY_LAD_FILE,
    DEMOGRAPHICS_METADATA_FILE,
    REQUIRED_CLIENT_FILES,
    REQUIRED_ROOT_FILES,
    REQUIRED_SHARED_FILES,
    TalentAnalyticsConfig,
)
from .geography import canonicalise_custom_geography_frame, constituent_authority_lists, load_custom_geography_lookup


@dataclass(frozen=True)
class TalentAnalyticsBundle:
    client_id: str
    manifest: dict[str, Any]
    readme: str
    target_roles: pd.DataFrame
    postings_by_geography: pd.DataFrame
    company_rankings: pd.DataFrame
    skills: pd.DataFrame
    industry_landscape: pd.DataFrame
    geography_lookup: pd.DataFrame
    london_custom_groups: gpd.GeoDataFrame
    lad_boundaries: gpd.GeoDataFrame
    demographics_by_lad: pd.DataFrame
    demographics_by_custom_geography: pd.DataFrame
    demographics_metadata: pd.DataFrame


def _normalise_soc_code(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64").astype(str).str.replace("<NA>", "", regex=False).str.zfill(4)


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Talent Analytics data pack is incomplete. Missing files:\n{joined}")


def _read_demographics_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    metadata["is_active"] = metadata["is_active"].astype(str).str.upper().eq("TRUE")
    metadata["display_order"] = pd.to_numeric(metadata["display_order"], errors="coerce").astype("Int64")
    return metadata.sort_values("display_order", na_position="last").reset_index(drop=True)


def _extract_upload_root(data_root: Path) -> Path:
    return data_root.parents[2] / ".codex_tmp" / "talent_analytics_uploads"


def _find_uploaded_client_dir(extract_root: Path, fallback_client_id: str) -> tuple[str, Path] | None:
    required = set(REQUIRED_CLIENT_FILES)
    candidate_dirs = [extract_root]
    candidate_dirs.extend(path for path in extract_root.rglob("*") if path.is_dir())
    matching_dirs = [path for path in candidate_dirs if required.issubset({item.name for item in path.iterdir() if item.is_file()})]

    if not matching_dirs:
        return None

    client_dir = matching_dirs[0]
    client_id = client_dir.name if client_dir != extract_root else fallback_client_id
    return client_id, client_dir


def _extract_zip_bytes(upload_name: str, upload_bytes: bytes, data_root: Path) -> tuple[str, Path]:
    digest = hashlib.sha256(upload_bytes).hexdigest()[:12]
    safe_name = Path(upload_name).stem or "uploaded_client_pack"
    upload_root = _extract_upload_root(data_root)
    upload_root.mkdir(parents=True, exist_ok=True)

    existing_extract_root = upload_root / f"{safe_name}_{digest}"
    if existing_extract_root.exists():
        existing_match = _find_uploaded_client_dir(existing_extract_root, safe_name)
        if existing_match is not None:
            return existing_match

    extract_root = upload_root / f"{safe_name}_{digest}_{uuid.uuid4().hex[:8]}"
    extract_root.mkdir(parents=True, exist_ok=True)

    with ZipFile(BytesIO(upload_bytes)) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("The uploaded Talent Analytics zip contains an unsafe file path.")
        archive.extractall(extract_root)

    match = _find_uploaded_client_dir(extract_root, safe_name)
    if match is not None:
        return match

    joined = ", ".join(REQUIRED_CLIENT_FILES)
    raise FileNotFoundError(
        "The uploaded zip does not contain a valid Talent Analytics client pack. "
        f"Expected these files in one folder: {joined}"
    )


@st.cache_data(show_spinner=False)
def list_available_clients(data_root: str = str(CONFIG.data_root)) -> list[str]:
    clients_dir = Path(data_root) / "clients"
    if not clients_dir.exists():
        return []
    return sorted(path.name for path in clients_dir.iterdir() if path.is_dir())


@st.cache_data(show_spinner=False)
def load_bundle(
    client_id: str = CONFIG.default_client_id,
    *,
    data_root: str = str(CONFIG.data_root),
) -> TalentAnalyticsBundle:
    return _load_bundle_internal(
        client_id=client_id,
        data_root=Path(data_root),
        client_dir_override=None,
    )


def _load_bundle_internal(
    *,
    client_id: str,
    data_root: Path,
    client_dir_override: Path | None,
) -> TalentAnalyticsBundle:
    config = TalentAnalyticsConfig(data_root=Path(data_root), default_client_id=CONFIG.default_client_id)
    shared_dir = config.shared_dir
    client_dir = client_dir_override or config.client_dir(client_id)

    required_paths = [config.data_root / name for name in REQUIRED_ROOT_FILES]
    required_paths.extend(shared_dir / name for name in REQUIRED_SHARED_FILES)
    required_paths.extend(client_dir / name for name in REQUIRED_CLIENT_FILES)
    _require_files(required_paths)

    manifest = json.loads((config.data_root / "manifest.json").read_text(encoding="utf-8"))
    readme = (config.data_root / "README.md").read_text(encoding="utf-8")

    target_roles = pd.read_csv(client_dir / "target_role_basket.csv")
    target_roles["soc_code_4d"] = _normalise_soc_code(target_roles["soc_code_4d"])
    target_roles["role_label"] = target_roles["soc_name_4d"] + " (" + target_roles["soc_code_4d"] + ")"

    postings = pd.read_csv(client_dir / "postings_by_geography.csv")
    postings["soc_code_4d"] = _normalise_soc_code(postings["soc_code_4d"])
    postings["role_label"] = postings["soc_name_4d"] + " (" + postings["soc_code_4d"] + ")"
    geography_lookup = load_custom_geography_lookup(config.data_root)
    postings = canonicalise_custom_geography_frame(postings, geography_lookup)

    geography_lists = constituent_authority_lists(geography_lookup)

    postings = postings.merge(
        geography_lists,
        on=["custom_geography_key", "custom_geography_name", "display_order"],
        how="left",
    )

    london_custom_groups = gpd.read_file(shared_dir / "london_custom_groups.geojson")
    if london_custom_groups.crs is None or str(london_custom_groups.crs).lower() != "epsg:4326":
        london_custom_groups = london_custom_groups.to_crs("EPSG:4326")
    london_custom_groups = canonicalise_custom_geography_frame(london_custom_groups, geography_lookup)

    lad_boundaries = gpd.read_file(shared_dir / "lad_uk_2024.geojson")
    if lad_boundaries.crs is None or str(lad_boundaries.crs).lower() != "epsg:4326":
        lad_boundaries = lad_boundaries.to_crs("EPSG:4326")

    demographics_by_lad = pd.read_csv(shared_dir / DEMOGRAPHICS_BY_LAD_FILE)
    demographics_by_custom_geography = pd.read_csv(shared_dir / DEMOGRAPHICS_BY_CUSTOM_GEOGRAPHY_FILE)
    demographics_by_custom_geography = canonicalise_custom_geography_frame(
        demographics_by_custom_geography,
        geography_lookup,
    )
    demographics_metadata = _read_demographics_metadata(shared_dir / DEMOGRAPHICS_METADATA_FILE)

    return TalentAnalyticsBundle(
        client_id=client_id,
        manifest=manifest,
        readme=readme,
        target_roles=target_roles,
        postings_by_geography=postings,
        company_rankings=pd.read_csv(client_dir / "company_rankings_london.csv"),
        skills=pd.read_csv(client_dir / "skills_london.csv"),
        industry_landscape=pd.read_csv(client_dir / "industry_landscape_london.csv"),
        geography_lookup=geography_lookup,
        london_custom_groups=london_custom_groups,
        lad_boundaries=lad_boundaries,
        demographics_by_lad=demographics_by_lad,
        demographics_by_custom_geography=demographics_by_custom_geography,
        demographics_metadata=demographics_metadata,
    )


@st.cache_data(show_spinner=False)
def load_uploaded_bundle(
    upload_name: str,
    upload_bytes: bytes,
    *,
    data_root: str = str(CONFIG.data_root),
) -> TalentAnalyticsBundle:
    data_root_path = Path(data_root)
    client_id, client_dir = _extract_zip_bytes(upload_name, upload_bytes, data_root_path)
    return _load_bundle_internal(
        client_id=client_id,
        data_root=data_root_path,
        client_dir_override=client_dir,
    )
