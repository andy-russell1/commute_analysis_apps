from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.runtime.paths import DATA_DIR


CANONICAL_LOOKUP_COLUMNS = (
    "custom_geography_key",
    "custom_geography_name",
    "display_order",
    "lad_code",
    "lad_name",
    "geography_level",
    "is_active",
)

CANONICAL_LOOKUP_RELATIVE_PATH = Path("shared/geography/custom_geography_lookup.csv")
LEGACY_LOOKUP_RELATIVE_PATH = Path("shared/geography_group_lookup.csv")


def canonical_lookup_path(data_root: Path | None = None) -> Path:
    root = data_root or (DATA_DIR / "talent_analytics")
    return Path(root) / CANONICAL_LOOKUP_RELATIVE_PATH


def legacy_lookup_path(data_root: Path | None = None) -> Path:
    root = data_root or (DATA_DIR / "talent_analytics")
    return Path(root) / LEGACY_LOOKUP_RELATIVE_PATH


def load_custom_geography_lookup(data_root: Path | None = None) -> pd.DataFrame:
    lookup = pd.read_csv(canonical_lookup_path(data_root), dtype=str)
    missing = [column for column in CANONICAL_LOOKUP_COLUMNS if column not in lookup.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Canonical custom geography lookup is missing required columns: {joined}")

    lookup["display_order"] = pd.to_numeric(lookup["display_order"], errors="raise").astype(int)
    lookup["is_active"] = lookup["is_active"].astype(str).str.upper().eq("TRUE")
    return lookup.loc[:, CANONICAL_LOOKUP_COLUMNS].copy()


def active_custom_geography_lookup(data_root: Path | None = None) -> pd.DataFrame:
    lookup = load_custom_geography_lookup(data_root)
    return lookup[lookup["is_active"]].copy()


def build_legacy_lookup(canonical_lookup: pd.DataFrame) -> pd.DataFrame:
    return canonical_lookup.loc[
        :,
        ["custom_geography_key", "custom_geography_name", "lad_code", "lad_name"],
    ].drop_duplicates()


def custom_geography_dimension(canonical_lookup: pd.DataFrame) -> pd.DataFrame:
    return (
        canonical_lookup.loc[:, ["custom_geography_key", "custom_geography_name", "display_order"]]
        .drop_duplicates()
        .sort_values("display_order")
        .reset_index(drop=True)
    )


def constituent_authority_lists(canonical_lookup: pd.DataFrame) -> pd.DataFrame:
    return (
        canonical_lookup.groupby(
            ["custom_geography_key", "custom_geography_name", "display_order"],
            as_index=False,
        )["lad_name"]
        .agg(lambda values: ", ".join(sorted(pd.Series(values).dropna().astype(str).unique().tolist())))
        .rename(columns={"lad_name": "constituent_authorities"})
    )


def canonicalise_custom_geography_frame(
    frame: pd.DataFrame,
    canonical_lookup: pd.DataFrame,
    *,
    key_col: str = "custom_geography_key",
    name_col: str = "custom_geography_name",
) -> pd.DataFrame:
    if name_col not in frame.columns:
        raise KeyError(f"'{name_col}' is required to canonicalise custom geographies.")

    dimension = custom_geography_dimension(canonical_lookup)
    working = frame.copy()
    working = working.drop(columns=[column for column in ("display_order",) if column in working.columns])
    if key_col in working.columns:
        working = working.drop(columns=[key_col])

    merged = working.merge(
        dimension,
        on=name_col,
        how="left",
        validate="many_to_one",
    )
    missing_names = merged.loc[merged["custom_geography_key"].isna(), name_col].dropna().unique().tolist()
    if missing_names:
        preview = ", ".join(sorted(missing_names)[:5])
        raise ValueError(f"Could not map custom geography names to canonical keys: {preview}")

    return merged
