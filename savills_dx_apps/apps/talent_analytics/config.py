from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.runtime.paths import DATA_DIR


DEFAULT_CLIENT_ID = ""

CANONICAL_GEOGRAPHY_LOOKUP_FILE = "geography/custom_geography_lookup.csv"
DEMOGRAPHICS_BY_LAD_FILE = "ons/processed/demographics_by_lad.csv"
DEMOGRAPHICS_BY_CUSTOM_GEOGRAPHY_FILE = "ons/processed/demographics_by_custom_geography.csv"
DEMOGRAPHICS_METADATA_FILE = "ons/processed/demographics_metadata.csv"


@dataclass(frozen=True)
class TalentAnalyticsConfig:
    data_root: Path = DATA_DIR / "talent_analytics"
    default_client_id: str = DEFAULT_CLIENT_ID

    @property
    def shared_dir(self) -> Path:
        return self.data_root / "shared"

    @property
    def clients_dir(self) -> Path:
        return self.data_root / "clients"

    def client_dir(self, client_id: str | None = None) -> Path:
        return self.clients_dir / (client_id or self.default_client_id)


CONFIG = TalentAnalyticsConfig()

REQUIRED_SHARED_FILES = (
    "lad_uk_2024.geojson",
    "london_custom_groups.geojson",
    "geography_group_lookup.csv",
    CANONICAL_GEOGRAPHY_LOOKUP_FILE,
    DEMOGRAPHICS_BY_LAD_FILE,
    DEMOGRAPHICS_BY_CUSTOM_GEOGRAPHY_FILE,
    DEMOGRAPHICS_METADATA_FILE,
)

REQUIRED_CLIENT_FILES = (
    "target_role_basket.csv",
    "postings_by_geography.csv",
    "company_rankings_london.csv",
    "skills_london.csv",
    "industry_landscape_london.csv",
)

REQUIRED_ROOT_FILES = (
    "README.md",
    "manifest.json",
)

TALENT_METRICS = {
    "unique_postings": "Unique postings",
    "latest_365_days_unique_postings": "Unique postings in latest 365 days",
    "latest_30_days_unique_postings": "Unique postings in latest 30 days",
    "number_of_companies_posting": "Companies posting",
    "online_profiles": "Online profiles",
}

LONDON_WIDE_METRICS = {
    "unique_postings": "Unique postings",
    "latest_365_days_unique_postings": "Unique postings in latest 365 days",
    "latest_30_days_unique_postings": "Unique postings in latest 30 days",
    "number_of_companies_posting": "Companies posting",
    "online_profiles": "Online profiles",
}
