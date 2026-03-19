from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from apps.talent_analytics.config import TalentAnalyticsConfig
from apps.talent_analytics.data import list_available_clients, load_uploaded_bundle


def test_talent_analytics_shared_pack_loads_without_private_clients():
    config = TalentAnalyticsConfig()
    clients = list_available_clients()

    assert "expedia_poc" not in clients

    demographics = pd.read_csv(config.shared_dir / "ons" / "processed" / "demographics_by_custom_geography.csv")
    metadata = pd.read_csv(config.shared_dir / "ons" / "processed" / "demographics_metadata.csv")

    assert demographics["custom_geography_key"].nunique() == 20
    assert set(metadata["metric_key"]) == {
        "total_population",
        "working_age_population",
        "employment_rate",
        "unemployment_rate",
        "economic_activity_rate",
        "nvq4_plus_share",
        "no_qualifications_share",
        "professional_occupations_share",
    }


def test_talent_analytics_bundle_loads_from_uploaded_zip():
    upload_buffer = BytesIO()
    with ZipFile(upload_buffer, "w") as archive:
        target_roles = pd.DataFrame(
            [
                {
                    "role_rank": 1,
                    "soc_code_4d": "2134",
                    "soc_name_4d": "Programmers and Software Development Professionals",
                }
            ]
        )
        postings = pd.DataFrame(
            [
                {
                    "soc_code_4d": "2134",
                    "soc_name_4d": "Programmers and Software Development Professionals",
                    "custom_geography_name": "Camden and City of London",
                    "unique_postings": 10,
                    "latest_365_days_unique_postings": 10,
                    "latest_30_days_unique_postings": 2,
                    "number_of_companies_posting": 3,
                    "online_profiles": 100,
                }
            ]
        )
        london_wide = pd.DataFrame(
            [
                {
                    "geography_name": "20 London County/Unitary Authorities",
                    "unique_postings": 10,
                    "latest_365_days_unique_postings": 10,
                    "latest_30_days_unique_postings": 2,
                    "number_of_companies_posting": 3,
                    "online_profiles": 100,
                    "company_name": "Example Co",
                    "skill_or_qualification": "Python",
                    "industry_name": "Software",
                    "sic_code": "6201",
                    "rank_london": 1,
                    "median_annual_advertised_salary": 50000,
                }
            ]
        )

        archive.writestr("uploaded_pack/target_role_basket.csv", target_roles.to_csv(index=False))
        archive.writestr("uploaded_pack/postings_by_geography.csv", postings.to_csv(index=False))
        archive.writestr(
            "uploaded_pack/company_rankings_london.csv",
            london_wide.drop(columns=["skill_or_qualification", "industry_name", "sic_code"]).to_csv(index=False),
        )
        archive.writestr(
            "uploaded_pack/skills_london.csv",
            london_wide.drop(columns=["company_name", "industry_name", "sic_code"]).to_csv(index=False),
        )
        archive.writestr(
            "uploaded_pack/industry_landscape_london.csv",
            london_wide.drop(columns=["company_name", "skill_or_qualification"]).to_csv(index=False),
        )

    bundle = load_uploaded_bundle("uploaded_pack.zip", upload_buffer.getvalue())

    assert bundle.client_id == "uploaded_pack"
    assert bundle.target_roles.shape[0] == 1
    assert bundle.postings_by_geography.shape[0] == 1
    assert bundle.demographics_by_custom_geography["custom_geography_key"].nunique() == 20
