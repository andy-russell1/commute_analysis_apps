from __future__ import annotations

"""Lightweight smoke checks for Amenity Analysis helper functions.

Run from repo root:
    python apps/amenity_analysis/smoke_test.py
"""

import pandas as pd

from apps.amenity_analysis.common import (
    build_amenity_points,
    build_office_scores,
)


def run_smoke_test() -> None:
    summary_df = pd.DataFrame(
        {
            "officeID": ["A", "B"],
            "office_name": ["Alpha House", "Beta House"],
            "address": ["Alpha House, London", "Beta House, London"],
            "lat": [51.5, 51.51],
            "lon": [-0.12, -0.1],
            "amenity_kpi": [72.5, 63.0],
            "normalised_lunch_and_coffee": [0.9, 0.4],
            "count_lunch_and_coffee": [10, 4],
            "normalised_green": [0.3, 0.8],
            "count_green": [2, 6],
            "normalised_fitness": [0.7, 0.2],
            "count_fitness": [5, 1],
            "nearest_public_transport_stop_distance_m": [120.0, 250.0],
        }
    )
    selected_metrics = ["Lunch & coffee", "Green", "Fitness", "Public transport"]

    office_scores = build_office_scores(summary_df=summary_df, selected_metrics=selected_metrics)

    assert "officeID" in office_scores.columns
    assert "total_score" in office_scores.columns
    assert "subscore_lunch_and_coffee" in office_scores.columns
    assert "count_lunch_and_coffee" in office_scores.columns

    poi_df = pd.DataFrame(
        {
            "officeID": ["A", "A", "B"],
            "office_name": ["Alpha House", "Alpha House", "Beta House"],
            "bucket": ["Lunch & coffee", "Green", "Lunch & coffee"],
            "name": ["Cafe One", "Park One", "Cafe Two"],
            "poi_lat": [51.5003, 51.5009, 51.509],
            "poi_lon": [-0.119, -0.121, -0.099],
            "distance_m": [80.0, 200.0, 120.0],
        }
    )
    weights_norm = {
        "Lunch & coffee": 0.4,
        "Green": 0.3,
        "Fitness": 0.2,
        "Public transport": 0.1,
    }

    amenity_points = build_amenity_points(
        poi_df=poi_df,
        office_scores_df=office_scores,
        selected_categories=["Lunch & coffee", "Green"],
        weights_norm=weights_norm,
    )

    required_columns = {
        "officeID",
        "office_name",
        "category",
        "name",
        "lat",
        "lon",
        "distance_m",
        "weight_contribution",
    }
    assert required_columns.issubset(set(amenity_points.columns))
    assert len(amenity_points) == 3

    print("Amenity Analysis smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()
