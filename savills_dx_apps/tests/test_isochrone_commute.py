from __future__ import annotations

import pandas as pd

from apps.isochrone_commute.wizard import _extract_worker_points


def test_extract_worker_points_from_successful_format():
    df = pd.DataFrame(
        {
            "Employee ID": ["E1", "E1", "E2"],
            "Metric": ["travel_time", "travel_time", "travel_time"],
            "Value": [1800, 1900, 2400],
            "Query Transport Method": ["bus", "train", "train"],
            "Employee - Lat": [51.5, 51.5, 51.6],
            "Employee - Long": [-0.1, -0.1, -0.12],
            "Office ID": ["O1", "O1", "O1"],
            "Office - Address": ["Office A", "Office A", "Office A"],
            "Office - Lat": [51.5, 51.5, 51.5],
            "Office - Long": [-0.1, -0.1, -0.1],
        }
    )

    out = _extract_worker_points(df)

    assert len(out) == 2
    assert {"employeeID", "lat", "lon"}.issubset(set(out.columns))
    assert sorted(out["employeeID"].tolist()) == ["E1", "E2"]


def test_extract_worker_points_from_geocoded_format():
    df = pd.DataFrame(
        {
            "Employee ID": ["E10", "E11"],
            "Latitude": [51.51, 51.52],
            "Longitude": [-0.11, -0.12],
            "Postcode": ["AB1", "AB2"],
        }
    )

    out = _extract_worker_points(df)

    assert len(out) == 2
    assert sorted(out["employeeID"].tolist()) == ["E10", "E11"]
    assert out["lat"].notna().all()
    assert out["lon"].notna().all()


def test_extract_worker_points_raises_without_coordinates():
    df = pd.DataFrame({"Employee ID": ["E1", "E2"], "Postcode": ["AB1", "AB2"]})

    try:
        _extract_worker_points(df)
    except KeyError:
        assert True
    else:
        assert False
