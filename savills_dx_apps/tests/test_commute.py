from __future__ import annotations

import pandas as pd

from core.commute import filter_travel_time_valid, match_columns


def test_commute_validation_passes():
    df = pd.DataFrame(
        {
            "Employee ID": ["E1", "E2"],
            "Metric": ["travel_time", "travel_time"],
            "Value": [1800, 2400],
            "Query Transport Method": ["bus", "train"],
            "Employee - Lat": [51.5, 51.6],
            "Employee - Long": [-0.1, -0.12],
            "Office ID": ["O1", "O1"],
            "Office - Address": ["Office A", "Office A"],
            "Office - Lat": [51.5, 51.5],
            "Office - Long": [-0.1, -0.1],
        }
    )
    cols = match_columns(df)
    out = filter_travel_time_valid(df, cols)
    assert not out.empty
    assert "travel_time_min" in out.columns


def test_commute_missing_column_raises():
    df = pd.DataFrame({"Metric": ["travel_time"], "Value": [100]})
    try:
        match_columns(df)
    except KeyError:
        assert True
    else:
        assert False
