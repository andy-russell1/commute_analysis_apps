from __future__ import annotations

import pandas as pd

from apps.lens.core import validate


def test_validate_weight_sums_success():
    macro = pd.DataFrame([{"macro": "Talent", "weight": 0.6}, {"macro": "Risk", "weight": 0.4}])
    major = pd.DataFrame(
        [
            {"macro": "Talent", "major": "Demographics", "weight": 0.5},
            {"macro": "Talent", "major": "Education", "weight": 0.5},
            {"macro": "Risk", "major": "Regulatory", "weight": 1.0},
        ]
    )
    minor = pd.DataFrame(
        [
            {"criterion_id": "c1", "macro": "Talent", "major": "Demographics", "weight": 0.5},
            {"criterion_id": "c2", "macro": "Talent", "major": "Demographics", "weight": 0.5},
            {"criterion_id": "c3", "macro": "Talent", "major": "Education", "weight": 1.0},
            {"criterion_id": "c4", "macro": "Risk", "major": "Regulatory", "weight": 1.0},
        ]
    )
    result = validate.validate_weight_sums(macro, major, minor)
    assert result.is_valid


def test_validate_weight_sums_failure():
    macro = pd.DataFrame([{"macro": "Talent", "weight": 0.7}, {"macro": "Risk", "weight": 0.4}])
    major = pd.DataFrame(
        [
            {"macro": "Talent", "major": "Demographics", "weight": 0.7},
            {"macro": "Talent", "major": "Education", "weight": 0.7},
            {"macro": "Risk", "major": "Regulatory", "weight": 1.0},
        ]
    )
    minor = pd.DataFrame(
        [
            {"criterion_id": "c1", "macro": "Talent", "major": "Demographics", "weight": 0.2},
            {"criterion_id": "c2", "macro": "Talent", "major": "Demographics", "weight": 0.2},
            {"criterion_id": "c3", "macro": "Talent", "major": "Education", "weight": 1.0},
            {"criterion_id": "c4", "macro": "Risk", "major": "Regulatory", "weight": 1.0},
        ]
    )
    result = validate.validate_weight_sums(macro, major, minor)
    assert not result.is_valid
    assert any("Macro weights sum" in err for err in result.errors)


def test_validate_direction_map():
    direction_map = {"c1": "higher", "c2": "invalid-value"}
    result = validate.validate_direction_map(direction_map, valid_criterion_ids={"c1", "c2"})
    assert not result.is_valid
    assert any("Invalid direction values" in err for err in result.errors)


def test_validate_input_data_detects_missing_criterion():
    criteria_df = pd.DataFrame(
        [
            {"criterion_id": "c1", "macro": "Talent", "major": "Demographics", "micro": "Population"},
            {"criterion_id": "c2", "macro": "Talent", "major": "Demographics", "micro": "Workforce"},
        ]
    )
    raw_df = pd.DataFrame(
        [
            {"criterion_id": "c1", "macro": "Talent", "major": "Demographics", "micro": "Population", "CityA": 10.0},
        ]
    )
    result = validate.validate_input_data(criteria_df, raw_df, city_columns=["CityA"])
    assert not result.is_valid
    assert any("missing in Data Sheet" in err for err in result.errors)


def test_validate_input_data_all_city_missing_row_is_warning():
    criteria_df = pd.DataFrame(
        [
            {"criterion_id": "c1", "macro": "Talent", "major": "Demographics", "micro": "Population"},
        ]
    )
    raw_df = pd.DataFrame(
        [
            {"criterion_id": "c1", "macro": "Talent", "major": "Demographics", "micro": "Population", "CityA": None, "CityB": None},
            {"criterion_id": "c2", "macro": "Talent", "major": "Demographics", "micro": "Working Age", "CityA": 1.0, "CityB": 2.0},
        ]
    )
    # Keep criterion sets aligned for this test.
    criteria_df = pd.concat(
        [
            criteria_df,
            pd.DataFrame([{"criterion_id": "c2", "macro": "Talent", "major": "Demographics", "micro": "Working Age"}]),
        ],
        ignore_index=True,
    )
    result = validate.validate_input_data(criteria_df, raw_df, city_columns=["CityA", "CityB"])
    assert result.is_valid
    assert any("no numeric values" in w for w in result.warnings)

