from __future__ import annotations

import numpy as np
import pandas as pd

from apps.lens.core import scoring
from apps.lens.core.constants import (
    DIRECTION_HIGHER,
    DIRECTION_LOWER,
    SCORING_METHOD_LOG_ROBUST_MINMAX,
    SCORING_METHOD_MINMAX,
    SCORING_METHOD_PERCENTILE,
    SCORING_METHOD_PERCENTILE_RANK,
    SCORING_METHOD_RANK,
    SCORING_METHOD_ROBUST_MINMAX,
)


def test_infer_direction_from_source():
    assert scoring.infer_direction_from_source("unemployment; Lower is better") == DIRECTION_LOWER
    assert scoring.infer_direction_from_source("count; Higher is better") == DIRECTION_HIGHER
    assert scoring.infer_direction_from_source(None) == DIRECTION_HIGHER


def test_normalize_method_key_maps_legacy_percentile():
    assert scoring.normalize_scoring_method_key("percentile") == SCORING_METHOD_PERCENTILE
    assert scoring.normalize_scoring_method_key("percentile_rank") == SCORING_METHOD_PERCENTILE_RANK


def test_rank_score_normalization():
    values = pd.Series([10.0, 20.0, 30.0], index=["A", "B", "C"])
    scores = scoring.score_series(values, method=SCORING_METHOD_RANK, direction="higher")
    assert scores["A"] == 0.0
    assert scores["B"] == 0.5
    assert scores["C"] == 1.0


def test_percentile_rank_score_normalization():
    values = pd.Series([10.0, 20.0, 30.0], index=["A", "B", "C"])
    scores = scoring.score_series(values, method=SCORING_METHOD_PERCENTILE, direction="higher")
    assert scores["A"] == 1.0 / 3.0
    assert scores["B"] == 2.0 / 3.0
    assert scores["C"] == 1.0


def test_competition_rank_has_no_half_steps_and_supports_ties():
    values = pd.Series([100.0, 100.0, 5000.0], index=["A", "B", "C"])
    ranks = scoring.compute_rank_series(values, direction="higher")
    assert ranks["C"] == 1.0
    assert ranks["A"] == 2.0
    assert ranks["B"] == 2.0
    assert all(float(value).is_integer() for value in ranks.dropna())


def test_percentile_is_distinct_from_rank_for_tied_distribution():
    values = pd.Series([10.0, 10.0, 20.0, 30.0], index=["A", "B", "C", "D"])
    rank_scores = scoring.score_series(values, method=SCORING_METHOD_RANK, direction="higher")
    percentile_scores = scoring.score_series(values, method=SCORING_METHOD_PERCENTILE, direction="higher")
    assert rank_scores["A"] == 0.0
    assert percentile_scores["A"] == 0.5
    assert percentile_scores["C"] == 0.75
    assert percentile_scores["D"] == 1.0


def test_percentile_applies_direction_before_ecdf():
    values = pd.Series([10.0, 20.0, 30.0], index=["A", "B", "C"])
    scores = scoring.score_series(values, method=SCORING_METHOD_PERCENTILE, direction="lower")
    assert scores["A"] == 1.0
    assert scores["B"] == 2.0 / 3.0
    assert scores["C"] == 1.0 / 3.0


def test_minmax_and_robust_and_log_robust_ranges():
    values = pd.Series([100.0, 101.0, 5000.0, np.nan], index=["A", "B", "C", "D"])
    for method in [SCORING_METHOD_MINMAX, SCORING_METHOD_ROBUST_MINMAX, SCORING_METHOD_LOG_ROBUST_MINMAX]:
        scores = scoring.score_series(values, method=method, direction="higher")
        valid = scores.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()


def test_constant_column_returns_half():
    values = pd.Series([3.0, 3.0, 3.0, np.nan], index=["A", "B", "C", "D"])
    for method in [
        SCORING_METHOD_RANK,
        SCORING_METHOD_PERCENTILE,
        SCORING_METHOD_MINMAX,
        SCORING_METHOD_ROBUST_MINMAX,
        SCORING_METHOD_LOG_ROBUST_MINMAX,
    ]:
        scores = scoring.score_series(values, method=method, direction="higher")
        valid = scores.dropna()
        assert all(abs(float(v) - 0.5) < 1e-9 for v in valid.values)


def test_direction_inversion_works():
    values = pd.Series([10.0, 20.0, 30.0], index=["A", "B", "C"])
    high_scores = scoring.score_series(values, method=SCORING_METHOD_MINMAX, direction="higher")
    low_scores = scoring.score_series(values, method=SCORING_METHOD_MINMAX, direction="lower")
    assert abs(float(low_scores["A"]) - (1.0 - float(high_scores["A"]))) < 1e-9
    assert abs(float(low_scores["C"]) - (1.0 - float(high_scores["C"]))) < 1e-9


def test_normalize_weight_map():
    normalized = scoring.normalize_weight_map({"Talent": 2.0, "Risk": 1.0})
    assert normalized["Talent"] == 2.0 / 3.0
    assert normalized["Risk"] == 1.0 / 3.0


def test_aggregation_math():
    micro_scores = pd.DataFrame(
        [
            {
                "criterion_id": "m1",
                "macro": "Talent",
                "major": "Demographics",
                "micro": "Population",
                "city": "London",
                "score": 0.8,
            },
            {
                "criterion_id": "m2",
                "macro": "Talent",
                "major": "Demographics",
                "micro": "Working age",
                "city": "London",
                "score": 0.4,
            },
            {
                "criterion_id": "m1",
                "macro": "Talent",
                "major": "Demographics",
                "micro": "Population",
                "city": "Paris",
                "score": 0.2,
            },
            {
                "criterion_id": "m2",
                "macro": "Talent",
                "major": "Demographics",
                "micro": "Working age",
                "city": "Paris",
                "score": 0.6,
            },
        ]
    )

    weighted_criteria = pd.DataFrame(
        [
            {
                "criterion_id": "m1",
                "macro": "Talent",
                "major": "Demographics",
                "macro_weight": 1.0,
                "major_weight": 1.0,
                "minor_weight": 0.25,
                "effective_micro_weight": 0.25,
            },
            {
                "criterion_id": "m2",
                "macro": "Talent",
                "major": "Demographics",
                "macro_weight": 1.0,
                "major_weight": 1.0,
                "minor_weight": 0.75,
                "effective_micro_weight": 0.75,
            },
        ]
    )

    out = scoring.aggregate_scores(micro_scores, weighted_criteria)
    overall = out["overall_scores"].set_index("city")["overall_score"].to_dict()
    assert abs(overall["London"] - 0.5) < 1e-9
    assert abs(overall["Paris"] - 0.5) < 1e-9


def test_compute_micro_scores_respects_direction_override():
    raw = pd.DataFrame(
        [
            {
                "criterion_id": "c1",
                "macro": "Risk",
                "major": "Regulatory",
                "micro": "Permitting delay",
                "source": "days; Lower is better",
                "CityA": 10,
                "CityB": 20,
            }
        ]
    )
    out = scoring.compute_micro_scores(
        raw_data=raw,
        city_columns=["CityA", "CityB"],
        direction_map={"c1": DIRECTION_HIGHER},
        method=SCORING_METHOD_RANK,
    )
    city_scores = out.set_index("city")["score"].to_dict()
    assert city_scores["CityB"] == 1.0
    assert city_scores["CityA"] == 0.0


def test_compute_micro_scores_uses_python_competition_ranks():
    raw = pd.DataFrame(
        [
            {
                "criterion_id": "c1",
                "macro": "Talent",
                "major": "Demographics",
                "micro": "Population",
                "source": "count; Higher is better",
                "CityA": 100,
                "CityB": 100,
                "CityC": 300,
            }
        ]
    )
    out = scoring.compute_micro_scores(
        raw_data=raw,
        city_columns=["CityA", "CityB", "CityC"],
        direction_map={},
        method=SCORING_METHOD_RANK,
    )
    ranks = out.set_index("city")["rank"].to_dict()
    assert ranks == {"CityA": 2.0, "CityB": 2.0, "CityC": 1.0}


