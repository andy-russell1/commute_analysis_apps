from __future__ import annotations

import numpy as np
import pandas as pd

from apps.lens.core import model
from apps.lens.core.constants import MODE_CLIENT


def test_default_mode_is_client():
    assert model.get_default_mode() == MODE_CLIENT


def test_apply_macro_preset_normalizes():
    macro_weights = pd.DataFrame(
        [
            {"macro": "Talent", "weight": 0.25},
            {"macro": "Operating Environment", "weight": 0.25},
            {"macro": "Risk", "weight": 0.25},
            {"macro": "Cost", "weight": 0.25},
        ]
    )
    out = model.apply_macro_preset("Cost-led", macro_weights)
    assert abs(float(out["weight"].sum()) - 1.0) < 1e-9
    assert float(out.loc[out["macro"] == "Cost", "weight"].iloc[0]) > float(
        out.loc[out["macro"] == "Risk", "weight"].iloc[0]
    )


def test_macro_scenario_roundtrip():
    macro_weights = pd.DataFrame(
        [
            {"macro": "Talent", "weight": 0.4},
            {"macro": "Risk", "weight": 0.6},
        ]
    )
    payload = model.serialize_macro_scenario("Scenario A", macro_weights)
    loaded = model.load_macro_scenario(payload, macro_weights.copy())
    assert abs(float(loaded["weight"].sum()) - 1.0) < 1e-9
    assert payload["name"] == "Scenario A"


def test_build_city_drilldown_shape():
    bundle = {
        "overall_scores": pd.DataFrame(
            [
                {"city": "CityA", "overall_score": 0.8},
                {"city": "CityB", "overall_score": 0.6},
            ]
        ),
        "macro_scores": pd.DataFrame(
            [
                {"city": "CityA", "macro": "Talent", "macro_score": 0.9, "macro_weight": 0.6},
                {"city": "CityA", "macro": "Risk", "macro_score": 0.7, "macro_weight": 0.4},
                {"city": "CityB", "macro": "Talent", "macro_score": 0.7, "macro_weight": 0.6},
                {"city": "CityB", "macro": "Risk", "macro_score": 0.5, "macro_weight": 0.4},
            ]
        ),
        "major_scores": pd.DataFrame(
            [
                {"city": "CityA", "macro": "Talent", "major": "Demographics", "major_score": 0.9, "major_weight": 1.0},
                {"city": "CityA", "macro": "Risk", "major": "Regulatory", "major_score": 0.7, "major_weight": 1.0},
                {"city": "CityB", "macro": "Talent", "major": "Demographics", "major_score": 0.7, "major_weight": 1.0},
                {"city": "CityB", "macro": "Risk", "major": "Regulatory", "major_score": 0.5, "major_weight": 1.0},
            ]
        ),
        "contributions": pd.DataFrame(
            [
                {
                    "city": "CityA",
                    "macro": "Talent",
                    "major": "Demographics",
                    "micro": "Population",
                    "criterion_id": "c1",
                    "score": 1.0,
                    "effective_micro_weight": 0.6,
                    "contribution": 0.6,
                    "direction": "higher",
                },
                {
                    "city": "CityA",
                    "macro": "Risk",
                    "major": "Regulatory",
                    "micro": "Permits",
                    "criterion_id": "c2",
                    "score": 0.5,
                    "effective_micro_weight": 0.4,
                    "contribution": 0.2,
                    "direction": "lower",
                },
                {
                    "city": "CityB",
                    "macro": "Talent",
                    "major": "Demographics",
                    "micro": "Population",
                    "criterion_id": "c1",
                    "score": 0.5,
                    "effective_micro_weight": 0.6,
                    "contribution": 0.3,
                    "direction": "higher",
                },
                {
                    "city": "CityB",
                    "macro": "Risk",
                    "major": "Regulatory",
                    "micro": "Permits",
                    "criterion_id": "c2",
                    "score": 0.75,
                    "effective_micro_weight": 0.4,
                    "contribution": 0.3,
                    "direction": "lower",
                },
            ]
        ),
    }
    drill = model.build_city_drilldown(bundle, "CityA", top_n=3)
    assert "summary" in drill
    assert "compact_breakdown" in drill
    expected_cols = ["level", "name", "weight", "score", "contribution", "direction", "notes"]
    assert list(drill["compact_breakdown"].columns) == expected_cols
    assert drill["summary"]["city"] == "CityA"


def test_add_overall_index_bounded_01():
    overall = pd.DataFrame(
        [
            {"city": "A", "overall_score": 0.2},
            {"city": "B", "overall_score": 0.8},
        ]
    )
    out = model.add_overall_index(overall)
    assert float(out.loc[out["city"] == "A", "overall_index"].iloc[0]) == 20.0
    assert float(out.loc[out["city"] == "B", "overall_index"].iloc[0]) == 80.0


def test_add_overall_index_unbounded_minmax_and_rank_unchanged():
    overall = pd.DataFrame(
        [
            {"city": "A", "overall_score": 2.5},
            {"city": "B", "overall_score": 1.0},
            {"city": "C", "overall_score": 0.5},
        ]
    )
    before = model.build_city_ranks(overall)[["city", "overall_rank"]]
    out = model.add_overall_index(overall)
    after = model.build_city_ranks(out)[["city", "overall_rank"]]
    assert before.equals(after)
    assert float(out.loc[out["city"] == "A", "overall_index"].iloc[0]) == 100.0
    assert float(out.loc[out["city"] == "C", "overall_index"].iloc[0]) == 0.0


def test_add_overall_index_constant_scores_set_to_50():
    overall = pd.DataFrame(
        [
            {"city": "A", "overall_score": 2.0},
            {"city": "B", "overall_score": 2.0},
        ]
    )
    out = model.add_overall_index(overall)
    assert all(np.isclose(out["overall_index"], 50.0))

