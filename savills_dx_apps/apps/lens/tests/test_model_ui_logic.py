from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from apps.lens.core import model, visuals
from apps.lens.core.constants import MODE_CLIENT


def test_default_mode_is_client():
    assert model.get_default_mode() == MODE_CLIENT


def test_supported_scoring_methods_are_rank_and_percentile_only():
    assert model.get_supported_scoring_method_keys() == ["rank", "percentile"]


def test_clamp_user_scoring_method_preserves_supported_choice():
    assert model.clamp_user_scoring_method("percentile") == "percentile"
    assert model.clamp_user_scoring_method("robust_minmax") == "rank"


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
    st.session_state["lens_scoring_method"] = "percentile"
    payload = model.serialize_macro_scenario("Scenario A", macro_weights)
    loaded = model.load_macro_scenario(payload, macro_weights.copy())
    assert abs(float(loaded["weight"].sum()) - 1.0) < 1e-9
    assert payload["name"] == "Scenario A"
    assert payload["scoring_method"] == "percentile"


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


def _sample_results_bundle() -> dict[str, pd.DataFrame]:
    return {
        "overall_scores": pd.DataFrame(
            [
                {"city": "CityA", "overall_score": 0.8, "overall_index": 80.0},
                {"city": "CityB", "overall_score": 0.6, "overall_index": 60.0},
            ]
        ),
        "macro_scores": pd.DataFrame(
            [
                {"city": "CityA", "macro": "Talent", "macro_score": 0.9, "macro_index": 90.0, "macro_weight": 0.6},
                {"city": "CityA", "macro": "Risk", "macro_score": 0.7, "macro_index": 70.0, "macro_weight": 0.4},
                {"city": "CityB", "macro": "Talent", "macro_score": 0.7, "macro_index": 70.0, "macro_weight": 0.6},
                {"city": "CityB", "macro": "Risk", "macro_score": 0.5, "macro_index": 50.0, "macro_weight": 0.4},
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
        "capability_cost": pd.DataFrame(
            [
                {
                    "city": "CityA",
                    "capability_index": 82.0,
                    "cost_index": 44.0,
                    "overall_index": 80.0,
                    "market_tier": "Primary",
                },
                {
                    "city": "CityB",
                    "capability_index": 64.0,
                    "cost_index": 58.0,
                    "overall_index": 60.0,
                    "market_tier": "Secondary",
                },
            ]
        ),
    }


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


def test_add_indexed_score_column_scales_to_100_basis():
    scores = pd.DataFrame([{"city": "A", "score": 0.25}, {"city": "B", "score": 1.0}])
    out = model.add_indexed_score_column(scores, "score", "score_index")
    assert float(out.loc[out["city"] == "A", "score_index"].iloc[0]) == 25.0
    assert float(out.loc[out["city"] == "B", "score_index"].iloc[0]) == 100.0


def test_build_city_profile_comparison_uses_macro_index_outputs():
    profile = model.build_city_profile_comparison(_sample_results_bundle(), "CityA")
    assert set(profile["series"]) == {"CityA", "Portfolio median"}
    assert list(profile.columns) == [
        "series",
        "item_key",
        "item_label",
        "score_index",
        "macro",
        "major",
        "group_key",
        "group_label",
        "sort_order",
        "detail_level",
    ]
    assert float(profile.loc[(profile["series"] == "CityA") & (profile["item_key"] == "Talent"), "score_index"].iloc[0]) == 90.0


def test_build_city_profile_comparison_supports_major_and_micro_levels():
    major_profile = model.build_city_profile_comparison(_sample_results_bundle(), "CityA", level="Major")
    micro_profile = model.build_city_profile_comparison(_sample_results_bundle(), "CityA", level="Micro")
    assert "group_label" in major_profile.columns
    assert "group_label" in micro_profile.columns
    assert set(major_profile["series"]) == {"CityA", "Portfolio median"}
    assert set(micro_profile["series"]) == {"CityA", "Portfolio median"}


def test_build_city_profile_summary_returns_strengths_and_weaknesses():
    profile = model.build_city_profile_comparison(_sample_results_bundle(), "CityA", level="Macro")
    summary = model.build_city_profile_summary(profile, "CityA", "Macro", top_n=2)
    assert "strengths" in summary
    assert "weaknesses" in summary
    assert len(summary["comparison_table"]) > 0
    assert all("portfolio median" in item.lower() for item in summary["strengths"] + summary["weaknesses"])


def test_profile_group_key_returns_macro_colours():
    profile = model.build_city_profile_comparison(_sample_results_bundle(), "CityA", level="Major")
    key_df = visuals.profile_group_key(profile)
    assert list(key_df.columns) == ["Group", "Colour"]
    assert len(key_df) > 0


def test_profile_legend_items_returns_series_and_groups():
    profile = model.build_city_profile_comparison(_sample_results_bundle(), "CityA", level="Major")
    legend = visuals.profile_legend_items(profile)
    assert set(legend.keys()) == {"series", "groups"}
    assert len(legend["series"]) == 2


def test_capability_cost_bubble_uses_selected_size_and_median_labels():
    fig = visuals.capability_cost_bubble(
        _sample_results_bundle()["capability_cost"],
        size_col="overall_index",
        size_label="Overall Index (0-100)",
    )
    assert fig.data[0]["marker"]["sizeref"] is not None
    annotation_text = " ".join(str(annotation["text"]) for annotation in fig.layout.annotations)
    assert "Median capability:" in annotation_text
    assert "Median cost:" in annotation_text
    assert fig.layout.xaxis.range == (0, 100)
    assert fig.layout.yaxis.range == (0, 100)


def test_city_profile_visual_trials_render():
    profile = model.build_city_profile_comparison(_sample_results_bundle(), "CityA")
    radar_fig = visuals.city_profile_radar(profile)
    polar_fig = visuals.city_profile_polar_area(profile)
    assert len(radar_fig.data) >= 2
    assert len(polar_fig.data) >= 2

