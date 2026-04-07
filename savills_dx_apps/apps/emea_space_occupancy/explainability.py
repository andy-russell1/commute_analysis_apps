from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
import streamlit as st

from . import engine, ui, visuals
from .config import SCENARIO_SCORE_WEIGHTS


def _timestamp_label(value: Any) -> str:
    if value in (None, ""):
        return "Live working view"
    text = str(value)
    return text.replace("T", " ").replace("+00:00", " UTC")


def _filter_summary(filters: dict[str, Any] | None) -> str:
    if not filters:
        return "All portfolio filters active."
    parts: list[str] = []
    for key, label in [
        ("region", "Region"),
        ("country", "Country"),
        ("city", "City"),
        ("site_name", "Site"),
        ("site_type", "Site type"),
        ("building_name", "Building"),
        ("business_unit", "Business unit"),
    ]:
        values = filters.get(key) or []
        if values:
            values_text = ", ".join(map(str, values[:2]))
            if len(values) > 2:
                values_text = f"{values_text} +{len(values) - 2}"
            parts.append(f"{label}: {values_text}")
    month_range = filters.get("month_range")
    if month_range:
        start, end = month_range
        if start is not None and end is not None:
            parts.append(f"Period: {pd.Timestamp(start):%b %Y} to {pd.Timestamp(end):%b %Y}")
    return " | ".join(parts) if parts else "All portfolio filters active."


def _active_filter_count(filters: dict[str, Any] | None) -> int:
    if not filters:
        return 0
    count = 0
    for key, value in filters.items():
        if key == "month_range":
            if value:
                count += 1
            continue
        if value:
            count += 1
    return count


def _metric_delta_label(current: float, baseline: float, kind: str = "int") -> str:
    delta = float(current or 0.0) - float(baseline or 0.0)
    if kind == "pct_pts":
        return f"vs seed {delta:+.1%}"
    if kind == "ratio":
        return f"vs seed {delta:+.2f}x"
    if kind == "sqm":
        return f"vs seed {delta:+,.1f} sqm"
    if kind == "months":
        return f"vs seed {delta:+.0f} months"
    if kind == "score":
        return f"vs baseline {delta:+.1f}"
    return f"vs baseline {delta:+,.0f}"


def _score_component_strength(summary: dict[str, Any]) -> tuple[str, str]:
    components = {
        "capacity fit": float(summary.get("capacity_fit_score", 0.0) or 0.0),
        "utilisation fit": float(summary.get("utilisation_fit_score", 0.0) or 0.0),
        "standards alignment": float(summary.get("standards_compliance_score", 0.0) or 0.0),
        "implementation simplicity": float(summary.get("implementation_simplicity_score", 0.0) or 0.0),
        "consolidation efficiency": float(summary.get("consolidation_efficiency_score", 0.0) or 0.0),
    }
    strongest = max(components, key=components.get)
    weakest = min(components, key=components.get)
    return strongest, weakest


def _priority_site(outputs: pd.DataFrame) -> pd.Series | None:
    if outputs.empty:
        return None
    ranked = outputs.assign(_risk_rank=outputs["risk_rating"].map(ui.risk_rank)).sort_values(
        ["_risk_rank", "seat_gap", "scenario_score"],
        ascending=[True, True, False],
    )
    if ranked.empty:
        return None
    return ranked.iloc[0]


def priority_site(outputs: pd.DataFrame) -> pd.Series | None:
    return _priority_site(outputs)


def _risk_driver_text(outputs: pd.DataFrame) -> str:
    if outputs.empty or "key_risk" not in outputs.columns:
        return "No risk drivers are available."
    risk_rows = outputs.assign(_risk_rank=outputs["risk_rating"].map(ui.risk_rank)).sort_values(
        ["_risk_rank", "seat_gap"],
        ascending=[True, True],
    )
    top_risk = risk_rows.iloc[0] if not risk_rows.empty else None
    if top_risk is None:
        return "No risk drivers are available."
    return f"{top_risk['site_name']} leads the watchlist: {top_risk['key_risk']}"


def _standards_outcome_text(outputs: pd.DataFrame) -> str:
    if outputs.empty or "standards_compliance_score" not in outputs.columns:
        return "No standards outcome is available."
    below_75 = int((pd.to_numeric(outputs["standards_compliance_score"], errors="coerce").fillna(0.0) < 75).sum())
    below_60 = int((pd.to_numeric(outputs["standards_compliance_score"], errors="coerce").fillna(0.0) < 60).sum())
    if below_60 > 0:
        return f"{below_60} site(s) are materially below the standards comfort band and need active mitigation."
    if below_75 > 0:
        return f"{below_75} site(s) sit below the standards target band and should be explained in the room."
    return "Standards scores remain in the comfort band for the filtered portfolio."


def build_key_assumption_metrics(
    clean_sheets: dict[str, pd.DataFrame],
    assumptions_df: pd.DataFrame,
    *,
    basis_scenario_name: str,
) -> list[tuple[str, str, str | None]]:
    basis_assumptions = engine.build_working_assumptions(clean_sheets, basis_scenario_name)
    current_horizon = float(engine._base_horizon_months(assumptions_df))
    basis_horizon = float(engine._base_horizon_months(basis_assumptions))

    def _resolve(df: pd.DataFrame, parameter_name: str, default: float) -> float:
        return float(engine.resolve_parameter_value(df, parameter_name, {}, default=default) or default)

    current_growth = _resolve(assumptions_df, "Headcount Growth Default", 0.035)
    basis_growth = _resolve(basis_assumptions, "Headcount Growth Default", 0.035)
    current_attendance = _resolve(assumptions_df, "Average Attendance Uplift", 0.0)
    basis_attendance = _resolve(basis_assumptions, "Average Attendance Uplift", 0.0)
    current_peak = _resolve(assumptions_df, "Peak Attendance Buffer", 0.08)
    basis_peak = _resolve(basis_assumptions, "Peak Attendance Buffer", 0.08)
    current_desk = _resolve(assumptions_df, "Desk Sharing Ratio Target", 1.2)
    basis_desk = _resolve(basis_assumptions, "Desk Sharing Ratio Target", 1.2)
    current_sqm = _resolve(assumptions_df, "sqm per Person Target", 9.0)
    basis_sqm = _resolve(basis_assumptions, "sqm per Person Target", 9.0)
    current_meeting = _resolve(assumptions_df, "Meeting Seats per 100 Staff", 14.0)
    basis_meeting = _resolve(basis_assumptions, "Meeting Seats per 100 Staff", 14.0)

    return [
        ("Planning horizon", f"{current_horizon:.0f} months", _metric_delta_label(current_horizon, basis_horizon, "months")),
        ("Headcount growth", f"{current_growth:.1%}", _metric_delta_label(current_growth, basis_growth, "pct_pts")),
        ("Attendance uplift", f"{current_attendance:+.1%}", _metric_delta_label(current_attendance, basis_attendance, "pct_pts")),
        ("Peak buffer", f"{current_peak:.1%}", _metric_delta_label(current_peak, basis_peak, "pct_pts")),
        ("Desk sharing", f"{current_desk:.2f}x", _metric_delta_label(current_desk, basis_desk, "ratio")),
        ("Space standard", f"{current_sqm:.1f} sqm", _metric_delta_label(current_sqm, basis_sqm, "sqm")),
        ("Meeting seats / 100", f"{current_meeting:.0f}", _metric_delta_label(current_meeting, basis_meeting)),
    ]


def render_assumption_summary_strip(
    clean_sheets: dict[str, pd.DataFrame],
    assumptions_df: pd.DataFrame,
    *,
    basis_scenario_name: str,
) -> None:
    visuals.render_metric_row(
        build_key_assumption_metrics(clean_sheets, assumptions_df, basis_scenario_name=basis_scenario_name),
        columns=4,
    )


def render_provenance_panel(
    *,
    context: dict[str, Any],
    scenario_name: str,
    scenario_origin: str,
    basis_scenario_name: str | None = None,
    basis_origin: str | None = None,
    calculation_timestamp: Any = None,
    assumption_count: int | None = None,
    notes: str | None = None,
) -> None:
    ui.render_section_heading(
        "Provenance and assumptions in play",
        "Makes it explicit which workbook, seed, filters, and working assumptions are driving the current view.",
        eyebrow="Trust signals",
    )
    badges: list[str | tuple[str, str]] = [
        ("View basis", scenario_name),
        ("Origin", scenario_origin),
        ("Workbook", str(context.get("workbook_name", "Workbook"))),
        ("Filters", f"{_active_filter_count(context.get('filters'))} active"),
        ("Calculated", _timestamp_label(calculation_timestamp or context.get("last_run"))),
    ]
    if basis_scenario_name:
        label = "Seed scenario" if basis_origin == "Seed Scenario" else "Reference"
        badges.insert(2, (label, basis_scenario_name))
    if assumption_count is not None:
        badges.append(("Assumptions", str(assumption_count)))
    badges.append(("Notes", "Yes" if str(notes or "").strip() else "No"))
    ui.render_badge_row(badges)
    ui.render_callout(
        "What this view is based on",
        (
            f"This page is showing {scenario_name} as a {scenario_origin.lower()} view. "
            f"Seed and workbook provenance stay visible so the room can distinguish workbook baseline, live edits, and saved snapshots."
        ),
        tone="info",
    )


def render_formula_cards(title: str, items: Sequence[tuple[str, str, str]], *, expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        cols = st.columns(min(len(items), 4))
        for index, (heading, value, body) in enumerate(items):
            with cols[index % len(cols)]:
                ui.render_stat_card(heading, value, body, tone="info")


def render_scenario_formula_cards(*, expanded: bool = False) -> None:
    render_formula_cards(
        "How this scenario model works",
        [
            (
                "Forecast headcount",
                "Current to horizon",
                "Current demand is projected to the chosen horizon, then adjusted for scenario growth, business-unit overrides, and any hub bias.",
            ),
            (
                "Required seats",
                "Peak / desk share",
                "Required seats equal peak attendance divided by the active desk-sharing target, so seat demand rises when peaks rise or sharing tightens.",
            ),
            (
                "Required area",
                "People x sqm + uplift",
                "Required area starts with headcount times the sqm-per-person target, then adds space support uplift where meeting, focus, or collaboration provision is short.",
            ),
            (
                "Overall score",
                "Weighted score",
                "Overall scenario score blends capacity fit 30%, utilisation fit 20%, standards alignment 20%, implementation simplicity 15%, and consolidation efficiency 15%.",
            ),
        ],
        expanded=expanded,
    )


def render_planning_formula_cards(*, expanded: bool = False) -> None:
    render_formula_cards(
        "How planning outputs are allocated",
        [
            (
                "Target seats",
                "Site need x share",
                "Target seats take the site-level requirement and allocate it to each building or floor using seat-capacity share first, then usable-area share if seat data is incomplete.",
            ),
            (
                "Target area",
                "Site area x share",
                "Target area follows the same share logic so implementation teams can see how much space each building or floor is expected to carry.",
            ),
            (
                "Implementation flag",
                "Gap vs thresholds",
                "Implementation flags show whether a building or floor is under expansion pressure, a release candidate, or best treated as a rebalance / retain case.",
            ),
            (
                "Planning confidence",
                "Data quality proxy",
                "Planning confidence is highest when seat-capacity evidence is present, medium when area drives the split, and low when the model must rely on equal allocation.",
            ),
        ],
        expanded=expanded,
    )


def build_driver_bullets(
    summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    outputs: pd.DataFrame,
    *,
    baseline_label: str,
) -> list[str]:
    strongest, weakest = _score_component_strength(summary)
    forecast_delta = float(summary.get("forecast_headcount", 0.0) or 0.0) - float(baseline_summary.get("forecast_headcount", 0.0) or 0.0)
    seat_delta = float(summary.get("required_seats", 0.0) or 0.0) - float(baseline_summary.get("required_seats", 0.0) or 0.0)
    area_delta = float(summary.get("required_area_sqm", 0.0) or 0.0) - float(baseline_summary.get("required_area_sqm", 0.0) or 0.0)
    score_delta = float(summary.get("scenario_score", 0.0) or 0.0) - float(baseline_summary.get("scenario_score", 0.0) or 0.0)
    risk_delta = int(summary.get("high_risk_sites", 0) or 0) - int(baseline_summary.get("high_risk_sites", 0) or 0)

    action_counts = outputs.groupby("action_flag", dropna=False).size().sort_values(ascending=False) if not outputs.empty else pd.Series(dtype=int)
    leading_action = action_counts.index[0] if not action_counts.empty else "No action output"
    leading_action_count = int(action_counts.iloc[0]) if not action_counts.empty else 0

    bullets = [
        (
            f"Against {baseline_label}, forecast demand moves by {forecast_delta:+,.0f}, which translates into "
            f"{seat_delta:+,.0f} required seats and {area_delta:+,.0f} sqm of required area."
        ),
        (
            f"Overall scenario score moves {score_delta:+.1f}; the strongest component is {strongest}, while the weakest remains {weakest}."
        ),
        (
            f"The dominant delivery response is {leading_action} across {leading_action_count} site(s), which tells the room where the scenario is leaning operationally."
        ),
        (
            f"High-risk exposure moves {risk_delta:+,.0f} site(s). {_standards_outcome_text(outputs)} {_risk_driver_text(outputs)}"
        ),
    ]
    return bullets


def render_driver_panel(
    *,
    title: str,
    summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    outputs: pd.DataFrame,
    baseline_label: str,
) -> None:
    ui.render_bullet_panel(
        title,
        build_driver_bullets(summary, baseline_summary, outputs, baseline_label=baseline_label),
        empty_message="Driver detail is not available for the current selection.",
        tone="info",
    )


def render_lead_site_rationale(site_row: pd.Series | None, *, title: str) -> None:
    if site_row is None:
        ui.render_empty_state(title, "There is no priority site to explain for the current selection.")
        return
    ui.render_bullet_panel(
        title,
        [
            f"{site_row['site_name']} in {site_row['region']} is the lead discussion site because it currently carries a {site_row['risk_rating']} risk rating and an action of {site_row['action_flag']}.",
            f"Action rationale: {site_row['action_reason']}",
            f"Why it changed: {site_row['why_this_changed']}",
            f"Key risk to defend in the room: {site_row['key_risk']}",
        ],
        empty_message="No site rationale is available.",
        tone="accent",
    )


def scenario_provenance_table(entries: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        rows.append(
            {
                "Scenario": entry.get("scenario_name"),
                "Origin": entry.get("origin"),
                "Seed / basis": entry.get("basis_scenario_name", "Not captured"),
                "Basis type": entry.get("basis_origin", "Not captured"),
                "Calculated": _timestamp_label(entry.get("calculation_timestamp") or entry.get("timestamp")),
                "Workbook": entry.get("workbook_name"),
                "Filters": _filter_summary(entry.get("filters", {})),
                "Assumptions": entry.get("assumption_count", 0),
            }
        )
    return pd.DataFrame(rows)


def build_recommendation_points(
    preferred_entry: dict[str, Any],
    baseline_entry: dict[str, Any] | None,
    *,
    comparison_entries: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    preferred_summary = preferred_entry.get("summary", {})
    outputs = preferred_entry.get("calculated_outputs", pd.DataFrame())
    strongest, weakest = _score_component_strength(preferred_summary)

    points = [
        (
            f"Decision: back {preferred_entry.get('scenario_name')} as the lead route for the filtered portfolio, "
            f"with origin shown as {preferred_entry.get('origin')}."
        ),
        (
            f"Why the model prefers it: overall score {preferred_summary.get('scenario_score', 0.0):.1f}, strongest on {strongest}, "
            f"and still weakest on {weakest}, which is where challenge should focus."
        ),
        (
            f"Risk and standards position: {preferred_summary.get('high_risk_sites', 0)} high-risk site(s). {_standards_outcome_text(outputs)}"
        ),
        f"Remaining delivery risk: {_risk_driver_text(outputs)}",
    ]

    if baseline_entry is not None:
        baseline_summary = baseline_entry.get("summary", {})
        seat_delta = float(preferred_summary.get("seat_gap", 0.0) or 0.0) - float(baseline_summary.get("seat_gap", 0.0) or 0.0)
        area_delta = float(preferred_summary.get("area_gap_sqm", 0.0) or 0.0) - float(baseline_summary.get("area_gap_sqm", 0.0) or 0.0)
        risk_delta = int(preferred_summary.get("high_risk_sites", 0) or 0) - int(baseline_summary.get("high_risk_sites", 0) or 0)
        points.insert(
            1,
            (
                f"Why it matters versus baseline: seat gap moves {seat_delta:+,.0f}, area gap moves {area_delta:+,.0f} sqm, "
                f"and high-risk exposure moves {risk_delta:+,.0f} site(s)."
            ),
        )

    if comparison_entries:
        ranked = sorted(
            comparison_entries,
            key=lambda entry: float(entry.get("summary", {}).get("scenario_score", 0.0) or 0.0),
            reverse=True,
        )
        if len(ranked) > 1:
            gap = float(ranked[0].get("summary", {}).get("scenario_score", 0.0) or 0.0) - float(
                ranked[1].get("summary", {}).get("scenario_score", 0.0) or 0.0
            )
            if gap <= 3.0:
                points.append(
                    "Sensitivity note: the shortlist is finely balanced on score, so headcount growth, attendance uplift, desk-sharing, and space-standard assumptions could still change the answer."
                )
            else:
                points.append(
                    "Sensitivity note: headcount growth, attendance uplift, desk-sharing, and sqm-per-person settings remain the assumptions most likely to move the recommendation."
                )
    return points


def render_recommendation_panel(
    *,
    preferred_entry: dict[str, Any],
    baseline_entry: dict[str, Any] | None,
    comparison_entries: Sequence[dict[str, Any]] | None = None,
    title: str = "Recommendation rationale",
) -> None:
    ui.render_bullet_panel(
        title,
        build_recommendation_points(preferred_entry, baseline_entry, comparison_entries=comparison_entries),
        empty_message="Recommendation rationale is not available.",
        tone="success",
    )


def render_score_component_glossary(*, expanded: bool = False) -> None:
    items = [
        (
            "Capacity fit",
            f"{SCENARIO_SCORE_WEIGHTS['capacity_fit']:.0f}%",
            "Tests whether seat and area supply stay close to required demand without creating material deficits or excessive surplus.",
        ),
        (
            "Utilisation fit",
            f"{SCENARIO_SCORE_WEIGHTS['utilisation_fit']:.0f}%",
            "Checks whether peak live utilisation remains inside the active planning threshold rather than crowding or leaving space materially under-used.",
        ),
        (
            "Standards alignment",
            f"{SCENARIO_SCORE_WEIGHTS['standards_alignment']:.0f}%",
            "Measures whether seat ratio, collaboration space, meeting seats, and focus provision stay close to the active planning standards.",
        ),
        (
            "Implementation simplicity",
            f"{SCENARIO_SCORE_WEIGHTS['implementation_simplicity']:.0f}%",
            "Penalises scenarios that depend on expansion, exit, strategic moves, complex transfers, or multi-building change programmes.",
        ),
        (
            "Consolidation efficiency",
            f"{SCENARIO_SCORE_WEIGHTS['consolidation_efficiency']:.0f}%",
            "Rewards scenarios that release surplus efficiently and transfer demand cleanly into recipient locations.",
        ),
    ]
    render_formula_cards("Plain-English score definitions", items, expanded=expanded)
