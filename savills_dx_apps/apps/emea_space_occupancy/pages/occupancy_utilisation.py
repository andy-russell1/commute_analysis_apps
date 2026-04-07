from __future__ import annotations

import streamlit as st

from apps.emea_space_occupancy import engine, ui, visuals
from apps.emea_space_occupancy.common import prepare_page


def render_page() -> None:
    context = prepare_page(
        "Occupancy & Utilisation",
        "Observed attendance and utilisation patterns across the selected EMEA portfolio scope.",
    )
    if not context.get("ready"):
        ui.render_callout("Workbook not ready", "Load a valid workbook before reviewing utilisation patterns.", tone="warning")
        return

    baseline = engine.build_portfolio_baseline(context["clean_sheets"], context["filters"])
    trend = baseline["occupancy_trend"]
    latest = baseline["site_table"]

    if trend.empty:
        ui.render_empty_state(
            "No utilisation trends available",
            "The current filters remove all occupancy and utilisation rows from the portfolio trend view.",
        )
        return

    latest_row = trend.sort_values("month").iloc[-1]
    ui.render_hero_panel(
        eyebrow="Observed Utilisation",
        title="Translate occupancy signals into a simple pressure narrative",
        body=(
            "Use the latest reporting month to anchor the conversation, then move to the trends and site rankings "
            "to explain where the portfolio is under pressure or underused."
        ),
        badges=[
            ("Latest month", latest_row.get("month")),
            ("Average desk utilisation", visuals.format_number(latest_row.get("avg_desk_utilisation_pct", 0), "pct")),
            ("Peak desk utilisation", visuals.format_number(latest_row.get("peak_desk_utilisation_pct", 0), "pct")),
        ],
    )

    visuals.render_metric_row(
        [
            ("Average Daily Attendance", visuals.format_number(latest_row.get("avg_daily_attendance", 0)), None),
            ("Peak Daily Attendance", visuals.format_number(latest_row.get("peak_daily_attendance", 0)), None),
            ("Average Desk Utilisation", visuals.format_number(latest_row.get("avg_desk_utilisation_pct", 0), "pct"), None),
            ("Peak Desk Utilisation", visuals.format_number(latest_row.get("peak_desk_utilisation_pct", 0), "pct"), None),
            (
                "Meeting Room Utilisation",
                visuals.format_number(latest_row.get("avg_meeting_room_utilisation_pct", 0), "pct"),
                None,
            ),
        ],
        columns=5,
    )

    ui.render_callout(
        "Current read-out",
        (
            f"The latest month shows average daily attendance of {visuals.format_number(latest_row.get('avg_daily_attendance', 0))} "
            f"and peak desk utilisation of {visuals.format_number(latest_row.get('peak_desk_utilisation_pct', 0), 'pct')}. "
            "Use the rankings below to separate genuine pressure points from underused space."
        ),
        tone="info",
    )

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        with st.container(border=True):
            ui.render_section_heading(
                "Attendance trends",
                "Shows how typical and peak attendance have moved through the reporting period.",
                eyebrow="Demand over time",
            )
            fig = visuals.line_chart(trend, x="month", y="avg_daily_attendance", title="Average attendance trend")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            fig = visuals.line_chart(trend, x="month", y="peak_daily_attendance", title="Peak attendance trend")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
    with chart_col_2:
        with st.container(border=True):
            ui.render_section_heading(
                "Utilisation trends",
                "Helps explain whether pressure is being driven by desks, meeting rooms, or collaboration space.",
                eyebrow="Space use over time",
            )
            fig = visuals.line_chart(trend, x="month", y="avg_desk_utilisation_pct", title="Average desk utilisation trend")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            fig = visuals.line_chart(
                trend,
                x="month",
                y="avg_meeting_room_utilisation_pct",
                title="Meeting room utilisation trend",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    pressure_col, underused_col = st.columns(2)
    if not latest.empty:
        site_rank = latest[
            [
                "site_name",
                "region",
                "avg_desk_utilisation_pct",
                "peak_desk_utilisation_pct",
                "avg_meeting_room_utilisation_pct",
                "collaboration_space_utilisation_pct",
                "avg_desk_utilisation_delta",
            ]
        ].rename(
            columns={
                "site_name": "Site",
                "region": "Region",
                "avg_desk_utilisation_pct": "Average desk utilisation",
                "peak_desk_utilisation_pct": "Peak desk utilisation",
                "avg_meeting_room_utilisation_pct": "Meeting room utilisation",
                "collaboration_space_utilisation_pct": "Collaboration utilisation",
                "avg_desk_utilisation_delta": "Change vs mean",
            }
        )
        with pressure_col:
            with st.container(border=True):
                ui.render_section_heading(
                    "High-pressure sites",
                    "The most stretched sites in the current portfolio scope.",
                    eyebrow="Pressure ranking",
                )
                st.dataframe(
                    site_rank.sort_values(["Peak desk utilisation", "Average desk utilisation"], ascending=[False, False]).head(12),
                    use_container_width=True,
                    hide_index=True,
                )
        with underused_col:
            with st.container(border=True):
                ui.render_section_heading(
                    "Underused sites",
                    "Useful when the conversation turns to consolidation, rebalance, or opportunity capture.",
                    eyebrow="Opportunity ranking",
                )
                st.dataframe(
                    site_rank.sort_values(["Average desk utilisation", "Meeting room utilisation"], ascending=[True, True]).head(12),
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        ui.render_empty_state(
            "No site ranking available",
            "There are no site-level utilisation rows for the current filter scope.",
        )
