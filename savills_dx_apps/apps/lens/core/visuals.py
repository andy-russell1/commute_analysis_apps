from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .constants import SAVILLS_COLOR_SEQUENCE, SAVILLS_MACRO_COLOR_MAP, SAVILLS_MARKET_TIER_COLOR_MAP

_CHART_BG = "rgba(0, 0, 0, 0)"
_GRID_COLOUR = "rgba(127, 137, 160, 0.24)"
_LINE_COLOUR = "rgba(127, 137, 160, 0.45)"
_SURFACE_TEXT_COLOUR = "#F8FAFC"
_BENCHMARK_PROFILE_HEIGHT = 700


def _apply_executive_chart_layout(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        legend_title_text="",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def _format_polar_tick_labels(labels: list[str]) -> list[str]:
    formatted: list[str] = []
    for label in labels:
        text = str(label)
        if " | " in text:
            formatted.append(text.replace(" | ", "<br>"))
            continue
        if len(text) > 18 and " " in text:
            parts = text.split()
            midpoint = max(1, len(parts) // 2)
            formatted.append(" ".join(parts[:midpoint]) + "<br>" + " ".join(parts[midpoint:]))
            continue
        formatted.append(text)
    return formatted


def _format_major_label(label: str) -> str:
    text = str(label).strip()
    replacements = {
        "Operating Environment": "Operating Env.",
        "Complementary Services": "Complementary Serv.",
        "Regulatory considerations": "Regulatory consid.",
        "Available Grants and Incentives": "Grants and Incentives",
        "Proximity to suppliers and market": "Supplier and market prox.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _detail_level_from_profile(profile_df: pd.DataFrame) -> str:
    if "detail_level" not in profile_df.columns or profile_df.empty:
        return "Macro"
    values = profile_df["detail_level"].dropna().astype(str).unique().tolist()
    return values[0] if values else "Macro"


def _abbreviate_micro_label(label: str) -> str:
    text = str(label).strip()
    replacements = {
        "proximity": "prox.",
        "international": "intl.",
        "public-private": "PPP",
        "partnership": "partner.",
        "opportunities": "opps.",
        "radiochemists": "radiochem.",
        "engineers": "eng.",
        "considerations": "consid.",
        "infrastructure": "infra.",
        "university": "univ.",
        "institution": "inst.",
        "working": "working",
        "population": "pop.",
        "utilities": "utilities",
        "presence": "presence",
    }
    for old, new in replacements.items():
        text = text.replace(old, new).replace(old.capitalize(), new.capitalize())
    words = text.split()
    if len(words) > 4:
        text = " ".join(words[:4]) + "..."
    return text


def _profile_display_labels(axis_meta: pd.DataFrame, detail_level: str) -> list[str]:
    labels = axis_meta["item_label"].astype(str).tolist()
    if detail_level == "Micro":
        labels = [_abbreviate_micro_label(label) for label in labels]
    elif detail_level == "Major":
        labels = [_format_major_label(label) for label in labels]
    return _format_polar_tick_labels(labels)


def _apply_profile_polar_layout(fig: go.Figure, axis_meta: pd.DataFrame, *, height: int = 620) -> go.Figure:
    tickvals = axis_meta["theta"].tolist()
    ticktext = _format_polar_tick_labels(axis_meta["item_label"].astype(str).tolist())
    fig.update_layout(
        barmode="overlay",
        polar=dict(
            radialaxis=dict(
                range=[0, 100],
                tickmode="array",
                tickvals=[0, 25, 50, 75, 100],
                gridcolor=_GRID_COLOUR,
                angle=0,
                showline=False,
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                rotation=90,
                direction="clockwise",
                gridcolor=_GRID_COLOUR,
                linecolor=_LINE_COLOUR,
                tickfont=dict(size=11),
            ),
            bgcolor=_CHART_BG,
            domain=dict(x=[0.06, 0.94], y=[0.08, 0.92]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=40, r=40, t=80, b=70),
    )
    return _apply_executive_chart_layout(fig, height=height)


def _build_profile_axis_meta(profile_df: pd.DataFrame) -> pd.DataFrame:
    axis_meta = (
        profile_df.sort_values("sort_order")[["item_key", "item_label", "group_key", "group_label", "macro", "sort_order"]]
        .drop_duplicates("item_key")
        .reset_index(drop=True)
    )
    theta_step = 360.0 / max(len(axis_meta), 1)
    axis_meta["theta"] = [idx * theta_step for idx in range(len(axis_meta))]
    return axis_meta


def _add_profile_axis_annotations(
    fig: go.Figure,
    axis_meta: pd.DataFrame,
    *,
    labels: list[str] | None = None,
    radius: float = 0.49,
    x_radius: float | None = None,
    y_radius: float | None = None,
    font_size: int = 12,
    bg_alpha: float = 0.82,
    borderpad: int = 1,
) -> None:
    if axis_meta.empty:
        return
    centre_x = 0.5
    centre_y = 0.5
    resolved_x_radius = x_radius if x_radius is not None else radius
    resolved_y_radius = y_radius if y_radius is not None else radius
    label_values = labels if labels is not None else _format_polar_tick_labels(axis_meta["item_label"].astype(str).tolist())
    for theta_deg, label in zip(axis_meta["theta"].tolist(), label_values, strict=False):
        theta_rad = np.deg2rad(90.0 - float(theta_deg))
        x = centre_x + (resolved_x_radius * np.cos(theta_rad))
        y = centre_y + (resolved_y_radius * np.sin(theta_rad))
        if x > 0.58:
            xanchor = "left"
        elif x < 0.42:
            xanchor = "right"
        else:
            xanchor = "center"
        if y > 0.58:
            yanchor = "bottom"
        elif y < 0.42:
            yanchor = "top"
        else:
            yanchor = "middle"
        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor=xanchor,
            yanchor=yanchor,
            align="center",
            font=dict(size=font_size, color=_SURFACE_TEXT_COLOUR),
            bgcolor=f"rgba(38,42,67,{bg_alpha:.2f})",
            borderpad=borderpad,
        )


def _hex_to_rgba(hex_value: str, alpha: float) -> str:
    cleaned = hex_value.lstrip("#")
    red, green, blue = (int(cleaned[idx : idx + 2], 16) for idx in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha:.3f})"


def _add_profile_group_underlays(fig: go.Figure, axis_meta: pd.DataFrame) -> None:
    if axis_meta.empty:
        return
    macro_count = int(axis_meta["macro"].nunique(dropna=True))
    if macro_count <= 1 or macro_count >= len(axis_meta):
        return

    theta_step = 360.0 / max(len(axis_meta), 1)
    grouped = axis_meta.groupby("macro", sort=False)
    for macro_name, group_df in grouped:
        if pd.isna(macro_name):
            continue
        start_theta = float(group_df["theta"].min()) - (theta_step / 2.0)
        width = float(len(group_df) * theta_step)
        colour = SAVILLS_MACRO_COLOR_MAP.get(str(macro_name), "#D5DDE5")
        fig.add_trace(
            go.Barpolar(
                r=[100],
                base=[0],
                theta=[start_theta + (width / 2.0)],
                width=[width],
                marker=dict(color=_hex_to_rgba(colour, 0.12), line=dict(color=_hex_to_rgba(colour, 0.0), width=0)),
                opacity=1.0,
                hoverinfo="skip",
                showlegend=False,
            )
        )


def profile_group_key(profile_df: pd.DataFrame) -> pd.DataFrame:
    if profile_df.empty:
        return pd.DataFrame(columns=["Group", "Colour"])
    groups = (
        profile_df[["macro"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"macro": "Group"})
        .reset_index(drop=True)
    )
    groups["Colour"] = groups["Group"].map(lambda value: SAVILLS_MACRO_COLOR_MAP.get(str(value), "#D5DDE5"))
    return groups


def profile_legend_items(profile_df: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    groups = profile_group_key(profile_df)
    return {
        "series": [
            {"label": str(profile_df["series"].drop_duplicates().tolist()[0]), "colour": "#F2D500", "style": "primary"},
            {"label": "Portfolio median", "colour": "#4A9A8D", "style": "secondary"},
        ],
        "groups": [{"label": str(row["Group"]), "colour": str(row["Colour"])} for _, row in groups.iterrows()],
    }


def overall_stacked_bar(
    macro_scores: pd.DataFrame,
    overall_scores: pd.DataFrame,
    macro_order: list[str] | None = None,
) -> go.Figure:
    order_metric = "overall_index" if "overall_index" in overall_scores.columns else "overall_score"
    order_df = overall_scores.sort_values(order_metric, ascending=False)
    city_order = order_df["city"].tolist()

    plot_df = macro_scores.copy()
    score_cols = ["city", "overall_score"]
    if "overall_index" in overall_scores.columns:
        score_cols.append("overall_index")
    plot_df = plot_df.merge(overall_scores[score_cols], on="city", how="left")
    plot_df["macro_contribution_raw"] = plot_df["macro_score"] * plot_df["macro_weight"]
    raw_totals = (
        plot_df.groupby("city", as_index=False)["macro_contribution_raw"]
        .sum()
        .rename(columns={"macro_contribution_raw": "total_raw"})
    )
    plot_df = plot_df.merge(raw_totals, on="city", how="left")

    if "overall_index" in plot_df.columns:
        plot_df["macro_share"] = 0.0
        valid_raw = plot_df["total_raw"] > 0
        plot_df.loc[valid_raw, "macro_share"] = (
            plot_df.loc[valid_raw, "macro_contribution_raw"] / plot_df.loc[valid_raw, "total_raw"]
        )

        weight_sums = plot_df.groupby("city", dropna=False)["macro_weight"].transform("sum")
        valid_weight = (~valid_raw) & (weight_sums > 0)
        plot_df.loc[valid_weight, "macro_share"] = (
            plot_df.loc[valid_weight, "macro_weight"] / weight_sums.loc[valid_weight]
        )

        plot_df["segment_value"] = plot_df["macro_share"] * plot_df["overall_index"]
        x_col = "segment_value"
        x_label = "Overall Score (0-100)"
    else:
        plot_df["segment_value"] = plot_df["macro_contribution_raw"]
        x_col = "segment_value"
        x_label = "Score"

    if macro_order:
        remaining = [m for m in plot_df["macro"].unique().tolist() if m not in macro_order]
        category_order = macro_order + sorted(remaining)
    else:
        category_order = sorted(plot_df["macro"].unique().tolist())
    plot_df["macro"] = pd.Categorical(plot_df["macro"], categories=category_order, ordered=True)

    fig = px.bar(
        plot_df,
        x=x_col,
        y="city",
        color="macro",
        orientation="h",
        category_orders={"city": city_order, "macro": category_order},
        barmode="stack",
        labels={x_col: x_label, "city": "City", "macro": "Macro"},
        color_discrete_map=SAVILLS_MACRO_COLOR_MAP,
        color_discrete_sequence=SAVILLS_COLOR_SEQUENCE,
    )
    if "overall_index" in plot_df.columns:
        fig.update_layout(xaxis_range=[0, 100])
    fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def macro_drilldown_bar(major_scores: pd.DataFrame, macro_scores: pd.DataFrame, selected_macro: str) -> go.Figure:
    plot_df = major_scores[major_scores["macro"] == selected_macro].copy()
    plot_df = plot_df.sort_values(["city", "major"])
    macro_totals = macro_scores[macro_scores["macro"] == selected_macro].copy()
    score_cols = ["city", "macro_score"]
    if "macro_index" in macro_totals.columns:
        score_cols.append("macro_index")
    macro_totals = macro_totals[score_cols].drop_duplicates("city")
    plot_df = plot_df.merge(macro_totals, on="city", how="left")

    plot_df["major_contribution_raw"] = plot_df["major_score"] * plot_df["major_weight"]
    raw_totals = (
        plot_df.groupby("city", as_index=False)["major_contribution_raw"]
        .sum()
        .rename(columns={"major_contribution_raw": "total_raw"})
    )
    plot_df = plot_df.merge(raw_totals, on="city", how="left")

    if "macro_index" in plot_df.columns:
        plot_df["major_share"] = 0.0
        valid_raw = plot_df["total_raw"] > 0
        plot_df.loc[valid_raw, "major_share"] = (
            plot_df.loc[valid_raw, "major_contribution_raw"] / plot_df.loc[valid_raw, "total_raw"]
        )

        weight_sums = plot_df.groupby("city", dropna=False)["major_weight"].transform("sum")
        valid_weight = (~valid_raw) & (weight_sums > 0)
        plot_df.loc[valid_weight, "major_share"] = plot_df.loc[valid_weight, "major_weight"] / weight_sums.loc[valid_weight]

        plot_df["segment_value"] = plot_df["major_share"] * plot_df["macro_index"]
        x_col = "segment_value"
        x_label = "Macro Score (0-100)"
        order_metric = "macro_index"
    else:
        plot_df["segment_value"] = plot_df["major_contribution_raw"]
        x_col = "segment_value"
        x_label = "Macro Score"
        order_metric = "macro_score"

    city_order = macro_totals.sort_values(order_metric, ascending=False)["city"].tolist()
    major_order = sorted(plot_df["major"].dropna().unique().tolist())

    fig = px.bar(
        plot_df,
        x=x_col,
        y="city",
        color="major",
        orientation="h",
        category_orders={"city": city_order, "major": major_order},
        barmode="stack",
        labels={x_col: x_label, "city": "City", "major": "Major"},
        color_discrete_sequence=SAVILLS_COLOR_SEQUENCE,
    )
    if "macro_index" in plot_df.columns:
        fig.update_layout(xaxis_range=[0, 100])
    fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def capability_cost_bubble(
    capability_cost_df: pd.DataFrame,
    *,
    size_col: str = "overall_index",
    size_label: str = "Overall Index (0-100)",
) -> go.Figure:
    plot_df = capability_cost_df.copy()
    x_col = "capability_index" if "capability_index" in plot_df.columns else "capability_score"
    y_col = "cost_index" if "cost_index" in plot_df.columns else "cost_score"
    if "population" not in plot_df.columns:
        plot_df["population"] = 1.0
    resolved_size_col = size_col if size_col in plot_df.columns else "population"
    resolved_size_label = size_label if resolved_size_col == size_col else "Population"
    x_median = float(plot_df[x_col].median()) if not plot_df.empty else 0.0
    y_median = float(plot_df[y_col].median()) if not plot_df.empty else 0.0

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        size=resolved_size_col,
        color="market_tier",
        hover_name="city",
        text="city",
        labels={
            x_col: "Capability Score (0-100)",
            y_col: "Cost Score (0-100)",
            "market_tier": "Market Tier",
            resolved_size_col: resolved_size_label,
        },
        size_max=55,
        color_discrete_map=SAVILLS_MARKET_TIER_COLOR_MAP,
        category_orders={"market_tier": ["Primary", "Secondary", "Tertiary"]},
    )
    fig.update_traces(textposition="top center")
    fig.add_vline(x=x_median, line_dash="dash", line_color=_LINE_COLOUR, line_width=1.5)
    fig.add_hline(y=y_median, line_dash="dash", line_color=_LINE_COLOUR, line_width=1.5)

    fig.add_annotation(
        x=x_median,
        y=100,
        text=f"Median capability: {x_median:.1f}",
        showarrow=False,
        yshift=12,
        font=dict(size=11, color=_SURFACE_TEXT_COLOUR),
        bgcolor="rgba(38,42,67,0.16)",
    )
    fig.add_annotation(
        x=100,
        y=y_median,
        text=f"Median cost: {y_median:.1f}",
        showarrow=False,
        xshift=-50,
        font=dict(size=11, color=_SURFACE_TEXT_COLOUR),
        bgcolor="rgba(38,42,67,0.16)",
    )

    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, text="Lower capability / stronger cost position", showarrow=False)
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98, text="Higher capability / stronger cost position", showarrow=False)
    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.02, text="Lower capability / weaker cost position", showarrow=False)
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.02, text="Higher capability / weaker cost position", showarrow=False)

    fig.update_xaxes(range=[0, 100], ticksuffix="", dtick=20, gridcolor=_GRID_COLOUR, zeroline=False)
    fig.update_yaxes(range=[0, 100], ticksuffix="", dtick=20, gridcolor=_GRID_COLOUR, zeroline=False)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return _apply_executive_chart_layout(fig)


def macro_contribution_bar(macro_df: pd.DataFrame) -> go.Figure:
    plot_df = macro_df.copy()
    fig = px.bar(
        plot_df,
        x="macro_contribution",
        y="macro",
        orientation="h",
        color="macro",
        labels={"macro_contribution": "Contribution", "macro": "Macro"},
        color_discrete_map=SAVILLS_MACRO_COLOR_MAP,
        color_discrete_sequence=SAVILLS_COLOR_SEQUENCE,
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def benchmark_delta_bar(comparison_table: pd.DataFrame, top_n: int = 12) -> go.Figure:
    plot_df = comparison_table.copy()
    if plot_df.empty:
        return _apply_executive_chart_layout(go.Figure(), height=420)
    plot_df["delta_to_benchmark"] = pd.to_numeric(plot_df["delta_to_benchmark"], errors="coerce")
    plot_df = plot_df.dropna(subset=["delta_to_benchmark"]).copy()
    if plot_df.empty:
        return _apply_executive_chart_layout(go.Figure(), height=420)

    plot_df["display_label"] = np.where(
        plot_df["group_label"].astype(str) == plot_df["item_label"].astype(str),
        plot_df["item_label"].astype(str),
        plot_df["item_label"].astype(str) + " | " + plot_df["group_label"].astype(str),
    )
    plot_df["gap_direction"] = np.where(plot_df["delta_to_benchmark"] >= 0, "Ahead", "Behind")
    plot_df["gap_abs"] = plot_df["delta_to_benchmark"].abs()
    plot_df = plot_df.nlargest(top_n, "gap_abs").sort_values("delta_to_benchmark")

    fig = px.bar(
        plot_df,
        x="delta_to_benchmark",
        y="display_label",
        orientation="h",
        color="gap_direction",
        labels={"delta_to_benchmark": "Index Gap", "display_label": ""},
        color_discrete_map={"Ahead": "#4A9A8D", "Behind": "#C96B5C"},
    )
    fig.add_vline(x=0, line_width=1.2, line_color=_LINE_COLOUR)
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Gap to Benchmark (0-100 index points)",
        yaxis_title="",
    )
    return _apply_executive_chart_layout(fig, height=max(420, 40 * len(plot_df) + 80))


def city_profile_radar(profile_df: pd.DataFrame) -> go.Figure:
    plot_df = profile_df.copy()
    if plot_df.empty:
        return _apply_executive_chart_layout(go.Figure(), height=520)
    axis_meta = _build_profile_axis_meta(plot_df)
    detail_level = _detail_level_from_profile(plot_df)
    display_labels = _profile_display_labels(axis_meta, detail_level)
    theta_map = dict(zip(axis_meta["item_key"], axis_meta["theta"], strict=False))
    fig = go.Figure()
    _add_profile_group_underlays(fig, axis_meta)
    series_order = plot_df["series"].drop_duplicates().tolist()
    colour_map = {
        series_order[0]: "#F2D500",
        series_order[1] if len(series_order) > 1 else "Portfolio median": "#4A9A8D",
    }
    for series in series_order:
        series_df = (
            plot_df[plot_df["series"] == series]
            .copy()
            .sort_values("sort_order")
        )
        theta = [theta_map[key] for key in series_df["item_key"]]
        r = pd.to_numeric(series_df["score_index"], errors="coerce").tolist()
        if theta:
            theta.append(theta[0])
            r.append(r[0])
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                name=series,
                mode="lines+markers",
                fill="toself",
                fillcolor="rgba(242, 213, 0, 0.24)" if series == series_order[0] else "rgba(74, 154, 141, 0.10)",
                opacity=1.0,
                line=dict(
                    color=colour_map.get(series, "#F2D500"),
                    width=3.5 if series == series_order[0] else 2.4,
                    dash="solid" if series == series_order[0] else "dash",
                ),
                marker=dict(
                    color=colour_map.get(series, "#F2D500"),
                    size=8 if series == series_order[0] else 6,
                    line=dict(
                        color="#262A43" if series == series_order[0] else colour_map.get(series, "#4A9A8D"),
                        width=1,
                    ),
                ),
                hovertemplate="%{text}<br>%{fullData.name}: %{r:.1f}<extra></extra>",
                text=axis_meta["item_label"].tolist() + ([axis_meta["item_label"].tolist()[0]] if not axis_meta.empty else []),
            )
        )
    dense_view = len(axis_meta) >= 14
    ultra_dense = detail_level == "Micro"
    major_view = detail_level == "Major"
    if ultra_dense:
        polar_domain = dict(x=[0.10, 0.90], y=[0.08, 0.94])
        chart_margin = dict(l=122, r=122, t=48, b=40)
        label_radius = 0.37
        label_x_radius = 0.29
        label_y_radius = 0.37
        label_font_size = 8
        label_bg_alpha = 0.78
        label_borderpad = 0
        chart_height = _BENCHMARK_PROFILE_HEIGHT
    elif major_view:
        polar_domain = dict(x=[0.10, 0.90], y=[0.08, 0.95])
        chart_margin = dict(l=108, r=108, t=32, b=34)
        label_radius = 0.42
        label_x_radius = 0.31
        label_y_radius = 0.42
        label_font_size = 10
        label_bg_alpha = 0.72
        label_borderpad = 1
        chart_height = _BENCHMARK_PROFILE_HEIGHT
    elif dense_view:
        polar_domain = dict(x=[0.11, 0.89], y=[0.08, 0.95])
        chart_margin = dict(l=108, r=108, t=30, b=32)
        label_radius = 0.40
        label_x_radius = 0.30
        label_y_radius = 0.40
        label_font_size = 9
        label_bg_alpha = 0.88
        label_borderpad = 1
        chart_height = _BENCHMARK_PROFILE_HEIGHT
    else:
        polar_domain = dict(x=[0.08, 0.92], y=[0.08, 0.96])
        chart_margin = dict(l=74, r=74, t=24, b=26)
        label_radius = 0.43
        label_x_radius = 0.27
        label_y_radius = 0.43
        label_font_size = 12
        label_bg_alpha = 0.72
        label_borderpad = 1
        chart_height = _BENCHMARK_PROFILE_HEIGHT
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 100],
                tickmode="array",
                tickvals=[0, 25, 50, 75, 100],
                gridcolor=_GRID_COLOUR,
                gridwidth=1,
                angle=0,
                showline=False,
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=axis_meta["theta"].tolist(),
                ticktext=[""] * len(axis_meta),
                rotation=90,
                direction="clockwise",
                gridcolor=_GRID_COLOUR,
                linecolor=_LINE_COLOUR,
                tickfont=dict(size=12, color="rgba(0,0,0,0)"),
            ),
            bgcolor=_CHART_BG,
            domain=polar_domain,
        ),
        showlegend=False,
        margin=chart_margin,
    )
    _add_profile_axis_annotations(
        fig,
        axis_meta,
        labels=display_labels,
        radius=label_radius,
        x_radius=label_x_radius,
        y_radius=label_y_radius,
        font_size=label_font_size,
        bg_alpha=label_bg_alpha,
        borderpad=label_borderpad,
    )
    return _apply_executive_chart_layout(fig, height=chart_height)


def city_profile_polar_area(profile_df: pd.DataFrame) -> go.Figure:
    plot_df = profile_df.copy()
    if plot_df.empty:
        return _apply_executive_chart_layout(go.Figure(), height=520)
    axis_meta = _build_profile_axis_meta(plot_df)
    theta_step = 360.0 / max(len(axis_meta), 1)
    theta_map = dict(zip(axis_meta["item_key"], axis_meta["theta"], strict=False))
    series_order = plot_df["series"].drop_duplicates().tolist()
    selected_label = series_order[0]
    benchmark_label = series_order[1] if len(series_order) > 1 else "Portfolio median"
    selected_df = plot_df[plot_df["series"] == selected_label].copy().sort_values("sort_order")
    benchmark_df = plot_df[plot_df["series"] == benchmark_label].copy().sort_values("sort_order")
    benchmark_lookup = benchmark_df.set_index("item_key")["score_index"].to_dict() if not benchmark_df.empty else {}
    selected_df["median_score"] = selected_df["item_key"].map(benchmark_lookup)
    bar_width = theta_step * 0.86
    fig = go.Figure()
    _add_profile_group_underlays(fig, axis_meta)
    if not benchmark_df.empty:
        fig.add_trace(
            go.Barpolar(
                r=pd.to_numeric(benchmark_df["score_index"], errors="coerce").tolist(),
                base=[0] * len(benchmark_df),
                theta=[theta_map[key] for key in benchmark_df["item_key"]],
                width=[bar_width] * len(benchmark_df),
                name=benchmark_label,
                marker=dict(color="rgba(74, 154, 141, 0.18)", line=dict(color="#4A9A8D", width=1.0)),
                opacity=0.95,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Barpolar(
            r=pd.to_numeric(selected_df["score_index"], errors="coerce").tolist(),
            base=[0] * len(selected_df),
            theta=[theta_map[key] for key in selected_df["item_key"]],
            width=[bar_width] * len(selected_df),
            name=selected_label,
            marker=dict(color="#F2D500", line=dict(color="#262A43", width=1.2)),
            opacity=0.9,
            customdata=np.column_stack(
                [
                    selected_df["item_label"].astype(str).to_numpy(),
                    pd.to_numeric(selected_df["median_score"], errors="coerce").fillna(np.nan).to_numpy(),
                ]
            ),
            hovertemplate=(
                "%{customdata[0]}<br>"
                + selected_label
                + ": %{r:.1f}<br>"
                + benchmark_label
                + ": %{customdata[1]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(bargap=0.0)
    return _apply_profile_polar_layout(fig, axis_meta, height=620)


def data_matrix_heatmap(
    matrix_df: pd.DataFrame,
    title: str = "Data Matrix",
    *,
    value_label: str = "Value",
    reverse_scale: bool = False,
) -> go.Figure:
    fig = px.imshow(
        matrix_df,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"color": value_label},
    )
    if reverse_scale:
        fig.update_coloraxes(reversescale=True)
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=50, b=10))
    return fig
