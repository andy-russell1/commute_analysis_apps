from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .constants import SAVILLS_COLOR_SEQUENCE, SAVILLS_MACRO_COLOR_MAP, SAVILLS_MARKET_TIER_COLOR_MAP


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
        x_label = "Overall Index (0-100)"
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


def macro_drilldown_bar(major_scores: pd.DataFrame, selected_macro: str) -> go.Figure:
    plot_df = major_scores[major_scores["macro"] == selected_macro].copy()
    plot_df = plot_df.sort_values(["city", "major"])
    city_order = (
        plot_df.groupby("city", as_index=False)["major_score"].sum().sort_values("major_score", ascending=False)["city"].tolist()
    )
    major_order = sorted(plot_df["major"].dropna().unique().tolist())

    fig = px.bar(
        plot_df,
        x="major_score",
        y="city",
        color="major",
        orientation="h",
        category_orders={"city": city_order, "major": major_order},
        barmode="stack",
        labels={"major_score": "Score", "city": "City", "major": "Major"},
        color_discrete_sequence=SAVILLS_COLOR_SEQUENCE,
    )
    fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def capability_cost_bubble(capability_cost_df: pd.DataFrame) -> go.Figure:
    plot_df = capability_cost_df.copy()
    x_median = float(plot_df["capability_score"].median()) if not plot_df.empty else 0.0
    y_median = float(plot_df["cost_score"].median()) if not plot_df.empty else 0.0

    fig = px.scatter(
        plot_df,
        x="capability_score",
        y="cost_score",
        size="population",
        color="market_tier",
        hover_name="city",
        text="city",
        labels={
            "capability_score": "Capability Score (higher = better)",
            "cost_score": "Cost Score (higher = more attractive)",
            "market_tier": "Market Tier",
        },
        size_max=55,
        color_discrete_map=SAVILLS_MARKET_TIER_COLOR_MAP,
        category_orders={"market_tier": ["Primary", "Secondary", "Tertiary"]},
    )
    fig.update_traces(textposition="top center")
    fig.add_vline(x=x_median, line_dash="dash", line_color="#757D84")
    fig.add_hline(y=y_median, line_dash="dash", line_color="#757D84")

    fig.add_annotation(x=x_median, y=plot_df["cost_score"].max(), text="Median Capability", showarrow=False, yshift=8)
    fig.add_annotation(x=plot_df["capability_score"].max(), y=y_median, text="Median Cost", showarrow=False, xshift=8)

    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, text="Low Capability / High Cost", showarrow=False)
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98, text="High Capability / High Cost", showarrow=False)
    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.02, text="Low Capability / Low Cost", showarrow=False)
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.02, text="High Capability / Low Cost", showarrow=False)

    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


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


def data_matrix_heatmap(matrix_df: pd.DataFrame, title: str = "Data Matrix") -> go.Figure:
    fig = px.imshow(
        matrix_df,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"color": "Value"},
    )
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=50, b=10))
    return fig
