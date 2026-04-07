from __future__ import annotations

import re
from typing import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.lens.core.constants import SAVILLS_COLOR_SEQUENCE
from shared.ui.kpi import render_kpi_strip

_CHART_BG = "rgba(0, 0, 0, 0)"
_GRID_COLOUR = "rgba(127, 137, 160, 0.24)"
_LINE_COLOUR = "rgba(127, 137, 160, 0.45)"
_SURFACE_TEXT_COLOUR = "#F8FAFC"


def _apply_executive_chart_layout(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        colorway=SAVILLS_COLOR_SEQUENCE,
        legend_title_text="",
        font=dict(color=_SURFACE_TEXT_COLOUR),
        margin=dict(l=0, r=0, t=50 if fig.layout.title.text else 10, b=0),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def format_number(value: float | int | None, kind: str = "int") -> str:
    if value is None or pd.isna(value):
        return "0"
    numeric = float(value)
    if kind == "pct":
        return f"{numeric:.1%}"
    if kind == "score":
        return f"{numeric:.0f}"
    if kind == "eur":
        return f"EUR {numeric:,.0f}"
    if kind == "sqm":
        return f"{numeric:,.0f} sqm"
    return f"{numeric:,.0f}"


def _format_metric_note(note: str) -> str:
    match = re.search(r"([+-]\d+(?:\.\d+)?)", note)
    if not match:
        return f"<div style='font-size:0.82rem;line-height:1.25;color:inherit;opacity:0.8;margin-top:0.28rem;'>{note}</div>"

    delta_token = match.group(1)
    delta_value = float(delta_token)
    if delta_value > 0:
        colour = "#35D07F"
        arrow = "&uarr;"
    elif delta_value < 0:
        colour = "#F26A73"
        arrow = "&darr;"
    else:
        colour = "inherit"
        arrow = "&rarr;"
    rendered_delta = (
        f"<span style='display:inline-flex;align-items:center;gap:0.18rem;color:{colour};font-weight:800;'>"
        f"<span style='font-size:0.82rem;line-height:1;'>{arrow}</span>{delta_token}</span>"
    )
    return (
        "<div style='font-size:0.82rem;line-height:1.25;color:inherit;opacity:0.88;margin-top:0.28rem;'>"
        f"{note.replace(delta_token, rendered_delta, 1)}</div>"
    )


def render_metric_row(metrics: Sequence[tuple[str, str, str | None]], *, columns: int | None = None) -> None:
    if not metrics:
        return
    render_kpi_strip(
        list(metrics),
        columns=columns or len(metrics),
        note_renderer=lambda note: _format_metric_note(str(note)),
    )


def bar_chart(df: pd.DataFrame, *, x: str, y: str, color: str | None = None, title: str | None = None, orientation: str = "v"):
    if df.empty:
        return None
    fig = px.bar(
        df,
        x=x if orientation == "v" else y,
        y=y if orientation == "v" else x,
        color=color,
        orientation=orientation,
        title=title,
        color_discrete_sequence=SAVILLS_COLOR_SEQUENCE,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=_LINE_COLOUR)
    fig.update_yaxes(gridcolor=_GRID_COLOUR, zeroline=False, linecolor=_LINE_COLOUR)
    return _apply_executive_chart_layout(fig, height=360)


def line_chart(df: pd.DataFrame, *, x: str, y: str, color: str | None = None, title: str | None = None):
    if df.empty:
        return None
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        markers=True,
        color_discrete_sequence=SAVILLS_COLOR_SEQUENCE,
    )
    fig.update_xaxes(gridcolor=_GRID_COLOUR, zeroline=False, linecolor=_LINE_COLOUR)
    fig.update_yaxes(gridcolor=_GRID_COLOUR, zeroline=False, linecolor=_LINE_COLOUR)
    return _apply_executive_chart_layout(fig, height=340)


def donut_chart(df: pd.DataFrame, *, names: str, values: str, title: str | None = None):
    if df.empty:
        return None
    fig = px.pie(
        df,
        names=names,
        values=values,
        hole=0.58,
        title=title,
        color=names,
        color_discrete_sequence=SAVILLS_COLOR_SEQUENCE,
    )
    fig.update_traces(
        marker=dict(line=dict(color=_CHART_BG, width=1.2)),
        textfont=dict(color=_SURFACE_TEXT_COLOUR),
    )
    return _apply_executive_chart_layout(fig, height=340)
