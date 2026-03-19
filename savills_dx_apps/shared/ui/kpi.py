from __future__ import annotations

from html import escape
from typing import Sequence

import streamlit as st

_DEFAULT_ACCENT_COLOUR = "#FFDF00"
_ROW_GAP_REM = 0.9
_MIN_CELL_WIDTH_REM = 10.5

KpiStripItem = tuple[object, object] | tuple[object, object, object | None]


def _chunk_items(items: Sequence[KpiStripItem], size: int) -> list[list[KpiStripItem]]:
    return [list(items[index:index + size]) for index in range(0, len(items), size)]


def _normalise_item(item: KpiStripItem) -> tuple[object, object, object | None]:
    if len(item) == 2:
        label, value = item
        return label, value, None
    if len(item) == 3:
        label, value, note = item
        return label, value, note
    raise ValueError("KPI strip items must contain a label and value, with an optional note.")


def _render_kpi_cell(label: object, value: object, note: object | None, *, accent_color: str) -> str:
    safe_label = escape(str(label), quote=True)
    safe_value = escape(str(value), quote=True)
    safe_accent = escape(accent_color, quote=True)
    note_html = ""
    if note not in (None, ""):
        safe_note = escape(str(note), quote=True)
        note_html = (
            "<div style='font-size:0.82rem;color:inherit;opacity:0.72;"
            f"line-height:1.25;margin-top:0.28rem;'>{safe_note}</div>"
        )
    return (
        "<div style='min-width:0;padding:0.15rem 0.6rem 0.1rem 0.8rem;"
        "color:inherit;"
        f"border-left:4px solid {safe_accent};'>"
        "<div style='font-size:0.78rem;color:inherit;opacity:0.72;"
        f"line-height:1.2;margin-bottom:0.2rem;'>{safe_label}</div>"
        f"<div style='font-size:1.9rem;color:inherit;line-height:1.05;font-weight:500;'>{safe_value}</div>"
        f"{note_html}"
        "</div>"
    )


def render_kpi_strip(
    items: Sequence[KpiStripItem],
    *,
    columns: int = 4,
    accent_color: str = _DEFAULT_ACCENT_COLOUR,
) -> None:
    if not items:
        return

    max_columns = max(int(columns), 1)
    row_blocks: list[str] = []

    # Chunk rows to preserve a sensible desktop maximum while still allowing narrow screens to wrap cleanly.
    for row_items in _chunk_items(items, max_columns):
        cells = "".join(
            _render_kpi_cell(label, value, note, accent_color=accent_color)
            for label, value, note in (_normalise_item(item) for item in row_items)
        )
        row_blocks.append(
            "<div style='display:grid;"
            f"grid-template-columns:repeat(auto-fit, minmax(min({_MIN_CELL_WIDTH_REM}rem, 100%), 1fr));"
            f"gap:{_ROW_GAP_REM}rem;align-items:stretch;'>"
            f"{cells}"
            "</div>"
        )

    st.markdown(
        "<div style='display:grid;gap:0.9rem;margin:0.15rem 0 1rem 0;'>"
        + "".join(row_blocks)
        + "</div>",
        unsafe_allow_html=True,
    )
