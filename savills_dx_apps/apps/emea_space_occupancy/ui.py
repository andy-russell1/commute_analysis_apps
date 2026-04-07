from __future__ import annotations

from html import escape
from typing import Sequence

import streamlit as st


RISK_ORDER = {"High": 0, "Medium": 1, "Low": 2}
ACTION_ORDER = {
    "Expand": 0,
    "Re-stack / Rebalance": 1,
    "Maintain": 2,
    "Maintain / Optimise": 2,
    "Consolidate": 3,
}


def risk_rank(value: object) -> int:
    return RISK_ORDER.get(str(value), 99)


def action_rank(value: object) -> int:
    return ACTION_ORDER.get(str(value), 99)


def _badge_html(item: str | tuple[str, str]) -> str:
    if isinstance(item, tuple):
        label, value = item
        return (
            "<span class='eso-badge'>"
            f"<span class='eso-badge__label'>{escape(str(label))}</span>"
            f"{escape(str(value))}"
            "</span>"
        )
    return f"<span class='eso-badge'>{escape(str(item))}</span>"


def render_badge_row(items: Sequence[str | tuple[str, str]]) -> None:
    if not items:
        return
    st.markdown(
        "<div class='eso-badge-row'>" + "".join(_badge_html(item) for item in items) + "</div>",
        unsafe_allow_html=True,
    )


def render_hero_panel(
    *,
    eyebrow: str,
    title: str,
    body: str,
    badges: Sequence[str | tuple[str, str]] | None = None,
    tone: str = "default",
) -> None:
    badges_html = ""
    if badges:
        badges_html = "<div class='eso-badge-row'>" + "".join(_badge_html(item) for item in badges) + "</div>"
    st.markdown(
        f"""
        <div class="eso-hero eso-tone-{escape(tone)}">
            <div class="eso-hero__eyebrow">{escape(eyebrow)}</div>
            <div class="eso-hero__title">{escape(title)}</div>
            <div class="eso-hero__body">{escape(body)}</div>
            {badges_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, body: str | None = None, *, eyebrow: str | None = None) -> None:
    eyebrow_html = ""
    if eyebrow:
        eyebrow_html = f"<div class='eso-section-heading__eyebrow'>{escape(eyebrow)}</div>"
    body_html = ""
    if body:
        body_html = f"<div class='eso-section-heading__body'>{escape(body)}</div>"
    st.markdown(
        f"""
        <div class="eso-section-heading">
            {eyebrow_html}
            <div class="eso-section-heading__title">{escape(title)}</div>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_callout(title: str, body: str, *, tone: str = "info") -> None:
    st.markdown(
        f"""
        <div class="eso-callout eso-tone-{escape(tone)}">
            <div class="eso-callout__title">{escape(title)}</div>
            <div class="eso-callout__body">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bullet_panel(
    title: str,
    items: Sequence[str],
    *,
    empty_message: str,
    tone: str = "default",
) -> None:
    bullet_items = list(items) if items else [empty_message]
    bullets_html = "".join(f"<li>{escape(str(item))}</li>" for item in bullet_items)
    st.markdown(
        f"""
        <div class="eso-callout eso-tone-{escape(tone)}">
            <div class="eso-callout__title">{escape(title)}</div>
            <div class="eso-callout__body">
                <ul class="eso-bullet-list">{bullets_html}</ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(title: str, value: str, body: str, *, tone: str = "default") -> None:
    st.markdown(
        f"""
        <div class="eso-mini-card eso-tone-{escape(tone)}">
            <div class="eso-mini-card__title">{escape(title)}</div>
            <div class="eso-mini-card__value">{escape(value)}</div>
            <div class="eso-mini-card__body">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="eso-empty-state">
            <div class="eso-empty-state__title">{escape(title)}</div>
            <div class="eso-empty-state__body">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
