from __future__ import annotations

from hashlib import md5
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from apps.lens.common import is_embedded_mode, render_nav_link

from . import io, scoring, validate
from .constants import (
    DEFAULT_COST_MACRO,
    MACRO_PRESET_TARGETS,
    MODE_ADVANCED,
    MODE_CLIENT,
    MODE_OPTIONS,
    SCORING_METHOD_ADVANCED_LABELS,
    SCORING_METHOD_HELP,
    SCORING_METHOD_LABELS,
    SCORING_METHOD_PERCENTILE,
    SCORING_METHOD_RANK,
    WEIGHTING_MODE_HELP,
)


def get_default_mode() -> str:
    return MODE_CLIENT


def get_supported_scoring_method_keys() -> list[str]:
    return [SCORING_METHOD_LABELS[label] for label in SCORING_METHOD_ADVANCED_LABELS]


def clamp_user_scoring_method(method: str | None) -> str:
    normalized = scoring.normalize_scoring_method_key(method)
    if normalized in get_supported_scoring_method_keys():
        return normalized
    return SCORING_METHOD_RANK


def _find_logo_path() -> Path | None:
    lens_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "assets" / "logos" / "Savills.png",
        lens_root / "assets" / "logos" / "Savills.png",
        lens_root / "assets" / "logo.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def render_page_header(title: str, caption: str | None = None, logo_width: int = 140) -> None:
    logo_path = _find_logo_path()
    title_col, logo_col = st.columns([6.0, 1.2])
    with title_col:
        st.title(title)
        if caption:
            st.caption(caption)
    with logo_col:
        if logo_path is not None:
            st.image(str(logo_path), width=int(logo_width))


def render_dashboard_chrome() -> None:
    st.markdown(
        """
        <style>
        .lens-surface {
            border: 1px solid rgba(127, 137, 160, 0.24);
            border-radius: 0.85rem;
            background: rgba(38, 42, 67, 0.08);
            color: inherit;
        }
        .lens-context-banner {
            margin: 0.15rem 0 0.9rem 0;
            padding: 0.8rem 1rem;
            border: 1px solid rgba(127, 137, 160, 0.24);
            border-radius: 0.8rem;
            background: rgba(38, 42, 67, 0.08);
            color: inherit;
            font-size: 0.98rem;
        }
        .lens-context-banner strong {
            display: block;
            margin-bottom: 0.2rem;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: inherit;
            opacity: 0.72;
        }
        .lens-insight-card {
            height: 100%;
            padding: 0.95rem 1rem;
            border: 1px solid rgba(127, 137, 160, 0.24);
            border-radius: 0.85rem;
            background: rgba(38, 42, 67, 0.08);
            color: inherit;
        }
        .lens-insight-card h4 {
            margin: 0 0 0.35rem 0;
            font-size: 0.95rem;
            color: inherit;
        }
        .lens-insight-card ul {
            margin: 0.35rem 0 0 1rem;
            padding: 0;
        }
        .lens-insight-card li {
            margin: 0.25rem 0;
        }
        div[data-testid="stMetric"] {
            background: rgba(38, 42, 67, 0.08);
            border: 1px solid rgba(127, 137, 160, 0.24);
            border-radius: 0.85rem;
            padding: 0.8rem 0.95rem;
        }
        div[data-testid="stMetric"] label {
            color: inherit;
            opacity: 0.72;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_context_sentence(question: str, sentence: str) -> None:
    st.markdown(
        f"<div class='lens-context-banner'><strong>{question}</strong>{sentence}</div>",
        unsafe_allow_html=True,
    )


def render_insight_card(title: str, items: list[str], empty_message: str, caption: str | None = None) -> None:
    bullet_items = items or [empty_message]
    bullets = "".join(f"<li>{item}</li>" for item in bullet_items)
    caption_html = f"<div style='color:#5A6370;font-size:0.9rem;margin-bottom:0.4rem;'>{caption}</div>" if caption else ""
    st.markdown(
        f"<div class='lens-insight-card'><h4>{title}</h4>{caption_html}<ul>{bullets}</ul></div>",
        unsafe_allow_html=True,
    )


def to_proper_case_label(text: str) -> str:
    cleaned = str(text).replace("_", " ").strip()
    if cleaned == "":
        return ""
    acronyms = {"id", "api", "url", "ui", "qa"}
    words = cleaned.split()
    output: list[str] = []
    for word in words:
        lower = word.lower()
        if lower in acronyms:
            output.append(lower.upper())
        elif lower == "vs":
            output.append("vs")
        else:
            output.append(lower.capitalize())
    return " ".join(output)


def format_table_for_display(
    df: pd.DataFrame,
    *,
    decimals: int | None = None,
    rename_map: dict[str, str] | None = None,
    criterion_id_mode: str = "rowid",
) -> pd.DataFrame:
    table = df.copy()
    if criterion_id_mode == "rowid" and "criterion_id" in table.columns:
        table.insert(0, "id", np.arange(1, len(table) + 1))
        table = table.drop(columns=["criterion_id"])
    elif criterion_id_mode == "label" and "criterion_id" in table.columns:
        table = table.rename(columns={"criterion_id": "id"})
    if decimals is not None:
        numeric_cols = table.select_dtypes(include=[np.number]).columns
        table[numeric_cols] = table[numeric_cols].round(decimals)
    if rename_map:
        table = table.rename(columns=rename_map)
    table = table.rename(columns={col: to_proper_case_label(col) for col in table.columns})
    return table


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    cleaned = hex_value.lstrip("#")
    return tuple(int(cleaned[idx : idx + 2], 16) for idx in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _blend_hex(start_hex: str, end_hex: str, fraction: float) -> str:
    start_rgb = _hex_to_rgb(start_hex)
    end_rgb = _hex_to_rgb(end_hex)
    frac = _clamp01(fraction)
    blended = tuple(int(round(start + ((end - start) * frac))) for start, end in zip(start_rgb, end_rgb, strict=False))
    return _rgb_to_hex(blended)


def get_active_theme_base() -> str:
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = None
    resolved = str(base or "dark").strip().lower()
    if resolved not in {"dark", "light"}:
        return "dark"
    return resolved


def _rag_gradient_hex(fraction: float, *, theme_base: str = "dark") -> str:
    frac = _clamp01(fraction)
    is_dark = str(theme_base).strip().lower() == "dark"
    if is_dark:
        if frac <= 0.5:
            return _blend_hex("#B14A4A", "#C3A34A", frac / 0.5)
        return _blend_hex("#C3A34A", "#47A56E", (frac - 0.5) / 0.5)
    if frac <= 0.5:
        return _blend_hex("#F3B0A7", "#F1D787", frac / 0.5)
    return _blend_hex("#F1D787", "#BDE6C9", (frac - 0.5) / 0.5)


def _build_relative_row_styles(row: pd.Series, value_columns: list[str], reverse: bool, *, theme_base: str = "dark") -> list[str]:
    styles = [""] * len(row)
    numeric = pd.to_numeric(row[value_columns], errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return styles

    min_value = float(valid.min())
    max_value = float(valid.max())
    if max_value > min_value:
        relative = (numeric - min_value) / (max_value - min_value)
    else:
        relative = pd.Series(0.5, index=numeric.index, dtype=float)

    if reverse:
        relative = 1.0 - relative

    for column in value_columns:
        if column not in row.index:
            continue
        raw_value = numeric.get(column)
        if pd.isna(raw_value):
            continue
        weight = _clamp01(float(relative.get(column, 0.5)))
        background = _rag_gradient_hex(weight, theme_base=theme_base)
        border = _blend_hex("#d96b6b", "#6aa84f", weight)
        styles[row.index.get_loc(column)] = f"background-color: {background}; border-left: 3px solid {border};"
    return styles


def style_relative_value_table(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    decimals: int | None = None,
    reverse_rows: pd.Series | dict[Any, bool] | None = None,
    theme_base: str | None = None,
) -> pd.io.formats.style.Styler:
    table = df.copy()
    subset = [column for column in value_columns if column in table.columns]
    reverse_lookup: dict[Any, bool] = {}
    if isinstance(reverse_rows, pd.Series):
        reverse_lookup = {idx: bool(value) for idx, value in reverse_rows.items()}
    elif isinstance(reverse_rows, dict):
        reverse_lookup = {idx: bool(value) for idx, value in reverse_rows.items()}
    resolved_theme = str(theme_base or get_active_theme_base()).strip().lower()
    if resolved_theme not in {"dark", "light"}:
        resolved_theme = "dark"

    styler = table.style
    if decimals is not None:
        format_map = {
            column: (lambda value, precision=decimals: "-" if pd.isna(value) else f"{float(value):,.{precision}f}")
            for column in subset
        }
        styler = styler.format(format_map)

    def _apply_row_styles(row: pd.Series) -> list[str]:
        reverse = reverse_lookup.get(row.name, False)
        return _build_relative_row_styles(row, subset, reverse=reverse, theme_base=resolved_theme)

    if subset:
        styler = styler.apply(_apply_row_styles, axis=1)
    return styler


def build_matrix_preview_table(
    long_df: pd.DataFrame,
    metric_col: str,
    *,
    value_label: str,
) -> pd.DataFrame:
    matrix = long_df.pivot(index="micro_label", columns="city", values=metric_col).sort_index().reset_index()
    return matrix.rename(columns={"micro_label": value_label})


def build_reference_rank_audit_table(rank_reference: pd.DataFrame) -> pd.DataFrame:
    if rank_reference.empty:
        return rank_reference.copy()
    audit = rank_reference.copy()
    audit.insert(0, "reference_status", "Reference input only")
    return audit


def get_active_scoring_basis(method: str | None) -> dict[str, str]:
    method_key = scoring.normalize_scoring_method_key(method)
    if method_key == SCORING_METHOD_PERCENTILE:
        return {
            "key": SCORING_METHOD_PERCENTILE,
            "column": "score_index",
            "label": "Percentile Score (0-100)",
            "short_label": "Percentile Score",
            "narrative_label": "percentile-based",
        }
    return {
        "key": SCORING_METHOD_RANK,
        "column": "rank",
        "label": "Rank",
        "short_label": "Rank",
        "narrative_label": "rank-based",
    }


def format_driver_basis_value(row: pd.Series, scoring_method: str | None) -> str:
    basis = get_active_scoring_basis(scoring_method)
    value = pd.to_numeric(row.get(basis["column"]), errors="coerce")
    if pd.isna(value):
        return basis["short_label"]
    if basis["key"] == SCORING_METHOD_PERCENTILE:
        return f"{basis['short_label']} {float(value):.1f}"
    return f"{basis['short_label']} {int(round(float(value)))}"


def get_micro_display_name(macro: str, major: str, micro: str) -> str:
    raw = "" if pd.isna(micro) else str(micro).strip()
    lowered = raw.lower()
    is_placeholder = lowered in {"", "tbd", "to be determined", "to be defined", "n/a", "na"}
    if not is_placeholder:
        placeholder_tokens = ("tbd", "tbc", "to be determined", "to be confirmed", "client")
        if "data point" in lowered and any(token in lowered for token in placeholder_tokens):
            is_placeholder = True
    if is_placeholder:
        return f"{major} ({macro})"
    return raw


def _normalize_weights(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    copy_df = df.copy()
    if group_cols:
        sums = copy_df.groupby(group_cols, dropna=False)["weight"].transform("sum")
        valid = sums > 0
        copy_df.loc[valid, "weight"] = copy_df.loc[valid, "weight"] / sums.loc[valid]
        copy_df.loc[~valid, "weight"] = 1.0 / copy_df.groupby(group_cols, dropna=False)["weight"].transform("count").loc[
            ~valid
        ]
        return copy_df

    total = float(copy_df["weight"].sum())
    if total <= 0:
        copy_df["weight"] = 1.0 / len(copy_df)
        return copy_df
    copy_df["weight"] = copy_df["weight"] / total
    return copy_df


def build_default_weight_tables(criteria_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    macro_weights = (
        criteria_df[["macro", "macro_weight_template"]]
        .drop_duplicates()
        .rename(columns={"macro_weight_template": "weight"})
        .reset_index(drop=True)
    )
    major_weights = (
        criteria_df[["macro", "major", "major_weight_template"]]
        .drop_duplicates()
        .rename(columns={"major_weight_template": "weight"})
        .reset_index(drop=True)
    )
    minor_weights = (
        criteria_df[["criterion_id", "macro", "major", "micro", "minor_weight_template"]]
        .drop_duplicates()
        .rename(columns={"minor_weight_template": "weight"})
        .reset_index(drop=True)
    )

    macro_weights["weight"] = pd.to_numeric(macro_weights["weight"], errors="coerce").fillna(0.0)
    major_weights["weight"] = pd.to_numeric(major_weights["weight"], errors="coerce").fillna(0.0)
    minor_weights["weight"] = pd.to_numeric(minor_weights["weight"], errors="coerce").fillna(0.0)

    macro_weights = _normalize_weights(macro_weights)
    major_weights = _normalize_weights(major_weights, group_cols=["macro"])
    minor_weights = _normalize_weights(minor_weights, group_cols=["macro", "major"])
    return macro_weights, major_weights, minor_weights


def build_simple_weight_tables(
    criteria_df: pd.DataFrame,
    macro_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    majors = criteria_df[["macro", "major"]].drop_duplicates().copy()
    major_counts = majors.groupby("macro", dropna=False)["major"].transform("count")
    majors["weight"] = 1.0 / major_counts

    minors = criteria_df[["criterion_id", "macro", "major", "micro"]].drop_duplicates().copy()
    minor_counts = minors.groupby(["macro", "major"], dropna=False)["micro"].transform("count")
    minors["weight"] = 1.0 / minor_counts

    macro_weights = _normalize_weights(macro_weights[["macro", "weight"]].copy())
    majors = _normalize_weights(majors, group_cols=["macro"])
    minors = _normalize_weights(minors, group_cols=["macro", "major"])
    return majors, minors


def compose_weighted_criteria(
    criteria_df: pd.DataFrame,
    macro_weights: pd.DataFrame,
    major_weights: pd.DataFrame,
    minor_weights: pd.DataFrame,
) -> pd.DataFrame:
    weighted = criteria_df[["criterion_id", "macro", "major", "micro"]].copy()
    weighted = weighted.merge(macro_weights[["macro", "weight"]].rename(columns={"weight": "macro_weight"}), on="macro", how="left")
    weighted = weighted.merge(
        major_weights[["macro", "major", "weight"]].rename(columns={"weight": "major_weight"}),
        on=["macro", "major"],
        how="left",
    )
    weighted = weighted.merge(
        minor_weights[["criterion_id", "weight"]].rename(columns={"weight": "minor_weight"}),
        on="criterion_id",
        how="left",
    )
    weighted["effective_micro_weight"] = weighted["macro_weight"] * weighted["major_weight"] * weighted["minor_weight"]
    return weighted


def build_default_direction_map(criteria_df: pd.DataFrame, raw_data_df: pd.DataFrame) -> dict[str, str]:
    source_map = raw_data_df.set_index("criterion_id")["source"].to_dict()
    directions: dict[str, str] = {}
    for criterion_id in criteria_df["criterion_id"]:
        source = source_map.get(criterion_id)
        directions[criterion_id] = scoring.infer_direction_from_source(source)
    return directions


def weight_dataframes_to_dict(
    macro_weights: pd.DataFrame,
    major_weights: pd.DataFrame,
    minor_weights: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "macro": macro_weights.sort_values("macro").reset_index(drop=True),
        "major": major_weights.sort_values(["macro", "major"]).reset_index(drop=True),
        "minor": minor_weights.sort_values(["macro", "major", "micro"]).reset_index(drop=True),
    }


def get_macro_presets(macros: list[str]) -> dict[str, dict[str, float]]:
    if not macros:
        return {"Balanced": {}}
    equal = 1.0 / len(macros)
    balanced = {macro: equal for macro in macros}

    presets: dict[str, dict[str, float]] = {"Balanced": balanced}
    for preset_name, template in MACRO_PRESET_TARGETS.items():
        mapped = {macro: float(template.get(macro, 0.0)) for macro in macros}
        if sum(mapped.values()) <= 0:
            mapped = balanced.copy()
        presets[preset_name] = scoring.normalize_weight_map(mapped)
    return presets


def apply_macro_preset(preset_name: str, macro_weights_df: pd.DataFrame) -> pd.DataFrame:
    output = macro_weights_df.copy()
    presets = get_macro_presets(output["macro"].tolist())
    target = presets.get(preset_name, presets["Balanced"])
    if not target:
        return _normalize_weights(output)
    output["weight"] = output["macro"].map(target).fillna(0.0)
    return _normalize_weights(output)


def serialize_macro_scenario(name: str, macro_weights_df: pd.DataFrame) -> dict[str, Any]:
    table = _normalize_weights(macro_weights_df[["macro", "weight"]].copy())
    return {
        "version": 1,
        "name": (name or "Scenario").strip() or "Scenario",
        "scoring_method": clamp_user_scoring_method(st.session_state.get("lens_scoring_method", SCORING_METHOD_RANK)),
        "macro_weights": {str(row["macro"]): float(row["weight"]) for _, row in table.iterrows()},
    }


def get_missing_scenario_macros(payload: dict[str, Any], macro_weights_df: pd.DataFrame) -> list[str]:
    scenario_macros = set(payload.get("macro_weights", {}).keys()) if isinstance(payload, dict) else set()
    valid_macros = set(macro_weights_df["macro"].tolist())
    return sorted(scenario_macros - valid_macros)


def load_macro_scenario(payload: dict[str, Any], macro_weights_df: pd.DataFrame) -> pd.DataFrame:
    output = macro_weights_df.copy()
    if not isinstance(payload, dict):
        return _normalize_weights(output)
    macro_weights = payload.get("macro_weights")
    if not isinstance(macro_weights, dict):
        return _normalize_weights(output)

    legacy_method = payload.get("scoring_method")
    if isinstance(legacy_method, str):
        resolved_method = clamp_user_scoring_method(legacy_method)
        st.session_state["lens_scoring_method"] = resolved_method
        st.session_state["lens_last_advanced_scoring_method"] = resolved_method

    macro_set = set(output["macro"].tolist())
    for macro, weight in macro_weights.items():
        if macro in macro_set:
            output.loc[output["macro"] == macro, "weight"] = pd.to_numeric(weight, errors="coerce")
    output["weight"] = pd.to_numeric(output["weight"], errors="coerce").fillna(0.0)
    return _normalize_weights(output)


def _build_city_scores_table(
    overall_scores: pd.DataFrame,
    macro_scores: pd.DataFrame,
    major_scores: pd.DataFrame,
) -> pd.DataFrame:
    city_scores = build_city_ranks(overall_scores).copy()
    macro_wide = macro_scores.pivot(index="city", columns="macro", values="macro_score").add_prefix("macro__")
    major_wide = (
        major_scores.assign(major_label=major_scores["macro"] + " > " + major_scores["major"])
        .pivot(index="city", columns="major_label", values="major_score")
        .add_prefix("major__")
    )
    city_scores = city_scores.set_index("city").join(macro_wide, how="left").join(major_wide, how="left").reset_index()
    return city_scores


def build_top_drivers(contributions: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    table = contributions[["city", "macro", "major", "micro", "criterion_id", "score", "contribution", "direction"]].copy()
    baseline = table.groupby("criterion_id", as_index=False)["contribution"].mean().rename(
        columns={"contribution": "mean_contribution"}
    )
    table = table.merge(baseline, on="criterion_id", how="left")
    table["delta"] = table["contribution"] - table["mean_contribution"]

    rows: list[pd.DataFrame] = []
    for city, city_df in table.groupby("city"):
        top_pos = city_df.nlargest(top_n, "delta").copy()
        top_pos["driver_type"] = "Positive"
        top_neg = city_df.nsmallest(top_n, "delta").copy()
        top_neg["driver_type"] = "Negative"
        rows.append(pd.concat([top_pos, top_neg], ignore_index=True))
    if not rows:
        return pd.DataFrame(
            columns=["city", "driver_type", "macro", "major", "micro", "direction", "delta", "contribution", "score"]
        )
    return pd.concat(rows, ignore_index=True).sort_values(["city", "driver_type", "delta"], ascending=[True, True, False])


def build_city_ranks(overall_scores: pd.DataFrame) -> pd.DataFrame:
    table = overall_scores.copy()
    table["overall_rank"] = table["overall_score"].rank(method="min", ascending=False).astype(int)
    return table.sort_values(["overall_rank", "city"]).reset_index(drop=True)


def build_home_recommendations_table(bundle: dict[str, Any], top_n: int = 3) -> pd.DataFrame:
    ranks = build_city_ranks(bundle["overall_scores"]).head(top_n).copy()
    capability_cost = bundle["capability_cost"][["city", "capability_index", "cost_index"]].drop_duplicates("city")
    table = ranks.merge(capability_cost, on="city", how="left")
    return table[
        [
            "overall_rank",
            "city",
            "overall_index",
            "capability_index",
            "cost_index",
        ]
    ].rename(
        columns={
            "overall_rank": "rank",
            "overall_index": "overall_index",
            "capability_index": "capability_score",
            "cost_index": "cost_score",
        }
    )


def add_indexed_score_column(df: pd.DataFrame, score_col: str, index_col: str) -> pd.DataFrame:
    table = df.copy()
    table[score_col] = pd.to_numeric(table[score_col], errors="coerce")
    table[index_col] = table[score_col] * 100.0
    return table


def add_overall_index(overall_scores: pd.DataFrame) -> pd.DataFrame:
    table = overall_scores.copy()
    table["overall_score"] = pd.to_numeric(table["overall_score"], errors="coerce")
    valid = table["overall_score"].dropna()
    if valid.empty:
        table["overall_index"] = np.nan
        table["distance_to_leader"] = np.nan
        table["overall_tier"] = np.nan
        return table

    bounded_01 = bool(((valid >= 0.0) & (valid <= 1.0)).all())
    if bounded_01:
        table["overall_index"] = table["overall_score"] * 100.0
    else:
        min_score = float(valid.min())
        max_score = float(valid.max())
        if max_score > min_score:
            table["overall_index"] = ((table["overall_score"] - min_score) / (max_score - min_score)) * 100.0
        else:
            table["overall_index"] = np.where(table["overall_score"].notna(), 50.0, np.nan)

    leader_index = float(table["overall_index"].max(skipna=True))
    if np.isfinite(leader_index) and leader_index > 0:
        leader_norm = (table["overall_index"] / leader_index) * 100.0
        table["distance_to_leader"] = leader_norm - 100.0
    else:
        table["distance_to_leader"] = np.where(table["overall_index"].notna(), 0.0, np.nan)

    table["overall_tier"] = np.select(
        [
            table["overall_index"] >= 80.0,
            table["overall_index"] >= 65.0,
            table["overall_index"] >= 50.0,
        ],
        ["Leading", "Strong", "Competitive"],
        default="Challenged",
    )
    table.loc[table["overall_index"].isna(), "overall_tier"] = np.nan
    return table


def build_city_drilldown(bundle: dict[str, Any], city: str, top_n: int = 5) -> dict[str, Any]:
    ranks = build_city_ranks(bundle["overall_scores"])
    city_rank_row = ranks[ranks["city"] == city].copy()
    if city_rank_row.empty:
        raise ValueError(f"City '{city}' is not available in overall scores.")

    macro_scores = bundle["macro_scores"]
    major_scores = bundle["major_scores"]
    contributions = bundle["contributions"]
    city_macro = macro_scores[macro_scores["city"] == city].copy()
    city_major = major_scores[major_scores["city"] == city].copy()
    city_micro = contributions[contributions["city"] == city].copy()
    if "macro_index" not in city_macro.columns and "macro_score" in city_macro.columns:
        city_macro["macro_index"] = pd.to_numeric(city_macro["macro_score"], errors="coerce") * 100.0
    if "major_index" not in city_major.columns and "major_score" in city_major.columns:
        city_major["major_index"] = pd.to_numeric(city_major["major_score"], errors="coerce") * 100.0
    if "score_index" not in city_micro.columns and "score" in city_micro.columns:
        city_micro["score_index"] = pd.to_numeric(city_micro["score"], errors="coerce") * 100.0
    if "rank" not in city_micro.columns:
        city_micro["rank"] = np.nan
    city_micro["micro_display"] = [
        get_micro_display_name(macro, major, micro)
        for macro, major, micro in zip(city_micro["macro"], city_micro["major"], city_micro["micro"], strict=False)
    ]

    baseline = contributions.groupby("criterion_id", as_index=False)["contribution"].mean().rename(
        columns={"contribution": "mean_contribution"}
    )
    city_micro = city_micro.merge(baseline, on="criterion_id", how="left")
    city_micro["delta"] = city_micro["contribution"] - city_micro["mean_contribution"]

    macro_contrib = city_micro.groupby("macro", as_index=False)["contribution"].sum().rename(
        columns={"contribution": "macro_contribution"}
    )
    major_contrib = city_micro.groupby(["macro", "major"], as_index=False)["contribution"].sum().rename(
        columns={"contribution": "major_contribution"}
    )

    city_macro = city_macro.merge(macro_contrib, on="macro", how="left")
    city_major = city_major.merge(major_contrib, on=["macro", "major"], how="left")

    strengths = city_micro.nlargest(top_n, "delta")[
        [
            "macro",
            "major",
            "micro_display",
            "direction",
            "rank",
            "score",
            "score_index",
            "effective_micro_weight",
            "contribution",
            "delta",
        ]
    ].reset_index(drop=True)
    weaknesses = city_micro.nsmallest(top_n, "delta")[
        [
            "macro",
            "major",
            "micro_display",
            "direction",
            "rank",
            "score",
            "score_index",
            "effective_micro_weight",
            "contribution",
            "delta",
        ]
    ].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    macro_score_map = city_macro.set_index("macro")["macro_score"].to_dict()
    major_score_map = city_major.set_index(["macro", "major"])["major_score"].to_dict()
    major_weight_map = city_major.set_index(["macro", "major"])["major_weight"].to_dict()

    for _, macro_row in city_macro.sort_values("macro").iterrows():
        macro_name = str(macro_row["macro"])
        rows.append(
            {
                "level": "Macro",
                "macro": macro_name,
                "major": "",
                "name": macro_name,
                "weight": float(macro_row["macro_weight"]),
                "score": float(macro_row.get("macro_index", macro_row["macro_score"])),
                "contribution": float(macro_row.get("macro_contribution", np.nan)),
                "direction": "",
                "notes": "Weighted macro contribution to the overall 0-100 index.",
            }
        )

        macro_majors = city_major[city_major["macro"] == macro_name].copy().sort_values("major")
        for _, major_row in macro_majors.iterrows():
            major_name = str(major_row["major"])
            rows.append(
                {
                    "level": "Major",
                    "macro": macro_name,
                    "major": major_name,
                    "name": major_name,
                    "weight": float(major_row["major_weight"]),
                    "score": float(major_row.get("major_index", major_row["major_score"])),
                    "contribution": float(major_row.get("major_contribution", np.nan)),
                    "direction": "",
                    "notes": "Weighted major contribution within macro on the 0-100 index basis.",
                }
            )

            major_micros = city_micro[
                (city_micro["macro"] == macro_name) & (city_micro["major"] == major_name)
            ].copy().sort_values("micro")
            for _, micro_row in major_micros.iterrows():
                direction = "Higher is better" if micro_row["direction"] == "higher" else "Lower is better"
                rows.append(
                    {
                        "level": "Micro",
                        "macro": macro_name,
                        "major": major_name,
                        "name": str(micro_row["micro_display"]),
                        "weight": float(micro_row["effective_micro_weight"]),
                        "score": float(micro_row.get("score_index", micro_row["score"])),
                        "contribution": float(micro_row["contribution"]),
                        "direction": direction,
                        "notes": "Micro contribution = indexed micro score x effective micro weight.",
                    }
                )

    hierarchy = pd.DataFrame(rows)
    compact = hierarchy[["level", "name", "weight", "score", "contribution", "direction", "notes"]].copy()
    capability_cost_source = bundle.get("capability_cost")
    if isinstance(capability_cost_source, pd.DataFrame):
        capability_cost = capability_cost_source[capability_cost_source["city"] == city].copy()
    else:
        capability_cost = pd.DataFrame()

    summary = {
        "city": city,
        "overall_score": float(city_rank_row.iloc[0]["overall_score"]),
        "overall_index": float(city_rank_row.iloc[0]["overall_index"]) if "overall_index" in city_rank_row.columns else np.nan,
        "overall_rank": int(city_rank_row.iloc[0]["overall_rank"]),
        "distance_to_leader": float(city_rank_row.iloc[0]["distance_to_leader"])
        if "distance_to_leader" in city_rank_row.columns
        else np.nan,
        "overall_tier": str(city_rank_row.iloc[0]["overall_tier"]) if "overall_tier" in city_rank_row.columns else "",
        "capability_score": float(capability_cost.iloc[0]["capability_index"]) if not capability_cost.empty else np.nan,
        "cost_score": float(capability_cost.iloc[0]["cost_index"]) if not capability_cost.empty else np.nan,
        "macro_count": int(city_macro.shape[0]),
        "strongest_macro": str(city_macro.sort_values("macro_score", ascending=False).iloc[0]["macro"]) if not city_macro.empty else "",
    }
    if not city_macro.empty:
        tradeoff_macro = city_macro.sort_values("macro_score", ascending=True).iloc[0]["macro"]
        summary["tradeoff_macro"] = str(tradeoff_macro)
    else:
        summary["tradeoff_macro"] = ""

    return {
        "summary": summary,
        "macro_summary": city_macro[["macro", "macro_weight", "macro_score", "macro_contribution"]],
        "major_summary": city_major[["macro", "major", "major_weight", "major_score", "major_contribution"]],
        "micro_details": city_micro,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "compact_breakdown": compact,
        "hierarchy_breakdown": hierarchy,
        "major_score_map": major_score_map,
        "major_weight_map": major_weight_map,
        "macro_score_map": macro_score_map,
    }


def build_city_profile_comparison(bundle: dict[str, Any], city: str, level: str = "Macro") -> pd.DataFrame:
    benchmark = build_benchmark_profile(bundle, city, comparison_mode="median", level=level)
    return benchmark["profile"]


def _empty_profile_comparison_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "series",
            "item_key",
            "item_label",
            "score_index",
            "basis_value",
            "basis_label",
            "macro",
            "major",
            "group_key",
            "group_label",
            "sort_order",
            "detail_level",
        ]
    )


def _prepare_profile_source(
    bundle: dict[str, Any],
    level: str,
    *,
    scoring_method: str | None = None,
) -> tuple[pd.DataFrame, str]:
    normalized_level = str(level).strip().lower()
    active_basis = get_active_scoring_basis(scoring_method)

    if normalized_level == "major":
        source_df = bundle["major_scores"].copy()
        if "major_index" not in source_df.columns and "major_score" in source_df.columns:
            source_df["major_index"] = pd.to_numeric(source_df["major_score"], errors="coerce") * 100.0
        source_df["score_index"] = pd.to_numeric(source_df.get("major_index"), errors="coerce")
        source_df["basis_value"] = source_df["score_index"]
        source_df["basis_label"] = "Index (0-100)"
        source_df["item_key"] = source_df["macro"].astype(str) + "||" + source_df["major"].astype(str)
        source_df["item_label"] = source_df["major"].astype(str)
        source_df["group_key"] = source_df["macro"].astype(str)
        source_df["group_label"] = source_df["macro"].astype(str)
    elif normalized_level == "micro":
        source_df = bundle["contributions"].copy()
        if "score_index" not in source_df.columns and "score" in source_df.columns:
            source_df["score_index"] = pd.to_numeric(source_df["score"], errors="coerce") * 100.0
        if "rank" not in source_df.columns and "micro_scores" in bundle:
            rank_source = bundle["micro_scores"][["city", "criterion_id", "rank"]].drop_duplicates(["city", "criterion_id"])
            source_df = source_df.merge(rank_source, on=["city", "criterion_id"], how="left")
        source_df["basis_value"] = pd.to_numeric(source_df.get(active_basis["column"]), errors="coerce")
        source_df["basis_label"] = active_basis["short_label"]
        source_df["micro_display"] = [
            get_micro_display_name(macro, major, micro)
            for macro, major, micro in zip(source_df["macro"], source_df["major"], source_df["micro"], strict=False)
        ]
        source_df["item_key"] = source_df["criterion_id"].astype(str)
        source_df["item_label"] = source_df["micro_display"].astype(str)
        source_df["group_key"] = source_df["macro"].astype(str) + "||" + source_df["major"].astype(str)
        source_df["group_label"] = source_df["macro"].astype(str) + " | " + source_df["major"].astype(str)
    else:
        source_df = bundle["macro_scores"].copy()
        if "macro_index" not in source_df.columns and "macro_score" in source_df.columns:
            source_df["macro_index"] = pd.to_numeric(source_df["macro_score"], errors="coerce") * 100.0
        source_df["score_index"] = pd.to_numeric(source_df.get("macro_index"), errors="coerce")
        source_df["basis_value"] = source_df["score_index"]
        source_df["basis_label"] = "Index (0-100)"
        source_df["major"] = ""
        source_df["item_key"] = source_df["macro"].astype(str)
        source_df["item_label"] = source_df["macro"].astype(str)
        source_df["group_key"] = source_df["macro"].astype(str)
        source_df["group_label"] = source_df["macro"].astype(str)

    source_df = source_df.dropna(subset=["city", "score_index", "item_key", "item_label", "macro", "major"]).copy()
    if source_df.empty:
        return _empty_profile_comparison_frame(), normalized_level.capitalize()

    source_df["macro"] = source_df["macro"].astype(str)
    source_df["major"] = source_df["major"].astype(str)
    source_df["basis_label"] = source_df["basis_label"].fillna("Index (0-100)")
    return source_df, normalized_level.capitalize()


def _finalize_profile_comparison(
    selected: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    city: str,
    benchmark_label: str,
    detail_level: str,
) -> pd.DataFrame:
    selected = selected.copy()
    benchmark = benchmark.copy()
    selected["series"] = city
    benchmark["series"] = benchmark_label
    selected["series_order"] = 0
    benchmark["series_order"] = 1

    columns = [
        "series",
        "item_key",
        "item_label",
        "score_index",
        "basis_value",
        "basis_label",
        "macro",
        "major",
        "group_key",
        "group_label",
        "sort_order",
        "series_order",
    ]
    output = pd.concat([selected[columns], benchmark[columns]], ignore_index=True)
    output = output.sort_values(["series_order", "sort_order"]).drop(columns=["series_order"]).reset_index(drop=True)
    output["detail_level"] = detail_level
    return output


def build_benchmark_profile(
    bundle: dict[str, Any],
    city: str,
    *,
    comparison_mode: str = "average",
    benchmark_city: str | None = None,
    level: str = "Macro",
    scoring_method: str | None = None,
) -> dict[str, Any]:
    source_df, detail_level = _prepare_profile_source(bundle, level, scoring_method=scoring_method)
    if source_df.empty:
        return {"profile": _empty_profile_comparison_frame(), "benchmark_label": "Benchmark"}

    selected = source_df[source_df["city"] == city].copy()
    if selected.empty:
        return {"profile": _empty_profile_comparison_frame(), "benchmark_label": "Benchmark"}
    selected = selected.sort_values(["macro", "major", "item_label"]).reset_index(drop=True)
    selected["sort_order"] = np.arange(len(selected))

    benchmark_label = "Portfolio average"
    resolved_mode = str(comparison_mode).strip().lower()
    if resolved_mode == "city":
        comparison_city = str(benchmark_city or "").strip()
        if not comparison_city:
            comparison_candidates = [value for value in source_df["city"].dropna().astype(str).unique().tolist() if value != city]
            comparison_city = comparison_candidates[0] if comparison_candidates else city
        benchmark = source_df[source_df["city"] == comparison_city].copy()
        benchmark_label = comparison_city
    elif resolved_mode == "best":
        top_city = str(build_city_ranks(bundle["overall_scores"]).iloc[0]["city"])
        benchmark = source_df[source_df["city"] == top_city].copy()
        benchmark_label = f"{top_city} (best performer)"
    elif resolved_mode == "median":
        benchmark = source_df.groupby(
            ["item_key", "item_label", "macro", "major", "group_key", "group_label", "basis_label"],
            as_index=False,
        )[["score_index", "basis_value"]].median()
        benchmark_label = "Portfolio median"
    else:
        benchmark = source_df.groupby(
            ["item_key", "item_label", "macro", "major", "group_key", "group_label", "basis_label"],
            as_index=False,
        )[["score_index", "basis_value"]].mean()

    benchmark = benchmark.merge(selected[["item_key", "sort_order"]], on="item_key", how="inner")
    if benchmark.empty:
        return {"profile": _empty_profile_comparison_frame(), "benchmark_label": benchmark_label}
    benchmark = benchmark.sort_values("sort_order").reset_index(drop=True)
    return {
        "profile": _finalize_profile_comparison(
            selected,
            benchmark,
            city=city,
            benchmark_label=benchmark_label,
            detail_level=detail_level,
        ),
        "benchmark_label": benchmark_label,
    }


def _build_profile_summary(
    profile_df: pd.DataFrame,
    city: str,
    benchmark_label: str,
    level: str,
    *,
    top_n: int = 3,
) -> dict[str, Any]:
    basis_label = "Index (0-100)"
    if not profile_df.empty and "basis_label" in profile_df.columns:
        basis_labels = profile_df["basis_label"].dropna().astype(str)
        if not basis_labels.empty:
            basis_label = basis_labels.iloc[0]

    empty = {
        "strengths": [],
        "weaknesses": [],
        "basis_label": basis_label,
        "comparison_table": pd.DataFrame(
            columns=[
                "item_key",
                "item_label",
                "group_label",
                "city_score",
                "benchmark_score",
                "delta_to_benchmark",
                "city_basis",
                "benchmark_basis",
                "delta_to_benchmark_basis",
            ]
        ),
    }
    if profile_df.empty:
        return empty

    level_label = str(level).strip().lower()
    pivot = (
        profile_df.pivot_table(
            index=["item_key", "item_label", "group_label", "sort_order"],
            columns="series",
            values="score_index",
            aggfunc="first",
        )
        .reset_index()
        .sort_values("sort_order")
    )
    basis_pivot = (
        profile_df.pivot_table(
            index=["item_key", "item_label", "group_label", "sort_order"],
            columns="series",
            values="basis_value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values("sort_order")
    )
    if city not in pivot.columns or benchmark_label not in pivot.columns:
        return empty

    pivot["city_score"] = pd.to_numeric(pivot[city], errors="coerce")
    pivot["benchmark_score"] = pd.to_numeric(pivot[benchmark_label], errors="coerce")
    if city in basis_pivot.columns:
        pivot["city_basis"] = pd.to_numeric(basis_pivot[city], errors="coerce")
    else:
        pivot["city_basis"] = np.nan
    if benchmark_label in basis_pivot.columns:
        pivot["benchmark_basis"] = pd.to_numeric(basis_pivot[benchmark_label], errors="coerce")
    else:
        pivot["benchmark_basis"] = np.nan
    pivot = pivot.dropna(subset=["city_score", "benchmark_score"]).copy()
    if pivot.empty:
        return empty

    pivot["delta_to_benchmark"] = pivot["city_score"] - pivot["benchmark_score"]
    pivot["delta_to_benchmark_basis"] = pivot["city_basis"] - pivot["benchmark_basis"]

    def _display_label(row: pd.Series) -> str:
        if level_label == "macro":
            return str(row["item_label"])
        return f"{row['item_label']} ({row['group_label']})"

    benchmark_label_in_sentence = str(benchmark_label).strip()

    def _format_strength(row: pd.Series) -> str:
        return f"{_display_label(row)} is ahead of {benchmark_label_in_sentence} by {row['delta_to_benchmark']:.1f} points."

    def _format_weakness(row: pd.Series) -> str:
        return f"{_display_label(row)} trails {benchmark_label_in_sentence} by {abs(row['delta_to_benchmark']):.1f} points."

    strengths_df = pivot[pivot["delta_to_benchmark"] > 0].nlargest(top_n, "delta_to_benchmark")
    weaknesses_df = pivot[pivot["delta_to_benchmark"] < 0].nsmallest(top_n, "delta_to_benchmark")

    return {
        "strengths": [_format_strength(row) for _, row in strengths_df.iterrows()],
        "weaknesses": [_format_weakness(row) for _, row in weaknesses_df.iterrows()],
        "basis_label": basis_label,
        "comparison_table": pivot[
            [
                "item_key",
                "item_label",
                "group_label",
                "city_score",
                "benchmark_score",
                "delta_to_benchmark",
                "city_basis",
                "benchmark_basis",
                "delta_to_benchmark_basis",
            ]
        ].reset_index(drop=True),
    }


def build_city_profile_summary(profile_df: pd.DataFrame, city: str, level: str, top_n: int = 3) -> dict[str, Any]:
    summary = _build_profile_summary(profile_df, city, "Portfolio median", level, top_n=top_n)
    comparison_table = summary["comparison_table"].rename(
        columns={
            "benchmark_score": "median_score",
            "delta_to_benchmark": "delta_to_median",
        }
    )
    return {
        "strengths": summary["strengths"],
        "weaknesses": summary["weaknesses"],
        "comparison_table": comparison_table[
            ["item_key", "item_label", "group_label", "city_score", "median_score", "delta_to_median"]
        ],
    }


def build_benchmark_profile_summary(
    profile_df: pd.DataFrame,
    city: str,
    benchmark_label: str,
    level: str,
    top_n: int = 3,
) -> dict[str, Any]:
    return _build_profile_summary(profile_df, city, benchmark_label, level, top_n=top_n)


def build_benchmark_overview(
    bundle: dict[str, Any],
    city: str,
    *,
    comparison_mode: str = "average",
    benchmark_city: str | None = None,
) -> dict[str, Any]:
    overall_scores = build_city_ranks(bundle["overall_scores"]).copy()
    capability_cost = bundle.get("capability_cost", pd.DataFrame()).copy()

    selected = overall_scores[overall_scores["city"] == city].copy()
    selected_capability = capability_cost[capability_cost["city"] == city].copy()
    if selected.empty:
        return {
            "selected_overall_index": np.nan,
            "benchmark_overall_index": np.nan,
            "overall_index_gap": np.nan,
            "selected_rank": np.nan,
        }

    resolved_mode = str(comparison_mode).strip().lower()
    benchmark_label = "Portfolio average"
    if resolved_mode == "city":
        resolved_city = str(benchmark_city or "").strip()
        if not resolved_city:
            other_cities = [value for value in overall_scores["city"].astype(str).tolist() if value != city]
            resolved_city = other_cities[0] if other_cities else city
        benchmark = overall_scores[overall_scores["city"] == resolved_city].copy()
        benchmark_label = resolved_city
    elif resolved_mode == "best":
        best_city = str(overall_scores.iloc[0]["city"])
        benchmark = overall_scores[overall_scores["city"] == best_city].copy()
        benchmark_label = f"{best_city} (best performer)"
    else:
        benchmark = pd.DataFrame(
            [
                {
                    "overall_index": pd.to_numeric(overall_scores["overall_index"], errors="coerce").mean(),
                }
            ]
        )

    selected_overall_index = float(selected.iloc[0]["overall_index"]) if "overall_index" in selected.columns else np.nan
    benchmark_overall_index = (
        float(pd.to_numeric(benchmark.iloc[0]["overall_index"], errors="coerce")) if not benchmark.empty else np.nan
    )

    benchmark_capability_index = np.nan
    benchmark_cost_index = np.nan
    if not selected_capability.empty and not capability_cost.empty:
        if resolved_mode == "city":
            benchmark_capability = capability_cost[capability_cost["city"] == benchmark_label].copy()
            if benchmark_capability.empty and benchmark_city:
                benchmark_capability = capability_cost[capability_cost["city"] == str(benchmark_city)].copy()
            if not benchmark_capability.empty:
                benchmark_capability_index = float(benchmark_capability.iloc[0].get("capability_index", np.nan))
                benchmark_cost_index = float(benchmark_capability.iloc[0].get("cost_index", np.nan))
        elif resolved_mode == "best":
            best_city = str(overall_scores.iloc[0]["city"])
            benchmark_capability = capability_cost[capability_cost["city"] == best_city].copy()
            if not benchmark_capability.empty:
                benchmark_capability_index = float(benchmark_capability.iloc[0].get("capability_index", np.nan))
                benchmark_cost_index = float(benchmark_capability.iloc[0].get("cost_index", np.nan))
        else:
            benchmark_capability_index = float(pd.to_numeric(capability_cost.get("capability_index"), errors="coerce").mean())
            benchmark_cost_index = float(pd.to_numeric(capability_cost.get("cost_index"), errors="coerce").mean())

    selected_capability_index = (
        float(selected_capability.iloc[0]["capability_index"])
        if not selected_capability.empty and "capability_index" in selected_capability.columns
        else np.nan
    )
    selected_cost_index = (
        float(selected_capability.iloc[0]["cost_index"])
        if not selected_capability.empty and "cost_index" in selected_capability.columns
        else np.nan
    )

    return {
        "selected_overall_index": selected_overall_index,
        "benchmark_overall_index": benchmark_overall_index,
        "overall_index_gap": selected_overall_index - benchmark_overall_index,
        "selected_rank": float(selected.iloc[0]["overall_rank"]) if "overall_rank" in selected.columns else np.nan,
        "selected_capability_index": selected_capability_index,
        "benchmark_capability_index": benchmark_capability_index,
        "capability_index_gap": selected_capability_index - benchmark_capability_index,
        "selected_cost_index": selected_cost_index,
        "benchmark_cost_index": benchmark_cost_index,
        "cost_index_gap": selected_cost_index - benchmark_cost_index,
        "benchmark_label": benchmark_label,
    }


def _normalize_macro_label(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in str(value)).strip()


def infer_capability_cost_macros(macros: list[str]) -> tuple[list[str], str]:
    if not macros:
        return [], DEFAULT_COST_MACRO

    normalized_map = {macro: _normalize_macro_label(macro) for macro in macros}
    by_norm = {norm: macro for macro, norm in normalized_map.items()}

    default_cost_norm = _normalize_macro_label(DEFAULT_COST_MACRO)
    if default_cost_norm in by_norm:
        cost_macro = by_norm[default_cost_norm]
    else:
        cost_keywords = (
            "cost",
            "costs",
            "expense",
            "expenses",
            "salary",
            "wage",
            "wages",
            "rent",
            "price",
            "pricing",
            "tax",
            "financial",
            "finance",
            "incentive",
            "utility",
        )
        scored: list[tuple[int, int, str]] = []
        for macro, norm in normalized_map.items():
            words = [w for w in norm.split() if w]
            score = sum(1 for w in words if any(keyword == w or keyword in w for keyword in cost_keywords))
            scored.append((score, len(words), macro))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        if scored[0][0] > 0:
            cost_macro = scored[0][2]
        elif len(macros) > 1:
            # Fallback: keep first macro for capability; assign last for cost when no cost-like label exists.
            cost_macro = sorted(macros)[-1]
        else:
            cost_macro = macros[0]

    capability_macros = [macro for macro in macros if macro != cost_macro]
    if not capability_macros:
        capability_macros = [cost_macro]
    return capability_macros, cost_macro


def compute_capability_cost(
    macro_scores: pd.DataFrame,
    capability_macros: list[str],
    cost_macro: str = DEFAULT_COST_MACRO,
) -> pd.DataFrame:
    table = macro_scores.copy()

    capability_block = table[table["macro"].isin(capability_macros)].copy()
    rows: list[dict[str, Any]] = []
    for city, city_df in capability_block.groupby("city", dropna=False):
        if city_df["macro_weight"].sum() > 0:
            capability_score = float(np.average(city_df["macro_score"], weights=city_df["macro_weight"]))
        else:
            capability_score = float(city_df["macro_score"].mean())
        rows.append({"city": city, "capability_score": capability_score})
    capability_scores = pd.DataFrame(rows)

    cost_scores = (
        table[table["macro"] == cost_macro][["city", "macro_score"]]
        .rename(columns={"macro_score": "cost_score"})
        .drop_duplicates("city")
    )

    output = capability_scores.merge(cost_scores, on="city", how="outer")
    output["capability_index"] = pd.to_numeric(output["capability_score"], errors="coerce") * 100.0
    output["cost_index"] = pd.to_numeric(output["cost_score"], errors="coerce") * 100.0
    return output


def add_market_tiers(score_df: pd.DataFrame) -> pd.DataFrame:
    table = score_df.copy()
    if table["overall_score"].nunique(dropna=True) < 3:
        table["market_tier"] = "Primary"
        return table
    table["market_tier"] = pd.qcut(
        table["overall_score"],
        q=3,
        labels=["Tertiary", "Secondary", "Primary"],
        duplicates="drop",
    ).astype(str)
    return table


def resolve_matrix_view_preference(current_view: str | None, has_rank_data: bool) -> str:
    options = ["Computed Ranks", "Score Index (0-100)", "Raw (units vary)"]
    if not has_rank_data:
        fallback = "Score Index (0-100)"
    else:
        fallback = "Computed Ranks"
    if current_view not in options:
        return fallback
    if current_view == "Computed Ranks" and not has_rank_data:
        return "Score Index (0-100)"
    return current_view


@st.cache_data(show_spinner=False)
def parse_workbook_cached(file_bytes: bytes) -> dict[str, Any]:
    return io.load_workbook_from_bytes(file_bytes)


@st.cache_data(show_spinner=False)
def compute_results_cached(
    criteria_df: pd.DataFrame,
    raw_data_df: pd.DataFrame,
    city_columns: tuple[str, ...],
    direction_items: tuple[tuple[str, str], ...],
    scoring_method: str,
    macro_weights: pd.DataFrame,
    major_weights: pd.DataFrame,
    minor_weights: pd.DataFrame,
) -> dict[str, Any]:
    direction_map = dict(direction_items)
    weighted_criteria = compose_weighted_criteria(criteria_df, macro_weights, major_weights, minor_weights)
    micro_scores = scoring.compute_micro_scores(
        raw_data=raw_data_df,
        city_columns=list(city_columns),
        direction_map=direction_map,
        method=scoring_method,
    )
    aggregations = scoring.aggregate_scores(micro_scores, weighted_criteria)

    micro_score_matrix = micro_scores.pivot(index="criterion_id", columns="city", values="score").reset_index()
    raw_value_matrix = micro_scores.pivot(index="criterion_id", columns="city", values="raw_value").reset_index()
    rank_matrix = micro_scores.pivot(index="criterion_id", columns="city", values="rank").reset_index()

    city_scores = _build_city_scores_table(
        overall_scores=aggregations["overall_scores"],
        macro_scores=aggregations["macro_scores"],
        major_scores=aggregations["major_scores"],
    )

    top_drivers = build_top_drivers(aggregations["contributions"])

    return {
        "weighted_criteria": weighted_criteria,
        "micro_scores": aggregations["micro_scores"],
        "major_scores": aggregations["major_scores"],
        "macro_scores": aggregations["macro_scores"],
        "overall_scores": aggregations["overall_scores"],
        "contributions": aggregations["contributions"],
        "city_scores": city_scores,
        "top_drivers": top_drivers,
        "micro_score_matrix": micro_score_matrix,
        "raw_value_matrix": raw_value_matrix,
        "rank_matrix": rank_matrix,
    }


def _load_workbook_bytes_from_session_or_default() -> tuple[bytes | None, str | None]:
    if "lens_file_bytes" in st.session_state:
        return st.session_state["lens_file_bytes"], st.session_state.get("lens_file_name", "Session file")

    default_path = io.find_default_workbook(Path.cwd())
    if default_path is None:
        return None, None

    file_bytes = default_path.read_bytes()
    st.session_state["lens_file_bytes"] = file_bytes
    st.session_state["lens_file_name"] = default_path.name
    return file_bytes, default_path.name


def initialize_state_for_new_workbook(
    workbook_hash: str,
    parsed: dict[str, Any],
) -> None:
    st.session_state["lens_workbook_hash"] = workbook_hash
    st.session_state["lens_parsed"] = parsed

    macro_weights, major_weights, minor_weights = build_default_weight_tables(parsed["criteria"])
    st.session_state["lens_macro_weights"] = macro_weights
    st.session_state["lens_default_macro_weights"] = macro_weights.copy()
    st.session_state["lens_major_weights"] = major_weights
    st.session_state["lens_minor_weights"] = minor_weights
    st.session_state["lens_direction_map"] = build_default_direction_map(parsed["criteria"], parsed["raw_data"])

    st.session_state.setdefault("lens_scoring_method", SCORING_METHOD_RANK)
    st.session_state["lens_scoring_method"] = clamp_user_scoring_method(st.session_state["lens_scoring_method"])
    st.session_state.setdefault("lens_last_advanced_scoring_method", st.session_state["lens_scoring_method"])
    st.session_state["lens_last_advanced_scoring_method"] = clamp_user_scoring_method(
        st.session_state["lens_last_advanced_scoring_method"]
    )
    st.session_state.setdefault("lens_weight_mode", "Simple")
    st.session_state.setdefault("lens_mode", get_default_mode())
    st.session_state.setdefault("lens_matrix_view", "Computed Ranks")


def render_sidebar() -> dict[str, Any]:
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.subheader("Navigation")
    render_nav_link(
        "LENS Location Evaluation",
        route="app",
        standalone_page_path="app.py",
        key="lens_sidebar_nav_home",
        sidebar=True,
    )
    render_nav_link(
        "Weights and Scoring",
        route="weights",
        standalone_page_path="pages/1_Weights_and_Scoring.py",
        key="lens_sidebar_nav_weights",
        sidebar=True,
    )
    render_nav_link(
        "Results Dashboard",
        route="results",
        standalone_page_path="pages/2_Results_Dashboard.py",
        key="lens_sidebar_nav_results",
        sidebar=True,
    )
    render_nav_link(
        "Benchmarking",
        route="benchmarking",
        standalone_page_path="pages/3_Benchmarking.py",
        key="lens_sidebar_nav_benchmarking",
        sidebar=True,
    )
    render_nav_link(
        "Data Matrix",
        route="matrix",
        standalone_page_path="pages/4_Data_Matrix.py",
        key="lens_sidebar_nav_matrix",
        sidebar=True,
    )
    render_nav_link(
        "Export",
        route="export",
        standalone_page_path="pages/5_Export.py",
        key="lens_sidebar_nav_export",
        sidebar=True,
    )
    render_nav_link(
        "Methodology and Glossary",
        route="methodology",
        standalone_page_path="pages/6_Methodology_and_Glossary.py",
        key="lens_sidebar_nav_methodology",
        sidebar=True,
    )
    if is_embedded_mode(st.session_state):
        st.sidebar.caption("Embedded mode: navigation is routed inside Savills DX.")
    st.sidebar.divider()

    st.sidebar.header("Model")
    st.session_state.setdefault("lens_mode", MODE_CLIENT)
    st.session_state["lens_mode"] = st.sidebar.radio(
        "Mode",
        options=MODE_OPTIONS,
        index=MODE_OPTIONS.index(st.session_state.get("lens_mode", MODE_CLIENT)),
    )

    file_bytes, file_name = _load_workbook_bytes_from_session_or_default()

    if file_bytes is None:
        st.sidebar.info("Upload a workbook to begin.")
        st.sidebar.divider()
        st.sidebar.subheader("Data Source")
        uploaded = st.sidebar.file_uploader("Upload Excel workbook", type=["xlsx"], key="lens_file_uploader")
        if uploaded is not None:
            st.session_state["lens_file_bytes"] = uploaded.getvalue()
            st.session_state["lens_file_name"] = uploaded.name
            st.rerun()
        return {"ready": False, "mode": st.session_state["lens_mode"]}

    workbook_hash = md5(file_bytes).hexdigest()
    parse_error = None
    parsed: dict[str, Any] | None = None
    try:
        parsed = parse_workbook_cached(file_bytes)
    except Exception as exc:  # pragma: no cover - Streamlit surface
        parse_error = str(exc)

    if parse_error:
        st.sidebar.error(f"Workbook parse error: {parse_error}")
        return {"ready": False, "mode": st.session_state["lens_mode"]}

    assert parsed is not None
    if st.session_state.get("lens_workbook_hash") != workbook_hash:
        initialize_state_for_new_workbook(workbook_hash, parsed)
    else:
        st.session_state["lens_parsed"] = parsed

    st.sidebar.caption(f"Loaded file: {file_name}")

    st.session_state["lens_scoring_method"] = clamp_user_scoring_method(
        st.session_state.get("lens_scoring_method", SCORING_METHOD_RANK)
    )
    st.session_state.setdefault("lens_last_advanced_scoring_method", st.session_state["lens_scoring_method"])
    st.session_state["lens_last_advanced_scoring_method"] = clamp_user_scoring_method(
        st.session_state.get("lens_last_advanced_scoring_method", st.session_state["lens_scoring_method"])
    )

    with st.sidebar.expander("Scoring and weighting", expanded=st.session_state["lens_mode"] == MODE_ADVANCED):
        default_weight_mode_index = 0 if st.session_state.get("lens_weight_mode") == "Simple" else 1
        st.session_state["lens_weight_mode"] = st.radio(
            "Weighting mode",
            options=["Simple", "Advanced"],
            index=default_weight_mode_index,
            help=WEIGHTING_MODE_HELP,
        )

        if st.session_state["lens_mode"] == MODE_ADVANCED:
            method_labels = SCORING_METHOD_ADVANCED_LABELS
            method_keys = get_supported_scoring_method_keys()
            selected_method = clamp_user_scoring_method(st.session_state["lens_last_advanced_scoring_method"])
            if selected_method not in method_keys:
                selected_method = SCORING_METHOD_RANK

            selected_method_label = st.radio(
                "Scoring method",
                options=method_labels,
                index=method_keys.index(selected_method),
                key="lens_scoring_method_label",
                help=SCORING_METHOD_HELP,
            )
            resolved_method = SCORING_METHOD_LABELS[selected_method_label]
            st.session_state["lens_scoring_method"] = resolved_method
            st.session_state["lens_last_advanced_scoring_method"] = resolved_method
            st.caption(SCORING_METHOD_HELP)
        else:
            st.session_state["lens_scoring_method"] = clamp_user_scoring_method(
                st.session_state.get("lens_last_advanced_scoring_method", st.session_state["lens_scoring_method"])
            )

    macros = sorted(parsed["criteria"]["macro"].dropna().unique().tolist())
    capability_macros, cost_macro = infer_capability_cost_macros(macros)
    st.session_state["lens_capability_macros"] = capability_macros
    st.session_state["lens_cost_macro"] = cost_macro

    data_validation = validate.validate_input_data(parsed["criteria"], parsed["raw_data"], parsed["city_columns"])
    if data_validation.errors:
        st.sidebar.error("Input data has blocking errors.")
    elif not data_validation.warnings:
        st.sidebar.success("Input data validated.")

    st.sidebar.divider()
    with st.sidebar.expander("Workbook", expanded=False):
        uploaded = st.file_uploader("Upload Excel workbook", type=["xlsx"], key="lens_file_uploader")
        if uploaded is not None:
            uploaded_bytes = uploaded.getvalue()
            uploaded_name = uploaded.name
            current_name = st.session_state.get("lens_file_name")
            current_bytes = st.session_state.get("lens_file_bytes")
            if current_name != uploaded_name or current_bytes != uploaded_bytes:
                st.session_state["lens_file_bytes"] = uploaded_bytes
                st.session_state["lens_file_name"] = uploaded_name
                st.rerun()

    return {
        "ready": True,
        "mode": st.session_state["lens_mode"],
        "parsed": parsed,
        "data_validation": data_validation,
        "scoring_method": st.session_state["lens_scoring_method"],
        "weight_mode": st.session_state["lens_weight_mode"],
        "capability_macros": st.session_state["lens_capability_macros"],
        "cost_macro": st.session_state["lens_cost_macro"],
    }


def ensure_context_ready(context: dict[str, Any], upload_message: str = "Upload a workbook from the sidebar to begin.") -> None:
    if not context.get("ready"):
        st.info(upload_message)
        st.stop()


def ensure_data_validation(context: dict[str, Any], prefix: str = "Input data has blocking errors.") -> None:
    validation_result = context.get("data_validation")
    if validation_result and validation_result.errors:
        st.error(prefix)
        for err in validation_result.errors:
            st.write(f"- {err}")
        st.stop()


def ensure_results_bundle(bundle: dict[str, Any] | None) -> dict[str, Any]:
    if bundle is None:
        st.stop()
    if "weight_validation" in bundle and not bundle["weight_validation"].is_valid:
        st.error("Weight sums are invalid. Fix them in 'Weights and Scoring'.")
        for err in bundle["weight_validation"].errors:
            st.write(f"- {err}")
        st.stop()
    if "direction_validation" in bundle and not bundle["direction_validation"].is_valid:
        st.error("Direction overrides are invalid. Fix them in 'Weights and Scoring'.")
        for err in bundle["direction_validation"].errors:
            st.write(f"- {err}")
        st.stop()
    return bundle


def get_active_weight_tables(criteria_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    macro_weights = st.session_state["lens_macro_weights"].copy()
    if st.session_state.get("lens_weight_mode", "Simple") == "Simple":
        major_weights, minor_weights = build_simple_weight_tables(criteria_df, macro_weights)
    else:
        major_weights = st.session_state["lens_major_weights"].copy()
        minor_weights = st.session_state["lens_minor_weights"].copy()
    return macro_weights, major_weights, minor_weights


def get_weight_validation(criteria_df: pd.DataFrame) -> validate.ValidationResult:
    macro_weights, major_weights, minor_weights = get_active_weight_tables(criteria_df)
    return validate.validate_weight_sums(macro_weights, major_weights, minor_weights)


def get_results_bundle(context: dict[str, Any]) -> dict[str, Any] | None:
    if not context.get("ready", False):
        return None

    parsed = context["parsed"]
    criteria_df = parsed["criteria"]
    raw_data_df = parsed["raw_data"]
    city_columns = tuple(parsed["city_columns"])

    weight_validation = get_weight_validation(criteria_df)
    if not weight_validation.is_valid:
        return {"weight_validation": weight_validation}

    direction_map = st.session_state.get("lens_direction_map", {})
    direction_validation = validate.validate_direction_map(direction_map, set(criteria_df["criterion_id"]))
    if not direction_validation.is_valid:
        return {"direction_validation": direction_validation}

    macro_weights, major_weights, minor_weights = get_active_weight_tables(criteria_df)
    results = compute_results_cached(
        criteria_df=criteria_df,
        raw_data_df=raw_data_df,
        city_columns=city_columns,
        direction_items=tuple(sorted(direction_map.items())),
        scoring_method=context["scoring_method"],
        macro_weights=macro_weights,
        major_weights=major_weights,
        minor_weights=minor_weights,
    )
    results["micro_scores"] = add_indexed_score_column(results["micro_scores"], "score", "score_index")
    results["contributions"] = add_indexed_score_column(results["contributions"], "score", "score_index")
    results["major_scores"] = add_indexed_score_column(results["major_scores"], "major_score", "major_index")
    results["macro_scores"] = add_indexed_score_column(results["macro_scores"], "macro_score", "macro_index")
    results["overall_scores"] = add_overall_index(results["overall_scores"])
    results["city_scores"] = _build_city_scores_table(
        overall_scores=results["overall_scores"],
        macro_scores=results["macro_scores"],
        major_scores=results["major_scores"],
    )

    capability_df = compute_capability_cost(
        macro_scores=results["macro_scores"],
        capability_macros=context["capability_macros"],
        cost_macro=context["cost_macro"],
    )

    overall_with_tier = add_market_tiers(results["overall_scores"])
    capability_df = capability_df.merge(
        overall_with_tier[["city", "overall_score", "overall_index", "market_tier"]],
        on="city",
        how="left",
    )

    population = parsed["raw_data"][parsed["raw_data"]["micro"].str.lower() == "population"]
    if not population.empty:
        pop_long = population.melt(
            id_vars=["criterion_id", "macro", "major", "micro", "source"],
            value_vars=list(city_columns),
            var_name="city",
            value_name="population",
        )
        pop_city = pop_long.groupby("city", as_index=False)["population"].mean()
        capability_df = capability_df.merge(pop_city, on="city", how="left")
    else:
        capability_df["population"] = 1.0
    capability_df["population"] = capability_df["population"].fillna(1.0)

    return {
        **results,
        "capability_cost": capability_df,
        "weight_validation": weight_validation,
    }
