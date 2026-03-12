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
    SCORING_METHOD_RANK,
    WEIGHTING_MODE_HELP,
)


def get_default_mode() -> str:
    return MODE_CLIENT


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
        "scoring_method": scoring.normalize_scoring_method_key(st.session_state.get("lens_scoring_method", SCORING_METHOD_RANK)),
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
        st.session_state["lens_scoring_method"] = scoring.normalize_scoring_method_key(legacy_method)

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
    city_scores = overall_scores.copy()
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
        ["macro", "major", "micro_display", "direction", "score", "effective_micro_weight", "contribution", "delta"]
    ].reset_index(drop=True)
    weaknesses = city_micro.nsmallest(top_n, "delta")[
        ["macro", "major", "micro_display", "direction", "score", "effective_micro_weight", "contribution", "delta"]
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
                "score": float(macro_row["macro_score"]),
                "contribution": float(macro_row.get("macro_contribution", np.nan)),
                "direction": "",
                "notes": "Weighted macro contribution to overall score.",
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
                    "score": float(major_row["major_score"]),
                    "contribution": float(major_row.get("major_contribution", np.nan)),
                    "direction": "",
                    "notes": "Weighted major contribution within macro.",
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
                        "score": float(micro_row["score"]),
                        "contribution": float(micro_row["contribution"]),
                        "direction": direction,
                        "notes": "Micro contribution = micro score x effective micro weight.",
                    }
                )

    hierarchy = pd.DataFrame(rows)
    compact = hierarchy[["level", "name", "weight", "score", "contribution", "direction", "notes"]].copy()

    summary = {
        "city": city,
        "overall_score": float(city_rank_row.iloc[0]["overall_score"]),
        "overall_index": float(city_rank_row.iloc[0]["overall_index"]) if "overall_index" in city_rank_row.columns else np.nan,
        "overall_rank": int(city_rank_row.iloc[0]["overall_rank"]),
        "distance_to_leader": float(city_rank_row.iloc[0]["distance_to_leader"])
        if "distance_to_leader" in city_rank_row.columns
        else np.nan,
        "overall_tier": str(city_rank_row.iloc[0]["overall_tier"]) if "overall_tier" in city_rank_row.columns else "",
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
    options = ["Computed Ranks", "Scores", "Raw (units vary)"]
    if not has_rank_data:
        fallback = "Scores"
    else:
        fallback = "Computed Ranks"
    if current_view not in options:
        return fallback
    if current_view == "Computed Ranks" and not has_rank_data:
        return "Scores"
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
    st.session_state["lens_scoring_method"] = scoring.normalize_scoring_method_key(st.session_state["lens_scoring_method"])
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
        "Data Matrix",
        route="matrix",
        standalone_page_path="pages/3_Data_Matrix.py",
        key="lens_sidebar_nav_matrix",
        sidebar=True,
    )
    render_nav_link(
        "Export",
        route="export",
        standalone_page_path="pages/4_Export.py",
        key="lens_sidebar_nav_export",
        sidebar=True,
    )
    render_nav_link(
        "Methodology and Glossary",
        route="methodology",
        standalone_page_path="pages/5_Methodology_and_Glossary.py",
        key="lens_sidebar_nav_methodology",
        sidebar=True,
    )
    if is_embedded_mode(st.session_state):
        st.sidebar.caption("Embedded mode: navigation is routed inside Savills DX.")
    st.sidebar.divider()

    st.sidebar.header("Model Controls")
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

    default_weight_mode_index = 0 if st.session_state.get("lens_weight_mode") == "Simple" else 1
    st.session_state["lens_weight_mode"] = st.sidebar.radio(
        "Weighting mode",
        options=["Simple", "Advanced"],
        index=default_weight_mode_index,
        help=WEIGHTING_MODE_HELP,
    )

    st.session_state["lens_scoring_method"] = scoring.normalize_scoring_method_key(
        st.session_state.get("lens_scoring_method", SCORING_METHOD_RANK)
    )

    if st.session_state["lens_mode"] == MODE_ADVANCED:
        method_labels = SCORING_METHOD_ADVANCED_LABELS
        method_keys = [SCORING_METHOD_LABELS[label] for label in method_labels]
        if st.session_state["lens_scoring_method"] not in method_keys:
            st.session_state["lens_scoring_method"] = SCORING_METHOD_RANK

        selected_method_label = st.sidebar.radio(
            "Scoring method",
            options=method_labels,
            index=method_keys.index(st.session_state["lens_scoring_method"]),
            key="lens_scoring_method_label",
            help=SCORING_METHOD_HELP,
        )
        st.session_state["lens_scoring_method"] = SCORING_METHOD_LABELS[selected_method_label]
        st.sidebar.caption(
            "Rank / Percentile Rank: ordering-based.\n"
            "Min-Max: magnitude-aware, outlier-sensitive.\n"
            "Robust Min-Max: magnitude-aware, more outlier-robust.\n"
            "Log + Robust Min-Max: for right-skewed metrics."
        )
    else:
        st.session_state.pop("lens_scoring_method_label", None)

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
    st.sidebar.subheader("Data Source")
    uploaded = st.sidebar.file_uploader("Upload Excel workbook", type=["xlsx"], key="lens_file_uploader")
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
    capability_df = capability_df.merge(overall_with_tier[["city", "overall_score", "market_tier"]], on="city", how="left")

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
