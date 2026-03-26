from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from apps.lens.common import safe_set_page_config

from apps.lens.core import model, validate
from apps.lens.core.constants import MODE_ADVANCED, MODE_CLIENT

safe_set_page_config(page_title="Weights and Scoring", layout="wide")


def _safe_key(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return cleaned[:120]


def _sync_weight_value(source_key: str, target_key: str) -> None:
    st.session_state[target_key] = float(st.session_state[source_key])


def _render_weight_control(label: str, slider_key: str, input_key: str) -> float:
    cols = st.columns([5, 1.2])
    with cols[0]:
        st.slider(
            label=label,
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key=slider_key,
            on_change=_sync_weight_value,
            args=(slider_key, input_key),
        )
    with cols[1]:
        st.number_input(
            label=f"{label} %",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            format="%.1f",
            key=input_key,
            label_visibility="collapsed",
            on_change=_sync_weight_value,
            args=(input_key, slider_key),
        )
    return float(st.session_state[slider_key])


def _normalize_slider_values(key_pairs: list[tuple[str, str]]) -> None:
    if not key_pairs:
        return
    total = sum(max(float(st.session_state.get(slider_key, 0.0)), 0.0) for slider_key, _ in key_pairs)
    if total <= 0:
        equal_value = 100.0 / len(key_pairs)
        for slider_key, input_key in key_pairs:
            st.session_state[slider_key] = float(equal_value)
            st.session_state[input_key] = float(equal_value)
        return
    for slider_key, input_key in key_pairs:
        normalized = float(max(float(st.session_state.get(slider_key, 0.0)), 0.0) / total * 100.0)
        st.session_state[slider_key] = normalized
        st.session_state[input_key] = normalized


def _push_macro_weights_to_inputs(macro_df: pd.DataFrame, key_prefix: str) -> None:
    for _, row in macro_df.iterrows():
        macro = str(row["macro"])
        slider_key = f"{key_prefix}_macro_{_safe_key(macro)}"
        input_key = f"{slider_key}_input"
        value = float(row["weight"] * 100.0)
        st.session_state[slider_key] = value
        st.session_state[input_key] = value


def _render_macro_sliders(macro_df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    st.subheader("Macro Weights")
    output = macro_df.copy()
    slider_meta: list[tuple[int, str, str, str]] = []

    for idx, row in output.iterrows():
        macro = str(row["macro"])
        slider_key = f"{key_prefix}_macro_{_safe_key(macro)}"
        input_key = f"{slider_key}_input"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = float(row["weight"] * 100.0)
        if input_key not in st.session_state:
            st.session_state[input_key] = float(st.session_state[slider_key])
        slider_meta.append((idx, macro, slider_key, input_key))

    if st.button("Normalize Macro Weights", key=f"{key_prefix}_macro_normalize"):
        _normalize_slider_values([(slider_key, input_key) for _, _, slider_key, input_key in slider_meta])

    for idx, macro, slider_key, input_key in slider_meta:
        value = _render_weight_control(macro, slider_key, input_key)
        output.loc[idx, "weight"] = float(value) / 100.0

    current_sum = float(output["weight"].sum())
    st.caption(f"Macro sum: {current_sum * 100:.2f}% ({'OK' if abs(current_sum - 1.0) <= 1e-6 else 'Needs fix'})")
    return output


def _render_major_and_minor_sliders(
    major_df: pd.DataFrame,
    minor_df: pd.DataFrame,
    key_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    updated_major = major_df.copy()
    updated_minor = minor_df.copy()

    for macro in updated_major["macro"].dropna().unique():
        with st.expander(f"Macro: {macro}", expanded=False):
            major_slice = updated_major[updated_major["macro"] == macro].copy()
            major_slider_meta: list[tuple[int, str, str, str]] = []

            st.markdown("**Major Weights**")
            for idx, row in major_slice.iterrows():
                major = str(row["major"])
                slider_key = f"{key_prefix}_major_{_safe_key(str(macro))}_{_safe_key(major)}"
                input_key = f"{slider_key}_input"
                if slider_key not in st.session_state:
                    st.session_state[slider_key] = float(row["weight"] * 100.0)
                if input_key not in st.session_state:
                    st.session_state[input_key] = float(st.session_state[slider_key])
                major_slider_meta.append((idx, major, slider_key, input_key))

            if st.button(f"Normalize Majors in {macro}", key=f"{key_prefix}_major_norm_{_safe_key(str(macro))}"):
                _normalize_slider_values(
                    [(slider_key, input_key) for _, _, slider_key, input_key in major_slider_meta]
                )

            for idx, major, slider_key, input_key in major_slider_meta:
                value = _render_weight_control(f"{macro} > {major}", slider_key, input_key)
                updated_major.loc[idx, "weight"] = float(value) / 100.0

            major_sum = float(updated_major[updated_major["macro"] == macro]["weight"].sum())
            st.caption(
                f"{macro} major sum: {major_sum * 100:.2f}% ({'OK' if abs(major_sum - 1.0) <= 1e-6 else 'Needs fix'})"
            )

            for major in major_slice["major"].dropna().unique():
                with st.expander(f"Major: {major}", expanded=False):
                    minor_slice = updated_minor[
                        (updated_minor["macro"] == macro) & (updated_minor["major"] == major)
                    ].copy()
                    minor_slider_meta: list[tuple[int, str, str, str]] = []
                    for idx, row in minor_slice.iterrows():
                        micro = str(row["micro"])
                        slider_key = f"{key_prefix}_minor_{_safe_key(str(row['criterion_id']))}"
                        input_key = f"{slider_key}_input"
                        if slider_key not in st.session_state:
                            st.session_state[slider_key] = float(row["weight"] * 100.0)
                        if input_key not in st.session_state:
                            st.session_state[input_key] = float(st.session_state[slider_key])
                        minor_slider_meta.append((idx, micro, slider_key, input_key))

                    if st.button(
                        f"Normalize Minors in {major}",
                        key=f"{key_prefix}_minor_norm_{_safe_key(str(macro))}_{_safe_key(str(major))}",
                    ):
                        _normalize_slider_values(
                            [(slider_key, input_key) for _, _, slider_key, input_key in minor_slider_meta]
                        )

                    for idx, micro, slider_key, input_key in minor_slider_meta:
                        value = _render_weight_control(f"{macro} > {major} > {micro}", slider_key, input_key)
                        updated_minor.loc[idx, "weight"] = float(value) / 100.0

                    minor_sum = float(
                        updated_minor[
                            (updated_minor["macro"] == macro) & (updated_minor["major"] == major)
                        ]["weight"].sum()
                    )
                    st.caption(
                        f"{macro} > {major} minor sum: {minor_sum * 100:.2f}% "
                        f"({'OK' if abs(minor_sum - 1.0) <= 1e-6 else 'Needs fix'})"
                    )
    return updated_major, updated_minor


def _render_client_presets(macro_weights: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    st.markdown("**Quick Presets**")
    preset_names = ["Balanced", "Cost-led", "Talent-led", "Risk-averse", "Growth-led"]
    cols = st.columns(len(preset_names))
    updated = macro_weights.copy()

    for idx, preset in enumerate(preset_names):
        with cols[idx]:
            if st.button(preset, use_container_width=True):
                if preset == "Balanced" and "lens_default_macro_weights" in st.session_state:
                    updated = st.session_state["lens_default_macro_weights"].copy()
                else:
                    updated = model.apply_macro_preset(preset, updated)
                st.session_state["lens_macro_weights"] = updated.copy()
                _push_macro_weights_to_inputs(updated, key_prefix=key_prefix)
    return st.session_state.get("lens_macro_weights", updated).copy()


def _render_direction_overrides(criteria_df: pd.DataFrame, raw_df: pd.DataFrame, key_prefix: str) -> None:
    st.subheader("Scoring Direction Overrides")
    source_map = raw_df.set_index("criterion_id")["source"].to_dict()
    direction_map = st.session_state.get("lens_direction_map", {})
    direction_table = criteria_df[["criterion_id", "macro", "major", "micro"]].copy()
    direction_table["source"] = direction_table["criterion_id"].map(source_map)
    direction_table["direction"] = direction_table["criterion_id"].map(direction_map).fillna("higher")
    direction_table.insert(0, "id", range(1, len(direction_table) + 1))
    editor_df = direction_table.rename(
        columns={
            "id": "ID",
            "criterion_id": "_criterion_id",
            "macro": "Macro",
            "major": "Major",
            "micro": "Micro",
            "source": "Source",
            "direction": "Direction",
        }
    )
    edited = st.data_editor(
        editor_df,
        hide_index=True,
        use_container_width=True,
        disabled=["ID", "Macro", "Major", "Micro", "Source", "_criterion_id"],
        column_config={
            "_criterion_id": None,
            "Direction": st.column_config.SelectboxColumn(
                "Direction",
                options=["higher", "lower"],
                help="Override inferred ranking direction.",
                required=True,
            )
        },
        key=f"{key_prefix}_direction_editor",
    )
    st.session_state["lens_direction_map"] = edited.set_index("_criterion_id")["Direction"].to_dict()

    direction_validation = validate.validate_direction_map(
        st.session_state["lens_direction_map"],
        set(criteria_df["criterion_id"]),
    )
    if direction_validation.errors:
        st.error("Direction overrides contain invalid values.")
        for err in direction_validation.errors:
            st.write(f"- {err}")
    if direction_validation.warnings:
        for warning in direction_validation.warnings:
            st.warning(warning)


model.render_page_header("Weights and Scoring")
context = model.render_sidebar()
model.ensure_context_ready(context)
model.ensure_data_validation(context, prefix="Input data has blocking errors. Fix the workbook before adjusting weights.")

parsed = context["parsed"]
criteria_df = parsed["criteria"]
raw_df = parsed["raw_data"]
weight_mode = context["weight_mode"]
mode = context["mode"]
workbook_hash = st.session_state.get("lens_workbook_hash", "default")
key_prefix = f"w_{workbook_hash[:10]}"

macro_weights = st.session_state["lens_macro_weights"].copy()
major_weights = st.session_state["lens_major_weights"].copy()
minor_weights = st.session_state["lens_minor_weights"].copy()

if mode == MODE_CLIENT:
    macro_weights = _render_client_presets(macro_weights, key_prefix=key_prefix)
updated_macro = _render_macro_sliders(macro_weights, key_prefix=key_prefix)
st.session_state["lens_macro_weights"] = updated_macro

if weight_mode == "Simple":
    simple_major, simple_minor = model.build_simple_weight_tables(criteria_df, updated_macro)
    validation_result = validate.validate_weight_sums(updated_macro, simple_major, simple_minor)
else:
    st.subheader("Advanced Hierarchy Weights")
    updated_major, updated_minor = _render_major_and_minor_sliders(
        major_df=major_weights,
        minor_df=minor_weights,
        key_prefix=key_prefix,
    )
    st.session_state["lens_major_weights"] = updated_major
    st.session_state["lens_minor_weights"] = updated_minor
    validation_result = validate.validate_weight_sums(updated_macro, updated_major, updated_minor)

if mode == MODE_ADVANCED:
    _render_direction_overrides(criteria_df, raw_df, key_prefix=key_prefix)

if validation_result.errors:
    st.error("Weight validation failed. Results pages are disabled until weights are fixed.")
    for err in validation_result.errors:
        st.write(f"- {err}")
else:
    st.success("All hierarchy weight sums are valid.")

if validation_result.warnings:
    for warning in validation_result.warnings:
        st.warning(warning)

bundle = model.get_results_bundle(context)
if bundle is None or ("weight_validation" in bundle and not bundle["weight_validation"].is_valid):
    st.info("Fix weight validation issues to generate scoring preview.")
    st.stop()
if "direction_validation" in bundle and not bundle["direction_validation"].is_valid:
    st.info("Fix direction validation issues to generate scoring preview.")
    st.stop()

leaderboard = model.build_city_ranks(bundle["overall_scores"]).copy()
if mode == MODE_ADVANCED:
    with st.expander("Scoring Preview (Advanced)", expanded=False):
        preview = leaderboard[["overall_rank", "city", "overall_index", "overall_score", "overall_tier"]].rename(
            columns={
                "overall_rank": "rank",
                "overall_index": "overall_index",
                "overall_score": "audit_score",
                "overall_tier": "tier",
            }
        )
        st.dataframe(model.format_table_for_display(preview, decimals=1), use_container_width=True, hide_index=True)
else:
    st.subheader("Current Top 3")
    preview = leaderboard.head(3)[["overall_rank", "city", "overall_index", "distance_to_leader"]].rename(
        columns={
            "overall_rank": "rank",
            "overall_index": "overall_index",
            "distance_to_leader": "distance_to_leader",
        }
    )
    st.dataframe(model.format_table_for_display(preview, decimals=1), use_container_width=True, hide_index=True)

