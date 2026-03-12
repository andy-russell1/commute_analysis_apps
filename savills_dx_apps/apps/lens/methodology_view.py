from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.lens.common import safe_set_page_config
from apps.lens.core import model, scoring
from apps.lens.core.constants import (
    MODE_ADVANCED,
    SCORING_METHOD_LOG_ROBUST_MINMAX,
    SCORING_METHOD_MINMAX,
    SCORING_METHOD_PERCENTILE_RANK,
    SCORING_METHOD_RANK,
    SCORING_METHOD_ROBUST_MINMAX,
)


def render_page() -> None:
    safe_set_page_config(page_title="Methodology and Glossary", layout="wide")

    model.render_page_header("Methodology and Glossary")
    context = model.render_sidebar()
    if not context.get("ready"):
        st.info("Upload a workbook from the sidebar for live outputs. This methodology page is available without data.")

    st.subheader("What This Model Does")
    st.write(
        "LENS converts multiple location indicators into one comparable overall score. "
        "Each city is scored at micro level, then rolled up through major and macro layers "
        "using selected weights."
    )

    st.subheader("Glossary")
    st.markdown(
        """
- **Macro**: top-level theme (for example, Talent or Cost).
- **Major**: sub-theme within a macro (for example, Demographics).
- **Micro**: individual metric used for scoring (for example, Population).
- **Weighting Mode (Simple)**: only macro weights are edited; major and micro weights are equally split.
- **Weighting Mode (Advanced)**: macro, major, and micro weights are directly editable.
- **Direction**: whether higher values are better or lower values are better.
- **Contribution / Driver**: impact of a micro criterion on a city outcome based on score and effective weight.
"""
    )

    if context.get("mode") == MODE_ADVANCED:
        st.subheader("Scoring Methods")
        method_table = pd.DataFrame(
            [
                {
                    "method": "Rank",
                    "what_it_does": "Normalises average rank positions to 0-1.",
                    "when_to_use": "When relative ordering matters most.",
                    "pros": "Stable and easy to explain.",
                    "cons": "Ignores magnitude gaps.",
                },
                {
                    "method": "Percentile Rank",
                    "what_it_does": "Normalised rank in 0-1 (ordering-based).",
                    "when_to_use": "When stakeholders prefer percentile language.",
                    "pros": "Interpretable rank position.",
                    "cons": "Does not preserve magnitude gaps.",
                },
                {
                    "method": "Min-Max",
                    "what_it_does": "Scales raw values by min and max to 0-1.",
                    "when_to_use": "When absolute distance between cities matters.",
                    "pros": "Magnitude-aware.",
                    "cons": "Sensitive to outliers.",
                },
                {
                    "method": "Robust Min-Max",
                    "what_it_does": "Winsorises at 5th/95th percentiles then min-max scales.",
                    "when_to_use": "When outliers exist but magnitude still matters.",
                    "pros": "Less outlier-sensitive.",
                    "cons": "Extreme values are compressed.",
                },
                {
                    "method": "Log + Robust Min-Max",
                    "what_it_does": "Applies log1p (with shift if negative), then robust min-max.",
                    "when_to_use": "Right-skewed metrics (for example population, income, postings).",
                    "pros": "Skew-aware and outlier-robust.",
                    "cons": "Less intuitive raw-value mapping.",
                },
            ]
        )
        st.dataframe(model.format_table_for_display(method_table), use_container_width=True, hide_index=True)

        st.warning(
            "Percentile Rank is a normalised rank. It does not preserve magnitude differences and is not a skewness "
            "correction in the magnitude sense."
        )

        st.subheader("Worked Example: Skewed Metric [100, 101, 5000]")
        values = pd.Series([100.0, 101.0, 5000.0], index=["City A", "City B", "City C"])
        example = pd.DataFrame({"city": values.index, "raw_value": values.values})
        for method_key, method_label in [
            (SCORING_METHOD_RANK, "rank"),
            (SCORING_METHOD_PERCENTILE_RANK, "percentile_rank"),
            (SCORING_METHOD_MINMAX, "minmax"),
            (SCORING_METHOD_ROBUST_MINMAX, "robust_minmax"),
            (SCORING_METHOD_LOG_ROBUST_MINMAX, "log_robust_minmax"),
        ]:
            scores = scoring.score_series(values, method=method_key, direction="higher")
            example[method_label] = [float(scores.loc[idx]) for idx in values.index]

        st.dataframe(model.format_table_for_display(example, decimals=4), use_container_width=True, hide_index=True)
        st.caption(
            "Rank and Percentile Rank produce evenly spaced ordering-based scores. "
            "Min-Max and Log + Robust Min-Max retain magnitude information differently."
        )
    else:
        st.subheader("Scoring Approach")
        st.write(
            "In Client mode, the app uses the configured scoring method in the background so results stay consistent "
            "across pages. Switch to Advanced mode to view and compare scoring methods."
        )

    st.subheader("How to Interpret Outputs")
    st.markdown(
        """
- Rankings are relative to the current set of cities and selected method.
- Missing values are skipped in scoring; fully empty criteria are ignored with warnings.
- For lower-is-better metrics, all methods invert scores so higher final score always means better.
- Use Client mode for decision communication and Advanced mode for technical audit.
"""
    )
