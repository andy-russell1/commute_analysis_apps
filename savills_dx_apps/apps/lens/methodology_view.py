from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.lens.common import safe_set_page_config
from apps.lens.core import model, scoring
from apps.lens.core.constants import (
    MODE_ADVANCED,
    SCORING_METHOD_PERCENTILE,
    SCORING_METHOD_RANK,
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
        "Each city is scored from raw metric values in Python at micro level, then rolled up "
        "through major and macro layers using selected weights."
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
- **Competition Rank**: tied cities share the same rank and the next rank skips accordingly (for example 1, 2, 2, 4).
- **Reference Rank Rows**: uploaded rank rows retained for audit/reference only; they never drive scoring.
- **Contribution / Driver**: impact of a micro criterion on a city outcome based on score and effective weight.
"""
    )

    if context.get("mode") == MODE_ADVANCED:
        st.subheader("Scoring Methods")
        method_table = pd.DataFrame(
            [
                {
                    "method": "Rank",
                    "what_it_does": "Computes competition ranks from raw values in Python.",
                    "when_to_use": "When relative ordering matters most.",
                    "pros": "Stable, auditable, and tie-aware.",
                    "cons": "Ignores magnitude gaps.",
                },
                {
                    "method": "Percentile",
                    "what_it_does": "Computes ECDF percentile position from direction-adjusted raw values.",
                    "when_to_use": "When stakeholders want an indexed comparison scale for visuals.",
                    "pros": "Distribution-position based and presentation-ready for radar/polar views.",
                    "cons": "Still does not preserve magnitude gaps between distinct values.",
                },
            ]
        )
        st.dataframe(model.format_table_for_display(method_table), use_container_width=True, hide_index=True)

        st.warning(
            "Uploaded rank rows are reference-only. LENS scoring always recomputes ranks in Python from raw metric values."
        )

        st.subheader("Worked Example: Tie Handling [100, 100, 5000]")
        values = pd.Series([100.0, 100.0, 5000.0], index=["City A", "City B", "City C"])
        example = pd.DataFrame({"city": values.index, "raw_value": values.values})
        computed_ranks = scoring.compute_rank_series(values, direction="higher")
        rank_scores = scoring.score_series(values, method=SCORING_METHOD_RANK, direction="higher")
        percentile_scores = scoring.score_series(values, method=SCORING_METHOD_PERCENTILE, direction="higher")
        example["computed_rank"] = [float(computed_ranks.loc[idx]) for idx in values.index]
        example["rank_score"] = [float(rank_scores.loc[idx]) for idx in values.index]
        example["percentile_index"] = [float(percentile_scores.loc[idx]) * 100.0 for idx in values.index]

        st.dataframe(model.format_table_for_display(example, decimals=4), use_container_width=True, hide_index=True)
        st.caption(
            "Rank stays ordinal, while Percentile reflects cumulative position in the adjusted raw-value distribution on a 0-100 basis."
        )
    else:
        st.subheader("Scoring Approach")
        st.write(
            "In Client mode, the app uses the configured scoring method in the background so results stay consistent "
            "across pages. Switch to Advanced mode to view or change the active Rank/Percentile method."
        )

    st.subheader("How to Interpret Outputs")
    st.markdown(
        """
- Rankings are relative to the current set of cities and selected method.
- Missing values are skipped in scoring; fully empty criteria are ignored with warnings.
- For lower-is-better metrics, scores are inverted after ranking so higher final score always means better.
- Use Client mode for decision communication and Advanced mode for technical audit.
"""
    )
