from __future__ import annotations

import streamlit as st


def apply_shell_theme() -> None:
    st.markdown(
        """
        <style>
          :root {
            --dx-ink-900: #111827;
            --dx-ink-700: #374151;
            --dx-ink-500: #6b7280;
            --dx-paper: #f8fafc;
            --dx-card: #ffffff;
            --dx-line: #d1d5db;
            --dx-brand-navy: #262a43;
            --dx-brand-teal: #4a9a8d;
            --dx-brand-blue: #6d769c;
            --dx-brand-yellow: #f2d500;
          }

          .dx-hero h1 {
            margin-bottom: 0.2rem;
            color: var(--dx-brand-navy);
          }

          .dx-hero p {
            color: var(--dx-ink-700);
            margin-top: 0;
          }

          .dx-section-title {
            margin-top: 1rem;
            color: var(--dx-brand-navy);
          }

          [data-testid="stVerticalBlock"] .stButton > button[kind="secondary"] {
            border-radius: 999px;
            border: 1px solid rgba(38, 42, 67, 0.28);
            background: transparent;
            color: inherit;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.35);
          }

          [data-testid="stVerticalBlock"] .stButton > button[kind="secondary"]:hover {
            border-color: var(--dx-brand-blue);
            background: rgba(109, 118, 156, 0.06);
          }

          [data-testid="stVerticalBlock"] .stButton > button[kind="primary"] {
            border-radius: 999px;
            border: 1px solid rgba(38, 42, 67, 0.34);
            background: transparent;
            color: inherit;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.35);
          }

          [data-testid="stVerticalBlock"] .stButton > button[kind="primary"]:hover {
            border-color: var(--dx-brand-blue);
            background: rgba(109, 118, 156, 0.06);
          }

          [data-testid="stSidebar"] {
            border-right: 1px solid var(--dx-line);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
