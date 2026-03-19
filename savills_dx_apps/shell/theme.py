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
            --dx-sidebar-bg: #1c2340;
            --dx-sidebar-ink: #f8fafc;
            --dx-sidebar-muted: #dbe4ff;
            --dx-sidebar-line: rgba(248, 250, 252, 0.18);
            --dx-sidebar-hover: rgba(248, 250, 252, 0.1);
            --dx-sidebar-selected: rgba(248, 250, 252, 0.16);
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
            background: var(--dx-sidebar-bg);
            border-right: 1px solid var(--dx-sidebar-line);
            color: var(--dx-sidebar-ink);
          }

          [data-testid="stHeader"] {
            background: var(--dx-sidebar-bg);
            color: var(--dx-sidebar-ink);
          }

          [data-testid="stHeader"] * {
            color: var(--dx-sidebar-ink);
          }

          [data-testid="stToolbar"] {
            background: transparent;
          }

          [data-testid="stSidebar"] > div:first-child {
            background: var(--dx-sidebar-bg);
          }

          [data-testid="stSidebar"] :where(
            p,
            label,
            span,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            .stMarkdown,
            .stText,
            .stCaption
          ) {
            color: var(--dx-sidebar-ink);
          }

          [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--dx-sidebar-muted);
          }

          [data-testid="stSidebar"] [data-testid="stDivider"] {
            border-color: var(--dx-sidebar-line);
          }

          [data-testid="stSidebar"] .stButton > button {
            border-color: rgba(248, 250, 252, 0.28);
            background: rgba(248, 250, 252, 0.04);
            color: var(--dx-sidebar-ink);
            box-shadow: none;
          }

          [data-testid="stSidebar"] .stButton > button:hover {
            border-color: rgba(248, 250, 252, 0.5);
            background: var(--dx-sidebar-hover);
            color: var(--dx-sidebar-ink);
          }

          [data-testid="stSidebar"] .stButton > button:focus-visible {
            outline: 2px solid var(--dx-brand-yellow);
            outline-offset: 2px;
          }

          [data-testid="stSidebar"] a {
            color: var(--dx-sidebar-ink);
          }

          [data-testid="stSidebar"] a:hover,
          [data-testid="stSidebar"] a[aria-current="page"] {
            color: var(--dx-sidebar-ink);
          }

          [data-testid="stSidebar"] [data-testid="stPageLink"] a {
            border-radius: 0.75rem;
            transition: background-color 0.15s ease, border-color 0.15s ease;
          }

          [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover,
          [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
            background: var(--dx-sidebar-selected);
          }

          [data-testid="stSidebar"] [data-baseweb="select"] > div,
          [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: rgba(248, 250, 252, 0.04);
            border-color: rgba(248, 250, 252, 0.24);
            color: var(--dx-sidebar-ink);
          }

          [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: var(--dx-sidebar-ink);
          }

          [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: var(--dx-sidebar-hover);
            border-radius: 0.5rem;
          }

          [data-testid="stSidebar"] [data-testid="stAlert"] {
            color: var(--dx-sidebar-ink);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
