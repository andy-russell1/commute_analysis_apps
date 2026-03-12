from __future__ import annotations

import base64

import streamlit as st

from shared.runtime.paths import LOGO_DIR


def _logo_path():
    combined = LOGO_DIR / "Savills Knowledge Cubed.png"
    if combined.exists():
        return combined
    savills = LOGO_DIR / "Savills.png"
    if savills.exists():
        return savills
    return None


def render_page_header(title: str, caption: str | None = None, *, logo_width: int = 170) -> None:
    title_col, logo_col = st.columns([6.0, 1.2])
    with title_col:
        st.title(title)
        if caption:
            st.caption(caption)
    with logo_col:
        logo_path = _logo_path()
        if logo_path is None:
            return
        encoded_logo = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        st.markdown(
            """
            <div style="display:flex; justify-content:flex-end;">
              <img src="data:image/png;base64,{0}" style="width:{1}px; max-width:100%; height:auto;" />
            </div>
            """.format(encoded_logo, int(logo_width)),
            unsafe_allow_html=True,
        )
