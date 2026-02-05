from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import streamlit as st

from apps.registry import get_plugin, get_plugins
from core.downloads import zip_bytes
from core.models import UploadPayload
from core.paths import LOGO_DIR
from core.session import (
    APP_KEY,
    STEP_KEY,
    append_log,
    build_upload_signature,
    clear_all_states,
    get_app_state,
)


def _set_step(step: int) -> None:
    st.session_state[STEP_KEY] = step


def _get_step() -> int:
    return int(st.session_state.get(STEP_KEY, 1))


def _set_selected_app(app_id: str) -> None:
    st.session_state[APP_KEY] = app_id


def _get_selected_app() -> Optional[str]:
    return st.session_state.get(APP_KEY)


def _is_print_view() -> bool:
    query = st.query_params
    query_print = query.get("print", "0")
    if isinstance(query_print, list):
        query_print = query_print[0] if query_print else "0"
    return str(query_print).lower() in {"1", "true", "yes"}


def _render_step_header(step: int) -> None:
    labels = {
        1: "",
        2: "Step 2 of 3: Upload and preprocess",
        3: "",
    }
    label = labels.get(step, "")
    if label:
        st.caption(label)


def _render_sidebar() -> None:
    with st.sidebar:
        logo_path = LOGO_DIR / "Savills.png"
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)


def _render_restart_button() -> None:
    if _is_print_view():
        return
    st.divider()
    st.markdown('<div class="print-hide">', unsafe_allow_html=True)
    if st.button("Restart Session", key="restart_session"):
        clear_all_states()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _run_pipeline(app_id: str, upload: UploadPayload) -> None:
    plugin = get_plugin(app_id)
    app_state = get_app_state(app_id)

    app_state["status"] = "validating"
    app_state["error"] = None
    append_log(app_state, "Validating upload")

    plugin.validate(upload)

    app_state["status"] = "building"
    append_log(app_state, "Building artifacts")
    artifacts = plugin.build(upload, lambda msg: append_log(app_state, msg))

    app_state["artifacts"] = artifacts
    app_state["status"] = "ready"
    append_log(app_state, "Build complete")


def _render_step_1() -> None:
    header_cols = st.columns([6, 1])
    with header_cols[0]:
        st.markdown(
            '<h1 style="margin-bottom:0.1rem;">Savills DX Apps</h1>'
            '<div style="margin-top:0;">Choose an app to get started.</div>',
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        kc_logo = LOGO_DIR / "Knowledge Cubed.png"
        if kc_logo.exists():
            st.image(str(kc_logo), use_container_width=True)

    plugins = get_plugins()
    cols = st.columns(3)
    for idx, plugin in enumerate(plugins):
        col = cols[idx % 3]
        with col:
            with st.container():
                st.subheader(plugin.metadata.name)
                st.write(plugin.metadata.description)
                if st.button("Open", key="open_{0}".format(plugin.metadata.id)):
                    _set_selected_app(plugin.metadata.id)
                    _set_step(2)
                    st.rerun()
    _render_restart_button()


def _render_step_2(app_id: str) -> None:
    plugin = get_plugin(app_id)
    app_state = get_app_state(app_id)

    st.title(plugin.metadata.name)
    _render_step_header(2)
    st.write(plugin.metadata.description)

    uploader_key = "upload_{0}".format(app_id)

    if app_id == "lightcast":
        mode = st.radio(
            "Upload mode",
            ["Multiple files", "Single ZIP"],
            horizontal=True,
            key="lightcast_upload_mode",
        )
        if mode == "Single ZIP":
            uploaded = st.file_uploader(
                "Upload Lightcast ZIP",
                type=["zip"],
                help="Upload a single ZIP containing multiple Lightcast exports.",
                key="{0}_zip".format(uploader_key),
            )
            uploaded_files = [uploaded] if uploaded is not None else []
        else:
            uploaded_files = st.file_uploader(
                "Upload Lightcast CSV/XLS/XLSX files",
                type=["csv", "xls", "xlsx"],
                help="Select multiple Lightcast exports to build the master table.",
                accept_multiple_files=True,
                key="{0}_multi".format(uploader_key),
            )
        if not uploaded_files:
            st.info("Upload a file to continue.")
            _render_restart_button()
            return

        allowed_exts = set(plugin.metadata.accepted_upload_types)
        data_files = []
        zip_files = []
        ignored = []
        for file_obj in uploaded_files:
            name = file_obj.name
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext == "zip":
                zip_files.append(file_obj)
            elif ext in allowed_exts:
                data_files.append(file_obj)
            else:
                ignored.append(name)

        if zip_files and len(uploaded_files) > 1:
            st.error("For Lightcast, upload either a single ZIP or multiple CSV/XLS/XLSX files, not both.")
            _render_restart_button()
            return

        if ignored:
            st.warning("Ignored files: {0}".format(", ".join(ignored)))

        if zip_files:
            file_obj = zip_files[0]
            bytes_data = file_obj.getvalue()
            ext = "zip"
            upload_name = file_obj.name
        else:
            if not data_files:
                st.error("No valid CSV/XLS/XLSX files found to process.")
                _render_restart_button()
                return
            if len(data_files) == 1:
                file_obj = data_files[0]
                bytes_data = file_obj.getvalue()
                ext = file_obj.name.rsplit(".", 1)[-1].lower() if "." in file_obj.name else ""
                upload_name = file_obj.name
            else:
                file_map = OrderedDict(
                    sorted(((f.name, f.getvalue()) for f in data_files), key=lambda item: item[0].lower())
                )
                bytes_data = zip_bytes(file_map)
                ext = "zip"
                upload_name = "lightcast_uploads.zip"
                st.caption("Files queued: {0}".format(len(data_files)))
    else:
        uploaded = st.file_uploader(
            plugin.metadata.upload_label,
            type=plugin.metadata.accepted_upload_types,
            help=plugin.metadata.upload_help,
            key=uploader_key,
        )
        if uploaded is None:
            st.info("Upload a file to continue.")
            _render_restart_button()
            return
        bytes_data = uploaded.getvalue()
        ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
        upload_name = uploaded.name

    sig = build_upload_signature(upload_name, bytes_data)

    if app_id == "commute" and upload_name.lower() != "successful.csv":
        st.warning("Expected file name is Successful.csv. Validation will still continue.")

    if sig != app_state.get("upload_sig"):
        app_state["upload_bytes"] = bytes_data
        app_state["upload_name"] = upload_name
        app_state["upload_ext"] = ext
        app_state["upload_sig"] = sig
        app_state["status"] = "idle"
        app_state["error"] = None
        app_state["artifacts"] = None
        app_state["logs"] = []
        append_log(app_state, "New upload detected")

    if app_state["status"] == "idle":
        with st.spinner("Validating and preprocessing..."):
            try:
                _run_pipeline(
                    app_id,
                    UploadPayload(name=upload_name, bytes_data=bytes_data, ext=ext),
                )
                st.success("Preprocessing complete. You can proceed to Step 3.")
            except Exception as exc:
                app_state["status"] = "failed"
                app_state["error"] = str(exc)
                append_log(app_state, "Error: {0}".format(exc))

    if app_state["status"] == "failed":
        st.error(app_state.get("error") or "Preprocessing failed.")
        if st.button("Retry preprocessing"):
            app_state["status"] = "idle"
            st.rerun()

    if app_state["status"] == "ready":
        st.success("Ready to run.")
        if st.button("Go to Step 3"):
            _set_step(3)
            st.rerun()

    _render_restart_button()


def _render_step_3(app_id: str) -> None:
    plugin = get_plugin(app_id)
    app_state = get_app_state(app_id)

    if app_state.get("status") != "ready" or app_state.get("artifacts") is None:
        st.error("Step 3 is locked until preprocessing succeeds.")
        _set_step(2)
        st.rerun()
        return

    _render_step_header(3)

    plugin.render(app_state["artifacts"])
    _render_restart_button()


def main() -> None:
    st.set_page_config(page_title="Savills DX Apps", layout="wide")
    app_id = _get_selected_app()
    _render_sidebar()

    step = _get_step()
    if step == 1 or not app_id:
        _set_step(1)
        _render_step_1()
    elif step == 2:
        _render_step_2(app_id)
    elif step == 3:
        _render_step_3(app_id)
    else:
        _set_step(1)
        _render_step_1()


if __name__ == "__main__":
    main()
