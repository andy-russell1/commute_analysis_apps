from __future__ import annotations

import streamlit as st

from shared.runtime.models import UploadPayload
from shared.runtime.session import append_log, build_upload_signature, get_app_state
from shared.ui.page_header import render_page_header
from shell import home, layout, registry, state
from shell.registry_models import Module, WizardModule


def _deactivate_standalone_context(module_id: str) -> None:
    if module_id == "amenity_analysis":
        from apps.amenity_analysis.common import set_embedded_mode, set_embedded_route

        set_embedded_mode(st.session_state, enabled=False)
        set_embedded_route(st.session_state, "app")
    if module_id == "lens":
        from apps.lens.common import set_embedded_mode, set_embedded_route

        set_embedded_mode(st.session_state, enabled=False)
        set_embedded_route(st.session_state, "app")


def _run_pipeline(module: WizardModule, module_id: str, upload: UploadPayload) -> None:
    app_state = get_app_state(module_id)
    app_state["status"] = "validating"
    app_state["error"] = None
    append_log(app_state, "Validating upload")

    module.validate(upload)

    app_state["status"] = "building"
    append_log(app_state, "Building artifacts")
    artifacts = module.build(upload, lambda message: append_log(app_state, message))

    app_state["artifacts"] = artifacts
    app_state["status"] = "ready"
    append_log(app_state, "Build complete")


def _read_generic_upload(module: WizardModule, uploader_key: str) -> UploadPayload | None:
    uploaded = st.file_uploader(
        module.upload_config.label,
        type=module.upload_config.accepted_types,
        help=module.upload_config.help,
        key=uploader_key,
    )
    if uploaded is None:
        st.info("Upload a file to continue.")
        return None
    ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
    return UploadPayload(name=uploaded.name, bytes_data=uploaded.getvalue(), ext=ext)


def _collect_upload(module: WizardModule, module_id: str) -> UploadPayload | None:
    uploader_key = f"upload_{module_id}"
    collector = getattr(module, "collect_upload", None)
    if callable(collector):
        return collector(uploader_key)
    return _read_generic_upload(module, uploader_key)


def _render_wizard_step_2(module: WizardModule, module_id: str) -> None:
    app_state = get_app_state(module_id)
    render_page_header(module.metadata.name, "Step 2 of 3: Upload and preprocess")
    st.write(module.metadata.description)

    if not module.metadata.supports_upload:
        if app_state["status"] == "idle":
            with st.spinner("Preparing app..."):
                try:
                    _run_pipeline(module, module_id, UploadPayload(name="", bytes_data=b"", ext=""))
                except Exception as exc:
                    app_state["status"] = "failed"
                    app_state["error"] = str(exc)
                    append_log(app_state, f"Error: {exc}")

        if app_state["status"] == "failed":
            st.error(app_state.get("error") or "App setup failed.")
            if st.button("Retry setup", key=f"retry_setup_{module_id}"):
                app_state["status"] = "idle"
                st.rerun()
            return

        if app_state["status"] == "ready":
            state.set_step(3)
            st.rerun()
            return

        st.info("Preparing app...")
        return

    upload = _collect_upload(module, module_id)
    if upload is None:
        return

    if module_id == "commute" and upload.name.lower() != "successful.csv":
        st.warning("Expected file name is Successful.csv. Validation will still continue.")

    sig = build_upload_signature(upload.name, upload.bytes_data)
    if sig != app_state.get("upload_sig"):
        app_state["upload_bytes"] = upload.bytes_data
        app_state["upload_name"] = upload.name
        app_state["upload_ext"] = upload.ext
        app_state["upload_sig"] = sig
        app_state["status"] = "idle"
        app_state["error"] = None
        app_state["artifacts"] = None
        app_state["logs"] = []
        append_log(app_state, "New upload detected")

    if app_state["status"] == "idle":
        with st.spinner("Validating and preprocessing..."):
            try:
                _run_pipeline(module, module_id, upload)
            except Exception as exc:
                app_state["status"] = "failed"
                app_state["error"] = str(exc)
                append_log(app_state, f"Error: {exc}")

    if app_state["status"] == "failed":
        st.error(app_state.get("error") or "Preprocessing failed.")
        if st.button("Retry preprocessing", key=f"retry_preprocess_{module_id}"):
            app_state["status"] = "idle"
            st.rerun()
        return

    if app_state["status"] == "ready":
        st.success("Ready to run.")
        if st.button("Go to Step 3", key=f"go_step_3_{module_id}"):
            state.set_step(3)
            st.rerun()


def _render_wizard_step_3(module: WizardModule, module_id: str) -> None:
    app_state = get_app_state(module_id)
    if app_state.get("status") != "ready" or app_state.get("artifacts") is None:
        st.error("Step 3 is locked until preprocessing succeeds.")
        state.set_step(2)
        st.rerun()
        return

    module.render(app_state["artifacts"])


def _render_wizard_module(module: WizardModule, module_id: str) -> None:
    step = state.get_step()
    if step <= 1:
        state.set_step(2)
        step = 2

    if step == 2:
        _render_wizard_step_2(module, module_id)
        return
    if step == 3:
        _render_wizard_step_3(module, module_id)
        return

    state.set_step(2)
    _render_wizard_step_2(module, module_id)


def _render_selected_module(module: Module) -> None:
    module_type = module.metadata.module_type
    if module_type == "wizard":
        _render_wizard_module(module, module.metadata.id)
        return
    module.run()


def render_shell() -> None:
    selected_id = state.get_selected_module_id()
    if not selected_id:
        home.render_home()
        return

    selected_module = registry.get_module(selected_id)
    layout.render_sidebar_branding()
    _render_selected_module(selected_module)

    actions = layout.render_sidebar_actions(selected_module.metadata.name)

    if actions["go_home"] and selected_id:
        _deactivate_standalone_context(selected_id)
        state.go_home()
        st.rerun()

    if actions["restart"]:
        if selected_id:
            _deactivate_standalone_context(selected_id)
        state.reset_all()
        st.rerun()
