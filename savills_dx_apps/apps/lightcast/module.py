from __future__ import annotations

from collections import OrderedDict

import streamlit as st

from shared.runtime.downloads import zip_bytes
from shared.runtime.models import AppArtifacts, LogFn, UploadPayload
from shell.registry_models import ModuleMetadata, UploadConfig

from .wizard import PLUGIN


class LightcastWizardModule:
    metadata = ModuleMetadata(
        id="lightcast",
        name=PLUGIN.metadata.name,
        description=PLUGIN.metadata.description,
        section="Talent Analytics",
        module_type="wizard",
        status="active",
        button_label="Upload data",
        supports_upload=True,
        tags=["wizard", "talent"],
    )
    upload_config = UploadConfig(
        accepted_types=list(PLUGIN.metadata.accepted_upload_types),
        label=PLUGIN.metadata.upload_label,
        help=PLUGIN.metadata.upload_help,
    )

    def collect_upload(self, uploader_key: str) -> UploadPayload | None:
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
                key=f"{uploader_key}_zip",
            )
            uploaded_files = [uploaded] if uploaded is not None else []
        else:
            uploaded_files = st.file_uploader(
                "Upload Lightcast CSV/XLS/XLSX files",
                type=["csv", "xls", "xlsx"],
                help="Select multiple Lightcast exports to build the master table.",
                accept_multiple_files=True,
                key=f"{uploader_key}_multi",
            )

        if not uploaded_files:
            st.info("Upload a file to continue.")
            return None

        allowed = set(self.upload_config.accepted_types)
        data_files = []
        zip_files = []
        ignored = []
        for file_obj in uploaded_files:
            name = file_obj.name
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext == "zip":
                zip_files.append(file_obj)
            elif ext in allowed:
                data_files.append(file_obj)
            else:
                ignored.append(name)

        if zip_files and len(uploaded_files) > 1:
            st.error("Upload either a single ZIP or multiple CSV/XLS/XLSX files, not both.")
            return None

        if ignored:
            st.warning("Ignored files: " + ", ".join(ignored))

        if zip_files:
            file_obj = zip_files[0]
            return UploadPayload(name=file_obj.name, bytes_data=file_obj.getvalue(), ext="zip")

        if not data_files:
            st.error("No valid CSV/XLS/XLSX files found to process.")
            return None

        if len(data_files) == 1:
            file_obj = data_files[0]
            ext = file_obj.name.rsplit(".", 1)[-1].lower() if "." in file_obj.name else ""
            return UploadPayload(name=file_obj.name, bytes_data=file_obj.getvalue(), ext=ext)

        file_map = OrderedDict(
            sorted(((file_obj.name, file_obj.getvalue()) for file_obj in data_files), key=lambda item: item[0].lower())
        )
        st.caption(f"Files queued: {len(data_files)}")
        return UploadPayload(name="lightcast_uploads.zip", bytes_data=zip_bytes(file_map), ext="zip")

    def validate(self, upload: UploadPayload) -> None:
        PLUGIN.validate(upload)

    def build(self, upload: UploadPayload, log: LogFn) -> AppArtifacts:
        return PLUGIN.build(upload, log)

    def render(self, artifacts: AppArtifacts) -> None:
        PLUGIN.render(artifacts)


MODULE = LightcastWizardModule()
