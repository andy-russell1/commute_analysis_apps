from __future__ import annotations

from collections import OrderedDict

import streamlit as st

from shared.runtime.downloads import zip_bytes
from shared.runtime.models import AppArtifacts, LogFn, UploadPayload
from shell.registry_models import ModuleMetadata, UploadConfig

from .wizard import PLUGIN


class IsochroneCommuteWizardModule:
    metadata = ModuleMetadata(
        id="isochrone_commute",
        name="Isochrone + Commute Impact Assessment",
        description=PLUGIN.metadata.description,
        section="Commuting Trends",
        module_type="wizard",
        status="active",
        button_label="Upload data",
        supports_upload=True,
        tags=["wizard", "isochrone", "commute"],
    )
    upload_config = UploadConfig(
        accepted_types=list(PLUGIN.metadata.accepted_upload_types),
        label=PLUGIN.metadata.upload_label,
        help=PLUGIN.metadata.upload_help,
    )

    def collect_upload(self, uploader_key: str) -> UploadPayload | None:
        isochrone_upload = st.file_uploader(
            "Upload isochrone ZIP",
            type=["zip"],
            help="ZIP should contain .shp, .dbf, and .shx files.",
            key=f"{uploader_key}_iso",
        )
        employee_upload = st.file_uploader(
            "Upload employee data (Successful.csv or geocoded CSV/XLS/XLSX)",
            type=["csv", "xls", "xlsx"],
            help="Employee file must include geocoded latitude/longitude fields.",
            key=f"{uploader_key}_emp",
        )
        if isochrone_upload is None or employee_upload is None:
            st.info("Upload both files to continue.")
            return None

        file_map = OrderedDict(
            [
                (f"isochrones__{isochrone_upload.name}", isochrone_upload.getvalue()),
                (f"employees__{employee_upload.name}", employee_upload.getvalue()),
            ]
        )
        st.caption(f"Isochrone: {isochrone_upload.name} | Employee data: {employee_upload.name}")
        return UploadPayload(name="isochrone_commute_inputs.zip", bytes_data=zip_bytes(file_map), ext="zip")

    def validate(self, upload: UploadPayload) -> None:
        PLUGIN.validate(upload)

    def build(self, upload: UploadPayload, log: LogFn) -> AppArtifacts:
        return PLUGIN.build(upload, log)

    def render(self, artifacts: AppArtifacts) -> None:
        PLUGIN.render(artifacts)


MODULE = IsochroneCommuteWizardModule()
