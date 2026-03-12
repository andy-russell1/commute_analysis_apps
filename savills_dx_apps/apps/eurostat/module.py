from __future__ import annotations

from shared.runtime.models import AppArtifacts, LogFn, UploadPayload
from shell.registry_models import ModuleMetadata, UploadConfig

from .wizard import PLUGIN


class EurostatWizardModule:
    metadata = ModuleMetadata(
        id="eurostat",
        name=PLUGIN.metadata.name,
        description=PLUGIN.metadata.description,
        section="Talent Analytics",
        module_type="wizard",
        status="active",
        button_label="Open",
        supports_upload=False,
        tags=["wizard", "eurostat"],
    )
    upload_config = UploadConfig(
        accepted_types=[],
        label="",
        help="",
    )

    def validate(self, upload: UploadPayload) -> None:
        PLUGIN.validate(upload)

    def build(self, upload: UploadPayload, log: LogFn) -> AppArtifacts:
        return PLUGIN.build(upload, log)

    def render(self, artifacts: AppArtifacts) -> None:
        PLUGIN.render(artifacts)


MODULE = EurostatWizardModule()
