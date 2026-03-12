from __future__ import annotations

from shared.runtime.models import AppArtifacts, LogFn, UploadPayload
from shell.registry_models import ModuleMetadata, UploadConfig

from .wizard import PLUGIN


class CommuteWizardModule:
    metadata = ModuleMetadata(
        id="commute",
        name="Commute Impact Assessment",
        description=PLUGIN.metadata.description,
        section="Commuting Trends",
        module_type="wizard",
        status="active",
        button_label="Upload data",
        supports_upload=True,
        tags=["wizard", "commuting"],
    )
    upload_config = UploadConfig(
        accepted_types=list(PLUGIN.metadata.accepted_upload_types),
        label=PLUGIN.metadata.upload_label,
        help=PLUGIN.metadata.upload_help,
    )

    def validate(self, upload: UploadPayload) -> None:
        PLUGIN.validate(upload)

    def build(self, upload: UploadPayload, log: LogFn) -> AppArtifacts:
        return PLUGIN.build(upload, log)

    def render(self, artifacts: AppArtifacts) -> None:
        PLUGIN.render(artifacts)


MODULE = CommuteWizardModule()
