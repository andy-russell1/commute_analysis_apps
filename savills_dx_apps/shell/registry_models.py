from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from shared.runtime.models import AppArtifacts, LogFn, UploadPayload


@dataclass(frozen=True)
class ModuleMetadata:
    id: str
    name: str
    description: str
    section: str
    module_type: Literal["wizard", "standalone", "workspace"]
    status: Literal["active", "beta", "hidden"] = "active"
    button_label: str = "Open"
    supports_upload: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UploadConfig:
    accepted_types: list[str]
    label: str
    help: str


class WizardModule(Protocol):
    metadata: ModuleMetadata
    upload_config: UploadConfig

    def validate(self, upload: UploadPayload) -> None:
        ...

    def build(self, upload: UploadPayload, log: LogFn) -> AppArtifacts:
        ...

    def render(self, artifacts: AppArtifacts) -> None:
        ...


class StandaloneModule(Protocol):
    metadata: ModuleMetadata

    def run(self) -> None:
        ...


class WorkspaceModule(Protocol):
    metadata: ModuleMetadata

    def run(self) -> None:
        ...


Module = WizardModule | StandaloneModule | WorkspaceModule
