from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol


@dataclass(frozen=True)
class AppMetadata:
    id: str
    name: str
    description: str
    accepted_upload_types: List[str]
    upload_label: str
    upload_help: str


@dataclass(frozen=True)
class UploadPayload:
    name: str
    bytes_data: bytes
    ext: str


AppArtifacts = Dict[str, Any]
LogFn = Callable[[str], None]


class AppPlugin(Protocol):
    metadata: AppMetadata

    def validate(self, upload: UploadPayload) -> None:
        ...

    def build(self, upload: UploadPayload, log: LogFn) -> AppArtifacts:
        ...

    def render(self, artifacts: AppArtifacts) -> None:
        ...
