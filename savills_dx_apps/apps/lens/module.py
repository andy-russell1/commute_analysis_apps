from __future__ import annotations

from apps.lens.runtime import run_lens
from shell.registry_models import ModuleMetadata


class LensStandaloneModule:
    metadata = ModuleMetadata(
        id="lens",
        name="LENS Location Evaluation",
        description="Decision-first location scoring with client and advanced workflows.",
        section="LENS",
        module_type="standalone",
        status="active",
        button_label="Open",
        supports_upload=False,
        tags=["standalone", "scoring"],
    )

    def run(self) -> None:
        run_lens()


MODULE = LensStandaloneModule()
