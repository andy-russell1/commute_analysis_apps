from __future__ import annotations

from shell.registry_models import ModuleMetadata

from .app import run_talent_analytics


class TalentAnalyticsStandaloneModule:
    metadata = ModuleMetadata(
        id="talent_analytics",
        name="Talent Analytics",
        description="Review role demand, competition, and industry context from a private uploaded client talent pack.",
        section="Talent Analytics",
        module_type="standalone",
        status="active",
        button_label="Open",
        supports_upload=True,
        tags=["standalone", "talent", "client pack"],
    )

    def run(self) -> None:
        run_talent_analytics()


MODULE = TalentAnalyticsStandaloneModule()
