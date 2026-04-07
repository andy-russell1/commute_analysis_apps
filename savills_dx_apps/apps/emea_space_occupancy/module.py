from __future__ import annotations

from shell.registry_models import ModuleMetadata

from .runtime import run_emea_space_occupancy


class EmeaSpaceOccupancyStandaloneModule:
    metadata = ModuleMetadata(
        id="emea_space_occupancy",
        name="EMEA Space & Occupancy Planning Studio",
        description=(
            "Dynamic EMEA space and occupancy planning with workbook validation, live scenario modelling, "
            "planning outputs, and decision-pack views."
        ),
        section="Workplace Planning",
        module_type="standalone",
        status="active",
        button_label="Open",
        supports_upload=False,
        tags=["standalone", "workplace", "scenario"],
    )

    def run(self) -> None:
        run_emea_space_occupancy()


MODULE = EmeaSpaceOccupancyStandaloneModule()
