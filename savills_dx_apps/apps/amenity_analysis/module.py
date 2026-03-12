from __future__ import annotations

from apps.amenity_analysis.runtime import run_amenity_analysis
from shell.registry_models import ModuleMetadata


class AmenityStandaloneModule:
    metadata = ModuleMetadata(
        id="amenity_analysis",
        name="Amenity Analysis",
        description=(
            "Assess office amenity access using OSM amenities and optional local "
            "NaPTAN transport data."
        ),
        section="Location Analytics",
        module_type="standalone",
        status="active",
        button_label="Open",
        supports_upload=False,
        tags=["standalone", "location"],
    )

    def run(self) -> None:
        run_amenity_analysis()


MODULE = AmenityStandaloneModule()
