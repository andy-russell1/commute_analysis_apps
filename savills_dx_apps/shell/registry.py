from __future__ import annotations

from typing import Iterable

import streamlit as st

from apps.amenity_analysis.module import MODULE as AMENITY_MODULE
from apps.commute.module import MODULE as COMMUTE_MODULE
from apps.eurostat.module import MODULE as EUROSTAT_MODULE
from apps.isochrone.module import MODULE as ISOCHRONE_MODULE
from apps.isochrone_commute.module import MODULE as ISOCHRONE_COMMUTE_MODULE
from apps.lens.module import MODULE as LENS_MODULE
from apps.lightcast.module import MODULE as LIGHTCAST_MODULE
from shell.registry_models import Module, ModuleMetadata


class PlaceholderWorkspaceModule:
    metadata = ModuleMetadata(
        id="location_workspace",
        name="Location Workspace",
        description="Workspace container for future grouped location modules.",
        section="Workspaces",
        module_type="workspace",
        status="hidden",
        button_label="Open",
        supports_upload=False,
        tags=["workspace"],
    )

    def run(self) -> None:
        st.info("Workspace modules are not yet configured.")


_MODULES: list[Module] = [
    COMMUTE_MODULE,
    ISOCHRONE_MODULE,
    ISOCHRONE_COMMUTE_MODULE,
    LIGHTCAST_MODULE,
    EUROSTAT_MODULE,
    AMENITY_MODULE,
    LENS_MODULE,
    PlaceholderWorkspaceModule(),
]

_REGISTRY = {module.metadata.id: module for module in _MODULES}


def get_modules(*, include_hidden: bool = False) -> list[Module]:
    if include_hidden:
        return list(_MODULES)
    return [module for module in _MODULES if module.metadata.status != "hidden"]


def get_module(module_id: str) -> Module:
    return _REGISTRY[module_id]


def ordered_sections(modules: Iterable[Module]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for module in modules:
        section = module.metadata.section
        if section not in seen:
            seen.add(section)
            order.append(section)
    return order
