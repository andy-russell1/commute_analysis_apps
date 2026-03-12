from __future__ import annotations

from apps.commute.module import MODULE as COMMUTE_MODULE
from apps.eurostat.module import MODULE as EUROSTAT_MODULE
from apps.isochrone.module import MODULE as ISOCHRONE_MODULE
from apps.isochrone_commute.module import MODULE as ISOCHRONE_COMMUTE_MODULE
from apps.lightcast.module import MODULE as LIGHTCAST_MODULE
from shell.registry_models import Module


REGISTRY: dict[str, Module] = {
    COMMUTE_MODULE.metadata.id: COMMUTE_MODULE,
    EUROSTAT_MODULE.metadata.id: EUROSTAT_MODULE,
    ISOCHRONE_MODULE.metadata.id: ISOCHRONE_MODULE,
    ISOCHRONE_COMMUTE_MODULE.metadata.id: ISOCHRONE_COMMUTE_MODULE,
    LIGHTCAST_MODULE.metadata.id: LIGHTCAST_MODULE,
}


def get_modules() -> list[Module]:
    return list(REGISTRY.values())


def get_module(app_id: str) -> Module:
    return REGISTRY[app_id]
