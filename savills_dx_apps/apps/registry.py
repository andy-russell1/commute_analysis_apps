from __future__ import annotations

from typing import Dict, List

from apps.commute import PLUGIN as COMMUTE_PLUGIN
from apps.eurostat import PLUGIN as EUROSTAT_PLUGIN
from apps.isochrone import PLUGIN as ISOCHRONE_PLUGIN
from apps.lightcast import PLUGIN as LIGHTCAST_PLUGIN
from core.models import AppPlugin


REGISTRY: Dict[str, AppPlugin] = {
    COMMUTE_PLUGIN.metadata.id: COMMUTE_PLUGIN,
    EUROSTAT_PLUGIN.metadata.id: EUROSTAT_PLUGIN,
    ISOCHRONE_PLUGIN.metadata.id: ISOCHRONE_PLUGIN,
    LIGHTCAST_PLUGIN.metadata.id: LIGHTCAST_PLUGIN,
}


def get_plugins() -> List[AppPlugin]:
    return list(REGISTRY.values())


def get_plugin(app_id: str) -> AppPlugin:
    return REGISTRY[app_id]
