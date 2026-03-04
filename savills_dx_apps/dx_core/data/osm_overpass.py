from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_TAG_MAP_PATH = Path(__file__).resolve().parents[1] / "config" / "osm_tag_map_v1.json"


class OverpassRateLimitError(RuntimeError):
    """Raised when Overpass returns a rate-limit response after retries."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def load_osm_tag_map(tag_map_path: Path | None = None) -> dict[str, Any]:
    path = tag_map_path or DEFAULT_TAG_MAP_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def selected_bucket_rules(tag_map: dict[str, Any], selected_buckets: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    selected_set = set(selected_buckets)
    buckets = tag_map.get("buckets", {})
    return {bucket: rules for bucket, rules in buckets.items() if bucket in selected_set}


def _regex_union(values: list[str]) -> str:
    return "^({0})$".format("|".join(re.escape(v) for v in values))


def build_overpass_query(lat: float, lon: float, radius_m: int, bucket_rules: dict[str, list[dict[str, Any]]]) -> str:
    fragments: list[str] = []
    for rules in bucket_rules.values():
        for rule in rules:
            key = str(rule["key"])
            values = [str(v) for v in rule.get("values", [])]
            if not key or not values:
                continue
            regex = _regex_union(values)
            fragments.append(f'node(around:{radius_m},{lat},{lon})["{key}"~"{regex}"];')
            fragments.append(f'way(around:{radius_m},{lat},{lon})["{key}"~"{regex}"];')
            fragments.append(f'relation(around:{radius_m},{lat},{lon})["{key}"~"{regex}"];')

    if not fragments:
        return "[out:json][timeout:25];(node(0,0,0,0););out body;"

    return "[out:json][timeout:25];\n(\n  {0}\n);\nout center tags;".format("\n  ".join(fragments))


def _parse_retry_after_seconds(response: requests.Response) -> float | None:
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return max(float(retry_after), 0.0)
    except ValueError:
        return None


def call_overpass(
    query: str,
    endpoint: str = OVERPASS_ENDPOINT,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = 4,
    base_backoff_seconds: float = 1.5,
) -> dict[str, Any]:
    """Call Overpass with retry/backoff for rate limits and transient server errors."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(endpoint, data={"data": query}, timeout=timeout_seconds)
            status = response.status_code

            if status == 429:
                retry_after = _parse_retry_after_seconds(response)
                if attempt >= max_retries:
                    raise OverpassRateLimitError(
                        "Overpass rate limit reached (HTTP 429). Please retry shortly.",
                        retry_after_seconds=retry_after,
                    )
                sleep_seconds = retry_after if retry_after is not None else base_backoff_seconds * (2**attempt)
                time.sleep(max(sleep_seconds, 0.5))
                continue

            if status in {500, 502, 503, 504}:
                if attempt >= max_retries:
                    response.raise_for_status()
                time.sleep(base_backoff_seconds * (2**attempt))
                continue

            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Unexpected Overpass response format")
            return payload
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(base_backoff_seconds * (2**attempt))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Overpass request failed unexpectedly")


def _matching_buckets(tags: dict[str, Any], bucket_rules: dict[str, list[dict[str, Any]]]) -> list[tuple[str, str, str]]:
    matches: list[tuple[str, str, str]] = []
    for bucket, rules in bucket_rules.items():
        for rule in rules:
            key = str(rule.get("key", ""))
            values = {str(v) for v in rule.get("values", [])}
            value = tags.get(key)
            if value is not None and str(value) in values:
                matches.append((bucket, key, str(value)))
    return matches


def parse_overpass_results(payload: dict[str, Any], bucket_rules: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {}) or {}
        matches = _matching_buckets(tags=tags, bucket_rules=bucket_rules)
        if not matches:
            continue

        if "lat" in element and "lon" in element:
            poi_lat = element.get("lat")
            poi_lon = element.get("lon")
        else:
            center = element.get("center") or {}
            poi_lat = center.get("lat")
            poi_lon = center.get("lon")

        if poi_lat is None or poi_lon is None:
            continue

        for bucket, tag_key, tag_value in matches:
            rows.append(
                {
                    "bucket": bucket,
                    "osm_type": element.get("type"),
                    "osm_id": element.get("id"),
                    "name": tags.get("name", ""),
                    "poi_lat": float(poi_lat),
                    "poi_lon": float(poi_lon),
                    "tag_key": tag_key,
                    "tag_value": tag_value,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["bucket", "osm_type", "osm_id", "name", "poi_lat", "poi_lon", "tag_key", "tag_value"])

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["bucket", "osm_type", "osm_id"]).reset_index(drop=True)


def fetch_pois(lat: float, lon: float, radius_m: int, selected_buckets_list: list[str], tag_map: dict[str, Any], endpoint: str = OVERPASS_ENDPOINT) -> pd.DataFrame:
    rules = selected_bucket_rules(tag_map=tag_map, selected_buckets=selected_buckets_list)
    query = build_overpass_query(lat=lat, lon=lon, radius_m=radius_m, bucket_rules=rules)
    payload = call_overpass(query=query, endpoint=endpoint)
    return parse_overpass_results(payload=payload, bucket_rules=rules)
