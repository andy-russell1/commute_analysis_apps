from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "amenity_analysis"


def ensure_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    target = cache_dir or DEFAULT_CACHE_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _normalise_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalise_for_hash(value[k]) for k in sorted(value.keys())}
    if isinstance(value, (list, tuple, set)):
        return [_normalise_for_hash(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, (np.integer, np.floating)):
            return value.item()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def make_hashed_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(_normalise_for_hash(payload), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def write_json_cache(cache_dir: Path, key: str, payload: Any, metadata: Optional[dict[str, Any]] = None) -> Path:
    target = _json_path(cache_dir, key)
    envelope = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "payload": payload,
    }
    target.write_text(json.dumps(envelope, default=_json_default), encoding="utf-8")
    return target


def read_json_cache(cache_dir: Path, key: str) -> Optional[dict[str, Any]]:
    path = _json_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_parquet_cache(cache_dir: Path, key: str, dataframe: pd.DataFrame, metadata: Optional[dict[str, Any]] = None) -> tuple[Path, Path]:
    parquet_path = cache_dir / f"{key}.parquet"
    meta_path = cache_dir / f"{key}.meta.json"
    dataframe.to_parquet(parquet_path, index=False)
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "row_count": int(len(dataframe)),
    }
    meta_path.write_text(json.dumps(meta, default=_json_default), encoding="utf-8")
    return parquet_path, meta_path
