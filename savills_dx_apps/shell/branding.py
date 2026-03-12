from __future__ import annotations

from pathlib import Path

from shared.runtime.paths import LOGO_DIR


def combined_logo_path() -> Path | None:
    path = LOGO_DIR / "Savills Knowledge Cubed.png"
    return path if path.exists() else None


def savills_logo_path() -> Path | None:
    path = LOGO_DIR / "Savills.png"
    return path if path.exists() else None


def knowledge_cubed_logo_path() -> Path | None:
    path = LOGO_DIR / "Knowledge Cubed.png"
    return path if path.exists() else None
