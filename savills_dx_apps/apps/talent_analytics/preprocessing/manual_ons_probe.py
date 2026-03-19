from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if __package__ in {None, ""}:
    # Support direct execution from this folder by exposing the repo root.
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from shared.runtime.paths import DATA_DIR


DEFAULT_OUTPUT_DIR = DATA_DIR / "talent_analytics" / "shared" / "ons" / "raw" / "_manual_probes"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRIES = 3


@dataclass(frozen=True)
class ProbeTarget:
    """A named endpoint that can be tested from the CLI."""

    name: str
    url: str
    description: str


PRESET_TARGETS: dict[str, ProbeTarget] = {
    "aps_page": ProbeTarget(
        name="aps_page",
        url="https://www.nomisweb.co.uk/datasets/apsnew",
        description="Annual Population Survey dataset page on Nomis.",
    ),
    "pest_page": ProbeTarget(
        name="pest_page",
        url="https://www.nomisweb.co.uk/datasets/pestnew",
        description="Population estimates dataset page on Nomis.",
    ),
    "api_help": ProbeTarget(
        name="api_help",
        url="https://www.nomisweb.co.uk/api/v01/help",
        description="Nomis API help page for endpoint discovery.",
    ),
    "aps_overview_candidate": ProbeTarget(
        name="aps_overview_candidate",
        url="https://www.nomisweb.co.uk/api/v01/dataset/APSNEW.overview.json",
        description="Candidate APS overview endpoint to verify manually.",
    ),
    "pest_overview_candidate": ProbeTarget(
        name="pest_overview_candidate",
        url="https://www.nomisweb.co.uk/api/v01/dataset/PESTNEW.overview.json",
        description="Candidate PEST overview endpoint to verify manually.",
    ),
}


def _build_session(retries: int) -> requests.Session:
    retry = Retry(
        total=retries,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Savills-DX-Talent-Analytics/1.0 (+manual-probe)",
            "Accept": "*/*",
        }
    )
    return session


def _sanitise_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe.strip("_") or "probe"


def _detect_extension(content_type: str | None, url: str) -> str:
    lowered = (content_type or "").lower()
    if "json" in lowered:
        return ".json"
    if "csv" in lowered or url.lower().endswith(".csv"):
        return ".csv"
    if "excel" in lowered or url.lower().endswith(".xls") or url.lower().endswith(".xlsx"):
        return ".xlsx"
    if "xml" in lowered:
        return ".xml"
    if "html" in lowered:
        return ".html"
    return ".txt"


def _preview_text(payload: bytes, limit: int = 1200) -> str:
    text = payload.decode("utf-8", errors="replace")
    return text[:limit]


def _write_probe_artifacts(
    *,
    name: str,
    url: str,
    response: requests.Response,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _sanitise_name(name)
    content_path = output_dir / f"{stem}{_detect_extension(response.headers.get('content-type'), url)}"
    metadata_path = output_dir / f"{stem}.metadata.json"

    content_path.write_bytes(response.content)
    metadata = {
        "name": name,
        "url": url,
        "final_url": response.url,
        "status_code": response.status_code,
        "reason": response.reason,
        "content_type": response.headers.get("content-type"),
        "content_length": len(response.content),
        "headers": dict(response.headers),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return content_path, metadata_path


def _probe_target(
    *,
    name: str,
    url: str,
    timeout_seconds: int,
    output_dir: Path,
    session: requests.Session,
) -> int:
    print(f"\n=== {name} ===")
    print(f"URL: {url}")
    try:
        response = session.get(url, timeout=timeout_seconds)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return 1

    content_path, metadata_path = _write_probe_artifacts(
        name=name,
        url=url,
        response=response,
        output_dir=output_dir,
    )

    print(f"Status: {response.status_code} {response.reason}")
    print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
    print(f"Final URL: {response.url}")
    print(f"Saved response: {content_path}")
    print(f"Saved metadata: {metadata_path}")
    print("Preview:")
    print(_preview_text(response.content))

    if response.status_code >= 400:
        return 1
    return 0


def _iter_requested_targets(
    *,
    presets: Iterable[str],
    urls: Iterable[str],
) -> list[ProbeTarget]:
    targets: list[ProbeTarget] = []
    for preset in presets:
        if preset not in PRESET_TARGETS:
            valid = ", ".join(sorted(PRESET_TARGETS))
            raise ValueError(f"Unknown preset '{preset}'. Valid presets: {valid}")
        targets.append(PRESET_TARGETS[preset])

    for index, url in enumerate(urls, start=1):
        targets.append(
            ProbeTarget(
                name=f"url_{index}",
                url=url,
                description="User-supplied probe URL.",
            )
        )
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manually probe ONS/Nomis endpoints from the Savills DX repo. "
            "Each response is saved under assets/data/talent_analytics/shared/ons/raw/_manual_probes/ "
            "with a metadata sidecar for later inspection."
        )
    )
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        help=f"Named probe to run. Repeat as needed. Available: {', '.join(sorted(PRESET_TARGETS))}",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Arbitrary URL to test. Repeat as needed.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Request timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Bounded retry count for transient failures. Default: {DEFAULT_RETRIES}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for raw probe outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the available named probes and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        for target in PRESET_TARGETS.values():
            print(f"{target.name}: {target.description}")
            print(f"  {target.url}")
        return 0

    if not args.preset and not args.url:
        parser.error("Specify at least one --preset or --url target to probe.")

    try:
        targets = _iter_requested_targets(presets=args.preset, urls=args.url)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = Path(args.output_dir)
    session = _build_session(retries=max(args.retries, 0))

    failures = 0
    for target in targets:
        failures += _probe_target(
            name=target.name,
            url=target.url,
            timeout_seconds=max(args.timeout, 1),
            output_dir=output_dir,
            session=session,
        )

    if failures:
        print(f"\nCompleted with {failures} failing probe(s).")
        return 1

    print("\nAll probes completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
