from __future__ import annotations

import argparse
import sys
from pathlib import Path

from apps.talent_analytics.geography import active_custom_geography_lookup

from ..config import TalentAnalyticsConfig
from .ons_extract import extract_nomis_raw
from .ons_publish import publish_outputs
from .ons_transform import aggregate_custom_geographies, build_lad_metrics, load_source_catalog
from .ons_validate import validate_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the live Talent Analytics ONS / Nomis demographics pipeline. "
            "This pulls raw Nomis extracts, standardises LAD metrics, aggregates to the "
            "20 custom London geographies, validates the result, and publishes processed CSVs."
        )
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override the Talent Analytics data root. Defaults to assets/data/talent_analytics.",
    )
    return parser


def run_pipeline(data_root: Path | None = None) -> dict[str, Path]:
    config = TalentAnalyticsConfig(data_root=data_root or TalentAnalyticsConfig().data_root)
    canonical_lookup = active_custom_geography_lookup(config.data_root)
    source_catalog = load_source_catalog(config.shared_dir / "ons" / "metadata" / "source_catalog.csv")
    raw_dir = config.shared_dir / "ons" / "raw"

    print("Extracting live Nomis data...")
    extracted = extract_nomis_raw(raw_dir)
    print(
        f"Using datasets {extracted.population_metadata.dataset_id} ({extracted.population_metadata.mnemonic}), "
        f"{extracted.aps_counts_metadata.dataset_id} ({extracted.aps_counts_metadata.mnemonic}), "
        f"and {extracted.aps_percentages_metadata.dataset_id} ({extracted.aps_percentages_metadata.mnemonic})."
    )

    print("Standardising LAD metrics...")
    lad_metrics = build_lad_metrics(extracted, canonical_lookup, source_catalog)

    print("Aggregating to custom geographies...")
    custom_metrics = aggregate_custom_geographies(lad_metrics, canonical_lookup)

    print("Validating outputs...")
    validate_outputs(
        canonical_lookup=canonical_lookup,
        source_catalog=source_catalog,
        lad_metrics=lad_metrics,
        custom_metrics=custom_metrics,
    )

    print("Publishing processed files...")
    published_paths = publish_outputs(
        data_root=config.data_root,
        canonical_lookup=canonical_lookup,
        source_catalog=source_catalog,
        lad_metrics=lad_metrics,
        custom_metrics=custom_metrics,
    )

    print("Pipeline completed successfully.")
    for label, path in published_paths.items():
        print(f"- {label}: {path}")
    return published_paths


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_root = Path(args.data_root) if args.data_root else None
    run_pipeline(data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
