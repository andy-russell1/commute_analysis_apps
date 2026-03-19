# Talent Analytics Shared Data Pack

This repository now keeps only the shared, publishable Talent Analytics assets. Private client-specific Lightcast packs are extracted to a separate zip archive before publication and are not committed here.

## Included datasets

### Shared
- `shared/lad_uk_2024.geojson` — original LAD boundary file from the Savills DX repo
- `shared/london_custom_groups.geojson` — dissolved geometry for the 20 custom London geography groupings used in the supplied posting exports
- `shared/geography_group_lookup.csv` — mapping from each custom geography group to its constituent LADs

### Client-specific
- No client-specific pack is checked into the public repository.
- Private client files should be restored locally from the extracted private zip when a secured working copy is needed.

## ONS / Nomis pipeline

The shared demographics layer now uses a live Nomis / ONS preprocessing pipeline rather than manual app-time calculations.

Live sources used for the London shared layer:

- `NM_31_1` (`pestnew`) for total population
- `NM_17_1` (`apsnew`) for working-age population counts
- `NM_17_5` (`apsnew`) for APS rates and shares, using the source numerator and denominator measures

Outputs:

- `shared/geography/custom_geography_lookup.csv` - canonical custom-region-to-LAD lookup
- `shared/ons/metadata/source_catalog.csv` - active public-data metric catalog
- `shared/ons/raw/` - reproducible raw Nomis extracts and metadata pulls
- `shared/ons/processed/demographics_by_lad.csv` - standardised LAD-level demographics metrics
- `shared/ons/processed/demographics_by_custom_geography.csv` - app-facing metrics aggregated to the 20 custom London regions
- `shared/ons/processed/demographics_metadata.csv` - display metadata for the Demographics view

Metrics included:

- total population
- working-age population
- employment rate
- unemployment rate
- economic activity rate
- NVQ4+ share
- no qualifications share
- professional occupations share

Aggregation rules:

- counts are summed across constituent boroughs
- rates and shares use explicit denominator-weighted aggregation in preprocessing rather than unweighted borough averages
- the APS percentage extract keeps all periods for the selected variables and publishes the latest populated source period for each metric

To rerun the pipeline:

```bash
python -m apps.talent_analytics.preprocessing.ons_pipeline
```

The processed outputs join to the app through the existing `custom_geography_key` and `custom_geography_name` contract.
