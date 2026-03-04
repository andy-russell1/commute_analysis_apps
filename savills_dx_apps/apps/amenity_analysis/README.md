# Amenity Analysis (MVP)

Standalone Streamlit app for Amenity KPI scoring.

## Run

From `savills_dx_apps/`:

```bash
python -m streamlit run apps/amenity_analysis/app.py
```

## Required upload headers

- `officeID`
- `address`
- `office - Latitude`
- `office - Longitude`

Supported uploads: CSV and XLSX (sheet picker shown for Excel).

## Workflow

1. Setup: upload and validate offices.
2. Controls: choose metrics (including Public transport) and set weights.
   Radius is a single select configured in Setup.
3. Overview:
   - `Map & Insights` tab (density map + single-office trade-off panel)
   - `Comparison` tab (A/B comparison visuals)
   - `Controls` tab (quick access back to full controls + scoring view)
4. Location Drilldown: office-level density map for a selected office.

Scoring details are available in the Controls page under the `Scoring system` view.

## NaPTAN setup

Place a local NaPTAN file under `data/reference/naptan/`.

Supported formats: CSV/TXT, XLS/XLSX, Parquet, Feather.

If missing, Public transport metric cannot be computed and will show as unavailable.

## KPI

- Amenity metrics: POI count within radius (higher is better)
- Public transport metric: nearest stop distance (lower is better)
- Metric scores are min-max normalised across offices
- If max == min, normalised score is 0.5
- KPI = weighted sum of normalised metrics * 100
- Weights are auto-normalised to 100%

## Map Modes

In Overview > `Map & Insights`:

- Density map with `Hex` or `Heatmap`.
- Office markers remain visible and radius rings are shown.

## Insights

- Trade-off visual (single office): selected office vs best office vs portfolio average by category.
- Comparison tab (office A vs office B):
  - overall score delta,
  - category delta chart,
  - counts table by category,
  - top unique advantages by amenity.

## Data sources

- OSM/Overpass API for amenities
- Local NaPTAN for transport distance

No paid APIs and no TfL integration in MVP.

## Caching

Overpass requests are disk-cached under `.cache/amenity_analysis`, keyed by office, radius, selected amenity buckets, and tag-map version.

## How To Test

1. Run app:
   - `python -m streamlit run apps/amenity_analysis/app.py`
2. Upload an XLSX, select worksheet, validate, and continue.
3. In Setup select a radius (default 1000m). In Controls run analysis.
4. In Overview > Map & Insights:
   - switch `Hex` and `Heatmap`.
5. In Overview > Comparison:
   - select Office A and B and verify deltas + unique advantages.
6. Re-run same settings and confirm cache message shows reused queries.

Optional smoke check:

- `python apps/amenity_analysis/smoke_test.py`

## Known limitations

- OSM coverage varies by place/category.
- Distances are straight-line proxies.
- Amenity counts reflect availability, not quality.
- Public Overpass rate limits may apply for large runs.
