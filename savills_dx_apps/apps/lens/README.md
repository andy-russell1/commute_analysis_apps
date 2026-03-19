# LENS Location Evaluation

Client-ready Streamlit app for hierarchical location scoring with two UX modes:

- `Client`: decision-first outputs, lightweight controls
- `Advanced`: full model controls and audit views

## Run

```bash
pip install -r requirements.txt
python scripts/create_template_with_help.py
streamlit run app/app.py
```

## Core Capabilities

- Excel ingestion (`Criteria Sheet`, `Data Sheet`)
- Rank-based or percentile-based micro scoring
- Hierarchical weighted rollups (micro -> major -> macro -> overall)
- Client-friendly dashboard with city drilldown
- Advanced model editing (hierarchy weights + direction overrides)
- Data matrix tabs (`Computed Ranks`, `Score Index (0-100)`, `Raw (units vary)`)
- CSV and Excel export
- Methodology & Glossary page

## Runtime Ownership

- Canonical DX runtime lives entirely in `apps/lens/`.
- `module.py` exposes the shell contract and calls a local runtime entrypoint only.
- `runtime.py` routes embedded DX navigation to local callable views; it does not dispatch by dynamic page-module import.
- Numbered `pages/` files remain for standalone Streamlit page compatibility and are not the shell runtime mechanism.

## Template Workbook

- Canonical workbook: `LENS.xlsx`
- Includes `How_To_Use` sheet for data-entry guidance
- Parser remains compatible by reading only:
  - `Criteria Sheet`
  - `Data Sheet`

## Page Map

- `Home`: top recommendations + short narrative
- `Weights and Scoring`: presets/scenarios (Client) and full controls (Advanced)
- `Results Dashboard`: charts + city drilldown, audit tables gated to Advanced
- `Data Matrix`: rank-first matrix views
- `Export`: download-first UX with optional advanced preview
- `Methodology & Glossary`: plain-English model explanation

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q -p no:cacheprovider
```
