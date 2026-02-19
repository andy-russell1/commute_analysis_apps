# Savills DX Apps

Self-contained Streamlit wizard with session-only state for multi-user use.

## Run

From the repo root:

```
pip install -r savills_dx_apps/requirements.txt
streamlit run savills_dx_apps/app.py
```

Or from inside `savills_dx_apps/`:

```
pip install -r requirements.txt
python -m streamlit run app.py
```

## Multi-user safety

- Uploads and derived outputs live only in `st.session_state`.
- No user data is written to disk.
- Static assets (if added under `assets/`) are safe to read.

## Eurostat

- A new **Eurostat** app is available from the main app selector.
- This is a flat/no-upload app: select Eurostat and it opens directly.
- It loads these bundled read-only assets:
  - `assets/data/eurostat/europe_talent_distribution.xlsx`
  - `assets/data/eurostat/boundary_lookup_selected.geojson`
- The Eurostat view provides:
  - one tab per workbook sheet,
  - sheet-level filtering,
  - table preview and CSV download,
  - choropleth map view with selectable numeric metric + aggregation.

## Add a new app

1. Create a new plugin module in `savills_dx_apps/apps/` implementing `metadata`, `validate`, `build`, and `render`.
2. Import and register it in `savills_dx_apps/apps/registry.py`.
3. The wizard will auto-list it in Step 1.
