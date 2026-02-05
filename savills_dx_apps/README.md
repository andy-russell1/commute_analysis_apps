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

## Add a new app

1. Create a new plugin module in `savills_dx_apps/apps/` implementing `metadata`, `validate`, `build`, and `render`.
2. Import and register it in `savills_dx_apps/apps/registry.py`.
3. The wizard will auto-list it in Step 1.
