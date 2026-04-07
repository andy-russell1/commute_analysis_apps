# EMEA Space & Occupancy Planning Studio

Standalone Savills DX module for EMEA workplace planning, portfolio diagnostics, live scenario modelling, and decision-support output generation.

## Where It Lives

- Module runtime: `apps/emea_space_occupancy/`
- Bundled demo workbook: `apps/emea_space_occupancy/assets/emea_space_occupancy_demo_dataset.xlsx`

## How Data Loads

- The module autoloads the bundled workbook by default.
- Users can replace it with a workbook that follows the same multi-sheet structure.
- Header rows are detected by token search because the demo workbook contains title/description rows above the true headers.

## Expected Workbook Structure

Supported sheets:

- `README`
- `portfolio_hierarchy`
- `property_metrics`
- `space_inventory`
- `people_demand`
- `occupancy_utilisation`
- `standards`
- `scenario_assumptions`
- `scenario_outputs`
- `data_dictionary`

Blocking core modelling sheets:

- `portfolio_hierarchy`
- `property_metrics`
- `space_inventory`
- `people_demand`
- `occupancy_utilisation`
- `standards`
- `scenario_assumptions`

Supported but non-blocking:

- `scenario_outputs`
- `README`
- `data_dictionary`

## Scenario Outputs Treatment

- `scenario_outputs` is treated as seed/example data only.
- Live modelling is always calculated by the module engine.
- If `scenario_outputs` is missing, the app still runs and simply omits seed-scenario comparison features.

## Dynamic Engine Notes

- Scenario assumptions drive forecast, attendance, desk sharing, area, meeting-room, compliance, action, risk, and score outputs.
- Explicit 12m and 24m forecast fields are used where present, with 18m interpolation.
- Workstyle standards are the primary default; site-type standards act as fallback.
- Consolidation scenarios can transfer demand out of small offices into hubs and HQs.

## Run In Savills DX

From the repo root:

```bash
python -m streamlit run app.py
```

Then open `EMEA Space & Occupancy Planning Studio` from the App Hub.

## Hardcoded vs Workbook-Driven

Workbook-driven:

- portfolio hierarchy
- property metrics
- space inventory
- people demand
- occupancy and utilisation
- standards
- scenario assumptions
- seed scenario outputs when present

App-defined demo logic:

- action flag thresholds
- risk rating thresholds
- move-complexity rules
- scenario score weighting
- proportional building/floor allocation for planning views

## Recommended Production Follow-Ups

- Replace proportional building/floor allocation with richer stacking logic.
- Add explicit map/location data if geographic mapping is required.
- Add formal export templating for board-ready PDF output.
- Expand automated tests around scenario-specific transfer and allocation rules.
