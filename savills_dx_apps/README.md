# Savills DX Apps

Savills DX Apps is a Streamlit-based Savills DX application shell for running multiple analytical modules in one environment.

## What This Repository Is

- A thin DX shell (`app.py` + `shell/`) that owns page setup, routing, shared layout/theme, and module dispatch.
- A module host (`apps/`) for isolated business apps.
- A shared foundation (`shared/`) for reusable cross-module standards, runtime contracts, and generic utilities only.

## Supported Module Types

The shell supports exactly three module types:

- `wizard`: upload/preprocess/render flow (`validate`, `build`, `render`).
- `standalone`: self-contained app runtime (`run`).
- `workspace`: grouped module families (`run`) for future multi-tool workspaces.

## Architecture Principles

- Keep the shell thin.
- Keep app logic app-local by default.
- Avoid cross-app coupling.
- Register modules through metadata and dispatch through the shell router.
- Do not add app-specific orchestration to root `app.py`.

## Design System Rule

Use 'LENS' styling for shell chrome, cards, headers, controls, and navigation, while preserving semantic analytics colours where colour encodes meaning:

- Keep red/amber/green semantics intact.
- Keep heatmaps/gradients/diverging analytic scales intact.
- Do not recolour analytical encodings to match product chrome.

## Shared Abstraction Rule

Do not move code into shared layers just because it might be useful.

A helper belongs in `shared/` only when one or more are true:

- It is already used by two or more modules.
- It is a deliberate Savills DX standard.
- It enforces shell-level consistency (theme, layout, navigation, branding).
- It is genuinely generic and not app-specific.

If unsure, keep it app-local first.

## Shared Abstractions Added In This Refactor

### `shared/ui/cards.py`
- What: shared DX module card renderer used by the home hub.
- Why shared: standardises shell card UX across all module types.
- Where: `shared/ui/cards.py`.
- Intended consumers: shell home and future workspace launchers.
- Constraints: card content should remain metadata-driven; do not add app-specific logic.

### `shell/theme.py`
- What: central shell theme/chrome layer.
- Why shared: enforces consistent DX product styling.
- Where: `shell/theme.py`.
- Intended consumers: root shell entrypoint only.
- Constraints: never override semantic analytical colouring in app plots/status visuals.

## Recommended Structure

```text
savills_dx_apps/
  app.py
  README.md
  requirements.txt

  shell/
    home.py
    router.py
    state.py
    layout.py
    theme.py
    branding.py
    registry.py
    registry_models.py

  shared/
    ui/
    io/
    geo/
    validation/
    downloads/
    charts/

  apps/
    commute/
      module.py
      wizard.py
    isochrone/
      module.py
      wizard.py
    isochrone_commute/
      module.py
      wizard.py
    lightcast/
      module.py
      wizard.py
    eurostat/
      module.py
      wizard.py
    amenity_analysis/
    lens/

  assets/
  tests/
```

## Current Module Registration

- `wizard`: Commute, Isochrone, Isochrone + Commute, Lightcast, Eurostat.
- `standalone`: Amenity Analysis, LENS Location Evaluation.
- `workspace`: placeholder support enabled in shell model and registry.

## Package Map

- `shell/`: shell-only routing, layout, branding, state, and module registry.
- `shared/`: generic DX runtime contracts, shared data/io/geo helpers, and shell-facing UI primitives.
- `apps/<module>/`: module-owned runtime and business logic.
- Legacy `core/` and `dx_core/` layers have been removed; imports should target `shared/` or `apps/<module>/` directly.

## Compatibility Shims

- Canonical wizard source lives in `apps/<module>/wizard.py` with DX-facing metadata in `apps/<module>/module.py`.
- Generic cross-module imports should target `shared/*`, and app-owned imports should target `apps.<module>.*`.
- Top-level `apps/*.py` wizard shim files have been removed after validation; package consumers should import from `apps.<module>.wizard` or `apps.<module>.module`.

## LENS Integration Notes

- Canonical DX LENS source of truth is now `apps/lens/`.
- LENS runs as a native `standalone` module through shell registry/router.
- LENS appears in its own homepage section (`LENS`) and is not grouped under Location Analytics.
- Embedded navigation is handled inside `apps/lens/` through a native callable router owned by the LENS package.

## Dependency Reconciliation

Main environment requirements now include the runtime needed by both DX and LENS:

- Added: `scipy>=1.11`.
- Updated floors to align with LENS runtime:
  - `pandas>=2.1`
  - `numpy>=1.26`

## Run

From this directory:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Validation from the main DX environment:

```bash
pip install -r requirements.txt
python -m pytest -q -p no:cacheprovider
```

## Pre-Build Prompt (For New DX-Compatible Apps)

Use this prompt before building a new Streamlit app outside DX:

```text
Build a new Streamlit analytical app that is integration-ready for Savills DX from day one.

Requirements:
1. Classify the app as exactly one module type: wizard, standalone, or workspace.
2. Use an integration-ready folder structure with a thin shell-facing entrypoint and app-local business logic separated from UI rendering.
3. Keep all domain logic app-local unless it is already proven reusable across multiple apps.
4. Avoid all cross-app imports/coupling.
5. Include a short README integration note describing module type and expected DX registration metadata.
6. Add tests for core logic and validation behavior.

Output must include:
- Final folder tree.
- Chosen module type and rationale.
- Which files remain app-local.
- Candidate shared abstractions (if any) with justification.
- DX integration notes (router/registry impact, dependencies).
```

## Integration Prompt (For Importing Existing Apps Into DX)

Use this prompt when integrating an existing Streamlit app into `savills_dx_apps`:

```text
Integrate this existing Streamlit app into Savills DX.

Requirements:
1. Classify the app as wizard, standalone, or workspace.
2. Map the app into the DX folder structure under apps/ while preserving functionality.
3. Identify what remains app-local vs what should be shared.
4. Keep root shell thin; do not add app-specific orchestration to app.py.
5. Reconcile dependencies with requirements.txt and call out conflicts.
6. Document any new shared abstractions (what, why, where, consumers, constraints).
7. Remove temporary migration scaffolding and duplicate source trees where safe.

Final output must list:
- Integration plan and completed changes.
- Registry/router updates.
- Dependency changes.
- Remaining risks.
- Follow-up cleanup items.
```

## Module Review Checklist

Use this checklist for every new/updated module:

- Architecture: module type is correct and shell remains thin.
- Structure: app code is isolated and registry metadata is complete.
- Shared discipline: only proven, generic abstractions moved to `shared/`.
- UX consistency: shell chrome and navigation are consistent with DX style rules.
- Maintainability: validation/build/render boundaries are clear and testable.
- Integration readiness: dependencies, routing, and homepage section placement are documented.
