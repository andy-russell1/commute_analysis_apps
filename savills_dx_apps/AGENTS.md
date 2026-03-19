# AGENTS.md

## Purpose
This repository contains Savills DX modular analytics applications built around a shared shell/runtime, with app-specific logic living inside isolated modules.

## Working Principles
- Keep the shell thin.
- Keep business logic app-local by default.
- Avoid cross-app coupling.
- Reuse existing runtime contracts and patterns before introducing new abstractions.
- Preserve behaviour outside the requested scope.
- Use UK English in UI text, documentation, and user-facing labels.
- Prioritise presentation-ready output and consistent UX.

## Repository Structure Expectations
- `app.py` and shell/routing layers should remain minimal and orchestration-focused.
- `apps/` contains app-specific modules and should own app-specific logic.
- `shared/` is only for genuinely reusable cross-app utilities, runtime contracts, styling helpers, or generic services.
- Do not move app-specific logic into `shared/` just to “tidy up” unless it is clearly reusable across multiple modules.
- New modules should follow the existing module registration and metadata patterns already present in the repo.

## Module Design Rules
- Prefer app-local transformations, configs, schemas, and render logic unless reused elsewhere.
- Preserve current module contracts (`wizard`, `standalone`, `workspace`) unless a change is explicitly requested.
- Avoid broad refactors when the request is scoped to one app or one workbook/tab.
- If a request mentions one workbook sheet/tab only, do not broaden the change to other sheets/tabs unless explicitly asked.

## UI / Styling Rules
- Maintain a polished, executive-ready, presentation-friendly feel.
- Preserve existing layout stability; avoid changes that cause unnecessary spacing shifts, sidebar clutter, or header movement.
- Reuse established theme/styling patterns where possible.
- Prefer refined, clean, business-facing outputs over developer-centric defaults.
- If applying visual updates, keep them consistent across the affected app and avoid unrelated visual churn elsewhere.

## Data Handling Rules
- Be explicit about schema assumptions, column mappings, workbook tabs, and upload requirements.
- Keep file-specific processing close to the app/module that owns it.
- Document assumptions clearly when handling uploaded Excel/CSV content.
- Do not silently broaden transformations beyond the requested dataset/sheet/scope.

## Safe Edit Boundaries
- Only edit files required for the requested task.
- Do not refactor unrelated modules or sibling apps.
- Do not rename files/folders unless necessary.
- Do not change shared contracts/interfaces unless required and clearly justified.
- If a requested change could affect multiple apps, explain the risk before implementing.

## Required Workflow
Before coding:
1. Inspect the relevant repo area and explain the current flow.
2. Identify the minimum touch points.
3. Produce a short implementation plan.
4. State risks and validation steps.

During implementation:
- Keep edits minimal and localised.
- Reuse existing patterns before adding new abstractions.
- Flag assumptions explicitly.
- Avoid speculative cleanup outside scope.

After implementation:
- List every changed file.
- Explain why each file changed.
- State exactly what was validated.
- State what remains unverified or needs manual checking.

## Validation Expectations
Where relevant, run the most appropriate checks available, such as:
- imports / static validation
- targeted module smoke tests
- app startup checks
- traceback inspection and fix verification
- workbook/tab-specific checks when file processing is involved

Do not claim a task is complete without stating what was actually validated.

## Output Format
Return results in this structure:
1. Current flow summary
2. Implementation plan
3. Files changed
4. What changed and why
5. Validation performed
6. Risks / follow-ups

## Environment Rules
- This repository uses Conda as the standard environment manager.
- The default development environment is `dxapps`.
- Do not create `.venv`, `venv`, or other local virtual environment folders unless explicitly requested.
- Do not add or commit environment directories to the repository.
- When running install, app, or test commands, assume the documented Conda environment is the correct runtime.
- If dependencies need to change, update `requirements.txt` and any relevant setup documentation rather than creating a separate environment.
- Keep setup instructions aligned with `README.md`; do not invent alternative environment workflows unless explicitly asked.