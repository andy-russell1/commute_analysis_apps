from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[3]


def _lens_app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"))
    at.default_timeout = 20
    at.run(timeout=20)
    return at.button[8].click().run(timeout=20)


def _click_by_label(app: AppTest, label: str) -> AppTest:
    for button in app.button:
        if button.label == label:
            return button.click().run(timeout=20)
    raise AssertionError(f"Could not find button with label '{label}'.")


def test_lens_home_and_navigation_pages_render_in_shell():
    lens = _lens_app()
    assert [item.value for item in lens.title] == ["LENS Location Evaluation"]

    weights = _click_by_label(lens, "Open Weights and Scoring")
    assert [item.value for item in weights.title] == ["Weights and Scoring"]
    assert "Macro Weights" in [item.value for item in weights.subheader]

    results = _click_by_label(_lens_app(), "Open Results Dashboard")
    assert [item.value for item in results.title] == ["Results Dashboard"]
    assert "Overall Weighted Scoring" in [item.value for item in results.subheader]

    matrix = _click_by_label(_lens_app(), "Data Matrix")
    assert [item.value for item in matrix.title] == ["Data Matrix"]

    export = _click_by_label(_lens_app(), "Open Export")
    assert [item.value for item in export.title] == ["Export Results"]
    assert "Downloads" in [item.value for item in export.subheader]

    methodology = _click_by_label(_lens_app(), "Open Methodology and Glossary")
    assert [item.value for item in methodology.title] == ["Methodology and Glossary"]
    assert "What This Model Does" in [item.value for item in methodology.subheader]
