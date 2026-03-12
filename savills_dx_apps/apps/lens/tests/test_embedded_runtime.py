from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _lens_app() -> AppTest:
    at = AppTest.from_file("app.py")
    at.default_timeout = 20
    at.run(timeout=20)
    return at.button[6].click().run(timeout=20)


def test_lens_home_and_navigation_pages_render_in_shell():
    lens = _lens_app()
    assert [item.value for item in lens.title] == ["LENS Location Evaluation"]

    weights = lens.button[1].click().run(timeout=20)
    assert [item.value for item in weights.title] == ["Weights and Scoring"]
    assert "Macro Weights" in [item.value for item in weights.subheader]

    results = _lens_app().button[0].click().run(timeout=20)
    assert [item.value for item in results.title] == ["Results Dashboard"]
    assert "Overall Weighted Scoring" in [item.value for item in results.subheader]

    matrix = _lens_app().button[9].click().run(timeout=20)
    assert [item.value for item in matrix.title] == ["Data Matrix"]

    export = _lens_app().button[2].click().run(timeout=20)
    assert [item.value for item in export.title] == ["Export Results"]
    assert "Downloads" in [item.value for item in export.subheader]

    methodology = _lens_app().button[3].click().run(timeout=20)
    assert [item.value for item in methodology.title] == ["Methodology and Glossary"]
    assert "What This Model Does" in [item.value for item in methodology.subheader]
