from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _app() -> AppTest:
    at = AppTest.from_file("app.py")
    at.default_timeout = 20
    return at


def test_shell_homepage_renders_sections_and_modules():
    at = _app()
    at.run(timeout=20)

    button_labels = [item.label for item in at.button]
    assert "Upload data" in button_labels
    assert "Open" in button_labels
    module_titles = [item.value for item in at.subheader]
    assert "Commute Impact Assessment" in module_titles
    assert "Amenity Analysis" in module_titles
    assert "LENS Location Evaluation" in module_titles


def test_shell_can_open_commute_amenity_and_lens_modules():
    home = _app().run(timeout=20)

    commute = home.button[0].click().run(timeout=20)
    assert [item.value for item in commute.title] == ["Commute Impact Assessment"]
    assert "Upload a file to continue." in [item.value for item in commute.info]

    amenity = _app().run(timeout=20).button[5].click().run(timeout=20)
    assert "Amenity Analysis" in [item.value for item in amenity.title]
    assert not amenity.exception
    assert not amenity.error

    lens = _app().run(timeout=20).button[6].click().run(timeout=20)
    assert [item.value for item in lens.title] == ["LENS Location Evaluation"]
    lens_subheaders = [item.value for item in lens.subheader]
    assert "Top Recommendations" in lens_subheaders
    assert "Next Steps" in lens_subheaders


def test_shell_restart_returns_to_homepage():
    lens = _app().run(timeout=20).button[6].click().run(timeout=20)
    restarted = lens.button[-1].click().run(timeout=20)
    module_titles = [item.value for item in restarted.subheader]
    assert "Commute Impact Assessment" in module_titles
    assert "LENS Location Evaluation" in module_titles
