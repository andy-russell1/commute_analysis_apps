from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def _app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"))
    at.default_timeout = 60
    return at


def _module() -> AppTest:
    return _app().run(timeout=60).button[7].click().run(timeout=60)


def _click_by_label(app: AppTest, label: str) -> AppTest:
    for button in app.button:
        if button.label == label or button.label.endswith(label):
            return button.click().run(timeout=60)
    raise AssertionError(f"Could not find button with label '{label}'.")


def test_emea_module_navigation_and_decision_pack_controls_render():
    module = _module()
    assert [item.value for item in module.title] == ["Home"]
    assert "EMEA Space & Occupancy Planning Studio" in [item.value for item in module.subheader]

    comparison = _click_by_label(module, "Scenario Comparison")
    assert [item.value for item in comparison.title] == ["Scenario Comparison"]
    assert "Choose 2 or more scenarios" in [item.label for item in comparison.multiselect]

    decision_pack = _click_by_label(module, "Decision Pack")
    assert [item.value for item in decision_pack.title] == ["Decision Pack"]
    selectbox_labels = [item.label for item in decision_pack.selectbox]
    assert "Preferred scenario pin" in selectbox_labels
