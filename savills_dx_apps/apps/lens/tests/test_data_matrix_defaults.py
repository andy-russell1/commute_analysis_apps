from __future__ import annotations

from apps.lens.core import model


def test_matrix_default_is_ranks_when_available():
    assert model.resolve_matrix_view_preference(None, has_rank_data=True) == "Computed Ranks"


def test_matrix_falls_back_to_scores_when_rank_unavailable():
    assert model.resolve_matrix_view_preference("Computed Ranks", has_rank_data=False) == "Score Index (0-100)"


def test_matrix_keeps_valid_non_rank_selection():
    assert model.resolve_matrix_view_preference("Raw (units vary)", has_rank_data=False) == "Raw (units vary)"


