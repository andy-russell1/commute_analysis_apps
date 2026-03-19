from __future__ import annotations

import numpy as np
import pandas as pd

from apps.lens.core import io


def test_parse_data_sheet_keeps_uploaded_rank_rows_as_reference_only():
    data_raw = pd.DataFrame(
        [
            ["metadata", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ["Macro", "Major", "Micro", "Source", "CityA", "CityB", "CityC"],
            ["Talent", "Demographics", "Population", "count; Higher is better", 100, 200, 300],
            [np.nan, np.nan, np.nan, "rank reference", 3, 2, 1],
        ]
    )

    raw_rows, rank_reference, city_columns = io.parse_data_sheet(data_raw)

    assert city_columns == ["CityA", "CityB", "CityC"]
    assert raw_rows.shape[0] == 1
    assert rank_reference.shape[0] == 1
    assert raw_rows.iloc[0]["source"] == "count; Higher is better"
    assert rank_reference.iloc[0]["source"] == "rank reference"
    assert raw_rows.iloc[0]["CityA"] == 100
    assert rank_reference.iloc[0]["CityA"] == 3
