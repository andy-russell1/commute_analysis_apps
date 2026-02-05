from __future__ import annotations

from io import BytesIO

import pandas as pd

from core.lightcast import build_master_from_files, detect_lightcast_like


def test_lightcast_master_builds():
    df = pd.DataFrame({"Posting Intensity": [1.2, 0.9], "Unique Postings": [10, 5]})
    data = df.to_csv(index=False).encode("utf-8")
    files = [
        (
            "Job_Posting_Table_1_Test_in_Camden_abcd1234.csv",
            data,
        )
    ]
    master = build_master_from_files(files)
    assert not master.empty
    assert "lower district authority" in master.columns
    assert "source_file" in master.columns


def test_lightcast_detects_columns():
    df = pd.DataFrame({"Posting Intensity": [1.2], "Unique Postings": [3]})
    assert detect_lightcast_like(df)
