from pathlib import Path

import pytest


@pytest.fixture
def stats_dir(tmp_path: Path) -> Path:
    """A minimal dataset directory with one zip code listed twice in the statistics."""
    (tmp_path / "properties.csv").write_text(
        "Property_ID,Zip_Code,Price\n"
        "1,90001,100\n"
        "2,90001,200\n"
        "3,90002,300\n"
    )
    (tmp_path / "crime.csv").write_text(
        "Zip_Code,Theft\n90001,1.00\n90001,2.00\n90002,3.00\n"
    )
    (tmp_path / "income.csv").write_text(
        "Zip_Code,Median_Household_Income\n90001,10.00\n90002,\n"
    )
    return tmp_path
