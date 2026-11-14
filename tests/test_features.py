import pandas as pd

from property_appraisal.config import INFERENCE_DATA_DIR, TRAINING_DATA_DIR
from property_appraisal.features import (
    FEATURE_COLUMNS,
    listed_zip_codes,
    load_dataset,
    remove_outliers,
)


def test_load_dataset_keeps_one_row_per_listing(stats_dir):
    dataset = load_dataset(stats_dir)

    assert dataset["Property_ID"].tolist() == [1, 2, 3]
    # The last statistics row for a zip code wins.
    assert dataset.loc[dataset["Zip_Code"] == 90001, "Theft"].tolist() == [2.0, 2.0]
    # Missing statistics are filled with 0.
    assert dataset.loc[dataset["Zip_Code"] == 90002, "Median_Household_Income"].item() == 0


def test_listed_zip_codes(stats_dir, tmp_path_factory):
    assert listed_zip_codes(stats_dir) == {90001, 90002}
    assert listed_zip_codes(tmp_path_factory.mktemp("empty")) == set()


def test_remove_outliers_drops_extreme_values():
    df = pd.DataFrame({"Price": [100] * 19 + [10_000]})
    assert remove_outliers(df, "Price")["Price"].max() == 100


def test_bundled_datasets_have_every_feature():
    for data_dir in (TRAINING_DATA_DIR, INFERENCE_DATA_DIR):
        dataset = load_dataset(data_dir)
        assert set(FEATURE_COLUMNS) <= set(dataset.columns)
        assert len(dataset) == len(pd.read_csv(data_dir / "properties.csv"))
