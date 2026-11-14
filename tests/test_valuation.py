import numpy as np
import pandas as pd
import pytest

from property_appraisal import valuation
from property_appraisal.features import FEATURE_COLUMNS
from property_appraisal.valuation import Appraisal, appraise_zip_code, find_undervalued


class FixedModel:
    """Stand-in for the regressor that predicts a fixed value per row."""

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, X):
        assert list(X.columns) == FEATURE_COLUMNS
        return self.predictions[: len(X)]


def _listings(rows):
    df = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        df[column] = 0
    return df


def test_find_undervalued_returns_only_discounted_listings_sorted():
    candidates = _listings(
        [
            {"Property_ID": 1, "Price": 100.0},
            {"Property_ID": 2, "Price": 100.0},
            {"Property_ID": 3, "Price": 100.0},
        ]
    )

    results = find_undervalued(FixedModel([110, 90, 150]), candidates)

    assert [r.property_id for r in results] == [3, 1]
    assert results[1].discount_pct == pytest.approx(10.0)


def test_appraise_zip_code_never_trains_on_the_listings_it_appraises(monkeypatch):
    training = _listings(
        [
            {"Property_ID": 1, "Zip_Code": 90001, "Price": 100.0},
            {"Property_ID": 2, "Zip_Code": 90001, "Price": 100.0},
            {"Property_ID": 3, "Zip_Code": 90002, "Price": 100.0},
        ]
    )
    listings = _listings(
        [
            {"Property_ID": 1, "Zip_Code": 90001, "Price": 100.0},
            {"Property_ID": 4, "Zip_Code": 90001, "Price": 100.0},
            {"Property_ID": 2, "Zip_Code": 90002, "Price": 100.0},
        ]
    )
    trained_on = []

    def fake_train(data):
        trained_on.extend(data["Property_ID"])
        return FixedModel([150, 150])

    monkeypatch.setattr(valuation, "train_model", fake_train)
    results = appraise_zip_code(training, listings, 90001)

    # Listing 1 is being appraised, so it is held out; listing 2 is in another zip code.
    assert sorted(trained_on) == [2, 3]
    assert len(results) == 2


def test_appraise_zip_code_unknown_zip_code():
    listings = _listings([{"Property_ID": 1, "Zip_Code": 90001, "Price": 1.0}])
    assert appraise_zip_code(listings, listings, 99999) == []


def test_discount_pct():
    appraisal = Appraisal(1, 3, 2, 1500, 1950, asking_price=200, predicted_price=250)
    assert appraisal.discount_pct == pytest.approx(25.0)
