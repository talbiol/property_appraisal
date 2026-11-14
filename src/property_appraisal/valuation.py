"""Train the valuation model and flag listings priced below their predicted value."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from xgboost import XGBRegressor

from property_appraisal.config import RANDOM_STATE
from property_appraisal.features import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class Appraisal:
    """A listing whose predicted market value exceeds its asking price."""

    property_id: int
    beds: float
    baths: float
    sqft: float
    year_built: int
    asking_price: float
    predicted_price: float

    @property
    def discount_pct(self) -> float:
        """How far below the predicted value the listing is priced, in percent."""
        return (self.predicted_price / self.asking_price - 1) * 100


def train_model(training_data: pd.DataFrame) -> XGBRegressor:
    """Fit the XGBoost regressor (the best performer in ``research/compare_models.py``)."""
    model = XGBRegressor(random_state=RANDOM_STATE)
    model.fit(training_data[FEATURE_COLUMNS], training_data[TARGET_COLUMN])
    return model


def appraise_zip_code(training_data: pd.DataFrame, listings: pd.DataFrame, zip_code: int) -> list[Appraisal]:
    """Find undervalued listings in ``zip_code``.

    The model is trained without any listing being appraised: a model that has
    already seen a listing's asking price would simply predict it back.
    """
    candidates = listings[listings["Zip_Code"] == zip_code]
    if candidates.empty:
        return []
    held_out = training_data[~training_data["Property_ID"].isin(candidates["Property_ID"])]
    return find_undervalued(train_model(held_out), candidates)


def find_undervalued(model: XGBRegressor, candidates: pd.DataFrame) -> list[Appraisal]:
    """Return the ``candidates`` whose predicted value exceeds their asking price.

    Results are sorted with the largest discount first.
    """
    if candidates.empty:
        return []

    predicted = model.predict(candidates[FEATURE_COLUMNS])
    appraisals = [
        Appraisal(
            property_id=int(row["Property_ID"]),
            beds=row["Beds"],
            baths=row.get("Baths", 0),
            sqft=row["Sqft"],
            year_built=int(row["Year_Built"]),
            asking_price=float(row[TARGET_COLUMN]),
            predicted_price=float(prediction),
        )
        for row, prediction in zip(candidates.to_dict("records"), predicted, strict=True)
        if prediction > row[TARGET_COLUMN] > 0
    ]
    return sorted(appraisals, key=lambda a: a.discount_pct, reverse=True)

