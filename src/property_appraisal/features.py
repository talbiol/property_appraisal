"""Load listing data and join it with zip-code level crime and income statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "Price"

PROPERTY_FEATURES = [
    "Beds",
    "Sqft",
    "Lot_Sqft",
    "Year_Built",
    "Garage",
    "Stories",
    "Noise",
    "Number_of_Schools",
    "Avg_Distance_Miles_School",
]
CRIME_FEATURES = [
    "Assault",
    "Robbery",
    "Rape",
    "Murder",
    "Theft",
    "Vehicle_Theft",
    "Burglary",
    "Arson",
    "Kidnapping",
    "Drug_Crimes",
    "Vandalism",
    "Identity_Theft",
    "Animal_Cruelty",
]
INCOME_FEATURES = [
    "Median_Household_Income",
    "Average_Household_Income",
    "Per-Capita_Income",
    "High_Income_Households",
    "Median_Income_Per_Householder_Age_under_25_years",
    "Median_Income_Per_Householder_Age_25_to_44_years",
    "Median_Income_Per_Householder_Age_45_to_64_years",
    "Median_Income_Per_Householder_Age_65_years_and_over",
]
FEATURE_COLUMNS = PROPERTY_FEATURES + CRIME_FEATURES + INCOME_FEATURES


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Return one row per listing in ``data_dir`` with its zip code's crime and income stats.

    ``data_dir`` must contain ``properties.csv``, ``crime.csv`` and ``income.csv``.
    If a zip code has more than one row of statistics, the last one is used, so every
    listing appears exactly once. Missing numeric values are filled with 0.
    """
    properties = pd.read_csv(data_dir / "properties.csv")
    crime = _latest_per_zip(pd.read_csv(data_dir / "crime.csv"))
    income = _latest_per_zip(pd.read_csv(data_dir / "income.csv"))

    dataset = properties.merge(crime, on="Zip_Code", how="left").merge(income, on="Zip_Code", how="left")
    numeric = dataset.select_dtypes("number").columns
    dataset[numeric] = dataset[numeric].fillna(0)
    return dataset


def listed_zip_codes(data_dir: Path) -> set[int]:
    """Zip codes that already have listings in ``data_dir`` (empty if there is no listings file)."""
    path = data_dir / "properties.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    return set(pd.read_csv(path, usecols=["Zip_Code"])["Zip_Code"].astype(int))


def remove_outliers(df: pd.DataFrame, column: str, quantile: float = 0.95) -> pd.DataFrame:
    """Drop rows whose absolute z-score in ``column`` is above the given quantile."""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores <= z_scores.quantile(quantile)]


def _latest_per_zip(stats: pd.DataFrame) -> pd.DataFrame:
    return stats.drop_duplicates(subset="Zip_Code", keep="last")
