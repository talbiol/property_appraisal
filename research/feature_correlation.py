"""Explore how each listing attribute relates to price.

For every attribute, prints its Pearson correlation with price (ignoring zero /
missing values) and shows a price-vs-attribute scatter plot. Price and lot size
outliers are removed first.

Run with ``python research/feature_correlation.py`` (needs the ``research`` extra).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from property_appraisal.config import TRAINING_DATA_DIR
from property_appraisal.features import remove_outliers

ATTRIBUTES = [
    "Beds",
    "Baths",
    "Sqft",
    "Lot_Sqft",
    "Type",
    "Year_Built",
    "Cooling",
    "Heating",
    "Garage",
    "Stories",
    "Pool",
    "Noise",
    "Number_of_Schools",
    "Avg_Distance_Miles_School",
]

# Ordinal encoding used to correlate property type with price.
PROPERTY_TYPE_CODES = {
    "duplex_triplex": 1,
    "multi_family": 2,
    "single_family": 3,
    "land": 4,
    "coop": 5,
    "townhomes": 6,
    "condos": 7,
}


def main() -> None:
    listings = pd.read_csv(TRAINING_DATA_DIR / "properties.csv").fillna(0)
    listings = remove_outliers(listings, "Price")
    listings = remove_outliers(listings, "Lot_Sqft")

    for attribute in ATTRIBUTES:
        plot_price_against(listings, attribute)

    print("Property types:", ", ".join(listings["Type"].unique()))
    type_codes = listings["Type"].map(PROPERTY_TYPE_CODES)
    print(f"{'Type':<26} Pearson r = {listings['Price'].corr(type_codes):.4f}")


def plot_price_against(listings: pd.DataFrame, attribute: str) -> None:
    pair = listings[["Price", attribute]].copy()
    try:
        pair[attribute] = pair[attribute].astype(float)
    except ValueError:
        pair[attribute] = pair[attribute].astype(str)
    else:
        pair = pair[pair[attribute] != 0]
        print(f"{attribute:<26} Pearson r = {pair['Price'].corr(pair[attribute]):.4f}")

    pair = pair.sort_values(by=attribute)
    plt.scatter(pair[attribute], pair["Price"])
    plt.ticklabel_format(style="plain", axis="y")
    plt.xticks(rotation=45)
    plt.xlabel(attribute)
    plt.ylabel("Price")
    plt.title(f"Price vs {attribute}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
