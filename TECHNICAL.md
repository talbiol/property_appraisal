# Technical Overview

This document describes how Property Appraisal works: the data, the features, the model, how listings are flagged, and how the model was chosen.

## Pipeline

```
 ┌──────────────────┐   ┌───────────────────┐   ┌──────────────────┐
 │  data/           │──►│  features.py      │──►│  valuation.py    │──► undervalued listings
 │ listings, crime, │   │ join listings     │   │ XGBoost predicts │
 │ income           │   │ with zip stats    │   │ value vs. price  │
 └──────────────────┘   └───────────────────┘   └──────────────────┘
```

| Module | Responsibility |
|---|---|
| `config.py` | Data paths and the random seed. Set `PROPERTY_APPRAISAL_DATA_DIR` to use another data folder. |
| `features.py` | Loads a dataset folder, joins each listing with its zip code's statistics, and defines the feature columns. |
| `valuation.py` | Trains the model and returns listings whose predicted value exceeds the asking price. |
| `web/` | Flask app factory, routes, template and stylesheet. |

## Data

`data/training/` holds the listings used to fit the model. `data/inference/` holds the listings the web app appraises. Each folder contains three CSV files, joined on `Zip_Code`.

| File | One row per | Columns |
|---|---|---|
| `properties.csv` | listing | `Property_ID`, `Zip_Code`, `Status`, `Price`, `Beds`, `Baths`, `Sqft`, `Lot_Sqft`, `Type`, `Year_Built`, `Cooling`, `Heating`, `Garage`, `Stories`, `Pool`, `Noise`, `Number_of_Schools`, `Avg_Distance_Miles_School`, `Distances_in_Miles_School` |
| `crime.csv` | zip code | Rates per 1,000 residents for 13 crime types (assault, robbery, theft, burglary, …) |
| `income.csv` | zip code | Median, average and per-capita household income, share of high-income households, and median income by householder age |

| Folder | Listings | Unique properties | Zip codes |
|---|---|---|---|
| `training/` | 806 | 663 | 90001–90004, 90006, 90008, 90010, 90013–90015, 90020, 90024 |
| `inference/` | 253 | 251 | 90001, 90002, 90003, 90005 |

Some properties appear more than once because they were recorded on more than one date. `Property_ID` is an anonymised identifier that is consistent across both folders: 101 properties appear in both.

## Features

The model uses 30 numeric features (`FEATURE_COLUMNS` in `features.py`):

- **Property (9):** beds, square footage, lot size, year built, garage spaces, stories, noise score, number of nearby schools, average distance to those schools.
- **Crime (13):** zip-code crime rates per crime type.
- **Income (8):** zip-code household income statistics.

Missing numeric values are filled with 0. If a zip code has more than one row of statistics, the last row is used.

The target is the asking price (`Price`).

## Valuation

For a selected zip code, `appraise_zip_code`:

1. Takes the inference listings in that zip code as candidates.
2. Removes those same properties (by `Property_ID`) from the training data. A model that has already seen a listing's price would just predict it back.
3. Trains an `XGBRegressor` with default hyperparameters (`random_state=42`).
4. Predicts a value for each candidate and keeps those where `predicted > asking price > 0`.
5. Sorts them by discount, where `discount = (predicted / asking − 1) × 100`.

The web app caches the result for each zip code, because the bundled data does not change at runtime.

## Model selection

`research/compare_models.py` compares six regressors on a random 80/20 split of the training data (`random_state=42`):

| Model | R² | RMSE ($) |
|---|---|---|
| **XGBoost** | **0.929** | **901,762** |
| Random Forest | 0.876 | 1,189,484 |
| Gradient Boosting | 0.768 | 1,629,268 |
| K-Nearest Neighbors | 0.324 | 2,779,989 |
| Linear Regression | 0.186 | 3,050,793 |
| Neural Network (3-layer MLP, Keras) | runs only if Keras is installed | |

XGBoost gives the most accurate valuations, so the app uses it. The charts in `project_screenshots/compare_models/` show predicted vs. actual prices for each model.

Because some properties appear more than once, a random row-level split can put the same property in both the training and the test set. The scores above are therefore likely optimistic. A split grouped by `Property_ID` would give a stricter estimate.

## Research scripts

Install the extra dependencies with `pip install -e ".[research]"`.

| Script | What it does |
|---|---|
| `research/compare_models.py` | Trains and scores the regressors above and plots their predictions and metrics. |
| `research/feature_correlation.py` | Prints the Pearson correlation between price and each attribute, and plots price against each one. |
| `research/neighborhood_clusters.py` | Standardises zip-code crime and income statistics, projects them with PCA and groups them into three k-means clusters. |

## Testing and linting

```bash
pytest          # unit tests for features and valuation, plus web route tests
ruff check .    # lint (configured in pyproject.toml)
```

## Limitations

- Predictions only use the features in the data. They do not account for property condition, renovations or interior quality.
- The data is a 2024 snapshot of a small part of Los Angeles, so predictions do not reflect current prices or other areas.
- The model predicts asking prices, not sale prices.
