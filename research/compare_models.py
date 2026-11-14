"""Compare regression models for predicting listing prices.

Trains linear regression, random forest, k-nearest neighbours, XGBoost, gradient
boosting and a small neural network on an 80/20 split of the training data, plots
predicted vs. actual prices for each, then plots R², MSE and RMSE side by side.
XGBoost performed best and is the model used by the web app.

Run with ``python research/compare_models.py`` (needs the ``research`` extra).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from property_appraisal.config import RANDOM_STATE, TRAINING_DATA_DIR
from property_appraisal.features import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, remove_outliers

PLOT_PRICE_LIMIT = 10_000_000


@dataclass
class Scores:
    model: str
    r2: float
    mse: float

    @property
    def rmse(self) -> float:
        return self.mse**0.5


def main() -> None:
    dataset = load_dataset(TRAINING_DATA_DIR)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset[FEATURE_COLUMNS], dataset[TARGET_COLUMN], test_size=0.2, random_state=RANDOM_STATE
    )

    estimators = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    results = []
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        results.append(evaluate(name, y_test, estimator.predict(X_test)))

    nn_scores = evaluate_neural_network(X_train, X_test, y_train, y_test)
    if nn_scores is not None:
        results.append(nn_scores)

    plot_metric_comparison(results)


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray, drop_outliers: bool = False) -> Scores:
    """Score predictions, print the metrics and plot predicted vs. actual prices."""
    mse = mean_squared_error(y_true, y_pred)
    scores = Scores(name, r2_score(y_true, y_pred), mse)
    print(f"{name:<20} RMSE: {scores.rmse:,.2f}  MSE: {scores.mse:,.2f}  R²: {scores.r2:.4f}")

    comparison = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    if drop_outliers:
        comparison = remove_outliers(comparison, "Actual")
    plot_predictions(name, comparison)
    return scores


def evaluate_neural_network(X_train, X_test, y_train, y_test) -> Scores | None:
    """Train a 3-layer MLP on standardised features; skipped if Keras is not installed."""
    try:
        from keras import Input, Sequential
        from keras.layers import Dense, Dropout
        from keras.optimizers import Adam
        from keras.regularizers import l2
    except ImportError:
        print("Keras is not installed; skipping the neural network.")
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

    y_pred = model.predict(X_test_scaled).flatten()
    return evaluate("Neural Network", y_test, y_pred, drop_outliers=True)


def plot_predictions(name: str, comparison: pd.DataFrame) -> None:
    plt.scatter(comparison["Actual"], comparison["Predicted"])
    plt.plot([0, PLOT_PRICE_LIMIT], [0, PLOT_PRICE_LIMIT], color="red")
    plt.xlim(0, PLOT_PRICE_LIMIT)
    plt.ylim(0, PLOT_PRICE_LIMIT)
    plt.ticklabel_format(style="plain")
    plt.xticks(rotation=45)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs. Predicted Prices - {name}")
    plt.tight_layout()
    plt.show()


def plot_metric_comparison(results: list[Scores]) -> None:
    names = [r.model for r in results]
    metrics = {
        "Coefficient of Determination": [r.r2 for r in results],
        "Mean Squared Error": [r.mse for r in results],
        "Root Mean Squared Error": [r.rmse for r in results],
    }
    for title, values in metrics.items():
        plt.bar(names, values)
        plt.xticks(rotation=45)
        plt.xlabel("Model")
        plt.ylabel(title)
        plt.title(f"{title} per model")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
