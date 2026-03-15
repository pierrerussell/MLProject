"""
Machine Learning Models for Financial Forecasting

Implements Linear Regression and Random Forest Regression for stock price prediction.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from schemas import StockPrice, ModelMetrics, ModelPrediction


@dataclass
class ForecastResult:
    """Container for forecast results from a model."""
    model_name: str
    predictions: list[float]
    prediction_dates: list[date]
    metrics: ModelMetrics
    feature_importance: Optional[dict[str, float]] = None


class FinancialForecaster:
    """
    Financial forecasting using ML models.

    Supports both Linear Regression and Random Forest Regression
    with configurable prediction horizons.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the forecaster.

        Args:
            test_size: Fraction of data to use for testing (0.0-1.0).
            random_state: Random seed for reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state
        self._linear_model: Optional[LinearRegression] = None
        self._rf_model: Optional[RandomForestRegressor] = None
        self._feature_names: list[str] = []

    def _create_features(
        self,
        df: pd.DataFrame,
        include_technical: bool = True
    ) -> pd.DataFrame:
        """
        Create features for ML models from price data.

        Args:
            df: DataFrame with OHLCV data.
            include_technical: Whether to include technical indicators.

        Returns:
            DataFrame with engineered features.
        """
        features = pd.DataFrame(index=df.index)

        # Time-based features
        features["day_of_week"] = df["date"].dt.dayofweek
        features["day_of_month"] = df["date"].dt.day
        features["month"] = df["date"].dt.month
        features["quarter"] = df["date"].dt.quarter

        # Price features
        features["open"] = df["open"]
        features["high"] = df["high"]
        features["low"] = df["low"]
        features["volume"] = df["volume"]

        # Price range
        features["daily_range"] = df["high"] - df["low"]
        features["daily_range_pct"] = features["daily_range"] / df["close"]

        if include_technical:
            # Lagged close prices
            for lag in [1, 2, 3, 5, 10]:
                features[f"close_lag_{lag}"] = df["close"].shift(lag)

            # Moving averages
            for window in [5, 10, 20]:
                features[f"sma_{window}"] = df["close"].rolling(window=window).mean()
                features[f"volume_sma_{window}"] = df["volume"].rolling(window=window).mean()

            # Volatility (standard deviation)
            for window in [5, 10, 20]:
                features[f"volatility_{window}"] = df["close"].rolling(window=window).std()

            # Price momentum
            for period in [5, 10, 20]:
                features[f"momentum_{period}"] = df["close"] - df["close"].shift(period)
                features[f"return_{period}"] = df["close"].pct_change(periods=period)

            # Relative position (where is price in recent range)
            for window in [10, 20]:
                rolling_min = df["low"].rolling(window=window).min()
                rolling_max = df["high"].rolling(window=window).max()
                features[f"rel_position_{window}"] = (
                    (df["close"] - rolling_min) / (rolling_max - rolling_min + 1e-10)
                )

        return features

    def _prepare_data(
        self,
        prices: list[StockPrice],
        include_technical: bool = True
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Prepare data for training.

        Args:
            prices: List of StockPrice objects.
            include_technical: Whether to include technical indicators.

        Returns:
            Tuple of (X features, y target, feature names).
        """
        # Convert to DataFrame
        df = pd.DataFrame([p.model_dump() for p in prices])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Create features
        features = self._create_features(df, include_technical)

        # Target is next day's close price
        target = df["close"].shift(-1)

        # Remove rows with NaN values
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_mask]
        target = target[valid_mask]

        feature_names = features.columns.tolist()

        return features.values, target.values, feature_names

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> ModelMetrics:
        """Calculate evaluation metrics."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # MAPE (avoiding division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = None

        return ModelMetrics(
            rmse=round(rmse, 4),
            mae=round(mae, 4),
            r2_score=round(r2, 4),
            mape=round(mape, 2) if mape else None
        )

    def _generate_future_dates(
        self,
        last_date: date,
        horizon: int
    ) -> list[date]:
        """Generate future trading dates (skip weekends)."""
        future_dates = []
        current_date = last_date

        while len(future_dates) < horizon:
            current_date += timedelta(days=1)
            # Skip weekends (Saturday=5, Sunday=6)
            if current_date.weekday() < 5:
                future_dates.append(current_date)

        return future_dates

    def train_and_predict(
        self,
        prices: list[StockPrice],
        horizon: int = 5,
        include_features: bool = True
    ) -> tuple[ForecastResult, ForecastResult]:
        """
        Train both models and generate predictions.

        Args:
            prices: Historical stock prices.
            horizon: Number of periods to forecast.
            include_features: Whether to include technical indicators.

        Returns:
            Tuple of (linear_regression_result, random_forest_result).
        """
        if len(prices) < 50:
            raise ValueError("Need at least 50 data points for training")

        # Prepare data
        X, y, self._feature_names = self._prepare_data(prices, include_features)

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            shuffle=False,  # Keep temporal order
            random_state=self.random_state
        )

        # Train Linear Regression
        self._linear_model = LinearRegression()
        self._linear_model.fit(X_train, y_train)
        lr_pred_test = self._linear_model.predict(X_test)
        lr_metrics = self._calculate_metrics(y_test, lr_pred_test)

        # Train Random Forest
        self._rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        self._rf_model.fit(X_train, y_train)
        rf_pred_test = self._rf_model.predict(X_test)
        rf_metrics = self._calculate_metrics(y_test, rf_pred_test)

        # Get feature importance for Random Forest
        feature_importance = dict(zip(
            self._feature_names,
            [round(float(imp), 4) for imp in self._rf_model.feature_importances_]
        ))
        # Sort by importance and keep top 10
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Generate future predictions
        last_date = max(p.date for p in prices)
        future_dates = self._generate_future_dates(last_date, horizon)

        # For predictions, we'll use iterative forecasting
        lr_predictions = self._forecast_iterative(
            prices, self._linear_model, horizon, include_features
        )
        rf_predictions = self._forecast_iterative(
            prices, self._rf_model, horizon, include_features
        )

        lr_result = ForecastResult(
            model_name="linear_regression",
            predictions=[round(p, 2) for p in lr_predictions],
            prediction_dates=future_dates,
            metrics=lr_metrics
        )

        rf_result = ForecastResult(
            model_name="random_forest",
            predictions=[round(p, 2) for p in rf_predictions],
            prediction_dates=future_dates,
            metrics=rf_metrics,
            feature_importance=feature_importance
        )

        return lr_result, rf_result

    def _forecast_iterative(
        self,
        prices: list[StockPrice],
        model,
        horizon: int,
        include_features: bool
    ) -> list[float]:
        """
        Generate multi-step forecasts iteratively.

        Uses predicted values as inputs for subsequent predictions.
        """
        # Convert to DataFrame for feature engineering
        df = pd.DataFrame([p.model_dump() for p in prices])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        predictions = []
        current_df = df.copy()

        for i in range(horizon):
            # Create features from current data
            features = self._create_features(current_df, include_features)

            # Get the last row of features (most recent)
            last_features = features.iloc[-1:].values

            # Handle any NaN values by filling with column means
            if np.isnan(last_features).any():
                col_means = np.nanmean(features.values, axis=0)
                nan_mask = np.isnan(last_features)
                last_features[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

            # Predict next value
            pred = float(model.predict(last_features)[0])
            predictions.append(pred)

            # Add predicted value to dataframe for next iteration
            last_date = current_df["date"].iloc[-1]
            next_date = last_date + timedelta(days=1)

            # Skip weekends
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)

            new_row = pd.DataFrame([{
                "symbol": current_df["symbol"].iloc[-1],
                "date": next_date,
                "open": pred,
                "high": pred * 1.01,  # Estimate
                "low": pred * 0.99,   # Estimate
                "close": pred,
                "volume": int(current_df["volume"].iloc[-20:].mean()),
                "change": pred - current_df["close"].iloc[-1],
                "change_percent": ((pred - current_df["close"].iloc[-1]) /
                                   current_df["close"].iloc[-1] * 100),
                "vwap": pred
            }])

            current_df = pd.concat([current_df, new_row], ignore_index=True)

        return predictions
