from pydantic import BaseModel, Field
import datetime
from typing import Optional
from enum import Enum


class DataSource(str, Enum):
    """Supported data sources for forecasting."""
    STOCK_PRICE = "stock_price"
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"


class Company(BaseModel):
    """Company information from FMP API."""
    symbol: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    currency: Optional[str] = Field(None, description="Trading currency")
    exchange: Optional[str] = Field(None, description="Exchange symbol")
    exchange_full_name: Optional[str] = Field(None, description="Full exchange name")


class StockPrice(BaseModel):
    """Historical stock price data."""
    symbol: str = Field(..., description="Stock ticker symbol")
    date: datetime.date = Field(..., description="Trading date")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")
    change: Optional[float] = Field(None, description="Price change")
    change_percent: Optional[float] = Field(None, description="Price change percentage")
    vwap: Optional[float] = Field(None, description="Volume weighted average price")


class HistoricalDataRequest(BaseModel):
    """Request for historical financial data."""
    symbol: str = Field(..., description="Stock ticker symbol", examples=["AAPL", "MSFT"])
    from_date: Optional[datetime.date] = Field(None, description="Start date for historical data")
    years: int = Field(default=2, ge=1, le=10, description="Number of years of data to fetch")


class HistoricalDataResponse(BaseModel):
    """Response containing historical financial data."""
    symbol: str
    company: Optional[Company] = None
    prices: list[StockPrice]
    count: int = Field(..., description="Number of data points returned")


class ModelMetrics(BaseModel):
    """Evaluation metrics for ML model performance."""
    rmse: float = Field(..., description="Root Mean Squared Error")
    mae: float = Field(..., description="Mean Absolute Error")
    r2_score: float = Field(..., description="R-squared score")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")


class ModelPrediction(BaseModel):
    """Prediction results from a single ML model."""
    model_name: str = Field(..., description="Name of the model (linear_regression or random_forest)")
    predictions: list[float] = Field(..., description="Predicted values for future periods")
    prediction_dates: list[datetime.date] = Field(..., description="Dates for predictions")
    metrics: ModelMetrics = Field(..., description="Model evaluation metrics")
    feature_importance: Optional[dict[str, float]] = Field(
        None, description="Feature importance scores (for Random Forest)"
    )


class PredictionRequest(BaseModel):
    """Request for stock price predictions."""
    symbol: str = Field(..., description="Stock ticker symbol", examples=["AAPL", "MSFT"])
    horizon: int = Field(
        default=5,
        ge=1,
        le=24,
        description="Number of periods to forecast (1-24)"
    )
    data_source: DataSource = Field(
        default=DataSource.STOCK_PRICE,
        description="Data source to use for prediction"
    )
    include_features: bool = Field(
        default=True,
        description="Include technical indicators as features"
    )


class PredictionResponse(BaseModel):
    """Response containing predictions from both ML models."""
    symbol: str = Field(..., description="Stock ticker symbol")
    company: Optional[Company] = None
    data_points_used: int = Field(..., description="Number of historical data points used")
    linear_regression: ModelPrediction = Field(..., description="Linear Regression model results")
    random_forest: ModelPrediction = Field(..., description="Random Forest model results")
    best_model: str = Field(..., description="Model with better RMSE score")


class CompanySearchResponse(BaseModel):
    """Response for company search."""
    query: str
    results: list[Company]
    count: int


class HealthResponse(BaseModel):
    """API health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
