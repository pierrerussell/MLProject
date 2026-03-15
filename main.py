"""
Financial Forecasting ML API

A FastAPI backend for financial data analysis and ML-based price predictions.
Uses FinancialModelingPrep API for data and scikit-learn for ML models.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from schemas import (
    Company,
    HistoricalDataRequest,
    HistoricalDataResponse,
    PredictionRequest,
    PredictionResponse,
    ModelPrediction,
    CompanySearchResponse,
    HealthResponse,
)
from services import FMPService
from models import FinancialForecaster

# Load environment variables
load_dotenv()

# Global service instances
fmp_service: Optional[FMPService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global fmp_service

    # Startup
    api_key = os.getenv("FMP_API_KEY")
    if api_key:
        fmp_service = FMPService(api_key=api_key)
        print("FMP Service initialized")
    else:
        print("Warning: FMP_API_KEY not set. API calls will fail.")

    yield

    # Shutdown
    if fmp_service:
        await fmp_service.close()


app = FastAPI(
    title="Financial Forecasting ML API",
    description=(
        "API for fetching financial data and generating ML-based stock price predictions. "
        "Supports Linear Regression and Random Forest models with configurable prediction horizons."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_fmp_service() -> FMPService:
    """Get FMP service or raise error if not configured."""
    if fmp_service is None:
        raise HTTPException(
            status_code=503,
            detail="FMP service not configured. Set FMP_API_KEY environment variable."
        )
    return fmp_service


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """API health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy" if fmp_service else "degraded",
        version="1.0.0"
    )


@app.get("/companies/search", response_model=CompanySearchResponse, tags=["Companies"])
async def search_companies(
    query: str = Query(..., min_length=1, description="Search query (symbol or name)"),
    search_type: str = Query(
        default="symbol",
        pattern="^(symbol|name)$",
        description="Search by 'symbol' or 'name'"
    )
):
    """
    Search for companies by symbol or name.

    Note: Free tier only supports major US stocks (AAPL, MSFT, GOOGL, etc.)
    """
    service = get_fmp_service()

    if search_type == "symbol":
        results = await service.search_by_symbol(query)
    else:
        results = await service.search_by_name(query)

    return CompanySearchResponse(
        query=query,
        results=results,
        count=len(results)
    )


@app.get(
    "/historical/{symbol}",
    response_model=HistoricalDataResponse,
    tags=["Historical Data"]
)
async def get_historical_data(
    symbol: str,
    years: int = Query(default=2, ge=1, le=10, description="Years of historical data")
):
    """
    Fetch historical stock price data for a symbol.

    Returns daily OHLCV data for the specified number of years.
    """
    service = get_fmp_service()

    # Get company info
    company = await service.get_company_info(symbol)

    # Get historical prices
    prices = await service.get_historical_prices(symbol, years=years)

    if not prices:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data found for symbol: {symbol}"
        )

    return HistoricalDataResponse(
        symbol=symbol.upper(),
        company=company,
        prices=prices,
        count=len(prices)
    )


@app.post("/historical", response_model=HistoricalDataResponse, tags=["Historical Data"])
async def get_historical_data_post(request: HistoricalDataRequest):
    """
    Fetch historical stock price data (POST version).

    Allows more control over the date range via request body.
    """
    service = get_fmp_service()

    company = await service.get_company_info(request.symbol)
    prices = await service.get_historical_prices(
        request.symbol,
        from_date=request.from_date,
        years=request.years
    )

    if not prices:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data found for symbol: {request.symbol}"
        )

    return HistoricalDataResponse(
        symbol=request.symbol.upper(),
        company=company,
        prices=prices,
        count=len(prices)
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_stock_price(request: PredictionRequest):
    """
    Generate stock price predictions using ML models.

    Trains both Linear Regression and Random Forest models on historical data
    and returns predictions with evaluation metrics.

    **Parameters:**
    - `symbol`: Stock ticker symbol (e.g., AAPL, MSFT)
    - `horizon`: Number of future periods to predict (1-24)
    - `include_features`: Whether to include technical indicators

    **Returns:**
    - Predictions from both models
    - Evaluation metrics (RMSE, MAE, R2)
    - Feature importance (for Random Forest)
    - Recommendation of best model based on RMSE
    """
    service = get_fmp_service()

    # Fetch historical data (need enough for training)
    company = await service.get_company_info(request.symbol)
    prices = await service.get_historical_prices(request.symbol, years=3)

    if len(prices) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for {request.symbol}. Need at least 50 data points, got {len(prices)}."
        )

    # Train models and generate predictions
    try:
        forecaster = FinancialForecaster()
        lr_result, rf_result = forecaster.train_and_predict(
            prices=prices,
            horizon=request.horizon,
            include_features=request.include_features
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {str(e)}"
        )

    # Convert to response models
    lr_prediction = ModelPrediction(
        model_name=lr_result.model_name,
        predictions=lr_result.predictions,
        prediction_dates=lr_result.prediction_dates,
        metrics=lr_result.metrics,
        feature_importance=None
    )

    rf_prediction = ModelPrediction(
        model_name=rf_result.model_name,
        predictions=rf_result.predictions,
        prediction_dates=rf_result.prediction_dates,
        metrics=rf_result.metrics,
        feature_importance=rf_result.feature_importance
    )

    # Determine best model based on RMSE
    best_model = (
        "linear_regression"
        if lr_result.metrics.rmse < rf_result.metrics.rmse
        else "random_forest"
    )

    return PredictionResponse(
        symbol=request.symbol.upper(),
        company=company,
        data_points_used=len(prices),
        linear_regression=lr_prediction,
        random_forest=rf_prediction,
        best_model=best_model
    )


@app.get("/predict/{symbol}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_stock_price_get(
    symbol: str,
    horizon: int = Query(default=5, ge=1, le=24, description="Prediction horizon (periods)")
):
    """
    Generate stock price predictions (GET version).

    Simplified endpoint for quick predictions with default settings.
    """
    request = PredictionRequest(
        symbol=symbol,
        horizon=horizon,
        include_features=True
    )
    return await predict_stock_price(request)


@app.get("/symbols", tags=["Companies"])
async def list_available_symbols():
    """
    List symbols available on free tier.

    The FMP free tier limits access to major US stocks.
    """
    return {
        "symbols": sorted(FMPService.FREE_TIER_ALLOWED_SYMBOLS),
        "count": len(FMPService.FREE_TIER_ALLOWED_SYMBOLS),
        "note": "Free tier only supports these major US stocks"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
