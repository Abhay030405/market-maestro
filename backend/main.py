"""
Market Maestro FastAPI Backend
Main application file with all API endpoints
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import logging

from backend.config import settings
from backend.utils import (
    data_fetcher,
    indicator_calculator,
    news_fetcher,
    risk_analyzer,
    cache_manager
)
from backend.agents import orchestrator
from backend.langchain_core import query_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="AI-driven multi-agent financial research platform"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Pydantic Models =============

class RiskMetricsRequest(BaseModel):
    """Request model for portfolio risk analysis"""
    symbols: List[str] = Field(..., description="List of stock symbols")
    weights: List[float] = Field(..., description="Portfolio weights (must sum to 1)")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    market_symbol: Optional[str] = Field("^GSPC", description="Market benchmark symbol")


class StressTestRequest(BaseModel):
    """Request model for stress testing"""
    symbols: List[str]
    weights: List[float]
    shock_percentage: float = Field(-0.20, description="Shock percentage (e.g., -0.20 for 20% drop)")
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ============= API Endpoints =============

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Welcome to Market Maestro API",
        "version": settings.API_VERSION,
        "status": "online",
        "endpoints": {
            "stock_data": "/stock/{symbol}",
            "indicators": "/indicators/{symbol}",
            "news": "/events/{symbol}",
            "risk_metrics": "/risk-metrics",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    # Validate API keys
    api_keys = settings.validate_keys()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.API_VERSION,
        "api_keys_configured": api_keys,
        "cache_stats": cache_manager.get_stats()
    }


@app.get("/stock/{symbol}")
async def get_stock_data(
    symbol: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    interval: str = Query("1d", description="Data interval (1d, 1h, 1m)")
):
    """
    Get historical stock data
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date
        end_date: End date
        interval: Data interval
    
    Returns:
        Stock data with OHLCV and metadata
    """
    try:
        # Check cache
        cache_key = f"stock_{symbol}_{start_date}_{end_date}_{interval}"
        cached_data = cache_manager.get(cache_key)
        
        if cached_data:
            logger.info(f"Returning cached data for {symbol}")
            return cached_data
        
        # Fetch fresh data
        result = data_fetcher.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=404, detail=result['message'])
        
        # Cache the result
        cache_manager.set(cache_key, result, ttl=3600)  # 1 hour cache
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_stock_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indicators/{symbol}")
async def get_indicators(
    symbol: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    indicators: Optional[str] = Query(
        None,
        description="Comma-separated list of indicators (e.g., 'sma_50,rsi_14,macd')"
    )
):
    """
    Get stock data with technical indicators
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date
        end_date: End date
        indicators: Specific indicators to compute
    
    Returns:
        Stock data with computed indicators
    """
    try:
        # Check cache
        cache_key = f"indicators_{symbol}_{start_date}_{end_date}_{indicators}"
        cached_data = cache_manager.get(cache_key)
        
        if cached_data:
            logger.info(f"Returning cached indicators for {symbol}")
            return cached_data
        
        # Fetch stock data
        stock_data = data_fetcher.get_stock_data(symbol, start_date, end_date)
        
        if stock_data['status'] == 'error':
            raise HTTPException(status_code=404, detail=stock_data['message'])
        
        # Convert to DataFrame
        df = pd.DataFrame(stock_data['data'])
        
        # Parse indicators list
        indicators_list = None
        if indicators:
            indicators_list = [ind.strip() for ind in indicators.split(',')]
        
        # Compute indicators
        computed_indicators = indicator_calculator.get_all_indicators(df, indicators_list)
        
        result = {
            'status': 'success',
            'symbol': symbol.upper(),
            'last_price': stock_data['last_price'],
            'data_points': len(df),
            'indicators': computed_indicators,
            'price_data': stock_data['data'][-50:],  # Last 50 data points
            'company_name': stock_data.get('company_name', symbol)
        }
        
        # Cache the result
        cache_manager.set(cache_key, result, ttl=1800)  # 30 minutes cache
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_indicators: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events/{symbol}")
async def get_events(
    symbol: str,
    days: int = Query(7, description="Number of days to look back"),
    use_llm: bool = Query(False, description="Use LLM for sentiment analysis")
):
    """
    Get news and events for a symbol
    
    Args:
        symbol: Stock ticker symbol
        days: Number of days to look back
        use_llm: Whether to use LLM for sentiment analysis
    
    Returns:
        News articles with sentiment and detected events
    """
    try:
        # Check cache (shorter TTL for news)
        cache_key = f"events_{symbol}_{days}_{use_llm}"
        cached_data = cache_manager.get(cache_key)
        
        if cached_data:
            logger.info(f"Returning cached events for {symbol}")
            return cached_data
        
        # Get news summary
        result = news_fetcher.get_news_summary(
            symbol=symbol,
            days=days,
            use_llm=use_llm
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        # Cache the result
        cache_manager.set(cache_key, result, ttl=900)  # 15 minutes cache
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_events: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk-metrics")
async def compute_risk_metrics(request: RiskMetricsRequest):
    """
    Compute portfolio risk metrics
    
    Args:
        request: RiskMetricsRequest with symbols, weights, dates
    
    Returns:
        Portfolio risk analysis
    """
    try:
        # Validate weights
        if len(request.symbols) != len(request.weights):
            raise HTTPException(
                status_code=400,
                detail="Number of symbols must match number of weights"
            )
        
        if abs(sum(request.weights) - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail="Weights must sum to 1.0"
            )
        
        # Set default dates
        end_date = request.end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = request.start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Fetch data for all symbols
        returns_data = {}
        
        for symbol in request.symbols:
            stock_data = data_fetcher.get_stock_data(symbol, start_date, end_date)
            
            if stock_data['status'] == 'error':
                raise HTTPException(
                    status_code=404,
                    detail=f"Could not fetch data for {symbol}: {stock_data['message']}"
                )
            
            df = pd.DataFrame(stock_data['data'])
            returns = df['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Compute portfolio risk
        portfolio_risk = risk_analyzer.compute_portfolio_risk(returns_df, request.weights)
        
        if portfolio_risk['status'] == 'error':
            raise HTTPException(status_code=500, detail=portfolio_risk['message'])
        
        # Compute correlation matrix
        correlation_result = risk_analyzer.compute_correlation_matrix(returns_df)
        
        # Fetch market data for beta calculation
        market_data = data_fetcher.get_stock_data(request.market_symbol, start_date, end_date)
        
        if market_data['status'] == 'success':
            market_df = pd.DataFrame(market_data['data'])
            market_returns = market_df['Close'].pct_change().dropna()
            
            # Calculate portfolio beta
            portfolio_returns = (returns_df * request.weights).sum(axis=1)
            beta = risk_analyzer.compute_beta(portfolio_returns, market_returns)
            portfolio_risk['portfolio_metrics']['beta'] = beta
        
        # Combine results
        result = {
            'status': 'success',
            'portfolio': {
                'symbols': request.symbols,
                'weights': request.weights,
                'metrics': portfolio_risk['portfolio_metrics']
            },
            'correlation': correlation_result,
            'date_range': {
                'start': start_date,
                'end': end_date
            }
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in compute_risk_metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stress-test")
async def stress_test_portfolio(request: StressTestRequest):
    """
    Perform stress test on portfolio
    
    Args:
        request: StressTestRequest with portfolio details
    
    Returns:
        Stress test results
    """
    try:
        # Validate
        if len(request.symbols) != len(request.weights):
            raise HTTPException(
                status_code=400,
                detail="Number of symbols must match number of weights"
            )
        
        # Set default dates
        end_date = request.end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = request.start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Fetch returns data
        returns_data = {}
        
        for symbol in request.symbols:
            stock_data = data_fetcher.get_stock_data(symbol, start_date, end_date)
            
            if stock_data['status'] == 'error':
                raise HTTPException(
                    status_code=404,
                    detail=f"Could not fetch data for {symbol}"
                )
            
            df = pd.DataFrame(stock_data['data'])
            returns = df['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        returns_df = pd.DataFrame(returns_data)
        
        # Perform stress test
        stress_result = risk_analyzer.stress_test(
            returns_df,
            request.weights,
            request.shock_percentage
        )
        
        if stress_result['status'] == 'error':
            raise HTTPException(status_code=500, detail=stress_result['message'])
        
        return stress_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stress_test_portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quote/{symbol}")
async def get_realtime_quote(symbol: str):
    """
    Get real-time quote for a symbol
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Current quote data
    """
    try:
        result = data_fetcher.get_realtime_quote(symbol)
        
        if result['status'] == 'error':
            raise HTTPException(status_code=404, detail=result['message'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_realtime_quote: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info/{symbol}")
async def get_stock_info(symbol: str):
    """
    Get company fundamental information
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Company info and fundamentals
    """
    try:
        # Check cache
        cache_key = f"info_{symbol}"
        cached_data = cache_manager.get(cache_key)
        
        if cached_data:
            return cached_data
        
        result = data_fetcher.get_stock_info(symbol)
        
        if result['status'] == 'error':
            raise HTTPException(status_code=404, detail=result['message'])
        
        # Cache for longer (company info doesn't change often)
        cache_manager.set(cache_key, result, ttl=86400)  # 24 hours
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_stock_info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cache entries"""
    try:
        cache_manager.clear()
        data_fetcher.clear_cache()
        
        return {
            'status': 'success',
            'message': 'Cache cleared successfully'
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= AI AGENT ENDPOINTS (PHASE 2) =============

class AnalyzeStockRequest(BaseModel):
    """Request model for stock analysis"""
    symbol: str = Field(..., description="Stock ticker symbol")
    user_goal: Optional[str] = Field(None, description="Investment goal or objective")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class AnalyzePortfolioRequest(BaseModel):
    """Request model for portfolio analysis"""
    symbols: List[str] = Field(..., description="List of stock symbols")
    weights: Optional[List[float]] = Field(None, description="Portfolio weights (optional)")
    user_goal: Optional[str] = Field(None, description="Investment goal")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date")
    optimization_method: str = Field("equal_weight", description="Optimization method")


@app.post("/analyze/stock")
async def analyze_stock(request: AnalyzeStockRequest):
    """
    Comprehensive AI-powered stock analysis
    Uses all agents to provide complete analysis and recommendations
    
    Args:
        request: AnalyzeStockRequest with symbol and parameters
    
    Returns:
        Comprehensive analysis report
    """
    try:
        logger.info(f"AI Analysis requested for {request.symbol}")
        
        # Check cache
        cache_key = f"analyze_stock_{request.symbol}_{request.start_date}_{request.end_date}"
        cached_result = cache_manager.get(cache_key)
        
        if cached_result:
            logger.info(f"Returning cached analysis for {request.symbol}")
            return cached_result
        
        # Run orchestrator
        result = orchestrator.analyze_stock(
            symbol=request.symbol,
            user_goal=request.user_goal,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('message', 'Analysis failed'))
        
        # Cache for 30 minutes
        cache_manager.set(cache_key, result, ttl=1800)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_stock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/portfolio")
async def analyze_portfolio(request: AnalyzePortfolioRequest):
    """
    Comprehensive AI-powered portfolio analysis
    Uses all agents to provide portfolio optimization and analysis
    
    Args:
        request: AnalyzePortfolioRequest with symbols and parameters
    
    Returns:
        Comprehensive portfolio analysis report
    """
    try:
        logger.info(f"AI Portfolio Analysis requested for {len(request.symbols)} assets")
        
        # Validate weights if provided
        if request.weights:
            if len(request.symbols) != len(request.weights):
                raise HTTPException(
                    status_code=400,
                    detail="Number of symbols must match number of weights"
                )
            
            if abs(sum(request.weights) - 1.0) > 0.01:
                raise HTTPException(
                    status_code=400,
                    detail="Weights must sum to 1.0"
                )
        
        # Run orchestrator
        result = orchestrator.analyze_portfolio(
            symbols=request.symbols,
            weights=request.weights,
            user_goal=request.user_goal,
            start_date=request.start_date,
            end_date=request.end_date,
            optimization_method=request.optimization_method
        )
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('message', 'Analysis failed'))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= NATURAL LANGUAGE QUERY ENDPOINT (PHASE 3) =============

class NaturalQueryRequest(BaseModel):
    """Request model for natural language queries"""
    query: str = Field(..., description="Natural language query")
    context: Optional[Dict] = Field(None, description="Additional context")


@app.post("/query")
async def natural_language_query(request: NaturalQueryRequest):
    """
    Process natural language queries using AI
    
    Examples:
    - "Analyze Apple stock for long-term investment"
    - "Compare TCS and Infosys for growth potential"
    - "Optimize portfolio with MSFT, GOOGL, AAPL"
    - "What's the latest news on Tesla?"
    
    Args:
        request: NaturalQueryRequest with natural language query
    
    Returns:
        Comprehensive analysis with natural language response
    """
    try:
        logger.info(f"Natural language query: {request.query}")
        
        # Check cache
        cache_key = f"nlp_query_{request.query.lower().replace(' ', '_')[:50]}"
        cached_result = cache_manager.get(cache_key)
        
        if cached_result:
            logger.info("Returning cached NLP query result")
            return cached_result
        
        # Process query
        result = query_processor.process_query(request.query)
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('message', 'Query processing failed'))
        
        # Cache for 30 minutes
        cache_manager.set(cache_key, result, ttl=1800)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in natural_language_query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.DEBUG_MODE)