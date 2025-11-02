"""
API Connector
Handles all communication with the FastAPI backend
"""

import requests
import streamlit as st
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIConnector:
    """Connects Streamlit frontend to FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.timeout = 120  # 2 minutes for long-running analyses
    
    def health_check(self) -> Dict:
        """Check if backend is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_stock_data(self, symbol: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Dict:
        """Get historical stock data"""
        try:
            params = {}
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            response = requests.get(
                f"{self.base_url}/stock/{symbol}",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get stock data error: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_indicators(self, symbol: str, start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      indicators: Optional[str] = None) -> Dict:
        """Get technical indicators"""
        try:
            params = {}
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            if indicators:
                params['indicators'] = indicators
            
            response = requests.get(
                f"{self.base_url}/indicators/{symbol}",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get indicators error: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_events(self, symbol: str, days: int = 7, use_llm: bool = False) -> Dict:
        """Get news and events"""
        try:
            params = {'days': days, 'use_llm': use_llm}
            
            response = requests.get(
                f"{self.base_url}/events/{symbol}",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get events error: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get company information"""
        try:
            response = requests.get(
                f"{self.base_url}/info/{symbol}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get stock info error: {e}")
            return {"status": "error", "message": str(e)}
    
    def compute_risk_metrics(self, symbols: List[str], weights: List[float],
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict:
        """Compute portfolio risk metrics"""
        try:
            payload = {
                "symbols": symbols,
                "weights": weights
            }
            if start_date:
                payload['start_date'] = start_date
            if end_date:
                payload['end_date'] = end_date
            
            response = requests.post(
                f"{self.base_url}/risk-metrics",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Compute risk metrics error: {e}")
            return {"status": "error", "message": str(e)}
    
    def analyze_stock(self, symbol: str, user_goal: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Dict:
        """AI-powered comprehensive stock analysis"""
        try:
            payload = {"symbol": symbol}
            if user_goal:
                payload['user_goal'] = user_goal
            if start_date:
                payload['start_date'] = start_date
            if end_date:
                payload['end_date'] = end_date
            
            with st.spinner('🤖 AI Agents analyzing... This may take a minute...'):
                response = requests.post(
                    f"{self.base_url}/analyze/stock",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Analyze stock error: {e}")
            return {"status": "error", "message": str(e)}
    
    def analyze_portfolio(self, symbols: List[str],
                         weights: Optional[List[float]] = None,
                         user_goal: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         optimization_method: str = "equal_weight") -> Dict:
        """AI-powered portfolio analysis"""
        try:
            payload = {
                "symbols": symbols,
                "optimization_method": optimization_method
            }
            if weights:
                payload['weights'] = weights
            if user_goal:
                payload['user_goal'] = user_goal
            if start_date:
                payload['start_date'] = start_date
            if end_date:
                payload['end_date'] = end_date
            
            with st.spinner('🤖 AI Agents optimizing portfolio... This may take a minute...'):
                response = requests.post(
                    f"{self.base_url}/analyze/portfolio",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Analyze portfolio error: {e}")
            return {"status": "error", "message": str(e)}


# Create global instance
@st.cache_resource
def get_api_connector():
    """Get cached API connector instance"""
    return APIConnector()