"""
Tools Registry
Central registry for all tools and APIs available to agents
Provides unified interface for tool discovery and invocation
"""

import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools"""
    DATA_FETCH = "data_fetch"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    COMMUNICATION = "communication"
    UTILITY = "utility"


@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    category: ToolCategory
    function: Callable
    parameters: Dict[str, Any]
    returns: str
    examples: Optional[List[str]] = None


class ToolsRegistry:
    """Central registry for all tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[Tool]:
        """List all tools or tools in a specific category"""
        if category:
            return [tool for tool in self.tools.values() if tool.category == category]
        return list(self.tools.values())
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool with given parameters"""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        try:
            logger.info(f"Executing tool: {name}")
            result = tool.function(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}")
            raise
    
    def _register_default_tools(self):
        """Register default tools from backend utilities"""
        from backend.utils import (
            data_fetcher,
            indicator_calculator,
            news_fetcher,
            risk_analyzer
        )
        
        # Data Fetching Tools
        self.register_tool(Tool(
            name="fetch_stock_data",
            description="Fetch historical stock price data",
            category=ToolCategory.DATA_FETCH,
            function=data_fetcher.get_stock_data,
            parameters={
                "symbol": "str - Stock ticker symbol",
                "start_date": "str - Start date (optional)",
                "end_date": "str - End date (optional)"
            },
            returns="Dict with OHLCV data",
            examples=["fetch_stock_data('AAPL', '2024-01-01', '2024-12-31')"]
        ))
        
        self.register_tool(Tool(
            name="get_stock_info",
            description="Get company fundamental information",
            category=ToolCategory.DATA_FETCH,
            function=data_fetcher.get_stock_info,
            parameters={"symbol": "str - Stock ticker symbol"},
            returns="Dict with company info",
            examples=["get_stock_info('AAPL')"]
        ))
        
        self.register_tool(Tool(
            name="get_last_price",
            description="Get current/latest stock price",
            category=ToolCategory.DATA_FETCH,
            function=data_fetcher.get_last_price,
            parameters={"symbol": "str - Stock ticker symbol"},
            returns="Float - Last price",
            examples=["get_last_price('AAPL')"]
        ))
        
        # Analysis Tools
        self.register_tool(Tool(
            name="compute_indicators",
            description="Compute technical indicators for a stock",
            category=ToolCategory.ANALYSIS,
            function=indicator_calculator.get_all_indicators,
            parameters={
                "df": "DataFrame - Price data",
                "indicators": "List[str] - Specific indicators (optional)"
            },
            returns="Dict with computed indicators",
            examples=["compute_indicators(df, ['sma_50', 'rsi_14'])"]
        ))
        
        self.register_tool(Tool(
            name="compute_sma",
            description="Compute Simple Moving Average",
            category=ToolCategory.ANALYSIS,
            function=indicator_calculator.compute_sma,
            parameters={
                "df": "DataFrame - Price data",
                "period": "int - Period for SMA"
            },
            returns="Series with SMA values",
            examples=["compute_sma(df, period=50)"]
        ))
        
        self.register_tool(Tool(
            name="compute_rsi",
            description="Compute Relative Strength Index",
            category=ToolCategory.ANALYSIS,
            function=indicator_calculator.compute_rsi,
            parameters={
                "df": "DataFrame - Price data",
                "period": "int - Period for RSI"
            },
            returns="Series with RSI values",
            examples=["compute_rsi(df, period=14)"]
        ))
        
        self.register_tool(Tool(
            name="fetch_news",
            description="Fetch news articles for a symbol",
            category=ToolCategory.DATA_FETCH,
            function=news_fetcher.fetch_news,
            parameters={
                "query": "str - Search query",
                "start_date": "str - Start date (optional)",
                "end_date": "str - End date (optional)",
                "max_results": "int - Max articles (default 20)"
            },
            returns="List of article dicts",
            examples=["fetch_news('AAPL', max_results=10)"]
        ))
        
        self.register_tool(Tool(
            name="analyze_sentiment",
            description="Analyze sentiment of text",
            category=ToolCategory.ANALYSIS,
            function=news_fetcher.analyze_sentiment_vader,
            parameters={"text": "str - Text to analyze"},
            returns="Dict with sentiment scores",
            examples=["analyze_sentiment('Apple reports strong earnings')"]
        ))
        
        self.register_tool(Tool(
            name="get_news_summary",
            description="Get comprehensive news summary for a symbol",
            category=ToolCategory.ANALYSIS,
            function=news_fetcher.get_news_summary,
            parameters={
                "symbol": "str - Stock ticker",
                "days": "int - Days to look back (default 7)"
            },
            returns="Dict with news analysis",
            examples=["get_news_summary('AAPL', days=7)"]
        ))
        
        # Risk Analysis Tools
        self.register_tool(Tool(
            name="compute_volatility",
            description="Compute volatility of returns",
            category=ToolCategory.ANALYSIS,
            function=risk_analyzer.compute_volatility,
            parameters={
                "returns": "Series - Return series",
                "annualize": "bool - Annualize result (default True)"
            },
            returns="Float - Volatility",
            examples=["compute_volatility(returns, annualize=True)"]
        ))
        
        self.register_tool(Tool(
            name="compute_sharpe_ratio",
            description="Compute Sharpe ratio",
            category=ToolCategory.ANALYSIS,
            function=risk_analyzer.compute_sharpe_ratio,
            parameters={
                "returns": "Series - Return series",
                "risk_free_rate": "float - Risk-free rate (optional)"
            },
            returns="Float - Sharpe ratio",
            examples=["compute_sharpe_ratio(returns, risk_free_rate=0.02)"]
        ))
        
        self.register_tool(Tool(
            name="compute_max_drawdown",
            description="Compute maximum drawdown",
            category=ToolCategory.ANALYSIS,
            function=risk_analyzer.compute_max_drawdown,
            parameters={"prices": "Series - Price series"},
            returns="Dict with drawdown info",
            examples=["compute_max_drawdown(prices)"]
        ))
        
        self.register_tool(Tool(
            name="compute_var",
            description="Compute Value at Risk",
            category=ToolCategory.ANALYSIS,
            function=risk_analyzer.compute_var,
            parameters={
                "returns": "Series - Return series",
                "confidence": "float - Confidence level (default 0.95)"
            },
            returns="Float - VaR",
            examples=["compute_var(returns, confidence=0.95)"]
        ))
        
        self.register_tool(Tool(
            name="compute_portfolio_risk",
            description="Compute portfolio risk metrics",
            category=ToolCategory.ANALYSIS,
            function=risk_analyzer.compute_portfolio_risk,
            parameters={
                "returns_df": "DataFrame - Returns for each asset",
                "weights": "List[float] - Portfolio weights"
            },
            returns="Dict with portfolio risk metrics",
            examples=["compute_portfolio_risk(returns_df, [0.5, 0.5])"]
        ))
        
        self.register_tool(Tool(
            name="stress_test",
            description="Perform stress test on portfolio",
            category=ToolCategory.ANALYSIS,
            function=risk_analyzer.stress_test,
            parameters={
                "returns_df": "DataFrame - Returns data",
                "weights": "List[float] - Portfolio weights",
                "shock_pct": "float - Shock percentage (default -0.20)"
            },
            returns="Dict with stress test results",
            examples=["stress_test(returns_df, weights, shock_pct=-0.20)"]
        ))
        
        logger.info(f"Registered {len(self.tools)} default tools")
    
    def get_tools_description(self) -> str:
        """Get formatted description of all tools"""
        descriptions = []
        
        for category in ToolCategory:
            tools_in_category = self.list_tools(category)
            if tools_in_category:
                descriptions.append(f"\n{category.value.upper()} TOOLS:")
                for tool in tools_in_category:
                    descriptions.append(f"  - {tool.name}: {tool.description}")
        
        return "\n".join(descriptions)
    
    def search_tools(self, keyword: str) -> List[Tool]:
        """Search tools by keyword in name or description"""
        keyword_lower = keyword.lower()
        return [
            tool for tool in self.tools.values()
            if keyword_lower in tool.name.lower() or 
               keyword_lower in tool.description.lower()
        ]


# Create global instance
tools_registry = ToolsRegistry()


# Utility function for agents
def get_available_tools() -> List[str]:
    """Get list of available tool names"""
    return list(tools_registry.tools.keys())


def call_tool(tool_name: str, **kwargs) -> Any:
    """Call a tool by name with parameters"""
    return tools_registry.execute_tool(tool_name, **kwargs)