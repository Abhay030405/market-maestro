"""
Schema Definitions
Defines standardized input/output schemas for all agents
Ensures consistent message format across the system
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


# ============= ENUMS =============

class AgentType(str, Enum):
    """Types of agents in the system"""
    MARKET = "market"
    EVENT = "event"
    RISK = "risk"
    PORTFOLIO = "portfolio"
    ADVISOR = "advisor"
    ORCHESTRATOR = "orchestrator"


class MessageType(str, Enum):
    """Types of messages exchanged between agents"""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    INFO = "info"


class AnalysisStatus(str, Enum):
    """Status of analysis"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"


# ============= BASE SCHEMAS =============

class AgentMessage(BaseModel):
    """Base message format for agent communication"""
    agent_type: AgentType
    message_type: MessageType
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    """Standard request format for agents"""
    request_id: str
    agent_type: AgentType
    action: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentResponse(BaseModel):
    """Standard response format from agents"""
    request_id: str
    agent_type: AgentType
    status: AnalysisStatus
    data: Dict[str, Any]
    errors: Optional[List[str]] = Field(default_factory=list)
    warnings: Optional[List[str]] = Field(default_factory=list)
    execution_time: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============= MARKET AGENT SCHEMAS =============

class MarketAnalysisRequest(BaseModel):
    """Request schema for MarketAgent"""
    symbol: str = Field(..., description="Stock ticker symbol")
    start_date: Optional[str] = Field(None, description="Analysis start date")
    end_date: Optional[str] = Field(None, description="Analysis end date")
    indicators: Optional[List[str]] = Field(None, description="Specific indicators to compute")


class TechnicalIndicators(BaseModel):
    """Technical indicators data"""
    current_price: float
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[Dict[str, float]] = None
    volatility: Optional[float] = None
    atr_14: Optional[float] = None


class TrendAnalysis(BaseModel):
    """Trend analysis data"""
    direction: str
    strength: str
    short_term: str
    long_term: str


class MarketAnalysisResponse(BaseModel):
    """Response schema for MarketAgent"""
    symbol: str
    last_price: float
    company_name: str
    indicators: TechnicalIndicators
    trend_analysis: TrendAnalysis
    signals: Dict[str, Any]
    summary: str


# ============= EVENT AGENT SCHEMAS =============

class EventAnalysisRequest(BaseModel):
    """Request schema for EventAgent"""
    symbol: str
    days: int = Field(7, description="Number of days to look back")
    use_llm: bool = Field(False, description="Use LLM for sentiment analysis")


class NewsArticle(BaseModel):
    """News article data"""
    title: str
    description: Optional[str] = None
    url: Optional[str] = None
    source: str
    published_at: str
    sentiment: Optional[Dict[str, Any]] = None


class EventAnalysisResponse(BaseModel):
    """Response schema for EventAgent"""
    symbol: str
    article_count: int
    avg_sentiment: float
    sentiment_label: str
    articles: List[NewsArticle]
    events: List[Dict[str, Any]]
    summary: str


# ============= RISK AGENT SCHEMAS =============

class RiskAnalysisRequest(BaseModel):
    """Request schema for RiskAgent"""
    symbols: List[str]
    weights: Optional[List[float]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class RiskMetrics(BaseModel):
    """Risk metrics data"""
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    beta: Optional[float] = None


class RiskAssessment(BaseModel):
    """Risk assessment data"""
    level: str
    score: float
    factors: List[str]


class RiskAnalysisResponse(BaseModel):
    """Response schema for RiskAgent"""
    symbols: List[str]
    metrics: RiskMetrics
    risk_assessment: RiskAssessment
    recommendations: List[str]
    summary: str


# ============= PORTFOLIO AGENT SCHEMAS =============

class PortfolioOptimizationRequest(BaseModel):
    """Request schema for PortfolioAgent"""
    symbols: List[str]
    method: str = Field("equal_weight", description="Optimization method")
    constraints: Optional[Dict[str, float]] = Field(default_factory=dict)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class PortfolioAllocation(BaseModel):
    """Portfolio allocation data"""
    weights: Dict[str, float]
    method: str
    rationale: str


class PortfolioMetrics(BaseModel):
    """Portfolio metrics"""
    expected_return: float
    volatility: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None


class PortfolioOptimizationResponse(BaseModel):
    """Response schema for PortfolioAgent"""
    symbols: List[str]
    allocation: PortfolioAllocation
    portfolio_metrics: PortfolioMetrics
    recommendations: List[str]
    summary: str


# ============= ADVISOR AGENT SCHEMAS =============

class AdvisorRequest(BaseModel):
    """Request schema for AdvisorAgent"""
    agent_results: Dict[str, Any]
    user_goal: Optional[str] = None


class AdvisorResponse(BaseModel):
    """Response schema for AdvisorAgent"""
    recommendations: List[str]
    llm_advice: str
    executive_summary: str
    confidence_score: Optional[float] = None


# ============= ORCHESTRATOR SCHEMAS =============

class OrchestratorRequest(BaseModel):
    """Request schema for Orchestrator"""
    query: str
    symbols: List[str]
    user_goal: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class OrchestratorResponse(BaseModel):
    """Response schema for Orchestrator"""
    status: AnalysisStatus
    report_type: str
    symbols: List[str]
    market_analysis: Optional[Dict[str, Any]] = None
    event_analysis: Optional[Dict[str, Any]] = None
    risk_analysis: Optional[Dict[str, Any]] = None
    portfolio_allocation: Optional[Dict[str, Any]] = None
    advisor_recommendation: Optional[Dict[str, Any]] = None
    executive_summary: str
    key_recommendations: List[str]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============= UTILITY FUNCTIONS =============

def create_agent_request(agent_type: AgentType, action: str, 
                        parameters: Dict[str, Any]) -> AgentRequest:
    """Create a standardized agent request"""
    import uuid
    return AgentRequest(
        request_id=str(uuid.uuid4()),
        agent_type=agent_type,
        action=action,
        parameters=parameters
    )


def create_agent_response(request_id: str, agent_type: AgentType,
                         status: AnalysisStatus, data: Dict[str, Any],
                         errors: List[str] = None) -> AgentResponse:
    """Create a standardized agent response"""
    return AgentResponse(
        request_id=request_id,
        agent_type=agent_type,
        status=status,
        data=data,
        errors=errors or []
    )


def validate_schema(data: Dict[str, Any], schema_class: type) -> bool:
    """Validate data against a schema"""
    try:
        schema_class(**data)
        return True
    except Exception:
        return False