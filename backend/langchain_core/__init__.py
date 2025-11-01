"""
LangChain Core initialization
Natural Language Query Processing with Memory and Tools
"""

from backend.langchain_core.llm_manager import llm_manager, LLMManager, QueryIntent
from backend.langchain_core.query_processor import query_processor, QueryProcessor
from backend.langchain_core.schema import (
    AgentType, MessageType, AnalysisStatus,
    AgentMessage, AgentRequest, AgentResponse,
    MarketAnalysisRequest, MarketAnalysisResponse,
    EventAnalysisRequest, EventAnalysisResponse,
    RiskAnalysisRequest, RiskAnalysisResponse,
    PortfolioOptimizationRequest, PortfolioOptimizationResponse,
    create_agent_request, create_agent_response
)
from backend.langchain_core.tools_registry import (
    tools_registry, ToolsRegistry, Tool, ToolCategory,
    get_available_tools, call_tool
)
from backend.langchain_core.memory_manager import (
    memory_manager, MemoryManager,
    ConversationMemory, AgentWorkingMemory,
    store_result, get_result, get_all_results
)

__all__ = [
    # LLM Manager
    'llm_manager',
    'LLMManager',
    'QueryIntent',
    
    # Query Processor
    'query_processor',
    'QueryProcessor',
    
    # Schema
    'AgentType',
    'MessageType',
    'AnalysisStatus',
    'AgentMessage',
    'AgentRequest',
    'AgentResponse',
    'MarketAnalysisRequest',
    'MarketAnalysisResponse',
    'EventAnalysisRequest',
    'EventAnalysisResponse',
    'RiskAnalysisRequest',
    'RiskAnalysisResponse',
    'PortfolioOptimizationRequest',
    'PortfolioOptimizationResponse',
    'create_agent_request',
    'create_agent_response',
    
    # Tools Registry
    'tools_registry',
    'ToolsRegistry',
    'Tool',
    'ToolCategory',
    'get_available_tools',
    'call_tool',
    
    # Memory Manager
    'memory_manager',
    'MemoryManager',
    'ConversationMemory',
    'AgentWorkingMemory',
    'store_result',
    'get_result',
    'get_all_results'
]