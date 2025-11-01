"""
LLM Manager
Manages LangChain LLM interactions and prompt templates
Handles query parsing and intent detection
"""

import logging
from typing import Dict, List, Optional
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

from backend.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define query intent schema
class QueryIntent(BaseModel):
    """Structure for parsed query intent"""
    intent_type: str = Field(description="Type of query: analyze_stock, analyze_portfolio, compare_stocks, optimize_portfolio, market_news, risk_analysis")
    symbols: List[str] = Field(description="List of stock symbols mentioned")
    time_horizon: Optional[str] = Field(description="Time horizon: short-term, medium-term, long-term")
    user_goal: Optional[str] = Field(description="User's investment goal or objective")
    additional_context: Optional[Dict] = Field(description="Any additional context from the query")


class LLMManager:
    """Manages LLM interactions for natural language queries"""
    
    def __init__(self):
        self.model = None
        self.chat_model = None
        
        # Initialize Gemini
        if settings.GOOGLE_API_KEY:
            try:
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                
                # Initialize LangChain chat model
                self.chat_model = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    google_api_key=settings.GOOGLE_API_KEY,
                    temperature=0.1,  # Low temperature for consistent parsing
                    convert_system_message_to_human=True
                )
                
                logger.info("LLM Manager initialized successfully")
            except Exception as e:
                logger.error(f"LLM Manager initialization failed: {e}")
    
    def parse_query(self, user_query: str) -> QueryIntent:
        """
        Parse natural language query into structured intent
        
        Args:
            user_query: User's natural language query
        
        Returns:
            QueryIntent object with parsed information
        """
        try:
            # Create parser
            parser = JsonOutputParser(pydantic_object=QueryIntent)
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a financial query parser. Analyze the user's query and extract:
1. Intent type (analyze_stock, analyze_portfolio, compare_stocks, optimize_portfolio, market_news, risk_analysis)
2. Stock symbols mentioned (extract ticker symbols)
3. Time horizon (short-term, medium-term, long-term) if mentioned
4. User's investment goal or objective
5. Additional context

{format_instructions}

Examples:
Query: "Analyze Apple stock for long-term investment"
Output: {{"intent_type": "analyze_stock", "symbols": ["AAPL"], "time_horizon": "long-term", "user_goal": "long-term investment", "additional_context": {{}}}}

Query: "Compare TCS and Infosys for short-term growth"
Output: {{"intent_type": "compare_stocks", "symbols": ["TCS", "INFY"], "time_horizon": "short-term", "user_goal": "growth potential", "additional_context": {{}}}}

Query: "Optimize my portfolio with MSFT, GOOGL, and AAPL"
Output: {{"intent_type": "optimize_portfolio", "symbols": ["MSFT", "GOOGL", "AAPL"], "time_horizon": null, "user_goal": "portfolio optimization", "additional_context": {{}}}}
"""),
                ("human", "{query}")
            ])
            
            # Create chain
            chain = prompt | self.chat_model | parser
            
            # Parse query
            result = chain.invoke({
                "query": user_query,
                "format_instructions": parser.get_format_instructions()
            })
            
            # Convert to QueryIntent
            return QueryIntent(**result)
            
        except Exception as e:
            logger.error(f"Query parsing error: {e}")
            # Return default intent
            return QueryIntent(
                intent_type="analyze_stock",
                symbols=[],
                time_horizon=None,
                user_goal=user_query,
                additional_context={}
            )
    
    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """
        Generate natural language response
        
        Args:
            prompt: Prompt template
            context: Context data to include
        
        Returns:
            Generated response text
        """
        try:
            if context:
                # Format prompt with context
                formatted_prompt = prompt.format(**context)
            else:
                formatted_prompt = prompt
            
            # Create chain
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a professional financial advisor providing clear, actionable insights."),
                ("human", "{input}")
            ])
            
            chain = prompt_template | self.chat_model | StrOutputParser()
            
            response = chain.invoke({"input": formatted_prompt})
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I apologize, but I'm unable to generate a response at this time."
    
    def route_query(self, intent: QueryIntent) -> Dict:
        """
        Route parsed query to appropriate agents
        
        Args:
            intent: Parsed query intent
        
        Returns:
            Dict with routing information
        """
        routing = {
            "agents_to_call": [],
            "parameters": {}
        }
        
        # Determine which agents to call based on intent
        if intent.intent_type == "analyze_stock":
            routing["agents_to_call"] = ["market", "events", "risk"]
            routing["parameters"] = {
                "symbols": intent.symbols,
                "user_goal": intent.user_goal
            }
        
        elif intent.intent_type == "compare_stocks":
            routing["agents_to_call"] = ["market", "events", "risk"]
            routing["parameters"] = {
                "symbols": intent.symbols,
                "user_goal": intent.user_goal,
                "comparison_mode": True
            }
        
        elif intent.intent_type == "optimize_portfolio" or intent.intent_type == "analyze_portfolio":
            routing["agents_to_call"] = ["portfolio", "risk"]
            routing["parameters"] = {
                "symbols": intent.symbols,
                "user_goal": intent.user_goal
            }
        
        elif intent.intent_type == "market_news":
            routing["agents_to_call"] = ["events"]
            routing["parameters"] = {
                "symbols": intent.symbols
            }
        
        elif intent.intent_type == "risk_analysis":
            routing["agents_to_call"] = ["risk"]
            routing["parameters"] = {
                "symbols": intent.symbols
            }
        
        else:
            # Default: analyze stock
            routing["agents_to_call"] = ["market", "events", "risk"]
            routing["parameters"] = {
                "symbols": intent.symbols,
                "user_goal": intent.user_goal
            }
        
        return routing
    
    def create_comparison_prompt(self, symbol1: str, symbol2: str,
                                 analysis1: Dict, analysis2: Dict) -> str:
        """Create comparison prompt for two stocks"""
        
        prompt = f"""Compare {symbol1} and {symbol2} based on the following analysis:

**{symbol1} Analysis:**
- Market Trend: {analysis1.get('market_analysis', {}).get('trend_analysis', {}).get('direction', 'N/A')}
- Sentiment: {analysis1.get('event_analysis', {}).get('sentiment_label', 'N/A')}
- Risk Level: {analysis1.get('risk_analysis', {}).get('risk_assessment', {}).get('level', 'N/A')}

**{symbol2} Analysis:**
- Market Trend: {analysis2.get('market_analysis', {}).get('trend_analysis', {}).get('direction', 'N/A')}
- Sentiment: {analysis2.get('event_analysis', {}).get('sentiment_label', 'N/A')}
- Risk Level: {analysis2.get('risk_analysis', {}).get('risk_assessment', {}).get('level', 'N/A')}

Provide a clear comparison highlighting:
1. Which stock has better technical momentum
2. Which has more positive sentiment
3. Which has better risk-adjusted potential
4. Your recommendation for short-term vs long-term investment
"""
        return prompt
    
    def extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract potential stock symbols from text"""
        import re
        
        # Common stock symbol patterns
        words = text.upper().split()
        
        # Look for known patterns
        symbols = []
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Stock symbols are typically 1-5 uppercase letters
            if clean_word.isalpha() and 1 <= len(clean_word) <= 5:
                symbols.append(clean_word)
        
        return symbols


# Create global instance
llm_manager = LLMManager()