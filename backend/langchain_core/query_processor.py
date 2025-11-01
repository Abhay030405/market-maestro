"""
Query Processor
Processes natural language queries and orchestrates agent responses
"""

import logging
from typing import Dict, List
from backend.langchain_core.llm_manager import llm_manager, QueryIntent
from backend.agents import orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryProcessor:
    """Processes natural language queries and returns AI insights"""
    
    def __init__(self):
        self.llm = llm_manager
    
    def process_query(self, user_query: str) -> Dict:
        """
        Process natural language query end-to-end
        
        Args:
            user_query: User's natural language question
        
        Returns:
            Complete response with analysis and natural language answer
        """
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Step 1: Parse query intent
            intent = self.llm.parse_query(user_query)
            logger.info(f"Parsed intent: {intent.intent_type}, Symbols: {intent.symbols}")
            
            # Step 2: Route to appropriate agents
            routing = self.llm.route_query(intent)
            logger.info(f"Routing to agents: {routing['agents_to_call']}")
            
            # Step 3: Execute agent workflow
            analysis_results = self._execute_agents(intent, routing)
            
            # Step 4: Generate natural language response
            nl_response = self._generate_natural_response(user_query, intent, analysis_results)
            
            # Step 5: Compile final response
            final_response = {
                'status': 'success',
                'query': user_query,
                'intent': intent.dict(),
                'analysis': analysis_results,
                'natural_language_response': nl_response,
                'actions_taken': routing['agents_to_call']
            }
            
            return final_response
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                'status': 'error',
                'query': user_query,
                'message': str(e)
            }
    
    def _execute_agents(self, intent: QueryIntent, routing: Dict) -> Dict:
        """Execute the appropriate agents based on routing"""
        
        results = {}
        symbols = intent.symbols
        
        if not symbols:
            return {'error': 'No valid stock symbols found in query'}
        
        # Handle different query types
        if intent.intent_type == "compare_stocks" and len(symbols) >= 2:
            # Compare two stocks
            results = self._compare_stocks(symbols[0], symbols[1], intent.user_goal)
        
        elif intent.intent_type in ["optimize_portfolio", "analyze_portfolio"] and len(symbols) >= 2:
            # Portfolio analysis
            results = self._analyze_portfolio(symbols, intent.user_goal)
        
        elif intent.intent_type == "analyze_stock" and len(symbols) >= 1:
            # Single stock analysis
            results = self._analyze_single_stock(symbols[0], intent.user_goal)
        
        elif intent.intent_type == "market_news" and len(symbols) >= 1:
            # News only
            results = self._get_news_only(symbols[0])
        
        elif intent.intent_type == "risk_analysis" and len(symbols) >= 1:
            # Risk only
            results = self._get_risk_only(symbols, intent.user_goal)
        
        else:
            # Default: analyze first symbol
            results = self._analyze_single_stock(symbols[0] if symbols else "AAPL", intent.user_goal)
        
        return results
    
    def _analyze_single_stock(self, symbol: str, user_goal: str = None) -> Dict:
        """Analyze a single stock"""
        try:
            result = orchestrator.analyze_stock(
                symbol=symbol,
                user_goal=user_goal
            )
            return result
        except Exception as e:
            logger.error(f"Single stock analysis error: {e}")
            return {'error': str(e)}
    
    def _compare_stocks(self, symbol1: str, symbol2: str, user_goal: str = None) -> Dict:
        """Compare two stocks"""
        try:
            # Analyze both stocks
            analysis1 = orchestrator.analyze_stock(symbol=symbol1, user_goal=user_goal)
            analysis2 = orchestrator.analyze_stock(symbol=symbol2, user_goal=user_goal)
            
            # Generate comparison
            comparison_prompt = self.llm.create_comparison_prompt(
                symbol1, symbol2, analysis1, analysis2
            )
            
            comparison_text = self.llm.generate_response(comparison_prompt)
            
            return {
                'comparison_type': 'two_stocks',
                'symbol1': symbol1,
                'symbol2': symbol2,
                'analysis1': analysis1,
                'analysis2': analysis2,
                'comparison_summary': comparison_text
            }
        except Exception as e:
            logger.error(f"Stock comparison error: {e}")
            return {'error': str(e)}
    
    def _analyze_portfolio(self, symbols: List[str], user_goal: str = None) -> Dict:
        """Analyze portfolio"""
        try:
            result = orchestrator.analyze_portfolio(
                symbols=symbols,
                user_goal=user_goal,
                optimization_method='risk_parity'
            )
            return result
        except Exception as e:
            logger.error(f"Portfolio analysis error: {e}")
            return {'error': str(e)}
    
    def _get_news_only(self, symbol: str) -> Dict:
        """Get news and events only"""
        try:
            from backend.agents import event_agent
            result = event_agent.analyze(symbol, days=7)
            return {'news_analysis': result}
        except Exception as e:
            logger.error(f"News analysis error: {e}")
            return {'error': str(e)}
    
    def _get_risk_only(self, symbols: List[str], user_goal: str = None) -> Dict:
        """Get risk analysis only"""
        try:
            from backend.agents import risk_agent
            
            if len(symbols) == 1:
                result = risk_agent.analyze_single_asset(symbols[0])
            else:
                # Equal weights for simplicity
                weights = [1.0 / len(symbols)] * len(symbols)
                result = risk_agent.analyze_portfolio(symbols, weights)
            
            return {'risk_analysis': result}
        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            return {'error': str(e)}
    
    def _generate_natural_response(self, query: str, intent: QueryIntent, 
                                   analysis: Dict) -> str:
        """Generate natural language response from analysis"""
        
        try:
            # Extract key insights
            if 'comparison_summary' in analysis:
                # Comparison query
                return analysis['comparison_summary']
            
            elif 'error' in analysis:
                return f"I encountered an issue analyzing your query: {analysis['error']}"
            
            # Build context for response generation
            context = {
                'query': query,
                'intent': intent.intent_type,
                'symbols': ', '.join(intent.symbols) if intent.symbols else 'N/A'
            }
            
            # Add analysis summaries
            if 'executive_summary' in analysis:
                context['summary'] = analysis['executive_summary']
            
            # Generate response
            prompt = f"""The user asked: "{query}"

Based on comprehensive analysis, provide a clear, concise answer that:
1. Directly answers their question
2. Highlights the most important insights
3. Provides actionable recommendations
4. Uses simple language

Analysis summary: {context.get('summary', 'Analysis completed successfully.')}
"""
            
            response = self.llm.generate_response(prompt, context)
            return response
            
        except Exception as e:
            logger.error(f"Natural response generation error: {e}")
            return "Analysis completed. Please review the detailed results above."


# Create global instance
query_processor = QueryProcessor()