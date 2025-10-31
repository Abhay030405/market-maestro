"""
Orchestrator Agent
Coordinates all agents using LangGraph
Manages workflow and aggregates results
"""

from typing import Dict, List, Optional, TypedDict
import logging

from backend.agents.market_agent import market_agent
from backend.agents.event_agent import event_agent
from backend.agents.risk_agent import risk_agent
from backend.agents.portfolio_agent import portfolio_agent
from backend.agents.advisor_agent import advisor_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define state structure for LangGraph
class AgentState(TypedDict):
    """State shared between agents"""
    query: str
    symbols: List[str]
    user_goal: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    market_results: Dict
    event_results: Dict
    risk_results: Dict
    portfolio_results: Dict
    advisor_results: Dict
    final_report: Dict


class Orchestrator:
    """Main orchestrator coordinating all agents"""
    
    def __init__(self):
        self.name = "Orchestrator"
        self.description = "Coordinates multiple agents to provide comprehensive analysis"
    
    def analyze_stock(self, symbol: str, user_goal: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Dict:
        """
        Perform comprehensive stock analysis using all agents
        
        Args:
            symbol: Stock ticker
            user_goal: User's investment goal
            start_date: Start date for analysis
            end_date: End date for analysis
        
        Returns:
            Comprehensive analysis report
        """
        try:
            logger.info(f"Orchestrator: Analyzing {symbol}")
            
            # Initialize state
            initial_state = {
                'query': f"Analyze {symbol}",
                'symbols': [symbol],
                'user_goal': user_goal,
                'start_date': start_date,
                'end_date': end_date,
                'market_results': {},
                'event_results': {},
                'risk_results': {},
                'portfolio_results': {},
                'advisor_results': {},
                'final_report': {}
            }
            
            # Execute workflow
            result = self._execute_single_stock_workflow(initial_state)
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestrator error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def analyze_portfolio(self, symbols: List[str], 
                         weights: Optional[List[float]] = None,
                         user_goal: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         optimization_method: str = 'equal_weight') -> Dict:
        """
        Perform comprehensive portfolio analysis
        
        Args:
            symbols: List of stock tickers
            weights: Portfolio weights (if None, will optimize)
            user_goal: User's investment goal
            start_date: Start date
            end_date: End date
            optimization_method: Method for portfolio optimization
        
        Returns:
            Comprehensive portfolio analysis
        """
        try:
            logger.info(f"Orchestrator: Analyzing portfolio with {len(symbols)} assets")
            
            initial_state = {
                'query': f"Analyze portfolio: {', '.join(symbols)}",
                'symbols': symbols,
                'weights': weights,
                'optimization_method': optimization_method,
                'user_goal': user_goal,
                'start_date': start_date,
                'end_date': end_date,
                'market_results': {},
                'event_results': {},
                'risk_results': {},
                'portfolio_results': {},
                'advisor_results': {},
                'final_report': {}
            }
            
            result = self._execute_portfolio_workflow(initial_state)
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestrator portfolio error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _execute_single_stock_workflow(self, state: Dict) -> Dict:
        """Execute workflow for single stock analysis"""
        try:
            symbol = state['symbols'][0]
            start_date = state.get('start_date')
            end_date = state.get('end_date')
            user_goal = state.get('user_goal')
            
            # Step 1: Market Analysis
            logger.info("Step 1: Running MarketAgent")
            market_results = market_agent.analyze(symbol, start_date, end_date)
            
            # Step 2: Event Analysis (parallel to market)
            logger.info("Step 2: Running EventAgent")
            event_results = event_agent.analyze(symbol, days=7)
            
            # Step 3: Risk Analysis (depends on market data)
            logger.info("Step 3: Running RiskAgent")
            risk_results = risk_agent.analyze_single_asset(symbol, start_date, end_date)
            
            # Step 4: Synthesize with AdvisorAgent
            logger.info("Step 4: Running AdvisorAgent")
            agent_results = {
                'market': market_results,
                'events': event_results,
                'risk': risk_results
            }
            advisor_results = advisor_agent.synthesize(agent_results, user_goal)
            
            # Step 5: Compile final report
            final_report = self._compile_single_stock_report(
                symbol,
                market_results,
                event_results,
                risk_results,
                advisor_results
            )
            
            return final_report
            
        except Exception as e:
            logger.error(f"Single stock workflow error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _execute_portfolio_workflow(self, state: Dict) -> Dict:
        """Execute workflow for portfolio analysis"""
        try:
            symbols = state['symbols']
            weights = state.get('weights')
            method = state.get('optimization_method', 'equal_weight')
            start_date = state.get('start_date')
            end_date = state.get('end_date')
            user_goal = state.get('user_goal')
            
            # Step 1: Portfolio Optimization
            logger.info("Step 1: Running PortfolioAgent")
            portfolio_results = portfolio_agent.optimize(
                symbols, 
                method=method,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get optimized weights
            if weights is None and portfolio_results.get('status') == 'success':
                allocation = portfolio_results.get('allocation', {})
                weights_dict = allocation.get('weights', {})
                weights = [weights_dict.get(s, 1.0/len(symbols)) for s in symbols]
            
            # Step 2: Portfolio Risk Analysis
            logger.info("Step 2: Running RiskAgent for portfolio")
            risk_results = risk_agent.analyze_portfolio(
                symbols, 
                weights,
                start_date,
                end_date
            )
            
            # Step 3: Event Analysis for each symbol (aggregated)
            logger.info("Step 3: Running EventAgent for portfolio")
            event_results = self._aggregate_portfolio_events(symbols)
            
            # Step 4: Synthesize with AdvisorAgent
            logger.info("Step 4: Running AdvisorAgent")
            agent_results = {
                'portfolio': portfolio_results,
                'risk': risk_results,
                'events': event_results
            }
            advisor_results = advisor_agent.synthesize(agent_results, user_goal)
            
            # Step 5: Compile final report
            final_report = self._compile_portfolio_report(
                symbols,
                weights,
                portfolio_results,
                risk_results,
                event_results,
                advisor_results
            )
            
            return final_report
            
        except Exception as e:
            logger.error(f"Portfolio workflow error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _aggregate_portfolio_events(self, symbols: List[str]) -> Dict:
        """Aggregate event analysis for multiple symbols"""
        try:
            all_events = []
            total_sentiment = 0
            article_count = 0
            
            for symbol in symbols[:5]:  # Limit to top 5 to avoid API overuse
                event_result = event_agent.analyze(symbol, days=7)
                
                if event_result.get('status') == 'success':
                    all_events.extend(event_result.get('events', {}).get('high_impact', []))
                    total_sentiment += event_result.get('avg_sentiment', 0)
                    article_count += event_result.get('article_count', 0)
            
            avg_sentiment = total_sentiment / len(symbols) if symbols else 0
            
            # Classify sentiment
            if avg_sentiment >= 0.05:
                sentiment_label = "Bullish"
            elif avg_sentiment <= -0.05:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"
            
            return {
                'status': 'success',
                'article_count': article_count,
                'avg_sentiment': avg_sentiment,
                'sentiment_label': sentiment_label,
                'key_events': all_events[:10],
                'summary': f"Analyzed {article_count} articles across portfolio. Overall sentiment: {sentiment_label}"
            }
            
        except Exception as e:
            logger.error(f"Portfolio events aggregation error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _compile_single_stock_report(self, symbol: str, market_results: Dict,
                                     event_results: Dict, risk_results: Dict,
                                     advisor_results: Dict) -> Dict:
        """Compile comprehensive single stock report"""
        return {
            'status': 'success',
            'report_type': 'single_stock',
            'symbol': symbol,
            'timestamp': self._get_timestamp(),
            'market_analysis': market_results,
            'event_analysis': event_results,
            'risk_analysis': risk_results,
            'advisor_recommendation': advisor_results,
            'executive_summary': advisor_results.get('executive_summary', ''),
            'key_recommendations': advisor_results.get('recommendations', [])
        }
    
    def _compile_portfolio_report(self, symbols: List[str], weights: List[float],
                                  portfolio_results: Dict, risk_results: Dict,
                                  event_results: Dict, advisor_results: Dict) -> Dict:
        """Compile comprehensive portfolio report"""
        return {
            'status': 'success',
            'report_type': 'portfolio',
            'symbols': symbols,
            'weights': weights,
            'timestamp': self._get_timestamp(),
            'portfolio_allocation': portfolio_results,
            'risk_analysis': risk_results,
            'event_analysis': event_results,
            'advisor_recommendation': advisor_results,
            'executive_summary': advisor_results.get('executive_summary', ''),
            'key_recommendations': advisor_results.get('recommendations', [])
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


# Create global instance
orchestrator = Orchestrator()