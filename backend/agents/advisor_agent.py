"""
Advisor Agent
Synthesizes all agent outputs into natural language recommendations
Uses Gemini LLM for generating human-friendly advice
"""

from typing import Dict, List, Optional
import logging
import google.generativeai as genai
from backend.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvisorAgent:
    """Agent responsible for synthesizing insights and generating advice"""
    
    def __init__(self):
        self.name = "AdvisorAgent"
        self.description = "Synthesizes agent outputs into actionable recommendations"
        
        # Initialize Gemini
        self.model = None
        if settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != "your_gemini_api_key_here":
            try:
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                # Using latest Gemini Flash model
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("AdvisorAgent: Gemini initialized")
            except Exception as e:
                logger.warning(f"AdvisorAgent: Gemini init failed - {e}")
    
    def synthesize(self, agent_results: Dict, user_goal: Optional[str] = None) -> Dict:
        """
        Synthesize results from all agents into unified advice
        
        Args:
            agent_results: Dict containing results from all agents
            user_goal: User's investment goal/objective
        
        Returns:
            Dict with synthesized recommendations
        """
        try:
            logger.info("AdvisorAgent synthesizing results")
            
            # Extract results from each agent
            market_result = agent_results.get('market', {})
            event_result = agent_results.get('events', {})
            risk_result = agent_results.get('risk', {})
            portfolio_result = agent_results.get('portfolio', {})
            
            # Create structured analysis
            structured_analysis = self._create_structured_analysis(
                market_result, event_result, risk_result, portfolio_result
            )
            
            # Generate recommendations using LLM (if available)
            if self.model:
                llm_advice = self._generate_llm_advice(
                    structured_analysis, user_goal
                )
            else:
                llm_advice = self._generate_rule_based_advice(
                    structured_analysis, user_goal
                )
            
            # Generate final recommendations
            final_recommendations = self._compile_final_recommendations(
                structured_analysis, llm_advice
            )
            
            # Create executive summary
            executive_summary = self._create_executive_summary(
                structured_analysis, final_recommendations
            )
            
            return {
                'agent': self.name,
                'status': 'success',
                'structured_analysis': structured_analysis,
                'llm_advice': llm_advice,
                'recommendations': final_recommendations,
                'executive_summary': executive_summary
            }
            
        except Exception as e:
            logger.error(f"AdvisorAgent error: {str(e)}")
            return {
                'agent': self.name,
                'status': 'error',
                'message': str(e)
            }
    
    def _create_structured_analysis(self, market_result: Dict, 
                                    event_result: Dict,
                                    risk_result: Dict,
                                    portfolio_result: Dict) -> Dict:
        """Create structured analysis from agent results"""
        analysis = {
            'market': {
                'status': market_result.get('status', 'unknown'),
                'summary': market_result.get('summary', ''),
                'trend': market_result.get('trend_analysis', {}).get('direction', 'Unknown'),
                'momentum': market_result.get('momentum_analysis', {}).get('overall', 'Neutral'),
                'signals': market_result.get('signals', {})
            },
            'events': {
                'status': event_result.get('status', 'unknown'),
                'summary': event_result.get('summary', ''),
                'sentiment': event_result.get('sentiment_label', 'Neutral'),
                'sentiment_impact': event_result.get('sentiment_impact', {})
            },
            'risk': {
                'status': risk_result.get('status', 'unknown'),
                'summary': risk_result.get('summary', ''),
                'risk_level': risk_result.get('risk_assessment', {}).get('level', 'Moderate'),
                'metrics': risk_result.get('metrics', {})
            },
            'portfolio': {
                'status': portfolio_result.get('status', 'unknown'),
                'summary': portfolio_result.get('summary', ''),
                'allocation': portfolio_result.get('allocation', {}),
                'metrics': portfolio_result.get('portfolio_metrics', {})
            }
        }
        
        return analysis
    
    def _generate_llm_advice(self, analysis: Dict, user_goal: Optional[str]) -> str:
        """Generate advice using Gemini LLM"""
        try:
            # Construct prompt
            prompt = self._build_advisor_prompt(analysis, user_goal)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"LLM advice generation error: {e}")
            return self._generate_rule_based_advice(analysis, user_goal)
    
    def _build_advisor_prompt(self, analysis: Dict, user_goal: Optional[str]) -> str:
        """Build prompt for Gemini LLM"""
        prompt = """You are a professional financial advisor analyzing investment data. 
Based on the following analysis from multiple expert agents, provide clear, actionable advice.

"""
        
        # Add user goal if provided
        if user_goal:
            prompt += f"User's Investment Goal: {user_goal}\n\n"
        
        # Add market analysis
        market = analysis.get('market', {})
        if market.get('status') == 'success':
            prompt += f"MARKET ANALYSIS:\n"
            prompt += f"- Trend: {market.get('trend', 'Unknown')}\n"
            prompt += f"- Momentum: {market.get('momentum', 'Neutral')}\n"
            prompt += f"- Signal: {market.get('signals', {}).get('overall_signal', 'Hold')}\n"
            prompt += f"- Summary: {market.get('summary', '')}\n\n"
        
        # Add event analysis
        events = analysis.get('events', {})
        if events.get('status') == 'success':
            prompt += f"NEWS & EVENTS:\n"
            prompt += f"- Sentiment: {events.get('sentiment', 'Neutral')}\n"
            prompt += f"- Impact: {events.get('sentiment_impact', {}).get('potential_impact', 'Minimal')}\n"
            prompt += f"- Summary: {events.get('summary', '')}\n\n"
        
        # Add risk analysis
        risk = analysis.get('risk', {})
        if risk.get('status') == 'success':
            prompt += f"RISK ASSESSMENT:\n"
            prompt += f"- Risk Level: {risk.get('risk_level', 'Moderate')}\n"
            prompt += f"- Summary: {risk.get('summary', '')}\n\n"
        
        # Add portfolio analysis
        portfolio = analysis.get('portfolio', {})
        if portfolio.get('status') == 'success':
            prompt += f"PORTFOLIO RECOMMENDATION:\n"
            prompt += f"- Summary: {portfolio.get('summary', '')}\n\n"
        
        prompt += """
Based on this comprehensive analysis, provide:

1. INVESTMENT RECOMMENDATION: Clear buy/hold/sell recommendation with rationale (2-3 sentences)
2. KEY CONSIDERATIONS: 3-4 bullet points of critical factors
3. ACTION ITEMS: Specific next steps for the investor
4. RISK WARNINGS: Important risks to be aware of

Keep your advice:
- Actionable and specific
- Evidence-based (reference the data above)
- Balanced (mention both opportunities and risks)
- Professional but accessible

Format your response clearly with the 4 sections above.
"""
        
        return prompt
    
    def _generate_rule_based_advice(self, analysis: Dict, 
                                    user_goal: Optional[str]) -> str:
        """Generate advice using rule-based logic (fallback)"""
        advice = []
        
        # Market-based advice
        market = analysis.get('market', {})
        if market.get('status') == 'success':
            signal = market.get('signals', {}).get('overall_signal', 'Hold')
            trend = market.get('trend', 'Unknown')
            
            if signal == 'Buy' and trend == 'Uptrend':
                advice.append("Market indicators suggest favorable entry conditions with strong upward momentum.")
            elif signal == 'Sell' or trend == 'Downtrend':
                advice.append("Market indicators suggest caution or potential exit due to negative momentum.")
            else:
                advice.append("Market indicators suggest maintaining current positions.")
        
        # Event-based advice
        events = analysis.get('events', {})
        if events.get('status') == 'success':
            sentiment = events.get('sentiment', 'Neutral')
            impact = events.get('sentiment_impact', {}).get('potential_impact', 'Minimal')
            
            if sentiment == 'Bullish' and impact in ['Moderate', 'Significant']:
                advice.append("Positive news sentiment may provide near-term support for prices.")
            elif sentiment == 'Bearish' and impact in ['Moderate', 'Significant']:
                advice.append("Negative news sentiment could create headwinds in the short term.")
        
        # Risk-based advice
        risk = analysis.get('risk', {})
        if risk.get('status') == 'success':
            risk_level = risk.get('risk_level', 'Moderate')
            
            if risk_level in ['High', 'Very High']:
                advice.append("Elevated risk levels suggest limiting position size and using stop-losses.")
            elif risk_level in ['Low', 'Very Low']:
                advice.append("Lower risk profile makes this suitable for conservative portfolios.")
        
        # Portfolio advice
        portfolio = analysis.get('portfolio', {})
        if portfolio.get('status') == 'success':
            advice.append("Review the suggested allocation to optimize your portfolio's risk-return profile.")
        
        return "\n\n".join(advice) if advice else "Insufficient data to generate comprehensive advice."
    
    def _compile_final_recommendations(self, analysis: Dict, 
                                      llm_advice: str) -> List[str]:
        """Compile final list of actionable recommendations"""
        recommendations = []
        
        # Extract key recommendations from each agent
        market = analysis.get('market', {})
        if market.get('status') == 'success':
            signal = market.get('signals', {}).get('overall_signal', 'Hold')
            confidence = market.get('signals', {}).get('confidence', 0.5)
            recommendations.append(f"Technical Signal: {signal} (Confidence: {confidence:.0%})")
        
        events = analysis.get('events', {})
        if events.get('status') == 'success':
            sentiment = events.get('sentiment', 'Neutral')
            recommendations.append(f"News Sentiment: {sentiment}")
        
        risk = analysis.get('risk', {})
        if risk.get('status') == 'success':
            risk_level = risk.get('risk_level', 'Moderate')
            recommendations.append(f"Risk Assessment: {risk_level}")
        
        return recommendations
    
    def _create_executive_summary(self, analysis: Dict, 
                                  recommendations: List[str]) -> str:
        """Create brief executive summary"""
        summary_parts = []
        
        # Overall recommendation
        market_signal = analysis.get('market', {}).get('signals', {}).get('overall_signal', 'Hold')
        summary_parts.append(f"Overall Signal: {market_signal}")
        
        # Risk level
        risk_level = analysis.get('risk', {}).get('risk_level', 'Moderate')
        summary_parts.append(f"Risk Level: {risk_level}")
        
        # Sentiment
        sentiment = analysis.get('events', {}).get('sentiment', 'Neutral')
        summary_parts.append(f"Market Sentiment: {sentiment}")
        
        return " | ".join(summary_parts)


# Create global instance
advisor_agent = AdvisorAgent()