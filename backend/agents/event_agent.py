# Fetches and summarizes financial/news events

"""
Event Agent
Monitors and summarizes financial news and events
Uses news_fetcher utility
"""

from typing import Dict, Optional
import logging
from backend.utils import news_fetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventAgent:
    """Agent responsible for news and events monitoring"""
    
    def __init__(self):
        self.name = "EventAgent"
        self.description = "Monitors financial news and events with sentiment analysis"
    
    def analyze(self, symbol: str, days: int = 7, use_llm: bool = False) -> Dict:
        """
        Analyze news and events for a symbol
        
        Args:
            symbol: Stock ticker
            days: Number of days to look back
            use_llm: Whether to use LLM for sentiment analysis
        
        Returns:
            Dict with news analysis results
        """
        try:
            logger.info(f"EventAgent analyzing news for {symbol}")
            
            # Get news summary
            news_data = news_fetcher.get_news_summary(symbol, days, use_llm)
            
            if news_data['status'] == 'error':
                return {
                    'agent': self.name,
                    'status': 'error',
                    'message': news_data['message'],
                    'symbol': symbol
                }
            
            # Analyze sentiment impact
            sentiment_impact = self._analyze_sentiment_impact(news_data)
            
            # Identify key events
            key_events = self._identify_key_events(news_data.get('events', []))
            
            # Generate event summary
            event_summary = self._create_event_summary(
                symbol,
                news_data,
                sentiment_impact,
                key_events
            )
            
            return {
                'agent': self.name,
                'status': 'success',
                'symbol': symbol,
                'article_count': news_data['article_count'],
                'avg_sentiment': news_data['avg_sentiment'],
                'sentiment_label': news_data['sentiment_label'],
                'sentiment_distribution': news_data['sentiment_distribution'],
                'sentiment_impact': sentiment_impact,
                'events': key_events,
                'articles': news_data['articles'][:5],  # Top 5 articles
                'summary': event_summary
            }
            
        except Exception as e:
            logger.error(f"EventAgent error for {symbol}: {str(e)}")
            return {
                'agent': self.name,
                'status': 'error',
                'message': str(e),
                'symbol': symbol
            }
    
    def _analyze_sentiment_impact(self, news_data: Dict) -> Dict:
        """Analyze the potential impact of sentiment on stock"""
        try:
            avg_sentiment = news_data.get('avg_sentiment', 0.0)
            sentiment_label = news_data.get('sentiment_label', 'Neutral')
            distribution = news_data.get('sentiment_distribution', {})
            
            impact = {
                'score': avg_sentiment,
                'label': sentiment_label,
                'strength': 'Weak',
                'direction': 'Neutral',
                'potential_impact': 'Minimal'
            }
            
            # Determine strength based on distribution
            total = sum(distribution.values())
            if total > 0:
                dominant_pct = max(distribution.values()) / total
                
                if dominant_pct > 0.7:
                    impact['strength'] = 'Strong'
                elif dominant_pct > 0.5:
                    impact['strength'] = 'Moderate'
            
            # Determine direction and potential impact
            if abs(avg_sentiment) > 0.3:
                impact['strength'] = 'Strong'
                impact['potential_impact'] = 'Significant'
            elif abs(avg_sentiment) > 0.15:
                impact['strength'] = 'Moderate'
                impact['potential_impact'] = 'Moderate'
            
            if avg_sentiment > 0.05:
                impact['direction'] = 'Positive'
            elif avg_sentiment < -0.05:
                impact['direction'] = 'Negative'
            
            return impact
            
        except Exception as e:
            logger.error(f"Sentiment impact analysis error: {e}")
            return {
                'score': 0.0,
                'label': 'Neutral',
                'strength': 'Weak',
                'potential_impact': 'Minimal'
            }
    
    def _identify_key_events(self, events: list) -> list:
        """Identify and prioritize key events"""
        try:
            if not events:
                return []
            
            # Sort events by importance
            sorted_events = sorted(
                events, 
                key=lambda x: x.get('importance', 0), 
                reverse=True
            )
            
            # Categorize events
            key_events = {
                'high_impact': [],
                'medium_impact': [],
                'low_impact': []
            }
            
            for event in sorted_events[:10]:  # Top 10 events
                importance = event.get('importance', 0)
                
                if importance >= 0.8:
                    key_events['high_impact'].append(event)
                elif importance >= 0.6:
                    key_events['medium_impact'].append(event)
                else:
                    key_events['low_impact'].append(event)
            
            return key_events
            
        except Exception as e:
            logger.error(f"Key events identification error: {e}")
            return {'high_impact': [], 'medium_impact': [], 'low_impact': []}
    
    def _create_event_summary(self, symbol: str, news_data: Dict,
                             sentiment_impact: Dict, key_events: Dict) -> str:
        """Create human-readable event summary"""
        try:
            article_count = news_data.get('article_count', 0)
            sentiment_label = sentiment_impact.get('label', 'Neutral')
            strength = sentiment_impact.get('strength', 'Weak')
            
            if article_count == 0:
                return f"No recent news found for {symbol}."
            
            summary = f"Analyzed {article_count} recent articles for {symbol}. "
            summary += f"Overall sentiment is {sentiment_label.lower()} with {strength.lower()} conviction. "
            
            # Add sentiment impact
            impact = sentiment_impact.get('potential_impact', 'Minimal')
            summary += f"Sentiment may have {impact.lower()} impact on stock price. "
            
            # Add key events
            if isinstance(key_events, dict):
                high_impact = len(key_events.get('high_impact', []))
                if high_impact > 0:
                    summary += f"Identified {high_impact} high-impact event(s) that warrant attention. "
            
            return summary
            
        except Exception as e:
            logger.error(f"Event summary creation error: {e}")
            return f"Event analysis completed for {symbol}."


# Create global instance
event_agent = EventAgent()