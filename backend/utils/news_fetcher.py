"""
News & Events Fetcher Module
Fetches news articles, analyzes sentiment, and detects financial events
Uses NewsAPI and VADER sentiment analysis, with Gemini for advanced NLP
"""

import requests
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import google.generativeai as genai
from backend.config import settings, EVENT_KEYWORDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsFetcher:
    """Handles fetching and analyzing financial news"""
    
    def __init__(self):
        # Initialize NewsAPI client
        self.news_api = None
        if settings.NEWS_API_KEY and settings.NEWS_API_KEY != "your_newsapi_key_here":
            try:
                self.news_api = NewsApiClient(api_key=settings.NEWS_API_KEY)
                logger.info("NewsAPI initialized successfully")
            except Exception as e:
                logger.warning(f"NewsAPI initialization failed: {str(e)}")
        
        # Initialize VADER sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize Gemini
        self.gemini_model = None
        if settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != "your_gemini_api_key_here":
            try:
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Gemini initialized successfully")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {str(e)}")
    
    def fetch_news(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 20,
        language: str = 'en'
    ) -> List[Dict]:
        """
        Fetch news articles for a query
        
        Args:
            query: Search query (company name, ticker, keywords)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_results: Maximum number of articles
            language: Article language
        
        Returns:
            List of article dicts
        """
        articles = []
        
        # Set default dates
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        # Try NewsAPI first
        if self.news_api:
            articles.extend(self._fetch_from_newsapi(query, start_date, end_date, max_results, language))
        
        # Fallback to Yahoo Finance news via yfinance if needed
        if not articles:
            articles.extend(self._fetch_from_yfinance(query, max_results))
        
        return articles
    
    def _fetch_from_newsapi(
        self,
        query: str,
        start_date: str,
        end_date: str,
        max_results: int,
        language: str
    ) -> List[Dict]:
        """Fetch articles from NewsAPI"""
        try:
            response = self.news_api.get_everything(
                q=query,
                from_param=start_date,
                to=end_date,
                language=language,
                sort_by='relevancy',
                page_size=min(max_results, 100)
            )
            
            if response['status'] == 'ok':
                articles = []
                for article in response['articles']:
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'published_at': article.get('publishedAt', ''),
                        'content': article.get('content', '')
                    })
                return articles
            
        except Exception as e:
            logger.error(f"NewsAPI error: {str(e)}")
        
        return []
    
    def _fetch_from_yfinance(self, symbol: str, max_results: int) -> List[Dict]:
        """Fetch news from Yahoo Finance via yfinance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            articles = []
            for item in news[:max_results]:
                articles.append({
                    'title': item.get('title', ''),
                    'description': item.get('summary', ''),
                    'url': item.get('link', ''),
                    'source': item.get('publisher', 'Yahoo Finance'),
                    'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                    'content': ''
                })
            
            return articles
            
        except Exception as e:
            logger.error(f"yfinance news error: {str(e)}")
            return []
    
    def analyze_sentiment_vader(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER (rule-based)
        
        Args:
            text: Text to analyze
        
        Returns:
            Dict with sentiment scores
        """
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Classify sentiment
            compound = scores['compound']
            if compound >= 0.05:
                sentiment = "Bullish"
            elif compound <= -0.05:
                sentiment = "Bearish"
            else:
                sentiment = "Neutral"
            
            return {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': compound,
                'sentiment': sentiment,
                'confidence': abs(compound)
            }
            
        except Exception as e:
            logger.error(f"VADER sentiment error: {str(e)}")
            return {
                'sentiment': 'Neutral',
                'compound': 0.0,
                'confidence': 0.0
            }
    
    def analyze_sentiment_llm(self, text: str) -> Dict:
        """
        Analyze sentiment using Gemini LLM (more nuanced)
        
        Args:
            text: Text to analyze
        
        Returns:
            Dict with sentiment analysis
        """
        if not self.gemini_model:
            # Fallback to VADER
            return self.analyze_sentiment_vader(text)
        
        try:
            prompt = f"""Analyze the sentiment of this financial news headline/text:

"{text}"

Respond with:
1. Sentiment: Bullish/Neutral/Bearish
2. Confidence: 0.0 to 1.0
3. Brief reason (one sentence)

Format your response as:
Sentiment: [sentiment]
Confidence: [confidence]
Reason: [reason]
"""
            
            response = self.gemini_model.generate_content(prompt)
            result_text = response.text
            
            # Parse response
            lines = result_text.strip().split('\n')
            sentiment = "Neutral"
            confidence = 0.5
            reason = ""
            
            for line in lines:
                if line.startswith("Sentiment:"):
                    sentiment = line.split(":", 1)[1].strip()
                elif line.startswith("Confidence:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except:
                        confidence = 0.5
                elif line.startswith("Reason:"):
                    reason = line.split(":", 1)[1].strip()
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'reason': reason,
                'method': 'llm'
            }
            
        except Exception as e:
            logger.error(f"LLM sentiment error: {str(e)}")
            return self.analyze_sentiment_vader(text)
    
    def detect_events(self, articles: List[Dict]) -> List[Dict]:
        """
        Detect financial events from article headlines
        
        Args:
            articles: List of article dicts
        
        Returns:
            List of detected events
        """
        events = []
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            text = f"{title} {description}"
            
            # Check for each event type
            for event_type, keywords in EVENT_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in text:
                        events.append({
                            'event_type': event_type,
                            'headline': article.get('title', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('published_at', ''),
                            'source': article.get('source', ''),
                            'importance': self._calculate_importance(event_type, text)
                        })
                        break
        
        # Remove duplicates and sort by importance
        events = list({e['headline']: e for e in events}.values())
        events.sort(key=lambda x: x['importance'], reverse=True)
        
        return events
    
    def _calculate_importance(self, event_type: str, text: str) -> float:
        """Calculate event importance score (0-1)"""
        importance_weights = {
            'earnings': 0.9,
            'merger': 0.95,
            'leadership': 0.7,
            'regulatory': 0.85,
            'dividend': 0.6,
            'product': 0.65
        }
        
        base_score = importance_weights.get(event_type, 0.5)
        
        # Boost for certain keywords
        high_impact_words = ['ceo', 'acquisition', 'sec', 'earnings beat', 'earnings miss']
        for word in high_impact_words:
            if word in text:
                base_score = min(1.0, base_score + 0.1)
        
        return base_score
    
    def get_news_summary(
        self,
        symbol: str,
        days: int = 7,
        use_llm: bool = False
    ) -> Dict:
        """
        Get comprehensive news summary for a symbol
        
        Args:
            symbol: Stock ticker
            days: Number of days to look back
            use_llm: Whether to use LLM for sentiment (slower but more accurate)
        
        Returns:
            Dict with articles, sentiment, and events
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch articles
            articles = self.fetch_news(
                query=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                max_results=20
            )
            
            if not articles:
                return {
                    'status': 'success',
                    'symbol': symbol,
                    'article_count': 0,
                    'articles': [],
                    'avg_sentiment': 0.0,
                    'sentiment_label': 'Neutral',
                    'events': []
                }
            
            # Analyze sentiment for each article
            sentiments = []
            for article in articles:
                text = f"{article['title']} {article['description']}"
                
                if use_llm:
                    sentiment = self.analyze_sentiment_llm(text)
                else:
                    sentiment = self.analyze_sentiment_vader(text)
                
                article['sentiment'] = sentiment
                
                # Collect compound scores for average
                if 'compound' in sentiment:
                    sentiments.append(sentiment['compound'])
                elif sentiment['sentiment'] == 'Bullish':
                    sentiments.append(0.5)
                elif sentiment['sentiment'] == 'Bearish':
                    sentiments.append(-0.5)
                else:
                    sentiments.append(0.0)
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            
            # Classify overall sentiment
            if avg_sentiment >= 0.05:
                sentiment_label = "Bullish"
            elif avg_sentiment <= -0.05:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"
            
            # Detect events
            events = self.detect_events(articles)
            
            return {
                'status': 'success',
                'symbol': symbol,
                'article_count': len(articles),
                'articles': articles[:10],  # Return top 10
                'avg_sentiment': round(avg_sentiment, 3),
                'sentiment_label': sentiment_label,
                'sentiment_distribution': {
                    'bullish': len([s for s in sentiments if s > 0.05]),
                    'neutral': len([s for s in sentiments if -0.05 <= s <= 0.05]),
                    'bearish': len([s for s in sentiments if s < -0.05])
                },
                'events': events[:5],  # Return top 5 events
                'date_range': {
                    'start': start_date.strftime("%Y-%m-%d"),
                    'end': end_date.strftime("%Y-%m-%d")
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting news summary: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'symbol': symbol
            }


# Create global instance
news_fetcher = NewsFetcher()