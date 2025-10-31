# Analyzes stock trends

"""
Market Agent
Analyzes stock trends, technical indicators, and price movements
Uses data_fetcher and indicators utilities
"""

from typing import Dict, List, Optional
import pandas as pd
import logging
from backend.utils import data_fetcher, indicator_calculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketAgent:
    """Agent responsible for market data analysis and technical indicators"""
    
    def __init__(self):
        self.name = "MarketAgent"
        self.description = "Analyzes stock price trends and technical indicators"
    
    def analyze(self, symbol: str, start_date: Optional[str] = None, 
                end_date: Optional[str] = None) -> Dict:
        """
        Perform comprehensive market analysis for a symbol
        
        Args:
            symbol: Stock ticker
            start_date: Start date for analysis
            end_date: End date for analysis
        
        Returns:
            Dict with market analysis results
        """
        try:
            logger.info(f"MarketAgent analyzing {symbol}")
            
            # Fetch stock data
            stock_data = data_fetcher.get_stock_data(symbol, start_date, end_date)
            
            if stock_data['status'] == 'error':
                return {
                    'agent': self.name,
                    'status': 'error',
                    'message': stock_data['message'],
                    'symbol': symbol
                }
            
            # Convert to DataFrame for indicator calculation
            df = pd.DataFrame(stock_data['data'])
            
            # Compute all technical indicators
            indicators = indicator_calculator.get_all_indicators(df)
            
            # Analyze trend
            trend_analysis = self._analyze_trend(df, indicators)
            
            # Analyze momentum
            momentum_analysis = self._analyze_momentum(indicators)
            
            # Analyze volatility
            volatility_analysis = self._analyze_volatility(indicators)
            
            # Generate trading signals
            signals = self._generate_signals(indicators)
            
            # Create comprehensive summary
            summary = self._create_summary(
                symbol, 
                stock_data, 
                indicators,
                trend_analysis,
                momentum_analysis,
                volatility_analysis,
                signals
            )
            
            return {
                'agent': self.name,
                'status': 'success',
                'symbol': symbol,
                'last_price': stock_data['last_price'],
                'company_name': stock_data.get('company_name', symbol),
                'data_points': stock_data['data_points'],
                'indicators': indicators,
                'trend_analysis': trend_analysis,
                'momentum_analysis': momentum_analysis,
                'volatility_analysis': volatility_analysis,
                'signals': signals,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"MarketAgent error for {symbol}: {str(e)}")
            return {
                'agent': self.name,
                'status': 'error',
                'message': str(e),
                'symbol': symbol
            }
    
    def _analyze_trend(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze price trend using moving averages"""
        try:
            current_price = indicators.get('current_price')
            sma_50 = indicators.get('sma_50')
            sma_200 = indicators.get('sma_200')
            
            trend = {
                'direction': 'Unknown',
                'strength': 'Neutral',
                'short_term': 'Neutral',
                'long_term': 'Neutral'
            }
            
            if current_price and sma_50:
                if current_price > sma_50:
                    trend['short_term'] = 'Bullish'
                else:
                    trend['short_term'] = 'Bearish'
            
            if current_price and sma_200:
                if current_price > sma_200:
                    trend['long_term'] = 'Bullish'
                else:
                    trend['long_term'] = 'Bearish'
            
            # Overall trend
            if sma_50 and sma_200:
                if sma_50 > sma_200:
                    trend['direction'] = 'Uptrend'
                    trend['strength'] = 'Strong' if current_price > sma_50 else 'Moderate'
                else:
                    trend['direction'] = 'Downtrend'
                    trend['strength'] = 'Strong' if current_price < sma_50 else 'Moderate'
            
            return trend
            
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return {'direction': 'Unknown', 'strength': 'Neutral'}
    
    def _analyze_momentum(self, indicators: Dict) -> Dict:
        """Analyze momentum indicators"""
        try:
            rsi = indicators.get('rsi_14')
            macd = indicators.get('macd', {})
            
            momentum = {
                'rsi_signal': 'Neutral',
                'rsi_value': rsi,
                'macd_signal': 'Neutral',
                'overall': 'Neutral'
            }
            
            # RSI analysis
            if rsi:
                if rsi > 70:
                    momentum['rsi_signal'] = 'Overbought'
                elif rsi < 30:
                    momentum['rsi_signal'] = 'Oversold'
                elif rsi > 50:
                    momentum['rsi_signal'] = 'Bullish'
                else:
                    momentum['rsi_signal'] = 'Bearish'
            
            # MACD analysis
            if macd and 'histogram' in macd:
                histogram = macd['histogram']
                if histogram and histogram > 0:
                    momentum['macd_signal'] = 'Bullish'
                elif histogram and histogram < 0:
                    momentum['macd_signal'] = 'Bearish'
            
            # Overall momentum
            bullish_count = sum([
                momentum['rsi_signal'] in ['Bullish', 'Oversold'],
                momentum['macd_signal'] == 'Bullish'
            ])
            
            if bullish_count >= 2:
                momentum['overall'] = 'Strong Bullish'
            elif bullish_count == 1:
                momentum['overall'] = 'Moderately Bullish'
            elif momentum['rsi_signal'] == 'Overbought':
                momentum['overall'] = 'Bearish'
            
            return momentum
            
        except Exception as e:
            logger.error(f"Momentum analysis error: {e}")
            return {'overall': 'Neutral'}
    
    def _analyze_volatility(self, indicators: Dict) -> Dict:
        """Analyze volatility metrics"""
        try:
            volatility = indicators.get('volatility')
            atr = indicators.get('atr_14')
            
            vol_analysis = {
                'level': 'Unknown',
                'volatility': volatility,
                'atr': atr,
                'risk_assessment': 'Moderate'
            }
            
            if volatility:
                if volatility > 0.40:
                    vol_analysis['level'] = 'Very High'
                    vol_analysis['risk_assessment'] = 'High Risk'
                elif volatility > 0.30:
                    vol_analysis['level'] = 'High'
                    vol_analysis['risk_assessment'] = 'Above Average Risk'
                elif volatility > 0.20:
                    vol_analysis['level'] = 'Moderate'
                    vol_analysis['risk_assessment'] = 'Moderate Risk'
                else:
                    vol_analysis['level'] = 'Low'
                    vol_analysis['risk_assessment'] = 'Low Risk'
            
            return vol_analysis
            
        except Exception as e:
            logger.error(f"Volatility analysis error: {e}")
            return {'level': 'Unknown', 'risk_assessment': 'Moderate'}
    
    def _generate_signals(self, indicators: Dict) -> Dict:
        """Generate trading signals based on indicators"""
        try:
            signals = {
                'buy_signals': [],
                'sell_signals': [],
                'neutral_signals': [],
                'overall_signal': 'Hold',
                'confidence': 0.5
            }
            
            rsi = indicators.get('rsi_14')
            current_price = indicators.get('current_price')
            sma_50 = indicators.get('sma_50')
            sma_200 = indicators.get('sma_200')
            
            # RSI signals
            if rsi:
                if rsi < 30:
                    signals['buy_signals'].append('RSI oversold (< 30)')
                elif rsi > 70:
                    signals['sell_signals'].append('RSI overbought (> 70)')
            
            # Moving average signals
            if current_price and sma_50 and sma_200:
                if sma_50 > sma_200 and current_price > sma_50:
                    signals['buy_signals'].append('Golden cross with price above SMA50')
                elif sma_50 < sma_200 and current_price < sma_50:
                    signals['sell_signals'].append('Death cross with price below SMA50')
            
            # Calculate overall signal
            buy_count = len(signals['buy_signals'])
            sell_count = len(signals['sell_signals'])
            
            if buy_count > sell_count:
                signals['overall_signal'] = 'Buy'
                signals['confidence'] = min(0.9, 0.5 + (buy_count * 0.15))
            elif sell_count > buy_count:
                signals['overall_signal'] = 'Sell'
                signals['confidence'] = min(0.9, 0.5 + (sell_count * 0.15))
            else:
                signals['overall_signal'] = 'Hold'
                signals['confidence'] = 0.5
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return {'overall_signal': 'Hold', 'confidence': 0.5}
    
    def _create_summary(self, symbol: str, stock_data: Dict, indicators: Dict,
                       trend: Dict, momentum: Dict, volatility: Dict, 
                       signals: Dict) -> str:
        """Create human-readable summary"""
        try:
            price = stock_data['last_price']
            company = stock_data.get('company_name', symbol)
            
            summary = f"{company} ({symbol}) is currently trading at ${price:.2f}. "
            
            # Trend summary
            summary += f"The stock is in a {trend['direction'].lower()} with {trend['strength'].lower()} strength. "
            
            # Momentum summary
            summary += f"Momentum indicators show {momentum['overall'].lower()} sentiment. "
            
            # Volatility summary
            summary += f"Volatility is {volatility['level'].lower()}, indicating {volatility['risk_assessment'].lower()}. "
            
            # Signal summary
            summary += f"Technical signals suggest a '{signals['overall_signal']}' position with {signals['confidence']:.0%} confidence."
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary creation error: {e}")
            return f"Analysis completed for {symbol} with limited data."


# Create global instance
market_agent = MarketAgent()