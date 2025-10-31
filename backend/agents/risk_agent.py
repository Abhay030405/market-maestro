"""
Risk Agent
Evaluates portfolio and asset risk metrics
Uses risk_analyzer and data_fetcher utilities
"""

from typing import Dict, List, Optional
import pandas as pd
import logging
from backend.utils import risk_analyzer, data_fetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAgent:
    """Agent responsible for risk evaluation and analysis"""
    
    def __init__(self):
        self.name = "RiskAgent"
        self.description = "Evaluates risk metrics and portfolio risk analysis"
    
    def analyze_single_asset(self, symbol: str, start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict:
        """
        Analyze risk metrics for a single asset
        
        Args:
            symbol: Stock ticker
            start_date: Start date
            end_date: End date
        
        Returns:
            Dict with risk analysis results
        """
        try:
            logger.info(f"RiskAgent analyzing {symbol}")
            
            # Fetch stock data
            stock_data = data_fetcher.get_stock_data(symbol, start_date, end_date)
            
            if stock_data['status'] == 'error':
                return {
                    'agent': self.name,
                    'status': 'error',
                    'message': stock_data['message'],
                    'symbol': symbol
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(stock_data['data'])
            prices = df['Close']
            returns = risk_analyzer.compute_returns(prices)
            
            # Compute risk metrics
            volatility = risk_analyzer.compute_volatility(returns)
            sharpe = risk_analyzer.compute_sharpe_ratio(returns)
            max_dd = risk_analyzer.compute_max_drawdown(prices)
            var_95 = risk_analyzer.compute_var(returns, 0.95)
            cvar_95 = risk_analyzer.compute_cvar(returns, 0.95)
            
            # Assess risk level
            risk_assessment = self._assess_risk_level(volatility, sharpe, max_dd)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(
                risk_assessment, volatility, sharpe, max_dd
            )
            
            # Create summary
            summary = self._create_risk_summary(
                symbol, risk_assessment, volatility, sharpe, max_dd
            )
            
            return {
                'agent': self.name,
                'status': 'success',
                'symbol': symbol,
                'metrics': {
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd['max_drawdown_pct'],
                    'var_95': var_95 * 100 if var_95 else None,
                    'cvar_95': cvar_95 * 100 if cvar_95 else None
                },
                'risk_assessment': risk_assessment,
                'recommendations': recommendations,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"RiskAgent error for {symbol}: {str(e)}")
            return {
                'agent': self.name,
                'status': 'error',
                'message': str(e),
                'symbol': symbol
            }
    
    def analyze_portfolio(self, symbols: List[str], weights: List[float],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict:
        """
        Analyze risk metrics for a portfolio
        
        Args:
            symbols: List of stock tickers
            weights: Portfolio weights (must sum to 1)
            start_date: Start date
            end_date: End date
        
        Returns:
            Dict with portfolio risk analysis
        """
        try:
            logger.info(f"RiskAgent analyzing portfolio: {symbols}")
            
            # Fetch data for all symbols
            returns_data = {}
            
            for symbol in symbols:
                stock_data = data_fetcher.get_stock_data(symbol, start_date, end_date)
                
                if stock_data['status'] == 'error':
                    return {
                        'agent': self.name,
                        'status': 'error',
                        'message': f"Failed to fetch data for {symbol}",
                        'symbols': symbols
                    }
                
                df = pd.DataFrame(stock_data['data'])
                returns = df['Close'].pct_change().dropna()
                returns_data[symbol] = returns
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Compute portfolio risk
            portfolio_risk = risk_analyzer.compute_portfolio_risk(returns_df, weights)
            
            if portfolio_risk['status'] == 'error':
                return {
                    'agent': self.name,
                    'status': 'error',
                    'message': portfolio_risk['message'],
                    'symbols': symbols
                }
            
            # Compute correlation
            correlation = risk_analyzer.compute_correlation_matrix(returns_df)
            
            # Assess diversification
            diversification_assessment = self._assess_diversification(correlation)
            
            # Generate portfolio recommendations
            recommendations = self._generate_portfolio_recommendations(
                portfolio_risk['portfolio_metrics'],
                diversification_assessment
            )
            
            # Create summary
            summary = self._create_portfolio_summary(
                symbols, weights, portfolio_risk['portfolio_metrics'],
                diversification_assessment
            )
            
            return {
                'agent': self.name,
                'status': 'success',
                'symbols': symbols,
                'weights': weights,
                'metrics': portfolio_risk['portfolio_metrics'],
                'correlation': correlation,
                'diversification': diversification_assessment,
                'recommendations': recommendations,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"RiskAgent portfolio error: {str(e)}")
            return {
                'agent': self.name,
                'status': 'error',
                'message': str(e),
                'symbols': symbols
            }
    
    def _assess_risk_level(self, volatility: float, sharpe: float, 
                          max_dd: Dict) -> Dict:
        """Assess overall risk level"""
        try:
            assessment = {
                'level': 'Moderate',
                'score': 5.0,  # 0-10 scale
                'factors': []
            }
            
            score = 5.0  # Start at moderate
            
            # Volatility assessment
            if volatility:
                if volatility > 0.40:
                    score += 2
                    assessment['factors'].append('High volatility')
                elif volatility > 0.30:
                    score += 1
                    assessment['factors'].append('Above average volatility')
                elif volatility < 0.15:
                    score -= 1
                    assessment['factors'].append('Low volatility')
            
            # Sharpe ratio assessment
            if sharpe:
                if sharpe > 1.5:
                    score -= 1.5
                    assessment['factors'].append('Excellent risk-adjusted returns')
                elif sharpe > 1.0:
                    score -= 0.5
                    assessment['factors'].append('Good risk-adjusted returns')
                elif sharpe < 0.5:
                    score += 1
                    assessment['factors'].append('Poor risk-adjusted returns')
            
            # Max drawdown assessment
            max_dd_pct = abs(max_dd.get('max_drawdown_pct', 0))
            if max_dd_pct > 40:
                score += 2
                assessment['factors'].append('Severe historical drawdown')
            elif max_dd_pct > 25:
                score += 1
                assessment['factors'].append('Significant historical drawdown')
            
            # Normalize score
            score = max(0, min(10, score))
            assessment['score'] = round(score, 1)
            
            # Determine level
            if score >= 7.5:
                assessment['level'] = 'Very High'
            elif score >= 6:
                assessment['level'] = 'High'
            elif score >= 4:
                assessment['level'] = 'Moderate'
            elif score >= 2:
                assessment['level'] = 'Low'
            else:
                assessment['level'] = 'Very Low'
            
            return assessment
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {'level': 'Moderate', 'score': 5.0, 'factors': []}
    
    def _assess_diversification(self, correlation: Dict) -> Dict:
        """Assess portfolio diversification"""
        try:
            if correlation['status'] == 'error':
                return {'level': 'Unknown', 'score': 0.5}
            
            div_score = correlation.get('diversification_score', 0.5)
            avg_corr = correlation.get('average_correlation', 0.5)
            
            assessment = {
                'score': div_score,
                'average_correlation': avg_corr,
                'level': 'Moderate'
            }
            
            if div_score > 0.7:
                assessment['level'] = 'Excellent'
            elif div_score > 0.5:
                assessment['level'] = 'Good'
            elif div_score > 0.3:
                assessment['level'] = 'Moderate'
            else:
                assessment['level'] = 'Poor'
            
            return assessment
            
        except Exception as e:
            logger.error(f"Diversification assessment error: {e}")
            return {'level': 'Unknown', 'score': 0.5}
    
    def _generate_risk_recommendations(self, risk_assessment: Dict,
                                      volatility: float, sharpe: float,
                                      max_dd: Dict) -> List[str]:
        """Generate risk-based recommendations"""
        recommendations = []
        
        risk_level = risk_assessment.get('level', 'Moderate')
        
        if risk_level in ['High', 'Very High']:
            recommendations.append('Consider reducing position size due to high risk')
            recommendations.append('Implement stop-loss orders to limit downside')
        
        if volatility and volatility > 0.35:
            recommendations.append('High volatility suggests using wider stop-losses')
        
        if sharpe and sharpe < 0.5:
            recommendations.append('Poor risk-adjusted returns - consider alternatives')
        
        max_dd_pct = abs(max_dd.get('max_drawdown_pct', 0))
        if max_dd_pct > 30:
            recommendations.append(f'Historical drawdown of {max_dd_pct:.1f}% - be prepared for volatility')
        
        if not recommendations:
            recommendations.append('Risk profile appears acceptable for moderate investors')
        
        return recommendations
    
    def _generate_portfolio_recommendations(self, metrics: Dict,
                                           diversification: Dict) -> List[str]:
        """Generate portfolio-specific recommendations"""
        recommendations = []
        
        sharpe = metrics.get('sharpe_ratio')
        volatility = metrics.get('volatility')
        div_level = diversification.get('level', 'Moderate')
        
        if div_level in ['Poor', 'Moderate']:
            recommendations.append('Consider adding more uncorrelated assets for better diversification')
        
        if sharpe and sharpe > 1.5:
            recommendations.append('Excellent risk-adjusted returns - portfolio is well-optimized')
        elif sharpe and sharpe < 0.8:
            recommendations.append('Sub-optimal risk-adjusted returns - review asset allocation')
        
        if volatility and volatility > 0.25:
            recommendations.append('Portfolio volatility is elevated - consider adding defensive assets')
        
        return recommendations
    
    def _create_risk_summary(self, symbol: str, risk_assessment: Dict,
                            volatility: float, sharpe: float, max_dd: Dict) -> str:
        """Create risk summary text"""
        risk_level = risk_assessment.get('level', 'Moderate')
        risk_score = risk_assessment.get('score', 5.0)
        
        summary = f"{symbol} has a {risk_level.lower()} risk level (score: {risk_score}/10). "
        
        if volatility:
            summary += f"Annualized volatility is {volatility*100:.1f}%. "
        
        if sharpe:
            summary += f"Sharpe ratio of {sharpe:.2f} indicates "
            if sharpe > 1:
                summary += "good risk-adjusted returns. "
            else:
                summary += "below-average risk-adjusted returns. "
        
        max_dd_pct = abs(max_dd.get('max_drawdown_pct', 0))
        summary += f"Maximum historical drawdown was {max_dd_pct:.1f}%."
        
        return summary
    
    def _create_portfolio_summary(self, symbols: List[str], weights: List[float],
                                 metrics: Dict, diversification: Dict) -> str:
        """Create portfolio risk summary"""
        summary = f"Portfolio of {len(symbols)} assets "
        summary += f"with {diversification.get('level', 'moderate').lower()} diversification. "
        
        vol = metrics.get('volatility')
        if vol:
            summary += f"Portfolio volatility is {vol*100:.1f}%. "
        
        sharpe = metrics.get('sharpe_ratio')
        if sharpe:
            summary += f"Sharpe ratio of {sharpe:.2f}. "
        
        return summary


# Create global instance
risk_agent = RiskAgent()