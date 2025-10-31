# Optimizes asset allocation

"""
Portfolio Agent
Suggests optimal asset allocation strategies
Uses risk_analyzer and data_fetcher utilities
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging
from backend.utils import risk_analyzer, data_fetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioAgent:
    """Agent responsible for portfolio optimization and allocation"""
    
    def __init__(self):
        self.name = "PortfolioAgent"
        self.description = "Generates optimal portfolio allocation strategies"
    
    def optimize(self, symbols: List[str], method: str = 'equal_weight',
                constraints: Optional[Dict] = None,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> Dict:
        """
        Generate optimal portfolio allocation
        
        Args:
            symbols: List of stock tickers
            method: Optimization method ('equal_weight', 'risk_parity', 'max_sharpe')
            constraints: Allocation constraints (max_weight, min_weight, etc.)
            start_date: Start date for analysis
            end_date: End date for analysis
        
        Returns:
            Dict with portfolio allocation and metrics
        """
        try:
            logger.info(f"PortfolioAgent optimizing portfolio: {symbols}, method: {method}")
            
            # Fetch data for all symbols
            returns_data = {}
            price_data = {}
            
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
                price_data[symbol] = df['Close'].iloc[-1]
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Apply constraints
            if constraints is None:
                constraints = {}
            
            # Generate allocation based on method
            if method == 'equal_weight':
                allocation = self._equal_weight_allocation(symbols)
            elif method == 'risk_parity':
                allocation = self._risk_parity_allocation(returns_df, constraints)
            elif method == 'max_sharpe':
                allocation = self._max_sharpe_allocation(returns_df, constraints)
            elif method == 'min_variance':
                allocation = self._min_variance_allocation(returns_df, constraints)
            else:
                allocation = self._equal_weight_allocation(symbols)
            
            # Calculate portfolio metrics
            weights = [allocation['weights'][symbol] for symbol in symbols]
            portfolio_metrics = risk_analyzer.compute_portfolio_risk(returns_df, weights)
            
            # Calculate expected returns for each asset
            expected_returns = self._calculate_expected_returns(returns_df)
            
            # Generate allocation recommendations
            recommendations = self._generate_allocation_recommendations(
                allocation, portfolio_metrics, expected_returns
            )
            
            # Create summary
            summary = self._create_allocation_summary(
                symbols, allocation, portfolio_metrics, method
            )
            
            return {
                'agent': self.name,
                'status': 'success',
                'symbols': symbols,
                'method': method,
                'allocation': allocation,
                'portfolio_metrics': portfolio_metrics.get('portfolio_metrics', {}),
                'expected_returns': expected_returns,
                'current_prices': price_data,
                'recommendations': recommendations,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"PortfolioAgent error: {str(e)}")
            return {
                'agent': self.name,
                'status': 'error',
                'message': str(e),
                'symbols': symbols
            }
    
    def _equal_weight_allocation(self, symbols: List[str]) -> Dict:
        """Equal weight allocation (1/n for each asset)"""
        n = len(symbols)
        weight = 1.0 / n
        
        allocation = {
            'weights': {symbol: weight for symbol in symbols},
            'method': 'Equal Weight',
            'rationale': f'Each asset receives equal weight of {weight*100:.2f}%'
        }
        
        return allocation
    
    def _risk_parity_allocation(self, returns_df: pd.DataFrame, 
                                constraints: Dict) -> Dict:
        """Risk parity allocation (equal risk contribution)"""
        try:
            # Calculate volatilities
            volatilities = returns_df.std()
            
            # Inverse volatility weighting
            inv_vol = 1 / volatilities
            weights_raw = inv_vol / inv_vol.sum()
            
            # Apply constraints
            max_weight = constraints.get('max_weight', 1.0)
            min_weight = constraints.get('min_weight', 0.0)
            
            weights = weights_raw.clip(lower=min_weight, upper=max_weight)
            weights = weights / weights.sum()  # Renormalize
            
            allocation = {
                'weights': weights.to_dict(),
                'method': 'Risk Parity',
                'rationale': 'Assets weighted inversely to their volatility for equal risk contribution'
            }
            
            return allocation
            
        except Exception as e:
            logger.error(f"Risk parity allocation error: {e}")
            return self._equal_weight_allocation(returns_df.columns.tolist())
    
    def _max_sharpe_allocation(self, returns_df: pd.DataFrame,
                               constraints: Dict) -> Dict:
        """Maximum Sharpe ratio allocation (simplified)"""
        try:
            # Calculate mean returns and covariance
            mean_returns = returns_df.mean()
            
            # Simple approach: weight by return/volatility ratio
            volatilities = returns_df.std()
            sharpe_ratios = mean_returns / volatilities
            
            # Positive Sharpe ratios only
            sharpe_ratios = sharpe_ratios.clip(lower=0)
            
            if sharpe_ratios.sum() > 0:
                weights = sharpe_ratios / sharpe_ratios.sum()
            else:
                return self._equal_weight_allocation(returns_df.columns.tolist())
            
            # Apply constraints
            max_weight = constraints.get('max_weight', 1.0)
            min_weight = constraints.get('min_weight', 0.0)
            
            weights = weights.clip(lower=min_weight, upper=max_weight)
            weights = weights / weights.sum()
            
            allocation = {
                'weights': weights.to_dict(),
                'method': 'Maximum Sharpe',
                'rationale': 'Assets weighted to maximize risk-adjusted returns'
            }
            
            return allocation
            
        except Exception as e:
            logger.error(f"Max Sharpe allocation error: {e}")
            return self._equal_weight_allocation(returns_df.columns.tolist())
    
    def _min_variance_allocation(self, returns_df: pd.DataFrame,
                                 constraints: Dict) -> Dict:
        """Minimum variance allocation"""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_df.cov()
            
            # Inverse of covariance matrix
            inv_cov = np.linalg.inv(cov_matrix.values)
            
            # Minimum variance weights
            ones = np.ones(len(returns_df.columns))
            weights = inv_cov @ ones
            weights = weights / weights.sum()
            
            # Convert to pandas Series
            weights = pd.Series(weights, index=returns_df.columns)
            
            # Apply constraints
            max_weight = constraints.get('max_weight', 1.0)
            min_weight = constraints.get('min_weight', 0.0)
            
            weights = weights.clip(lower=min_weight, upper=max_weight)
            weights = weights / weights.sum()
            
            allocation = {
                'weights': weights.to_dict(),
                'method': 'Minimum Variance',
                'rationale': 'Assets weighted to minimize portfolio volatility'
            }
            
            return allocation
            
        except Exception as e:
            logger.error(f"Min variance allocation error: {e}")
            return self._equal_weight_allocation(returns_df.columns.tolist())
    
    def _calculate_expected_returns(self, returns_df: pd.DataFrame) -> Dict:
        """Calculate expected returns for each asset"""
        try:
            # Annualized returns
            mean_returns = returns_df.mean() * 252
            
            # Volatility
            volatilities = returns_df.std() * np.sqrt(252)
            
            expected = {}
            for symbol in returns_df.columns:
                expected[symbol] = {
                    'expected_return': float(mean_returns[symbol]),
                    'volatility': float(volatilities[symbol]),
                    'return_pct': float(mean_returns[symbol] * 100)
                }
            
            return expected
            
        except Exception as e:
            logger.error(f"Expected returns calculation error: {e}")
            return {}
    
    def _generate_allocation_recommendations(self, allocation: Dict,
                                            portfolio_metrics: Dict,
                                            expected_returns: Dict) -> List[str]:
        """Generate portfolio allocation recommendations"""
        recommendations = []
        
        weights = allocation.get('weights', {})
        metrics = portfolio_metrics.get('portfolio_metrics', {})
        
        # Check concentration
        max_weight_val = max(weights.values()) if weights else 0
        if max_weight_val > 0.4:
            recommendations.append(f'Portfolio is concentrated in one asset ({max_weight_val*100:.1f}%) - consider diversification')
        
        # Check Sharpe ratio
        sharpe = metrics.get('sharpe_ratio')
        if sharpe and sharpe > 1.5:
            recommendations.append('Excellent risk-adjusted returns - allocation appears well-optimized')
        elif sharpe and sharpe < 0.8:
            recommendations.append('Consider rebalancing to improve risk-adjusted returns')
        
        # Check volatility
        vol = metrics.get('volatility')
        if vol and vol > 0.25:
            recommendations.append('High portfolio volatility - consider adding lower-risk assets')
        
        # Asset-specific recommendations
        for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]:
            if weight > 0.1:  # Only mention significant allocations
                exp_ret = expected_returns.get(symbol, {}).get('return_pct', 0)
                recommendations.append(f'{symbol}: {weight*100:.1f}% allocation with {exp_ret:.1f}% expected annual return')
        
        return recommendations
    
    def _create_allocation_summary(self, symbols: List[str], allocation: Dict,
                                   portfolio_metrics: Dict, method: str) -> str:
        """Create allocation summary text"""
        weights = allocation.get('weights', {})
        metrics = portfolio_metrics.get('portfolio_metrics', {})
        
        summary = f"Generated portfolio allocation for {len(symbols)} assets using {method} method. "
        
        # Top holdings
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_weights[:3]
        
        summary += "Top allocations: "
        summary += ", ".join([f"{symbol} ({weight*100:.1f}%)" for symbol, weight in top_3])
        summary += ". "
        
        # Portfolio metrics
        sharpe = metrics.get('sharpe_ratio')
        vol = metrics.get('volatility')
        
        if sharpe:
            summary += f"Expected Sharpe ratio: {sharpe:.2f}. "
        if vol:
            summary += f"Portfolio volatility: {vol*100:.1f}%. "
        
        return summary


# Create global instance
portfolio_agent = PortfolioAgent()