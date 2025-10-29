"""
Risk Analyzer Module
Computes portfolio and asset risk metrics including volatility, Sharpe ratio,
VaR, CVaR, max drawdown, beta, and correlation analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging
from backend.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Calculate risk metrics for assets and portfolios"""
    
    def __init__(self):
        self.trading_days = settings.TRADING_DAYS_PER_YEAR
        self.risk_free_rate = settings.RISK_FREE_RATE
    
    def compute_returns(self, prices: pd.Series) -> pd.Series:
        """
        Compute returns from price series
        
        Args:
            prices: Series of prices
        
        Returns:
            Series of returns
        """
        return prices.pct_change().dropna()
    
    def compute_volatility(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """
        Compute volatility (standard deviation of returns)
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize the result
        
        Returns:
            Volatility as float
        """
        try:
            vol = returns.std()
            if annualize:
                vol = vol * np.sqrt(self.trading_days)
            return float(vol)
        except Exception as e:
            logger.error(f"Error computing volatility: {str(e)}")
            return None
    
    def compute_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Compute Sharpe ratio (risk-adjusted return)
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default from config)
        
        Returns:
            Sharpe ratio as float
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            # Annualize returns
            mean_return = returns.mean() * self.trading_days
            
            # Annualize volatility
            volatility = self.compute_volatility(returns, annualize=True)
            
            if volatility == 0 or volatility is None:
                return None
            
            # Sharpe ratio
            sharpe = (mean_return - risk_free_rate) / volatility
            
            return float(sharpe)
            
        except Exception as e:
            logger.error(f"Error computing Sharpe ratio: {str(e)}")
            return None
    
    def compute_max_drawdown(self, prices: pd.Series) -> Dict:
        """
        Compute maximum drawdown
        
        Args:
            prices: Series of prices
        
        Returns:
            Dict with max_drawdown, peak, trough, duration
        """
        try:
            # Calculate cumulative returns
            cumulative = (1 + self.compute_returns(prices)).cumprod()
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            # Find maximum drawdown
            max_dd = drawdown.min()
            
            # Find when max drawdown occurred
            max_dd_idx = drawdown.idxmin()
            
            # Find peak before drawdown
            peak_idx = cumulative[:max_dd_idx].idxmax()
            
            # Find recovery point (if any)
            recovery_idx = None
            peak_value = cumulative[peak_idx]
            after_trough = cumulative[max_dd_idx:]
            recovery_points = after_trough[after_trough >= peak_value]
            if not recovery_points.empty:
                recovery_idx = recovery_points.index[0]
            
            # Calculate duration
            if recovery_idx:
                duration = (recovery_idx - peak_idx).days
            else:
                duration = (cumulative.index[-1] - peak_idx).days
            
            return {
                'max_drawdown': float(max_dd),
                'max_drawdown_pct': float(max_dd * 100),
                'peak_date': str(peak_idx),
                'trough_date': str(max_dd_idx),
                'recovery_date': str(recovery_idx) if recovery_idx else None,
                'duration_days': int(duration)
            }
            
        except Exception as e:
            logger.error(f"Error computing max drawdown: {str(e)}")
            return {
                'max_drawdown': None,
                'max_drawdown_pct': None
            }
    
    def compute_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Compute beta (sensitivity to market movements)
        
        Args:
            asset_returns: Returns of the asset
            market_returns: Returns of the market benchmark
        
        Returns:
            Beta as float
        """
        try:
            # Align the series
            aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
            
            if len(aligned) < 2:
                return None
            
            asset_col = aligned.iloc[:, 0]
            market_col = aligned.iloc[:, 1]
            
            # Calculate covariance and variance
            covariance = asset_col.cov(market_col)
            market_variance = market_col.var()
            
            if market_variance == 0:
                return None
            
            beta = covariance / market_variance
            
            return float(beta)
            
        except Exception as e:
            logger.error(f"Error computing beta: {str(e)}")
            return None
    
    def compute_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Compute Value at Risk (VaR)
        
        Args:
            returns: Series of returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            method: 'historical' or 'parametric'
        
        Returns:
            VaR as float (negative value represents loss)
        """
        try:
            if method == 'historical':
                # Historical VaR
                var = np.percentile(returns, (1 - confidence) * 100)
            
            elif method == 'parametric':
                # Parametric VaR (assumes normal distribution)
                mean = returns.mean()
                std = returns.std()
                var = mean - stats.norm.ppf(confidence) * std
            
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            return float(var)
            
        except Exception as e:
            logger.error(f"Error computing VaR: {str(e)}")
            return None
    
    def compute_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Compute Conditional Value at Risk (CVaR / Expected Shortfall)
        Average loss beyond VaR threshold
        
        Args:
            returns: Series of returns
            confidence: Confidence level
        
        Returns:
            CVaR as float (negative value represents loss)
        """
        try:
            # Calculate VaR
            var = self.compute_var(returns, confidence, method='historical')
            
            # Calculate CVaR (average of losses beyond VaR)
            cvar = returns[returns <= var].mean()
            
            return float(cvar)
            
        except Exception as e:
            logger.error(f"Error computing CVaR: {str(e)}")
            return None
    
    def compute_portfolio_risk(
        self,
        returns_df: pd.DataFrame,
        weights: List[float]
    ) -> Dict:
        """
        Compute risk metrics for a portfolio
        
        Args:
            returns_df: DataFrame with returns for each asset (columns = assets)
            weights: List of portfolio weights (must sum to 1)
        
        Returns:
            Dict with portfolio risk metrics
        """
        try:
            # Validate weights
            weights = np.array(weights)
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError("Weights must sum to 1")
            
            # Calculate portfolio returns
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            # Calculate metrics
            volatility = self.compute_volatility(portfolio_returns, annualize=True)
            sharpe = self.compute_sharpe_ratio(portfolio_returns)
            
            # Reconstruct prices from returns for drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            max_dd = self.compute_max_drawdown(cumulative_returns)
            
            var_95 = self.compute_var(portfolio_returns, confidence=0.95)
            cvar_95 = self.compute_cvar(portfolio_returns, confidence=0.95)
            
            # Annualized return
            mean_return = portfolio_returns.mean() * self.trading_days
            
            return {
                'status': 'success',
                'portfolio_metrics': {
                    'expected_return': float(mean_return),
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd['max_drawdown_pct'],
                    'var_95': float(var_95 * 100) if var_95 else None,  # Convert to percentage
                    'cvar_95': float(cvar_95 * 100) if cvar_95 else None,  # Convert to percentage
                },
                'weights': weights.tolist(),
                'asset_count': len(weights)
            }
            
        except Exception as e:
            logger.error(f"Error computing portfolio risk: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def compute_correlation_matrix(
        self,
        returns_df: pd.DataFrame
    ) -> Dict:
        """
        Compute correlation matrix for multiple assets
        
        Args:
            returns_df: DataFrame with returns for each asset
        
        Returns:
            Dict with correlation matrix and analysis
        """
        try:
            # Compute correlation
            corr_matrix = returns_df.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Threshold for high correlation
                        high_corr_pairs.append({
                            'asset1': corr_matrix.columns[i],
                            'asset2': corr_matrix.columns[j],
                            'correlation': float(corr_value)
                        })
            
            # Calculate average correlation
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            return {
                'status': 'success',
                'correlation_matrix': corr_matrix.to_dict(),
                'average_correlation': float(avg_corr),
                'high_correlation_pairs': high_corr_pairs,
                'diversification_score': float(1 - abs(avg_corr))  # Higher = better diversification
            }
            
        except Exception as e:
            logger.error(f"Error computing correlation matrix: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def stress_test(
        self,
        returns_df: pd.DataFrame,
        weights: List[float],
        shock_pct: float = -0.20
    ) -> Dict:
        """
        Perform stress test on portfolio
        
        Args:
            returns_df: DataFrame with returns
            weights: Portfolio weights
            shock_pct: Shock percentage (e.g., -0.20 for 20% drop)
        
        Returns:
            Dict with stress test results
        """
        try:
            weights = np.array(weights)
            
            # Calculate initial portfolio value (normalized to 100)
            initial_value = 100.0
            
            # Apply shock to each asset
            shocked_returns = returns_df.copy()
            shocked_returns = shocked_returns + shock_pct
            
            # Calculate portfolio value after shock
            portfolio_return_after_shock = (shocked_returns * weights).sum(axis=1).mean()
            final_value = initial_value * (1 + portfolio_return_after_shock)
            
            loss = final_value - initial_value
            loss_pct = (loss / initial_value) * 100
            
            return {
                'status': 'success',
                'shock_percentage': float(shock_pct * 100),
                'initial_value': float(initial_value),
                'final_value': float(final_value),
                'loss': float(loss),
                'loss_percentage': float(loss_pct),
                'scenario': 'Market-wide shock'
            }
            
        except Exception as e:
            logger.error(f"Error in stress test: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_asset_risk_metrics(
        self,
        prices: pd.Series,
        market_prices: Optional[pd.Series] = None
    ) -> Dict:
        """
        Get comprehensive risk metrics for a single asset
        
        Args:
            prices: Price series for the asset
            market_prices: Price series for market benchmark (optional)
        
        Returns:
            Dict with all risk metrics
        """
        try:
            returns = self.compute_returns(prices)
            
            metrics = {
                'volatility': self.compute_volatility(returns, annualize=True),
                'sharpe_ratio': self.compute_sharpe_ratio(returns),
                'max_drawdown': self.compute_max_drawdown(prices),
                'var_95': self.compute_var(returns, confidence=0.95),
                'cvar_95': self.compute_cvar(returns, confidence=0.95),
            }
            
            # Add beta if market data provided
            if market_prices is not None:
                market_returns = self.compute_returns(market_prices)
                metrics['beta'] = self.compute_beta(returns, market_returns)
            
            # Convert VaR and CVaR to percentages
            if metrics['var_95']:
                metrics['var_95_pct'] = float(metrics['var_95'] * 100)
            if metrics['cvar_95']:
                metrics['cvar_95_pct'] = float(metrics['cvar_95'] * 100)
            
            return {
                'status': 'success',
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting asset risk metrics: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }


# Create global instance
risk_analyzer = RiskAnalyzer()