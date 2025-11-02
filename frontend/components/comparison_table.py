"""
Comparison Table Component
Side-by-side comparison of multiple assets
"""

import streamlit as st
import pandas as pd
from typing import Dict, List


def create_comparison_table(assets_data: Dict[str, Dict]):
    """
    Create comparison table for multiple assets
    
    Args:
        assets_data: Dict mapping symbol to its data/metrics
    """
    if not assets_data:
        st.warning("No data available for comparison")
        return
    
    # Prepare comparison data
    comparison = {'Metric': []}
    
    for symbol in assets_data.keys():
        comparison[symbol] = []
    
    # Define metrics to compare
    metrics_to_display = [
        ('last_price', 'Current Price', '$', 2),
        ('volatility', 'Volatility', '%', 2),
        ('sharpe_ratio', 'Sharpe Ratio', '', 2),
        ('max_drawdown', 'Max Drawdown', '%', 2),
        ('rsi_14', 'RSI (14)', '', 1),
        ('sentiment_label', 'Sentiment', '', 0),
        ('risk_level', 'Risk Level', '', 0),
    ]
    
    for key, label, suffix, decimals in metrics_to_display:
        comparison['Metric'].append(label)
        
        for symbol, data in assets_data.items():
            value = extract_nested_value(data, key)
            
            if value is not None:
                if suffix == '$':
                    formatted = f"${value:.{decimals}f}"
                elif suffix == '%':
                    if isinstance(value, (int, float)) and value < 1:
                        formatted = f"{value*100:.{decimals}f}%"
                    else:
                        formatted = f"{value:.{decimals}f}%"
                elif decimals > 0 and isinstance(value, (int, float)):
                    formatted = f"{value:.{decimals}f}"
                else:
                    formatted = str(value)
                
                comparison[symbol].append(formatted)
            else:
                comparison[symbol].append("N/A")
    
    # Create DataFrame
    df = pd.DataFrame(comparison)
    
    # Display with styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def create_side_by_side_comparison(symbol1: str, data1: Dict,
                                   symbol2: str, data2: Dict):
    """
    Create detailed side-by-side comparison of two assets
    
    Args:
        symbol1: First symbol
        data1: Data for first symbol
        symbol2: Second symbol
        data2: Data for second symbol
    """
    st.markdown(f"## {symbol1} 🆚 {symbol2}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {symbol1}")
        display_asset_summary(data1)
    
    with col2:
        st.markdown(f"### {symbol2}")
        display_asset_summary(data2)
    
    # Comparison metrics
    st.markdown("### 📊 Head-to-Head Comparison")
    
    metrics = [
        ('last_price', 'Price', True),
        ('volatility', 'Volatility', False),
        ('sharpe_ratio', 'Sharpe Ratio', True),
        ('rsi_14', 'RSI', None),
        ('sentiment_score', 'Sentiment', True),
    ]
    
    for key, label, higher_better in metrics:
        val1 = extract_nested_value(data1, key)
        val2 = extract_nested_value(data2, key)
        
        if val1 is not None and val2 is not None:
            create_metric_comparison_bar(label, symbol1, val1, symbol2, val2, higher_better)


def display_asset_summary(data: Dict):
    """Display summary of asset data"""
    
    # Price
    price = extract_nested_value(data, 'last_price')
    if price:
        st.metric("Current Price", f"${price:.2f}")
    
    # Trend
    trend = extract_nested_value(data, 'trend_analysis.direction')
    if trend:
        st.info(f"**Trend:** {trend}")
    
    # Sentiment
    sentiment = extract_nested_value(data, 'sentiment_label')
    if sentiment:
        emoji = '🟢' if sentiment == 'Bullish' else '🔴' if sentiment == 'Bearish' else '🟡'
        st.info(f"**Sentiment:** {emoji} {sentiment}")
    
    # Risk
    risk = extract_nested_value(data, 'risk_level')
    if risk:
        st.warning(f"**Risk:** {risk}")


def create_metric_comparison_bar(label: str, symbol1: str, value1: float,
                                 symbol2: str, value2: float,
                                 higher_better: bool = None):
    """
    Create a visual comparison bar for a metric
    
    Args:
        label: Metric label
        symbol1: First symbol
        value1: First value
        symbol2: Second symbol
        value2: Second value
        higher_better: Whether higher is better (None for neutral)
    """
    st.markdown(f"**{label}**")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.metric(symbol1, f"{value1:.2f}")
    
    with col2:
        if higher_better is not None:
            if higher_better:
                winner = symbol1 if value1 > value2 else symbol2
            else:
                winner = symbol1 if value1 < value2 else symbol2
            
            st.markdown(f"<div style='text-align: center; padding-top: 10px;'>"
                       f"<strong style='color: green;'>✓ {winner}</strong></div>",
                       unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center; padding-top: 10px;'>-</div>",
                       unsafe_allow_html=True)
    
    with col3:
        st.metric(symbol2, f"{value2:.2f}")
    
    st.divider()


def extract_nested_value(data: Dict, key_path: str):
    """
    Extract value from nested dict using dot notation
    
    Args:
        data: Data dictionary
        key_path: Dot-separated path (e.g., 'market_analysis.last_price')
    
    Returns:
        Value or None
    """
    keys = key_path.split('.')
    value = data
    
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return None
        else:
            return None
    
    return value


def create_ranked_comparison(assets_data: Dict[str, Dict], metric_key: str,
                             metric_label: str, higher_better: bool = True):
    """
    Create ranked comparison for a specific metric
    
    Args:
        assets_data: Dict mapping symbol to data
        metric_key: Key of metric to rank
        metric_label: Display label for metric
        higher_better: Whether higher values are better
    """
    st.markdown(f"### 🏆 Ranked by {metric_label}")
    
    # Extract metric values
    rankings = []
    for symbol, data in assets_data.items():
        value = extract_nested_value(data, metric_key)
        if value is not None:
            rankings.append((symbol, value))
    
    # Sort
    rankings.sort(key=lambda x: x[1], reverse=higher_better)
    
    # Display rankings
    for rank, (symbol, value) in enumerate(rankings, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        st.markdown(f"{emoji} **{symbol}**: {value:.2f}")