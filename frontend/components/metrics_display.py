"""
Metrics Display Component
Display key financial metrics in organized layouts
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional


def display_technical_metrics(indicators: Dict):
    """
    Display technical analysis metrics
    
    Args:
        indicators: Dict with technical indicators
    """
    st.subheader("📊 Technical Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Moving Averages**")
        sma_50 = indicators.get('sma_50')
        sma_200 = indicators.get('sma_200')
        st.metric("SMA 50", f"${sma_50:.2f}" if sma_50 else "N/A")
        st.metric("SMA 200", f"${sma_200:.2f}" if sma_200 else "N/A")
    
    with col2:
        st.markdown("**Momentum**")
        rsi = indicators.get('rsi_14')
        st.metric("RSI (14)", f"{rsi:.2f}" if rsi else "N/A")
        
        macd = indicators.get('macd', {})
        if macd and macd.get('macd_line'):
            st.metric("MACD", f"{macd['macd_line']:.2f}")
    
    with col3:
        st.markdown("**Volatility**")
        vol = indicators.get('volatility')
        st.metric("Annual Vol", f"{vol*100:.2f}%" if vol else "N/A")
        
        atr = indicators.get('atr_14')
        st.metric("ATR (14)", f"{atr:.2f}" if atr else "N/A")
    
    with col4:
        st.markdown("**Returns**")
        ret_7d = indicators.get('return_7d')
        if ret_7d is not None:
            st.metric("7-Day Return", f"{ret_7d:.2f}%", delta=f"{ret_7d:.2f}%")
        
        ret_30d = indicators.get('return_30d')
        if ret_30d is not None:
            st.metric("30-Day Return", f"{ret_30d:.2f}%")


def display_risk_metrics(metrics: Dict):
    """
    Display risk analysis metrics
    
    Args:
        metrics: Dict with risk metrics
    """
    st.subheader("⚠️ Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        vol = metrics.get('volatility')
        if vol:
            st.metric(
                "Volatility",
                f"{vol*100:.2f}%",
                help="Annualized volatility (risk)"
            )
    
    with col2:
        sharpe = metrics.get('sharpe_ratio')
        if sharpe:
            delta_color = "normal" if sharpe > 1 else "inverse"
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta="Good" if sharpe > 1 else "Poor",
                help="Risk-adjusted return (higher is better)"
            )
    
    with col3:
        max_dd = metrics.get('max_drawdown')
        if max_dd:
            st.metric(
                "Max Drawdown",
                f"{max_dd:.2f}%",
                delta=f"{max_dd:.2f}%",
                delta_color="inverse",
                help="Largest historical loss"
            )
    
    with col4:
        var = metrics.get('var_95')
        if var:
            st.metric(
                "VaR (95%)",
                f"{var:.2f}%",
                help="Maximum expected daily loss (95% confidence)"
            )


def display_portfolio_metrics(metrics: Dict):
    """
    Display portfolio metrics
    
    Args:
        metrics: Dict with portfolio metrics
    """
    st.subheader("💼 Portfolio Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        exp_ret = metrics.get('expected_return')
        if exp_ret:
            st.metric(
                "Expected Return",
                f"{exp_ret*100:.2f}%",
                delta=f"{exp_ret*100:.2f}%",
                help="Annualized expected return"
            )
    
    with col2:
        vol = metrics.get('volatility')
        if vol:
            st.metric(
                "Portfolio Risk",
                f"{vol*100:.2f}%",
                help="Portfolio volatility"
            )
    
    with col3:
        sharpe = metrics.get('sharpe_ratio')
        if sharpe:
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta="Excellent" if sharpe > 1.5 else "Good" if sharpe > 1 else "Fair",
                help="Risk-adjusted return"
            )
    
    with col4:
        max_dd = metrics.get('max_drawdown')
        if max_dd:
            st.metric(
                "Max Drawdown",
                f"{max_dd:.2f}%",
                delta_color="inverse",
                help="Maximum historical loss"
            )


def display_comparison_metrics(symbol1: str, metrics1: Dict,
                               symbol2: str, metrics2: Dict):
    """
    Display comparison of metrics between two assets
    
    Args:
        symbol1: First symbol
        metrics1: Metrics for first symbol
        symbol2: Second symbol
        metrics2: Metrics for second symbol
    """
    st.subheader(f"🆚 {symbol1} vs {symbol2}")
    
    # Create comparison dataframe
    comparison_data = {
        'Metric': [],
        symbol1: [],
        symbol2: [],
        'Winner': []
    }
    
    # Compare key metrics
    metrics_to_compare = [
        ('volatility', 'Volatility', False, '%'),
        ('sharpe_ratio', 'Sharpe Ratio', True, ''),
        ('max_drawdown', 'Max Drawdown', False, '%'),
        ('rsi_14', 'RSI', None, ''),
    ]
    
    for key, label, higher_better, suffix in metrics_to_compare:
        val1 = metrics1.get(key)
        val2 = metrics2.get(key)
        
        if val1 is not None and val2 is not None:
            comparison_data['Metric'].append(label)
            
            # Format values
            if suffix == '%':
                comparison_data[symbol1].append(f"{val1*100:.2f}%")
                comparison_data[symbol2].append(f"{val2*100:.2f}%")
            else:
                comparison_data[symbol1].append(f"{val1:.2f}")
                comparison_data[symbol2].append(f"{val2:.2f}")
            
            # Determine winner
            if higher_better is not None:
                if higher_better:
                    winner = symbol1 if val1 > val2 else symbol2
                else:
                    winner = symbol1 if val1 < val2 else symbol2
                comparison_data['Winner'].append(f"✓ {winner}")
            else:
                comparison_data['Winner'].append("-")
    
    if comparison_data['Metric']:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def display_sentiment_metrics(sentiment_data: Dict):
    """
    Display sentiment analysis metrics
    
    Args:
        sentiment_data: Dict with sentiment data
    """
    st.subheader("😊 Sentiment Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment = sentiment_data.get('sentiment_label', 'Neutral')
        emoji = '🟢' if sentiment == 'Bullish' else '🔴' if sentiment == 'Bearish' else '🟡'
        st.metric("Overall Sentiment", f"{emoji} {sentiment}")
    
    with col2:
        avg_sent = sentiment_data.get('avg_sentiment', 0)
        st.metric("Sentiment Score", f"{avg_sent:.2f}")
    
    with col3:
        article_count = sentiment_data.get('article_count', 0)
        st.metric("Articles Analyzed", article_count)
    
    with col4:
        distribution = sentiment_data.get('sentiment_distribution', {})
        bullish = distribution.get('bullish', 0)
        bearish = distribution.get('bearish', 0)
        if bullish + bearish > 0:
            ratio = bullish / (bullish + bearish) * 100
            st.metric("Bullish Ratio", f"{ratio:.0f}%")


def display_key_stats_grid(stats: Dict):
    """
    Display a grid of key statistics
    
    Args:
        stats: Dict with key-value pairs
    """
    # Create columns based on number of stats
    num_stats = len(stats)
    cols_per_row = 4
    
    items = list(stats.items())
    for i in range(0, num_stats, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < num_stats:
                key, value = items[i + j]
                with col:
                    st.metric(key, value)


def create_metrics_table(data: Dict, title: str = "Metrics Summary"):
    """
    Create a formatted metrics table
    
    Args:
        data: Dict with metrics
        title: Table title
    """
    st.markdown(f"**{title}**")
    
    df = pd.DataFrame([
        {"Metric": k, "Value": v}
        for k, v in data.items()
    ])
    
    st.dataframe(df, use_container_width=True, hide_index=True)