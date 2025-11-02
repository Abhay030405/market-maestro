"""
Dashboard Cards Component
Reusable metric cards for displaying key information
"""

import streamlit as st
from typing import Optional


def metric_card(title: str, value: str, delta: Optional[str] = None,
                icon: str = "📊", help_text: Optional[str] = None):
    """
    Display a metric card with icon
    
    Args:
        title: Metric title
        value: Metric value
        delta: Change indicator (optional)
        icon: Emoji icon
        help_text: Tooltip help text
    """
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown(f"<div style='font-size: 3rem; text-align: center;'>{icon}</div>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.metric(label=title, value=value, delta=delta, help=help_text)


def info_card(title: str, content: str, icon: str = "ℹ️"):
    """
    Display an information card
    
    Args:
        title: Card title
        content: Card content
        icon: Emoji icon
    """
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
        <h4>{icon} {title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def status_card(status: str, message: str):
    """
    Display a status card with color coding
    
    Args:
        status: Status type (success, warning, error, info)
        message: Status message
    """
    colors = {
        'success': '#d4edda',
        'warning': '#fff3cd',
        'error': '#f8d7da',
        'info': '#d1ecf1'
    }
    
    icons = {
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
        'info': 'ℹ️'
    }
    
    color = colors.get(status, colors['info'])
    icon = icons.get(status, icons['info'])
    
    st.markdown(f"""
    <div style="background-color: {color}; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
        <p style="margin: 0;"><strong>{icon} {message}</strong></p>
    </div>
    """, unsafe_allow_html=True)


def signal_card(signal: str, confidence: float, description: str):
    """
    Display a trading signal card
    
    Args:
        signal: Signal type (Buy, Hold, Sell)
        confidence: Confidence score (0-1)
        description: Signal description
    """
    signal_colors = {
        'Buy': '#28a745',
        'Hold': '#ffc107',
        'Sell': '#dc3545'
    }
    
    signal_icons = {
        'Buy': '🟢',
        'Hold': '🟡',
        'Sell': '🔴'
    }
    
    color = signal_colors.get(signal, '#6c757d')
    icon = signal_icons.get(signal, '⚪')
    
    st.markdown(f"""
    <div style="background-color: {color}20; border-left: 5px solid {color}; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
        <h3 style="color: {color}; margin: 0;">{icon} {signal} Signal</h3>
        <p style="margin: 0.5rem 0;"><strong>Confidence:</strong> {confidence*100:.0f}%</p>
        <p style="margin: 0;">{description}</p>
    </div>
    """, unsafe_allow_html=True)


def sentiment_card(sentiment: str, score: float, details: str):
    """
    Display a sentiment analysis card
    
    Args:
        sentiment: Sentiment label (Bullish, Neutral, Bearish)
        score: Sentiment score
        details: Additional details
    """
    sentiment_colors = {
        'Bullish': '#28a745',
        'Neutral': '#ffc107',
        'Bearish': '#dc3545',
        'Positive': '#28a745',
        'Negative': '#dc3545'
    }
    
    sentiment_icons = {
        'Bullish': '🐂',
        'Neutral': '😐',
        'Bearish': '🐻',
        'Positive': '😊',
        'Negative': '😞'
    }
    
    color = sentiment_colors.get(sentiment, '#6c757d')
    icon = sentiment_icons.get(sentiment, '😐')
    
    st.markdown(f"""
    <div style="background-color: {color}20; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
        <h3 style="color: {color}; margin: 0;">{icon} {sentiment}</h3>
        <p style="margin: 0.5rem 0;"><strong>Score:</strong> {score:.2f}</p>
        <p style="margin: 0; font-size: 0.9rem;">{details}</p>
    </div>
    """, unsafe_allow_html=True)


def risk_card(risk_level: str, metrics: dict):
    """
    Display a risk assessment card
    
    Args:
        risk_level: Risk level (Very Low, Low, Moderate, High, Very High)
        metrics: Dict with risk metrics
    """
    risk_colors = {
        'Very Low': '#28a745',
        'Low': '#5cb85c',
        'Moderate': '#ffc107',
        'High': '#ff8c00',
        'Very High': '#dc3545'
    }
    
    color = risk_colors.get(risk_level, '#6c757d')
    
    st.markdown(f"""
    <div style="border: 2px solid {color}; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
        <h3 style="color: {color}; margin: 0;">⚠️ Risk Level: {risk_level}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display metrics
    cols = st.columns(len(metrics))
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(key, value)


def comparison_card(symbol1: str, symbol2: str, metric_name: str,
                   value1: float, value2: float):
    """
    Display a comparison card for two assets
    
    Args:
        symbol1: First symbol
        symbol2: Second symbol
        metric_name: Name of the metric being compared
        value1: Value for first symbol
        value2: Value for second symbol
    """
    winner = symbol1 if value1 > value2 else symbol2
    
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
        <h4 style="margin: 0 0 0.5rem 0;">{metric_name}</h4>
        <div style="display: flex; justify-content: space-between;">
            <div style="flex: 1; text-align: center;">
                <p style="font-size: 1.2rem; font-weight: bold; margin: 0;">{symbol1}</p>
                <p style="font-size: 1.5rem; margin: 0.5rem 0;">{value1:.2f}</p>
            </div>
            <div style="flex: 0; padding: 0 1rem; display: flex; align-items: center;">
                <p style="font-size: 2rem; margin: 0;">🆚</p>
            </div>
            <div style="flex: 1; text-align: center;">
                <p style="font-size: 1.2rem; font-weight: bold; margin: 0;">{symbol2}</p>
                <p style="font-size: 1.5rem; margin: 0.5rem 0;">{value2:.2f}</p>
            </div>
        </div>
        <p style="text-align: center; margin: 0.5rem 0; color: #28a745;">
            <strong>Winner: {winner} ✓</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)