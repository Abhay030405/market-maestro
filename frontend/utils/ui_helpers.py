"""
UI Helpers
Utility functions for Streamlit UI components
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional


def display_metric_card(label: str, value: str, delta: Optional[str] = None,
                       help_text: Optional[str] = None):
    """Display a metric card"""
    st.metric(label=label, value=value, delta=delta, help=help_text)


def display_error(message: str):
    """Display error message"""
    st.error(f"❌ {message}")


def display_success(message: str):
    """Display success message"""
    st.success(f"✅ {message}")


def display_warning(message: str):
    """Display warning message"""
    st.warning(f"⚠️ {message}")


def display_info(message: str):
    """Display info message"""
    st.info(f"ℹ️ {message}")


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format number as percentage"""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format number as currency"""
    return f"${value:,.{decimals}f}"


def create_download_button(data: pd.DataFrame, filename: str, label: str = "Download CSV"):
    """Create download button for DataFrame"""
    csv = data.to_csv(index=False)
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime='text/csv'
    )


def render_dataframe(df: pd.DataFrame, title: Optional[str] = None):
    """Render DataFrame with optional title"""
    if title:
        st.subheader(title)
    st.dataframe(df, use_container_width=True)


def create_two_columns():
    """Create two equal columns"""
    return st.columns(2)


def create_three_columns():
    """Create three equal columns"""
    return st.columns(3)


def create_four_columns():
    """Create four equal columns"""
    return st.columns(4)


def add_vertical_space(lines: int = 1):
    """Add vertical spacing"""
    for _ in range(lines):
        st.write("")


def display_sentiment_badge(sentiment: str) -> str:
    """Return emoji badge for sentiment"""
    sentiment_map = {
        'Bullish': '🟢',
        'Bearish': '🔴',
        'Neutral': '🟡',
        'Positive': '🟢',
        'Negative': '🔴'
    }
    return sentiment_map.get(sentiment, '⚪')


def display_risk_badge(risk_level: str) -> str:
    """Return emoji badge for risk level"""
    risk_map = {
        'Very Low': '🟢',
        'Low': '🟢',
        'Moderate': '🟡',
        'High': '🟠',
        'Very High': '🔴'
    }
    return risk_map.get(risk_level, '⚪')


def display_signal_badge(signal: str) -> str:
    """Return emoji badge for trading signal"""
    signal_map = {
        'Buy': '🟢',
        'Sell': '🔴',
        'Hold': '🟡',
        'Strong Buy': '🟢🟢',
        'Strong Sell': '🔴🔴'
    }
    return signal_map.get(signal, '⚪')


def create_expandable_section(title: str, content: str):
    """Create expandable section"""
    with st.expander(title):
        st.write(content)


def display_loading(message: str = "Loading..."):
    """Display loading spinner"""
    with st.spinner(message):
        pass


def sidebar_header(text: str):
    """Display sidebar header"""
    st.sidebar.markdown(f"## {text}")


def sidebar_divider():
    """Add divider in sidebar"""
    st.sidebar.divider()


def main_header(text: str, icon: str = "📊"):
    """Display main page header"""
    st.title(f"{icon} {text}")


def sub_header(text: str):
    """Display sub header"""
    st.subheader(text)


def display_key_value(key: str, value: str):
    """Display key-value pair"""
    st.write(f"**{key}:** {value}")


def create_tabs(tab_names: List[str]):
    """Create tabs"""
    return st.tabs(tab_names)


def display_json(data: Dict, expanded: bool = False):
    """Display JSON data"""
    st.json(data, expanded=expanded)


def create_color_by_value(value: float, threshold_low: float = 0, 
                         threshold_high: float = 0) -> str:
    """Return color based on value thresholds"""
    if value > threshold_high:
        return "green"
    elif value < threshold_low:
        return "red"
    else:
        return "orange"