"""
Trend Plot Component
Interactive stock charts with technical indicators
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional


def plot_candlestick_chart(data: List[Dict], symbol: str, indicators: Optional[Dict] = None):
    """
    Plot candlestick chart with optional indicators
    
    Args:
        data: List of OHLCV data dicts
        symbol: Stock symbol
        indicators: Optional technical indicators
    """
    df = pd.DataFrame(data)
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} Price Chart', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Add moving averages if available
    if indicators:
        # Get indicator data from the dataframe or indicators dict
        if 'sma_50' in indicators and indicators['sma_50']:
            # SMA 50 - approximate line
            sma_50_value = indicators['sma_50']
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=[sma_50_value] * len(df),
                    name='SMA 50',
                    line=dict(color='orange', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        if 'sma_200' in indicators and indicators['sma_200']:
            sma_200_value = indicators['sma_200']
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=[sma_200_value] * len(df),
                    name='SMA 200',
                    line=dict(color='blue', width=1, dash='dot')
                ),
                row=1, col=1
            )
    
    # Volume bars
    colors = ['red' if row['Close'] < row['Open'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,
        height=600,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_line_chart(data: List[Dict], symbol: str, column: str = 'Close'):
    """
    Plot simple line chart
    
    Args:
        data: List of price data dicts
        symbol: Stock symbol
        column: Column to plot (default: Close)
    """
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df[column],
            mode='lines',
            name=symbol,
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        )
    )
    
    fig.update_layout(
        title=f'{symbol} Price Trend',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_rsi_chart(data: List[Dict], rsi_data: List[float], symbol: str):
    """
    Plot RSI indicator chart
    
    Args:
        data: List of price data dicts
        rsi_data: List of RSI values
        symbol: Stock symbol
    """
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    # RSI line
    fig.add_trace(
        go.Scatter(
            x=df['Date'][-len(rsi_data):],
            y=rsi_data,
            name='RSI',
            line=dict(color='purple', width=2)
        )
    )
    
    # Overbought/Oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title=f'{symbol} RSI Indicator',
        xaxis_title='Date',
        yaxis_title='RSI',
        yaxis_range=[0, 100],
        height=300,
        hovermode='x'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_comparison_chart(data1: List[Dict], data2: List[Dict],
                         symbol1: str, symbol2: str):
    """
    Plot comparison chart for two stocks
    
    Args:
        data1: Price data for first stock
        data2: Price data for second stock
        symbol1: First symbol
        symbol2: Second symbol
    """
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    # Normalize to percentage change from first data point
    df1['Normalized'] = (df1['Close'] / df1['Close'].iloc[0] - 1) * 100
    df2['Normalized'] = (df2['Close'] / df2['Close'].iloc[0] - 1) * 100
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df1['Date'],
            y=df1['Normalized'],
            name=symbol1,
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df2['Date'],
            y=df2['Normalized'],
            name=symbol2,
            line=dict(color='#ff7f0e', width=2)
        )
    )
    
    fig.update_layout(
        title=f'{symbol1} vs {symbol2} Performance',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_portfolio_allocation(weights: Dict[str, float], title: str = "Portfolio Allocation"):
    """
    Plot pie chart for portfolio allocation
    
    Args:
        weights: Dict mapping symbol to weight
        title: Chart title
    """
    fig = go.Figure(data=[
        go.Pie(
            labels=list(weights.keys()),
            values=[w*100 for w in weights.values()],
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(correlation_matrix: Dict):
    """
    Plot correlation heatmap
    
    Args:
        correlation_matrix: Dict with correlation data
    """
    df = pd.DataFrame(correlation_matrix)
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale='RdYlGn',
        zmid=0,
        text=df.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Asset Correlation Matrix',
        height=400,
        xaxis_title='Assets',
        yaxis_title='Assets'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_returns_distribution(returns: List[float], symbol: str):
    """
    Plot returns distribution histogram
    
    Args:
        returns: List of return values
        symbol: Stock symbol
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker_color='#1f77b4'
        )
    )
    
    fig.update_layout(
        title=f'{symbol} Returns Distribution',
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)