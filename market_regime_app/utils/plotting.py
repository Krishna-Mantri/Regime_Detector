# utils/plotting.py
"""
Visualization utilities for market regime analysis.
"""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from regime_model import regime_labels

logger = logging.getLogger(__name__)


def plot_regimes(df: pd.DataFrame, regimes: np.ndarray) -> go.Figure:
    """
    Create interactive Plotly visualization of market regimes and price action.
    
    Displays:
    - Price line colored by current regime
    - Bull, Bear, and Sideways regime segments
    - 50-day and 200-day moving averages
    - Regime shading background
    - Interactive legend and hover information
    
    Args:
        df: DataFrame with 'Close' column (must have datetime index)
        regimes: Array of regime labels (0=Bull, 1=Bear, 2=Sideways)
        
    Returns:
        Plotly figure object with interactive chart
        
    Raises:
        ValueError: If inputs are invalid or shapes don't match
        KeyError: If 'Close' column not found
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    if 'Close' not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column")
    
    if regimes is None or len(regimes) == 0:
        raise ValueError("Regimes array is empty or None")
    
    if len(df) != len(regimes):
        raise ValueError(f"DataFrame length ({len(df)}) != regimes length ({len(regimes)})")
    
    # Convert regimes to proper integer array
    regimes = np.asarray(regimes, dtype=int)
    
    try:
        fig = go.Figure()

        # Create regime color mapping and labels
        regime_color_map = {0: 'green', 1: 'red', 2: 'blue'}
        regime_rgba_map = {0: 'rgba(0,255,0,0.15)', 1: 'rgba(255,0,0,0.15)', 2: 'rgba(0,0,255,0.15)'}
        regime_name_map = {0: 'Bull', 1: 'Bear', 2: 'Sideways'}
        
        # Find regime segments for background shading
        shapes = []
        start_idx = 0
        
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[start_idx]:
                shapes.append(dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=df.index[start_idx],
                    x1=df.index[i-1],
                    y0=0,
                    y1=1,
                    fillcolor=regime_rgba_map[regimes[start_idx]],
                    layer="below",
                    line_width=0,
                ))
                start_idx = i
        
        # Final segment
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=df.index[start_idx],
            x1=df.index[-1],
            y0=0,
            y1=1,
            fillcolor=regime_rgba_map[regimes[start_idx]],
            layer="below",
            line_width=0,
        ))

        # Add price line colored by regime
        regime_segments = []
        start_idx = 0
        
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[start_idx]:
                regime_color = regime_color_map[regimes[start_idx]]
                regime_name = regime_name_map[regimes[start_idx]]
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index[start_idx:i],
                        y=df['Close'].iloc[start_idx:i],
                        mode='lines',
                        name=f'{regime_name} Price',
                        line=dict(color=regime_color, width=2.5),
                        hovertemplate=f'<b>{regime_name} Regime</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:.2f}}<extra></extra>',
                        showlegend=(i == 1)  # Only show in legend for first occurrence
                    )
                )
                start_idx = i
        
        # Final segment
        regime_color = regime_color_map[regimes[start_idx]]
        regime_name = regime_name_map[regimes[start_idx]]
        fig.add_trace(
            go.Scatter(
                x=df.index[start_idx:],
                y=df['Close'].iloc[start_idx:],
                mode='lines',
                name=f'{regime_name} Price',
                line=dict(color=regime_color, width=2.5),
                hovertemplate=f'<b>{regime_name} Regime</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:.2f}}<extra></extra>',
                showlegend=False
            )
        )
        
        # Add 50-day moving average
        ma50 = df['Close'].rolling(50).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ma50,
                mode='lines',
                name='MA50 (50-day Moving Avg)',
                line=dict(color='orange', width=1.5, dash='dash'),
                hovertemplate='<b>MA50</b><br>Date: %{x|%Y-%m-%d}<br>Value: $%{y:.2f}<extra></extra>'
            )
        )

        # Add 200-day moving average
        ma200 = df['Close'].rolling(200).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ma200,
                mode='lines',
                name='MA200 (200-day Moving Avg)',
                line=dict(color='cyan', width=1.5, dash='dash'),
                hovertemplate='<b>MA200</b><br>Date: %{x|%Y-%m-%d}<br>Value: $%{y:.2f}<extra></extra>'
            )
        )
        
        # Add regime transition markers (using shapes instead of vlines for better compatibility)
        transition_indices = [0]
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                transition_indices.append(i)
        
        # Add vertical lines as shapes for regime transitions
        for idx in transition_indices[1:]:  # Skip first index
            shapes.append(dict(
                type="line",
                xref="x",
                yref="paper",
                x0=df.index[idx],
                x1=df.index[idx],
                y0=0,
                y1=1,
                line=dict(color="gray", width=1, dash="dot"),
                opacity=0.5
            ))

        # Update layout with enhanced styling
        fig.update_layout(
            shapes=shapes,
            template="plotly_white",
            height=700,
            title="<b>Market Regime Detection Analysis</b><br><sub>Green=Bull | Red=Bear | Blue=Sideways</sub>",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            font=dict(size=11),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            xaxis=dict(
                rangeslider=dict(visible=False),
                type="date"
            ),
            yaxis=dict(
                zeroline=False
            ),
            plot_bgcolor="rgba(240,240,240,0.5)",
            margin=dict(l=70, r=20, t=80, b=70)
        )

        logger.info("Enhanced chart generated successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        raise
