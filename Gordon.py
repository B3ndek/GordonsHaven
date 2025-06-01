import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="Stock & Currency Analysis",
    page_icon="📈",
    layout="wide"
)

# ----------------------------------------
# 1. Technical Indicators Functions
# ----------------------------------------

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period).mean()

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=period).mean()
    rolling_std = data.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, rolling_mean, lower_band

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ----------------------------------------
# 2. Caching & Data Fetching
# ----------------------------------------

@st.cache_data(show_spinner=False)
def fetch_data(symbol, timeframe):
    """Fetch historical data from yfinance (cached)"""
    try:
        ticker = yf.Ticker(symbol)
        period_info = get_period_mapping(timeframe)
        data = ticker.history(
            period=period_info['period'],
            interval=period_info['interval']
        )
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# ----------------------------------------
# 3. Add Technical Indicators (cached)
# ----------------------------------------

@st.cache_data(show_spinner=False)
def add_technical_indicators(data, bb_period):
    """
    Add technical indicators to the dataframe, using a dynamic Bollinger Bands period.
    Drop only the minimum warm-up rows so full history is preserved.
    """
    df = data.copy()
    # EMAs
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['EMA_200'] = calculate_ema(df['Close'], 200)
    # Bollinger Bands (period = bb_period, std_dev fixed at 2)
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'], bb_period, 2)
    # RSI
    df['RSI'] = calculate_rsi(df['Close'], 14)
    # Drop only warm-up rows
    df_clean = df.dropna()
    return df_clean

# ----------------------------------------
# 4. Timeframe-to-Period Mapping
# ----------------------------------------

def get_period_mapping(timeframe):
    """Map timeframe to yfinance period and interval - optimized for ~500-600 periods"""
    mapping = {
        'M15': {'period': '10d', 'interval': '15m'},
        'H1': {'period': '25d', 'interval': '1h'},
        'H4': {'period': '100d', 'interval': '4h'},
        'D1': {'period': '2y', 'interval': '1d'},
        'W1': {'period': '12y', 'interval': '1wk'}
    }
    return mapping.get(timeframe, {'period': '2y', 'interval': '1d'})

# ----------------------------------------
# 5. Plotting Functions
# ----------------------------------------

def plot_stock_data_interactive(data, symbol, timeframe, display_candles=100):
    """
    Create interactive chart with Plotly:
    - Price + EMAs + Bollinger Bands (row 1)
    - RSI (row 2)
    - Volume (row 3)
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        #subplot_titles=[f'{symbol} - {timeframe} Chart', 'RSI (14)', 'Volume']
    )
    # Row 1: Candlestick + EMAs + Bollinger Bands
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    # EMA 50
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['EMA_50'],
            name='EMA 50',
            line=dict(color='blue', width=2),
            hovertemplate='EMA50: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    # EMA 200
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['EMA_200'],
            name='EMA 200',
            line=dict(color='orange', width=2),
            hovertemplate='EMA200: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    # Bollinger Bands – Upper
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            name='BB Upper',
            line=dict(color='purple', width=1, dash='dash'),
            opacity=0.7,
            hovertemplate='BB Upper: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    # Bollinger Bands – Middle
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_Middle'],
            name='BB Middle',
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5,
            hovertemplate='BB Middle: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    # Bollinger Bands – Lower (filled)
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            name='BB Lower',
            line=dict(color='purple', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128, 0, 128, 0.1)',
            opacity=0.7,
            hovertemplate='BB Lower: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    # Row 2: RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='purple', width=2),
            hovertemplate='RSI: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
    fig.add_hline(y=50, line_dash="solid", line_color="gray", opacity=0.5, row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1, title_text="RSI")
    # Row 3: Volume
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='gray',
            opacity=0.5,
            hovertemplate='Volume: %{y}<extra></extra>'
        ),
        row=3, col=1
    )
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    # Layout
    fig.update_layout(
        title=f'{symbol} - {timeframe} Interactive Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1
        )
    )
    # Initial view: last display_candles
    if len(data) > display_candles:
        start_date = data.index[-display_candles]
        end_date = data.index[-1]
        fig.update_xaxes(range=[start_date, end_date])
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig

def plot_stock_data_static(data, symbol, timeframe):
    """
    Create static chart with mplfinance - fallback option
    (full history preserved, RSI panel included).
    """
    plot_data = data[['Open', 'High', 'Low', 'Close']].copy()
    apd = [
        mpf.make_addplot(data['EMA_50'], color='blue', width=2),
        mpf.make_addplot(data['EMA_200'], color='orange', width=2),
        mpf.make_addplot(data['BB_Upper'], color='purple', linestyle='--', alpha=0.7),
        mpf.make_addplot(data['BB_Middle'], color='gray', linestyle=':', alpha=0.5),
        mpf.make_addplot(data['BB_Lower'], color='purple', linestyle='--', alpha=0.7),
        mpf.make_addplot(data['RSI'], panel=1, color='purple', width=2, ylabel='RSI', ylim=(0, 100))
    ]
    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit',
        wick={'up': 'green', 'down': 'red'}
    )
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='-',
        gridcolor='lightgray',
        facecolor='white',
        figcolor='white'
    )
    fig, axes = mpf.plot(
        plot_data,
        type='candle',
        style=s,
        addplot=apd,
        volume=False,
        title=f'{symbol} - {timeframe} Chart with Technical Indicators',
        ylabel='Price',
        figsize=(15, 10),
        panel_ratios=(3, 1),
        returnfig=True,
        warn_too_much_data=1000
    )
    rsi_ax = axes[1]
    rsi_ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, linewidth=1)
    rsi_ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, linewidth=1)
    rsi_ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
    rsi_ax.set_ylim(0, 100)
    return fig

# ----------------------------------------
# 6. Streamlit App
# ----------------------------------------

def main():
    st.title("📈 Stock & Currency Analysis Dashboard")
    st.markdown("---")
    # Sidebar
    st.sidebar.header("Analysis Settings")
    st.sidebar.subheader("Symbol Selection")
    popular_stocks = [
        'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ'
    ]
    popular_forex = [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X'
    ]
    popular_crypto = [
        'BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'SOL-USD'
    ]
    category = st.sidebar.selectbox(
        "Select Category:",
        ["Stocks", "Forex", "Crypto", "Custom"]
    )
    if category == "Stocks":
        symbol = st.sidebar.selectbox("Select Stock:", popular_stocks)
    elif category == "Forex":
        symbol = st.sidebar.selectbox("Select Currency Pair:", popular_forex)
    elif category == "Crypto":
        symbol = st.sidebar.selectbox("Select Cryptocurrency:", popular_crypto)
    else:
        symbol = st.sidebar.text_input("Enter Symbol (e.g., AAPL, EURUSD=X, BTC-USD):", value="AAPL")

    # Bollinger Band period slider (dynamic BB window)
    bb_period = st.sidebar.slider(
        "Bollinger Band Window:",
        min_value=10,
        max_value=200,
        step=10,
        value=20,
        help="Adjust the lookback period for Bollinger Bands"
    )

    timeframe = st.sidebar.selectbox(
        "Select Timeframe:",
        ['M15', 'H1', 'H4', 'D1', 'W1'],
        index=3
    )
    st.sidebar.subheader("Chart Settings")
    chart_type = st.sidebar.radio(
        "Chart Type:",
        ["Interactive (Plotly)", "Static (mplfinance)"],
        index=0
    )
    if chart_type == "Interactive (Plotly)":
        display_candles = st.sidebar.slider(
            "Initial candles to display:",
            min_value=50,
            max_value=300,
            value=100,
            step=25,
            help="Chart will initially show this many recent candles, but you can scroll/zoom to see all data"
        )

    # Fetch data button
    if st.sidebar.button("📊 Analyze", type="primary"):
        if symbol:
            with st.spinner(f"Fetching data for {symbol}..."):
                data_raw = fetch_data(symbol, timeframe)
                if data_raw is not None and not data_raw.empty:
                    # Add technical indicators with dynamic BB period
                    data = add_technical_indicators(data_raw, bb_period)
                    st.session_state.data = data
                    st.session_state.symbol = symbol
                    st.session_state.timeframe = timeframe
                    st.session_state.bb_period = bb_period
                    st.success(f"✅ Data loaded successfully! {len(data)} periods analyzed (BB window = {bb_period}).")
                else:
                    st.error("❌ Failed to fetch data. Please check the symbol and try again.")
        else:
            st.warning("⚠️ Please enter a valid symbol.")

    # Display analysis if data exists
    if 'data' in st.session_state and st.session_state.data is not None:
        data = st.session_state.data
        symbol = st.session_state.symbol
        timeframe = st.session_state.timeframe
        # Use bb_period from session_state if available, otherwise fallback to current slider
        bb_period = st.session_state.get('bb_period', bb_period)
        st.subheader(f"📈 {symbol} - {timeframe} Analysis (BB window = {bb_period})")

        if chart_type == "Interactive (Plotly)":
            fig = plot_stock_data_interactive(data, symbol, timeframe, display_candles)
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            🖱️ **Interactive Chart Controls:**
            - **Zoom**: Use mouse wheel or zoom tools
            - **Pan**: Click and drag to scroll through historical data  
            - **Reset**: Double-click to reset view
            - **Legend**: Click legend items to show/hide indicators
            - **Crosshair**: Hover over chart for precise values
            """)
        else:
            fig = plot_stock_data_static(data, symbol, timeframe)
            st.pyplot(fig)

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        latest_data = data.iloc[-1]
        prev_data = data.iloc[-2] if len(data) > 1 else latest_data

        with col1:
            price_change = latest_data['Close'] - prev_data['Close']
            price_change_pct = (price_change / prev_data['Close']) * 100
            st.metric(
                "Current Price",
                f"${latest_data['Close']:.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )

        with col2:
            st.metric("RSI (14)", f"{latest_data['RSI']:.2f}")

        with col3:
            st.metric("EMA 50", f"${latest_data['EMA_50']:.2f}")

        with col4:
            st.metric("EMA 200", f"${latest_data['EMA_200']:.2f}")

        # Recent data table
        with st.expander("📋 Recent Data"):
            display_data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_50', 'EMA_200', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower']].tail(10)
            st.dataframe(display_data.round(4))

    else:
        st.info("👋 Welcome! Please select a symbol and timeframe from the sidebar to begin analysis.")
        st.markdown("""
        ### How to use this app:
        1. **Select Category**: Choose between Stocks, Forex, Crypto, or enter a custom symbol
        2. **Choose Symbol**: Pick from popular options or enter your own
        3. **Adjust Bollinger Band Window**: Use the slider to pick a lookback (e.g., 20 or 200)
        4. **Select Timeframe**: Choose from M15, H1, H4, D1, or W1
        5. **Click Analyze**: The app will fetch data and display charts with technical indicators
        
        ### Features included:
        - 🕯️ **Interactive Charts**: Full zoom, pan, and scroll capabilities with Plotly (including volume)
        - 📊 **Static Charts**: Professional mplfinance charts as alternative
        - 📈 50 & 200 period Exponential Moving Averages
        - 📉 Bollinger Bands (dynamic window)
        - 🔄 RSI indicator (14 period) with overbought/oversold levels
        - 🖱️ **Chart Controls**: Zoom, pan, crosshair, legend toggle
        - 📋 Key metrics and recent data table
        
        ### Installation Requirements:
        ```bash
        pip install streamlit yfinance pandas numpy matplotlib mplfinance plotly
        ```
        """)

if __name__ == "__main__":
    main()