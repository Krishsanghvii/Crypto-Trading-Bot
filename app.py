import streamlit as st
import pandas as pd
import os
import logging
from datetime import datetime
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv

from model import get_price_prediction_model
from bot import CryptoTradingBot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="Crypto Trading Bot App",
    page_icon="📈",
    layout="wide"
)

# Sidebar contents
with st.sidebar:
    st.title('Cryptocurrency Trading Bot 📈🤖')
    st.markdown('''
    ## About
    This web-app is a cryptocurrency price prediction and trading bot based on:
    - **Machine Learning**: SARIMAX model for price prediction
    - **Technical Analysis**: SMA crossover strategy  
    - **Risk Management**: Stop-loss and position sizing
    - **Real Trading**: Binance API integration
    
    ⚠️ **Warning**: This bot trades with real money. Use sandbox mode first!
    ''')
    
    add_vertical_space(3)
    
    # Trading mode selection
    trading_mode = st.selectbox(
        "Trading Mode",
        ["sandbox", "paper", "live"],
        help="Sandbox: Test mode, Paper: Simulated trades, Live: Real money"
    )
    
    add_vertical_space(2)
    st.write('Made by [Krish Sanghvi](https://github.com/Krishsanghvii)')

def validate_api_credentials():
    """Validate API credentials are set"""
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    
    if not api_key or api_key == 'your_binance_api_key_here':
        return False, "API_KEY not set in .env file"
    if not api_secret or api_secret == 'your_binance_secret_key_here':
        return False, "API_SECRET not set in .env file"
    
    return True, "API credentials validated"

def main():
    st.header("Cryptocurrency Trading Bot 📈")
    st.divider()
    
    # Validate API credentials
    creds_valid, creds_message = validate_api_credentials()
    
    if not creds_valid:
        st.error(f"❌ {creds_message}")
        st.info("Please update your .env file with valid Binance API credentials")
        st.code("""
# Add to .env file:
API_KEY=your_actual_binance_api_key
API_SECRET=your_actual_binance_secret_key
        """)
        return
    
    # Initialize trading bot
    try:
        bot = CryptoTradingBot(trading_mode=trading_mode)
        st.success(f"✅ Connected to Binance in {trading_mode} mode")
    except Exception as e:
        st.error(f"❌ Failed to connect to Binance: {str(e)}")
        return
    
    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Price Prediction", "Trading Signals", "Portfolio Status"])
    
    with tab1:
        st.subheader("🔮 Price Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            target_date = st.date_input(
                'Select prediction date',
                min_value=datetime.now().date(),
                help="Select a future date for price prediction"
            )
        with col2:
            predict_button = st.button("🎯 Get Prediction", type="primary")
        
        if predict_button:
            with st.spinner("Running ML model..."):
                try:
                    predictions = get_price_prediction_model(target_date)
                    
                    st.success("✅ Predictions generated!")
                    
                    # Display predictions in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Bitcoin (BTC)",
                            f"${predictions['Bitcoin']:,.2f}",
                            help="Predicted price using SARIMAX model"
                        )
                    
                    with col2:
                        st.metric(
                            "Ethereum (ETH)", 
                            f"${predictions['Ethereum']:,.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Litecoin (LTC)",
                            f"${predictions['Litecoin']:,.2f}"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
    
    with tab2:
        st.subheader("📊 Trading Signals")
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox("Select Symbol", ["BTC/USDT", "ETH/USDT", "LTC/USDT"])
        with col2:
            get_signals_button = st.button("📈 Get Signals", type="primary")
        
        if get_signals_button:
            with st.spinner("Analyzing market data..."):
                try:
                    signals = bot.get_trading_signals(symbol)
                    
                    # Display signal with colored indicator
                    if signals['action'] == 'BUY':
                        st.success(f"🟢 **BUY SIGNAL** for {symbol}")
                    elif signals['action'] == 'SELL':
                        st.error(f"🔴 **SELL SIGNAL** for {symbol}")
                    else:
                        st.info(f"🟡 **HOLD** for {symbol}")
                    
                    # Display technical indicators
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${signals['current_price']:,.2f}")
                    with col2:
                        st.metric("Short SMA (50)", f"${signals['sma_short']:,.2f}")
                    with col3:
                        st.metric("Long SMA (200)", f"${signals['sma_long']:,.2f}")
                    
                    # Execute trade option
                    if signals['action'] in ['BUY', 'SELL']:
                        st.divider()
                        st.subheader("🤖 Execute Trade")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            trade_amount = st.number_input(
                                "Trade Amount (BTC/ETH/LTC)", 
                                value=float(os.getenv('DEFAULT_TRADE_AMOUNT', 0.001)),
                                min_value=0.0001,
                                step=0.0001,
                                format="%.4f"
                            )
                        
                        with col2:
                            execute_button = st.button(
                                f"Execute {signals['action']} Order", 
                                type="primary"
                            )
                        
                        if execute_button:
                            with st.spinner(f"Executing {signals['action']} order..."):
                                result = bot.execute_trade(symbol, signals['action'], trade_amount)
                                
                                if result['success']:
                                    st.success(f"✅ Trade executed successfully!")
                                    st.json(result['order_info'])
                                else:
                                    st.error(f"❌ Trade failed: {result['error']}")
                
                except Exception as e:
                    st.error(f"❌ Failed to get signals: {str(e)}")
    
    with tab3:
        st.subheader("💼 Portfolio Status")
        
        refresh_button = st.button("🔄 Refresh Portfolio", type="secondary")
        
        if refresh_button or True:  # Auto-load on tab switch
            try:
                portfolio = bot.get_portfolio_status()
                
                if portfolio['success']:
                    st.success("✅ Portfolio data loaded")
                    
                    # Display balances
                    st.subheader("💰 Account Balances")
                    balances_df = pd.DataFrame(portfolio['balances'])
                    if not balances_df.empty:
                        st.dataframe(balances_df, use_container_width=True)
                    else:
                        st.info("No significant balances found")
                    
                    # Display recent trades
                    if portfolio.get('recent_trades'):
                        st.subheader("📋 Recent Trades")
                        trades_df = pd.DataFrame(portfolio['recent_trades'])
                        st.dataframe(trades_df, use_container_width=True)
                
                else:
                    st.error(f"❌ Failed to load portfolio: {portfolio['error']}")
            
            except Exception as e:
                st.error(f"❌ Portfolio error: {str(e)}")

    # Footer with warnings
    st.divider()
    st.warning("""
    ⚠️ **Risk Disclaimer**: 
    - Cryptocurrency trading involves significant risk
    - This bot is for educational purposes
    - Never invest more than you can afford to lose
    - Always test in sandbox/paper mode first
    """)

if __name__ == '__main__':
    main()
