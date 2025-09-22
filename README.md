# Crypto Trading Bot

A sophisticated cryptocurrency trading bot with machine learning price prediction and automated trading capabilities.

## 🚀 Features

- **Machine Learning Predictions**: SARIMAX time series models for BTC, ETH, LTC price forecasting
- **Technical Analysis**: SMA crossover strategy with customizable parameters
- **Risk Management**: Built-in stop-loss, take-profit, and position sizing
- **Multiple Trading Modes**: Sandbox, Paper Trading, and Live Trading
- **Real-time Data**: Binance API integration for live market data
- **Professional UI**: Streamlit web interface with portfolio tracking
- **Performance Optimized**: Data caching and efficient API calls

## 📋 Prerequisites

1. **Python 3.8+**
2. **Binance Account** with API access
3. **API Keys** with appropriate permissions

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Krishsanghvii/Crypto-Trading-Bot.git
   cd Crypto-Trading-Bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your credentials:
   ```env
   # Binance API Credentials
   API_KEY=your_binance_api_key_here
   API_SECRET=your_binance_secret_key_here
   
   # Trading Configuration
   TRADING_MODE=sandbox
   MAX_POSITION_SIZE=0.1
   STOP_LOSS_PERCENT=0.02
   TAKE_PROFIT_PERCENT=0.05
   ```

## 🔐 Getting Binance API Keys

1. **Create Binance Account**: [binance.com](https://www.binance.com/)
2. **Go to API Management**: Account → API Management
3. **Create New API Key**:
   - Label: "Trading Bot"
   - Restrictions: Enable Spot Trading
   - IP Restriction: Add your server IP (recommended)
4. **Copy API Key and Secret** to your `.env` file

⚠️ **Security**: Never share your API keys or commit them to version control!

## 🚦 Quick Start

1. **Test Connection** (Sandbox Mode)
   ```bash
   python -c "from bot import CryptoTradingBot; bot = CryptoTradingBot('sandbox'); print('✅ Connection successful')"
   ```

2. **Run the Web Interface**
   ```bash
   streamlit run app.py
   ```

3. **Access the Bot**: Open http://localhost:8501 in your browser

## 🎮 Trading Modes

### 1. **Sandbox Mode** (Recommended for Testing)
- Uses Binance testnet
- No real money involved
- Perfect for testing strategies

### 2. **Paper Trading Mode**
- Simulates trades with fake data
- No API calls to exchanges
- Good for strategy validation

### 3. **Live Trading Mode**
- Real money trading
- Requires funded Binance account
- ⚠️ **Use with caution!**

## 📊 Usage

### Price Predictions
```python
from model import get_price_prediction_model
from datetime import datetime, timedelta

# Predict prices for next week
future_date = datetime.now() + timedelta(days=7)
predictions = get_price_prediction_model(future_date)

print(f"BTC: ${predictions['Bitcoin']:,.2f}")
print(f"ETH: ${predictions['Ethereum']:,.2f}")
print(f"LTC: ${predictions['Litecoin']:,.2f}")
```

### Trading Signals
```python
from bot import CryptoTradingBot

bot = CryptoTradingBot('sandbox')
signals = bot.get_trading_signals('BTC/USDT')

print(f"Action: {signals['action']}")
print(f"Current Price: ${signals['current_price']:,.2f}")
print(f"Confidence: {signals['confidence']:.2%}")
```

### Execute Trades
```python
# Only in sandbox/live mode
result = bot.execute_trade('BTC/USDT', 'BUY', 0.001)

if result['success']:
    print("✅ Trade executed successfully")
    print(f"Order ID: {result['order_info']['id']}")
else:
    print(f"❌ Trade failed: {result['error']}")
```

## ⚙️ Configuration

### Risk Management Settings
```env
MAX_POSITION_SIZE=0.1      # Max 10% of portfolio per trade
STOP_LOSS_PERCENT=0.02     # 2% stop loss
TAKE_PROFIT_PERCENT=0.05   # 5% take profit
```

### Technical Analysis Parameters
```python
# In bot.py - customize as needed
self.sma_short_period = 50   # Short-term SMA
self.sma_long_period = 200   # Long-term SMA
```

### Performance Optimization
```env
DATA_CACHE_HOURS=1         # Cache market data for 1 hour
DEFAULT_TRADE_AMOUNT=0.001  # Default trade size
```

## 📈 Strategy Details

### SMA Crossover Strategy
- **Buy Signal**: When 50-period SMA crosses above 200-period SMA (Golden Cross)
- **Sell Signal**: When 50-period SMA crosses below 200-period SMA (Death Cross)
- **Confidence**: Based on distance between SMAs

### Machine Learning Predictions
- **Model**: SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)
- **Data**: Historical price data from Yahoo Finance
- **Caching**: Models cached for 1 hour to improve performance
- **Fallback**: Synthetic data generation if API fails

## 🔧 Troubleshooting

### Common Issues

1. **API Connection Failed**
   ```
   ❌ Failed to connect to Binance: Invalid API key
   ```
   - Check your API keys in `.env`
   - Ensure API key has correct permissions
   - Verify IP restrictions

2. **Insufficient Funds**
   ```
   ❌ Insufficient funds for trade
   ```
   - Check account balance
   - Reduce trade amount
   - Switch to sandbox mode for testing

3. **Model Prediction Errors**
   ```
   ❌ Failed to fetch market data
   ```
   - Check internet connection
   - Yahoo Finance API might be down
   - Bot will use fallback synthetic data

### Debug Mode
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
streamlit run app.py
```

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Cloud Deployment (Heroku)
1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy:
   ```bash
   heroku create your-crypto-bot
   git push heroku main
   ```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

## ⚠️ Risk Disclaimer

- **High Risk**: Cryptocurrency trading involves substantial risk of loss
- **Educational Purpose**: This bot is for educational and research purposes
- **No Guarantees**: Past performance does not guarantee future results
- **Start Small**: Always test with small amounts first
- **Use Stop-Losses**: Never trade without proper risk management

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Krish Sanghvi**
- GitHub: [@krishsanghvii](https://github.com/krishsanghvii)
- LinkedIn: [krishpsanghvi](https://linkedin.com/in/krishpsanghvi)
- Email: ksanghvi@gmu.edu

## 🙏 Acknowledgments

- Binance API for real-time market data
- CCXT library for exchange integration
- Streamlit for the web interface
- Yahoo Finance for historical data

---

⭐ **Star this repo if you found it helpful!** ⭐
