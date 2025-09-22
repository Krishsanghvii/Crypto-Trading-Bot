import os
import ccxt
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoTradingBot:
    def __init__(self, trading_mode='sandbox'):
        """
        Initialize the crypto trading bot
        
        Args:
            trading_mode (str): 'sandbox', 'paper', or 'live'
        """
        self.trading_mode = trading_mode
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')
        
        # Risk management parameters
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', 0.1))
        self.stop_loss_percent = float(os.getenv('STOP_LOSS_PERCENT', 0.02))
        self.take_profit_percent = float(os.getenv('TAKE_PROFIT_PERCENT', 0.05))
        
        # Technical analysis parameters
        self.sma_short_period = 50
        self.sma_long_period = 200
        
        # Initialize exchange
        self._initialize_exchange()
        
        # Cache for market data
        self.data_cache = {}
        self.cache_duration = int(os.getenv('DATA_CACHE_HOURS', 1)) * 3600  # Convert to seconds
        
        logger.info(f"Trading bot initialized in {trading_mode} mode")
    
    def _initialize_exchange(self):
        """Initialize Binance exchange connection"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': self.trading_mode == 'sandbox',
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'  # Use spot trading
                }
            })
            
            # Test connection
            if self.trading_mode != 'paper':
                self.exchange.load_markets()
                logger.info("✅ Successfully connected to Binance")
            else:
                logger.info("✅ Paper trading mode - no real connection needed")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize exchange: {e}")
            raise
    
    def get_market_data(self, symbol='BTC/USDT', timeframe='1h', limit=500):
        """
        Get market data with caching to improve performance
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Candle timeframe
            limit (int): Number of candles to fetch
        
        Returns:
            pd.DataFrame: OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}_{limit}"
        current_time = time.time()
        
        # Check cache
        if (cache_key in self.data_cache and 
            current_time - self.data_cache[cache_key]['timestamp'] < self.cache_duration):
            logger.info(f"Using cached data for {symbol}")
            return self.data_cache[cache_key]['data']
        
        try:
            if self.trading_mode == 'paper':
                # For paper trading, simulate some data
                logger.info(f"Simulating market data for {symbol}")
                dates = pd.date_range(end=datetime.now(), periods=limit, freq='H')
                base_price = 50000 if 'BTC' in symbol else 3000
                prices = base_price + np.random.randn(limit).cumsum() * 100
                
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': prices,
                    'high': prices * 1.02,
                    'low': prices * 0.98,
                    'close': prices,
                    'volume': np.random.randint(1000, 10000, limit)
                })
            else:
                # Fetch real data from Binance
                logger.info(f"Fetching market data for {symbol}")
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Cache the data
            self.data_cache[cache_key] = {
                'data': df,
                'timestamp': current_time
            }
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch market data: {e}")
            raise
    
    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    def get_trading_signals(self, symbol='BTC/USDT'):
        """
        Generate trading signals based on SMA crossover strategy
        
        Args:
            symbol (str): Trading pair symbol
        
        Returns:
            dict: Trading signals and technical indicators
        """
        try:
            # Get market data
            df = self.get_market_data(symbol)
            
            if len(df) < self.sma_long_period:
                raise ValueError(f"Not enough data points. Need at least {self.sma_long_period}")
            
            # Calculate SMAs
            df['sma_short'] = self.calculate_sma(df['close'], self.sma_short_period)
            df['sma_long'] = self.calculate_sma(df['close'], self.sma_long_period)
            
            # Get latest values
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            current_price = float(latest['close'])
            sma_short = float(latest['sma_short'])
            sma_long = float(latest['sma_long'])
            
            # Determine trading signals
            buy_signal = (sma_short > sma_long and 
                         previous['sma_short'] <= previous['sma_long'])  # Golden cross
            sell_signal = (sma_short < sma_long and 
                          previous['sma_short'] >= previous['sma_long'])  # Death cross
            
            # Determine action
            if buy_signal:
                action = 'BUY'
            elif sell_signal:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            signals = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'sma_short': sma_short,
                'sma_long': sma_long,
                'action': action,
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'confidence': abs(sma_short - sma_long) / sma_long  # Distance between SMAs as confidence
            }
            
            logger.info(f"Generated signals for {symbol}: {action}")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Failed to generate signals: {e}")
            raise
    
    def calculate_position_size(self, symbol, action, balance):
        """
        Calculate position size based on risk management rules
        
        Args:
            symbol (str): Trading pair
            action (str): BUY or SELL
            balance (float): Available balance
        
        Returns:
            float: Position size
        """
        if action == 'BUY':
            # For buying, use percentage of USDT balance
            max_usd_amount = balance * self.max_position_size
            return max_usd_amount
        else:
            # For selling, use percentage of crypto balance
            return balance * self.max_position_size
    
    def execute_trade(self, symbol, action, amount=None):
        """
        Execute a trade with risk management
        
        Args:
            symbol (str): Trading pair
            action (str): BUY or SELL
            amount (float): Trade amount (optional)
        
        Returns:
            dict: Trade execution result
        """
        try:
            logger.info(f"Attempting to execute {action} order for {symbol}")
            
            if self.trading_mode == 'paper':
                # Paper trading - simulate order
                order_info = {
                    'id': f"paper_{int(time.time())}",
                    'symbol': symbol,
                    'type': 'market',
                    'side': action.lower(),
                    'amount': amount or float(os.getenv('DEFAULT_TRADE_AMOUNT', 0.001)),
                    'price': self.get_current_price(symbol),
                    'status': 'filled',
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"✅ Paper trade executed: {order_info}")
                return {
                    'success': True,
                    'order_info': order_info,
                    'message': 'Paper trade executed successfully'
                }
            
            # Real trading
            current_price = self.get_current_price(symbol)
            
            # Get account balance
            balance = self.get_account_balance(symbol, action)
            if not balance:
                return {
                    'success': False,
                    'error': 'Insufficient balance'
                }
            
            # Calculate position size if not provided
            if not amount:
                if action == 'BUY':
                    amount = self.calculate_position_size(symbol, action, balance) / current_price
                else:
                    amount = self.calculate_position_size(symbol, action, balance)
            
            # Validate minimum order size
            market_info = self.exchange.market(symbol)
            min_amount = market_info['limits']['amount']['min']
            
            if amount < min_amount:
                return {
                    'success': False,
                    'error': f'Amount {amount} below minimum {min_amount}'
                }
            
            # Execute market order
            if action == 'BUY':
                order = self.exchange.create_market_buy_order(symbol, amount)
            else:
                order = self.exchange.create_market_sell_order(symbol, amount)
            
            # Set stop-loss and take-profit (if supported)
            self._set_risk_management_orders(symbol, action, current_price, amount)
            
            logger.info(f"✅ Trade executed successfully: {order['id']}")
            
            return {
                'success': True,
                'order_info': {
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'type': order['type'],
                    'side': order['side'],
                    'amount': order['amount'],
                    'price': order.get('average', current_price),
                    'status': order['status'],
                    'timestamp': order['timestamp']
                },
                'message': 'Trade executed successfully'
            }
            
        except ccxt.InsufficientFunds:
            error_msg = 'Insufficient funds for trade'
            logger.error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
            
        except ccxt.InvalidOrder as e:
            error_msg = f'Invalid order: {str(e)}'
            logger.error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
            
        except Exception as e:
            error_msg = f'Trade execution failed: {str(e)}'
            logger.error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def get_current_price(self, symbol):
        """Get current market price for symbol"""
        try:
            if self.trading_mode == 'paper':
                # Simulate price for paper trading
                base_prices = {
                    'BTC/USDT': 50000,
                    'ETH/USDT': 3000,
                    'LTC/USDT': 100
                }
                base_price = base_prices.get(symbol, 50000)
                return base_price + np.random.randn() * base_price * 0.01
            
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
            
        except Exception as e:
            logger.error(f"❌ Failed to get current price: {e}")
            raise
    
    def get_account_balance(self, symbol, action):
        """Get account balance for trading"""
        try:
            if self.trading_mode == 'paper':
                # Return simulated balance for paper trading
                return 1000.0 if action == 'BUY' else 0.1
            
            balance = self.exchange.fetch_balance()
            
            if action == 'BUY':
                # For buying, need USDT balance
                return balance['USDT']['free']
            else:
                # For selling, need crypto balance
                base_currency = symbol.split('/')[0]
                return balance[base_currency]['free']
                
        except Exception as e:
            logger.error(f"❌ Failed to get balance: {e}")
            return 0
    
    def _set_risk_management_orders(self, symbol, action, entry_price, amount):
        """Set stop-loss and take-profit orders (if supported by exchange)"""
        try:
            if self.trading_mode == 'paper':
                logger.info("Paper trading: Risk management orders simulated")
                return
            
            if action == 'BUY':
                # Set stop-loss below entry price
                stop_price = entry_price * (1 - self.stop_loss_percent)
                # Set take-profit above entry price  
                profit_price = entry_price * (1 + self.take_profit_percent)
            else:
                # For short positions (if supported)
                stop_price = entry_price * (1 + self.stop_loss_percent)
                profit_price = entry_price * (1 - self.take_profit_percent)
            
            logger.info(f"Risk management: Stop-loss at {stop_price:.2f}, Take-profit at {profit_price:.2f}")
            
            # Note: Implementation depends on exchange support for OCO orders
            # This is a simplified version
            
        except Exception as e:
            logger.warning(f"⚠️ Could not set risk management orders: {e}")
    
    def get_portfolio_status(self):
        """Get current portfolio status and recent trades"""
        try:
            if self.trading_mode == 'paper':
                return {
                    'success': True,
                    'balances': [
                        {'currency': 'USDT', 'free': 1000.0, 'used': 0.0, 'total': 1000.0},
                        {'currency': 'BTC', 'free': 0.1, 'used': 0.0, 'total': 0.1}
                    ],
                    'recent_trades': [],
                    'total_value_usd': 6000.0
                }
            
            # Get account balance
            balance = self.exchange.fetch_balance()
            
            # Filter out zero balances
            significant_balances = []
            for currency, amounts in balance.items():
                if isinstance(amounts, dict) and amounts.get('total', 0) > 0:
                    significant_balances.append({
                        'currency': currency,
                        'free': amounts['free'],
                        'used': amounts['used'], 
                        'total': amounts['total']
                    })
            
            # Get recent trades (last 10)
            try:
                recent_trades = self.exchange.fetch_my_trades(limit=10)
                trades_info = []
                for trade in recent_trades:
                    trades_info.append({
                        'id': trade['id'],
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'amount': trade['amount'],
                        'price': trade['price'],
                        'cost': trade['cost'],
                        'timestamp': trade['datetime']
                    })
            except:
                trades_info = []
            
            return {
                'success': True,
                'balances': significant_balances,
                'recent_trades': trades_info
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get portfolio status: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Backward compatibility functions
def get_crypto_trading(api_key, api_secret):
    """Legacy function for backward compatibility"""
    try:
        bot = CryptoTradingBot('paper')  # Default to paper trading for legacy calls
        signals = bot.get_trading_signals('BTC/USDT')
        
        signal_text = f"{signals['action']} signal detected. Execute {signals['action'].lower()} order."
        
        return (
            signal_text,
            signals['sma_short'], 
            signals['sma_long'],
            signals['buy_signal'],
            signals['sell_signal']
        )
        
    except Exception as e:
        logger.error(f"❌ Legacy function failed: {e}")
        return (
            "Error generating signal",
            0, 0, False, False
        )
