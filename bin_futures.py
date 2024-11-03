import os
from dotenv import load_dotenv
import time
import logging
import math
import pandas as pd
import traceback
import asyncio
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import (
    SIDE_BUY,
    SIDE_SELL,
    ORDER_TYPE_MARKET,
)

from indicators_calculator import IndicatorCalculator

# Load environment variables
load_dotenv()

MAX_PERIOD = 200
CHECK_INTERVAL = 900  # Main loop interval in seconds

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)

class BinanceClient:
    def __init__(self, api_key, api_secret, testnet=False):
        self.testnet = testnet    
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.position_mode = self.get_position_mode()
                
        if testnet:
            try:
                # Test connection by getting account info
                account = self.client.futures_account()
                logging.info(f"Successfully connected to futures testnet. Available balance: {account.get('totalWalletBalance', 'Unknown')} USDT")
                
            except BinanceAPIException as e:
                logging.error(f"Error connecting to futures testnet: {str(e)}")
                raise
        
        self.initialized = True
        
    def __del__(self):
        # Cleanup if needed
        pass

    def is_initialized(self):
        return self.initialized

    def get_account_info(self):
        """Get futures account information"""
        return self.client.futures_account()
    
    def get_order_book(self, symbol, limit=20):
        """Get futures order book"""
        return self.client.futures_order_book(symbol=symbol, limit=limit)
            
    def get_symbol_ticker(self, symbol):
        """Get futures symbol ticker"""
        return self.client.futures_symbol_ticker(symbol=symbol)
            
    def get_recent_trades(self, symbol, limit=100):
        """Get futures recent trades"""
        return self.client.futures_recent_trades(symbol=symbol, limit=limit)
            
    def get_klines(self, symbol, interval, limit=500):
        """Get futures klines/candlestick data"""
        return self.client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
    
    def futures_ticker(self, symbol=None):
        """Get futures ticker information"""
        try:
            if symbol:
                return self.client.futures_ticker(symbol=symbol)
            return self.client.futures_ticker()
        except BinanceAPIException as e:
            logging.error(f"Error getting futures ticker: {str(e)}")
            return None
        
    def get_symbol_info(self, symbol):
        """Get futures symbol information"""
        try:
            exchange_info = self.client.futures_exchange_info()
            for sym_info in exchange_info['symbols']:
                if sym_info['symbol'] == symbol:
                    return {
                        'symbol': sym_info['symbol'],
                        'quantityPrecision': sym_info['quantityPrecision'],
                        'pricePrecision': sym_info['pricePrecision'],
                        'filters': sym_info['filters'],
                        'minQty': next(
                            (float(f['minQty']) for f in sym_info['filters'] 
                             if f['filterType'] == 'LOT_SIZE'),
                            0.001
                        ),
                        'stepSize': next(
                            (float(f['stepSize']) for f in sym_info['filters'] 
                             if f['filterType'] == 'LOT_SIZE'),
                            0.001
                        ),
                        'minNotional': next(
                            (float(f['notional']) for f in sym_info['filters'] 
                             if f['filterType'] == 'MIN_NOTIONAL'),
                            5.0
                        )
                    }
            return None
        except BinanceAPIException as e:
            logging.error(f"Error fetching futures symbol info: {str(e)}")
            return None
        
    def futures_position_information(self, symbol=None):
        """Get futures position information"""
        try:
            if symbol:
                return self.client.futures_position_information(symbol=symbol)
            return self.client.futures_position_information()
        except BinanceAPIException as e:
            logging.error(f"Error getting futures position information: {str(e)}")
            return None
            
    def create_order(self, **kwargs):
        """Create an order with error handling"""
        try:
            return self.client.futures_create_order(**kwargs)
        except BinanceAPIException as e:
            logging.error(f"Error creating order: {str(e)}")
            return None
        
    def get_position_mode(self):
        """Get current position mode (Hedge or One-way)"""
        try:
            mode = self.client.futures_get_position_mode()
            return mode['dualSidePosition']  # True for Hedge Mode, False for One-way
        except BinanceAPIException as e:
            logging.error(f"Error getting position mode: {str(e)}")
            return False  # Default to One-way mode
        
    def query_futures_order(self, symbol, orderId):
        """Query a futures order status"""
        try:
            return self.client.futures_get_order(
                symbol=symbol,
                orderId=orderId
            )
        except BinanceAPIException as e:
            logging.error(f"Error querying futures order: {str(e)}")
            return None

    def cancel_order(self, symbol, orderId):
        """Cancel a futures order"""
        try:
            return self.client.futures_cancel_order(
                symbol=symbol,
                orderId=orderId
            )
        except BinanceAPIException as e:
            logging.error(f"Error cancelling futures order: {str(e)}")
            return None

    def set_position_mode(self, hedge_mode=False):
        """Set position mode to Hedge or One-way"""
        try:
            if self.position_mode != hedge_mode:
                self.client.futures_change_position_mode(dualSidePosition=hedge_mode)
                self.position_mode = hedge_mode
                logging.info(f"Position mode changed to: {'Hedge' if hedge_mode else 'One-way'}")
            return True
        except BinanceAPIException as e:
            logging.error(f"Error setting position mode: {str(e)}")
            return False

    def set_leverage(self, symbol, leverage):
        """Set leverage for a symbol"""
        try:
            response = self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            logging.info(f"Leverage set for {symbol}: {leverage}x")
            return response
        except BinanceAPIException as e:
            logging.error(f"Error setting leverage: {str(e)}")
            return None

    def set_margin_type(self, symbol, margin_type='ISOLATED'):
        """Set margin type for a symbol"""
        try:
            response = self.client.futures_change_margin_type(
                symbol=symbol,
                marginType=margin_type
            )
            logging.info(f"Margin type set for {symbol}: {margin_type}")
            return response
        except BinanceAPIException as e:
            logging.error(f"Error setting margin type: {str(e)}")
            return None

    def get_leverage_brackets(self, symbol):
        """Get leverage brackets for a symbol"""
        try:
            brackets = self.client.futures_leverage_bracket(symbol=symbol)
            return brackets[0]['brackets'] if brackets else None
        except BinanceAPIException as e:
            logging.error(f"Error getting leverage brackets: {str(e)}")
            return None
        
    def futures_exchange_info(self):
        """Get futures exchange information"""
        try:
            return self.client.futures_exchange_info()
        except BinanceAPIException as e:
            logging.error(f"Error getting futures exchange info: {str(e)}")
            return None
        
    def futures_funding_rate(self, symbol, limit=1):
        """Get funding rate information"""
        try:
            return self.client.futures_funding_rate(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            logging.error(f"Error getting funding rate: {str(e)}")
            return None

    def futures_mark_price(self, symbol):
        """Get mark price information"""
        try:
            return self.client.futures_mark_price(symbol=symbol)
        except BinanceAPIException as e:
            logging.error(f"Error getting mark price: {str(e)}")
            return None

    def futures_open_interest(self, symbol):
        """Get open interest information"""
        try:
            return self.client.futures_open_interest(symbol=symbol)
        except BinanceAPIException as e:
            logging.error(f"Error getting open interest: {str(e)}")
            return None

class MarketData:
    BINANCE_TIMEFRAME_MAP = {
        1: Client.KLINE_INTERVAL_1MINUTE,    
        5: Client.KLINE_INTERVAL_5MINUTE,    
        15: Client.KLINE_INTERVAL_15MINUTE,  
        30: Client.KLINE_INTERVAL_30MINUTE,   
    }

    def __init__(self, symbol, timeframes, client):
        self.symbol = symbol
        self.client = client
        self.original_timeframes = timeframes
        self.timeframes = [self.BINANCE_TIMEFRAME_MAP[tf] for tf in timeframes]
        self.num_candles = {tf: None for tf in self.timeframes}
        
        try:
            self.symbol_info = self.client.get_symbol_info(symbol)
        except Exception as e:
            logging.error(f"Error fetching symbol info for {symbol}: {str(e)}")
            self.symbol_info = None

    def calculate_num_candles(self, timeframe):
        """Convert Binance interval to minutes"""
        minutes_map = {
            Client.KLINE_INTERVAL_1MINUTE: 1,
            Client.KLINE_INTERVAL_5MINUTE: 5,
            Client.KLINE_INTERVAL_15MINUTE: 15,
            Client.KLINE_INTERVAL_30MINUTE: 30,
        }
        
        if timeframe not in minutes_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        timeframe_in_minutes = minutes_map[timeframe]
        num_candles = MAX_PERIOD * 2
        self.num_candles[timeframe] = int(num_candles / timeframe_in_minutes)

    def fetch_data(self, timeframe):
        """Fetch and clean market data"""
        if self.num_candles[timeframe] is None:
            self.calculate_num_candles(timeframe)

        try:
            # Get klines data
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=timeframe,
                limit=self.num_candles[timeframe]
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Basic data cleaning
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert price and volume columns to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Simple cleaning: forward fill then any remaining NaNs to 0
            df = df.ffill().fillna(0)
            
            return df

        except Exception as e:
            logging.error(f"Error fetching data for {self.symbol}: {str(e)}")
            return None

    def analyze_market_depth(self):
        """Analyze order book depth"""
        try:
            order_book = self.client.get_order_book(symbol=self.symbol, limit=20)
            
            # Calculate total bid/ask volumes
            bid_volume = sum(float(bid[1]) for bid in order_book['bids'])
            ask_volume = sum(float(ask[1]) for ask in order_book['asks'])
            
            # Calculate imbalance
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return {'liquidity_score': 0, 'imbalance': 0, 'spread': 0}
                
            imbalance = (bid_volume - ask_volume) / total_volume
            
            # Calculate spread
            best_bid = float(order_book['bids'][0][0])
            best_ask = float(order_book['asks'][0][0])
            spread = (best_ask - best_bid) / best_bid * 100
            
            # Calculate liquidity score
            liquidity_score = bid_volume * ask_volume / total_volume * (1 / (1 + spread))
            
            return {
                'liquidity_score': liquidity_score,
                'imbalance': imbalance,
                'spread': spread
            }
        except Exception as e:
            logging.error(f"Error in market depth analysis: {str(e)}")
            return None
               
    def get_funding_rate(self):
        """Get funding rate information"""
        try:
            # Get most recent funding rate
            funding_data = self.client.futures_funding_rate(
                symbol=self.symbol,
                limit=1
            )
            if funding_data:
                return funding_data[0]
            return None
        except Exception as e:
            logging.error(f"Error getting funding rate for {self.symbol}: {str(e)}")
            return None

    def get_next_funding_time(self):
        """Get next funding time"""
        try:
            # Get premium index info which includes next funding time
            premium_info = self.client.futures_mark_price(symbol=self.symbol)
            if premium_info:
                next_funding = pd.to_datetime(premium_info['nextFundingTime'], unit='ms')
                current_time = pd.Timestamp.now()
                
                # If next funding time is in the past, add 8 hours to get the next one
                while next_funding < current_time:
                    next_funding += pd.Timedelta(hours=8)
                
                hours_to_funding = (next_funding - current_time).total_seconds() / 3600
                
                return {
                    'next_time': next_funding,
                    'hours_remaining': hours_to_funding
                }
            return None
        except Exception as e:
            logging.error(f"Error getting next funding time for {self.symbol}: {str(e)}")
            return None

    def get_mark_price(self):
        """Get current mark price"""
        try:
            mark_price_info = self.client.futures_mark_price(symbol=self.symbol)
            if mark_price_info:
                return {
                    'mark_price': float(mark_price_info['markPrice']),
                    'index_price': float(mark_price_info['indexPrice']),
                    'basis': float(mark_price_info['markPrice']) - float(mark_price_info['indexPrice'])
                }
            return None
        except Exception as e:
            logging.error(f"Error getting mark price for {self.symbol}: {str(e)}")
            return None

    def get_open_interest(self):
        """Get open interest information"""
        try:
            open_interest = self.client.futures_open_interest(symbol=self.symbol)
            if open_interest:
                return {
                    'open_interest': float(open_interest['openInterest']),
                    'time': pd.to_datetime(open_interest['time'], unit='ms')
                }
            return None
        except Exception as e:
            logging.error(f"Error getting open interest for {self.symbol}: {str(e)}")
            return None

    def get_futures_metrics(self):
        """Get essential futures trading metrics"""
        try:
            metrics = {}
            
            # Get funding rate
            funding_info = self.get_funding_rate()
            if funding_info:
                metrics['funding_rate'] = funding_info
                
            # Get next funding time
            funding_time = self.get_next_funding_time()
            if funding_time:
                metrics['next_funding_time'] = funding_time
                
            # Get mark price and basis
            mark_price_info = self.get_mark_price()
            if mark_price_info:
                metrics['mark_price'] = mark_price_info
                
            # Get open interest
            open_interest = self.get_open_interest()
            if open_interest:
                metrics['open_interest'] = open_interest
                
            if not metrics:
                return None
                
            return metrics
            
        except Exception as e:
            logging.error(f"Error getting futures metrics for {self.symbol}: {str(e)}")
            return None
        
class FuturesStrategyLogic:
    def __init__(self):
        self.min_funding_threshold = 0.0005  # 0.05%
        self.max_funding_threshold = 0.01    # 1%
        self.min_hours_to_funding = 1
        
    def calculate_confidence(self, trend, funding_signal, time_to_funding, price_data):
        """Calculate trade confidence based on futures metrics and trend alignment"""
        try:
            confidence = 0
            funding_trend_aligned = False 
            
            # Trend strength (0-40 points)
            ema_diff = abs(price_data['ema_short'].iloc[-1] - price_data['ema_long'].iloc[-1])
            ema_trend_strength = min(40, (ema_diff / price_data['ema_long'].iloc[-1]) * 2000)
            
            # Add trend alignment bonus (up to 20 extra points)
            if funding_signal['rate'] != 0:  # Only if there's a funding rate
                # Positive funding rate means longs pay shorts
                # Negative funding rate means shorts pay longs
                funding_trend_aligned = (
                    (funding_signal['rate'] > 0 and trend == 'bearish') or  # High funding = good for shorts
                    (funding_signal['rate'] < 0 and trend == 'bullish')     # Negative funding = good for longs
                )
                if funding_trend_aligned:
                    confidence += min(20, abs(funding_signal['rate'] * 4000))  # Up to 20 points for alignment
            
            confidence += ema_trend_strength
            
            # Funding rate contribution (0-30 points)
            funding_rate = abs(float(funding_signal['rate']))
            if funding_rate > self.min_funding_threshold:
                # More points for higher funding rates (up to a limit)
                funding_points = min(30, funding_rate * 6000)
                confidence += funding_points
            
            # Time to funding contribution (0-30 points)
            hours_remaining = float(time_to_funding['hours_remaining'])
            if hours_remaining > 4:  # More than 4 hours to funding
                confidence += 30
            elif hours_remaining > 2:  # 2-4 hours to funding
                confidence += 15
            elif hours_remaining > 1:  # 1-2 hours to funding
                confidence += 5
            
            logging.info(f"Confidence breakdown:")
            logging.info(f"  - EMA Trend Strength: {ema_trend_strength:.1f}")
            logging.info(f"  - Trend-Funding Alignment: {20 if funding_trend_aligned else 0} points")
            logging.info(f"  - Funding Rate Contribution: {min(30, funding_rate * 6000):.1f}")
            logging.info(f"  - Time to Funding Contribution: {30 if hours_remaining > 4 else 15 if hours_remaining > 2 else 5 if hours_remaining > 1 else 0}")
            
            return confidence
            
        except Exception as e:
            logging.error(f"Error calculating confidence: {str(e)}")
            return 50  # Default middle confidenc
        
    def analyze_setup(self, price_data, futures_metrics):
        try:
            # Trend direction
            trend = self._get_trend_direction(price_data)
            
            # Funding rate check
            funding_signal = self._analyze_funding(
                futures_metrics['funding_rate'],
                trend
            )
            
            # Check if near funding time
            time_to_funding = futures_metrics['next_funding_time']
            hours_to_funding = float(time_to_funding['hours_remaining'])
            
            # Calculate confidence
            confidence = self.calculate_confidence(
                trend,
                funding_signal,
                time_to_funding,
                price_data
            )
            
            # Final trade decision
            should_trade = (
                funding_signal['is_valid'] and
                hours_to_funding >= self.min_hours_to_funding and  
                confidence > 60
            )
            
            return {
                'should_trade': should_trade,
                'trend': trend,
                'funding_signal': funding_signal,
                'time_to_funding': time_to_funding,
                'confidence': confidence,
                'hours_to_funding': hours_to_funding
            }
            
        except Exception as e:
            logging.error(f"Error in futures analysis: {str(e)}")
            return None
            
    def _get_trend_direction(self, data):
        """Get trend using EMA crossover"""
        return 'bullish' if data['ema_short'].iloc[-1] > data['ema_long'].iloc[-1] else 'bearish'
        
    def _analyze_funding(self, funding_rate, trend):
        """Analyze funding rate for trade direction"""
        rate = float(funding_rate['fundingRate'])
        
        return {
            'is_valid': abs(rate) > self.min_funding_threshold,
            'favor_long': rate < -self.min_funding_threshold and trend == 'bullish',
            'favor_short': rate > self.min_funding_threshold and trend == 'bearish',
            'rate': rate
        }
            
class TradeManager:
    def __init__(self, client, market_data):
        self.contract_manager = FuturesContractManager(client)    
        self.client = client
        self.market_data = market_data
        self.timeframes = market_data.timeframes
        self.indicator_calc = IndicatorCalculator(client)
        self.order_manager = OrderManager(client, market_data)
        self.risk_monitor = RiskMonitor(client)
        self.active_positions = {}
    
    async def check_for_signals(self, symbol):
        try:
            # First check account risk before any trading
            account_risk = await self.risk_monitor.check_account_risk()
            if not account_risk or not account_risk['is_safe']:
                logging.warning(f"[{symbol}] Account risk check failed:")
                for warning in account_risk.get('warnings', []):
                    logging.warning(f"- {warning}")
                return

            # Check contract status
            contract_status = self.contract_manager.check_contract_expiry(symbol)
            if not contract_status['is_valid']:
                logging.warning(f"[{symbol}] Contract validation failed: {contract_status.get('warning', 'Unknown error')}")
                return

            # Check liquidity
            base_asset = symbol.replace('USDT', '')
            liquidity_analysis = self.contract_manager.analyze_contract_liquidity(base_asset)
            if not liquidity_analysis['is_valid']:
                logging.warning(f"[{symbol}] Liquidity check failed: {liquidity_analysis.get('warning', 'Unknown error')}")
                return
            
            logging.info(f"\n[{symbol}] Starting signal analysis...")
            
            # Fetch and validate data
            data = {tf: self.market_data.fetch_data(tf) for tf in self.timeframes}
            if any(d is None for d in data.values()):
                logging.error(f"[{symbol}] Failed to fetch data for one or more timeframes")
                return
            
            # Calculate core indicators (EMAs, ATR)
            indicators = {tf: self.indicator_calc.calculate_indicators(d, symbol=symbol) for tf, d in data.items()}
            
            # Use lowest timeframe for entry signals
            entry_tf = min(self.timeframes)
            entry_indicators = indicators[entry_tf]

            # Get futures-specific metrics
            futures_metrics = self.market_data.get_futures_metrics()
            if not futures_metrics:
                logging.error(f"[{symbol}] Failed to get futures metrics")
                return

            # Log key metrics
            funding_rate = float(futures_metrics['funding_rate']['fundingRate'])
            hours_to_funding = float(futures_metrics['next_funding_time']['hours_remaining'])
            mark_price = float(futures_metrics['mark_price']['mark_price'])
            
            logging.info(f"[{symbol}] Current metrics:")
            logging.info(f"  - Mark Price: {mark_price:.2f}")
            logging.info(f"  - Funding Rate: {funding_rate:.4%}")
            logging.info(f"  - Hours to Funding: {hours_to_funding:.1f}")
            logging.info(f"  - EMA Difference: {(entry_indicators['ema_short'].iloc[-1] / entry_indicators['ema_long'].iloc[-1] - 1):.4%}")

            # Analyze potential trade setup
            strategy = FuturesStrategyLogic()
            analysis = strategy.analyze_setup(entry_indicators, futures_metrics)
            
            if not analysis:
                logging.info(f"[{symbol}] No valid setup found - analysis returned None")
                return

            logging.info(f"[{symbol}] Analysis results:")
            logging.info(f"  - Trend: {analysis['trend']}")
            logging.info(f"  - Confidence: {analysis['confidence']:.1f}")
            logging.info(f"  - Should Trade: {analysis['should_trade']}")

            if not analysis['should_trade']:
                reasons = []
                if not analysis['funding_signal']['is_valid']:
                    reasons.append(f"Funding rate {funding_rate:.4%} below threshold")
                if analysis['hours_to_funding'] < strategy.min_hours_to_funding:
                    reasons.append(f"Too close to funding time ({hours_to_funding:.1f}h)")
                if analysis['confidence'] <= 60:
                    reasons.append(f"Low confidence score ({analysis['confidence']:.1f})")
                
                logging.info(f"[{symbol}] No trade setup - Reasons:")
                for reason in reasons:
                    logging.info(f"  - {reason}")
                return

            logging.info(f"[{symbol}] Trade setup found:")
            logging.info(f"  Direction: {analysis['trend']}")
            logging.info(f"  Funding rate: {analysis['funding_signal']['rate']:.4%}")
            logging.info(f"  Hours to funding: {analysis['time_to_funding']['hours_remaining']:.1f}")

            # Validate market conditions
            market_conditions = self.validate_market_conditions(
                symbol, 
                setup_type=analysis['trend']
            )
            
            if not market_conditions['is_valid']:
                logging.warning(f"[{symbol}] Market conditions not suitable:")
                for reason in market_conditions['reasons']:
                    logging.warning(f"- {reason}")
                return

            try:
                # Initialize trading parameters before execution
                if not self.order_manager.initialize_trading_parameters(symbol):
                    logging.error(f"[{symbol}] Failed to initialize trading parameters")
                    return

                # Create trade setup structure
                setup = {
                    'type': 'buy' if analysis['trend'] == 'bullish' else 'sell',
                    'confidence': analysis['confidence'],   
                    'metrics': {
                        'symbol': symbol,
                        'funding_rate': analysis['funding_signal']['rate'],
                        'time_to_funding': analysis['time_to_funding']['hours_remaining']
                    }
                }
    
                result = await self.execute_trade(setup, indicators)
                
                if result and result.get('entry_order', {}).get('status') == 'FILLED':
                    entry_order = result['entry_order']
                    if 'fills' in entry_order and entry_order['fills']:
                        filled_price = float(entry_order['fills'][0]['price'])
                        filled_quantity = float(entry_order['fills'][0]['qty'])
                    elif 'avgPrice' in entry_order:  # Alternative way to get filled price
                        filled_price = float(entry_order['avgPrice'])
                        filled_quantity = float(entry_order['executedQty'])
                    else:
                        logging.error(f"[{symbol}] Unable to determine filled price from order result")
                        return
                    
                    logging.info(f"[{symbol}] Trade executed successfully:")
                    logging.info(f"  - Type: {setup['type']}")
                    logging.info(f"  - Entry: {filled_price}")
                    
                    # Add position monitoring after successful entry
                    current_position = {
                        'symbol': symbol,
                        'side': setup['type'],
                        'entry_price': filled_price,
                        'quantity': filled_quantity,
                        'leverage': self.order_manager.default_leverage
                    }
                    
                    # Store position info
                    self.active_positions[symbol] = current_position
                    
                    # Start monitoring in background task
                    asyncio.create_task(self.monitor_position(
                        symbol, 
                        current_position, 
                        entry_indicators
                    ))
                    
                    logging.info(f"[{symbol}] Position monitoring started")
                else:
                    logging.info(f"[{symbol}] No trade executed - Entry conditions not met")
            
            except Exception as e:
                logging.error(f"[{symbol}] Trade execution error: {str(e)}")
                logging.error(traceback.format_exc())

        except Exception as e:
            logging.error(f"[{symbol}] Error in signal analysis: {str(e)}")
            logging.error(traceback.format_exc())
        
    async def monitor_position(self, symbol, position, entry_indicators):
        """Monitor position with enhanced risk management"""
        try:
            initial_stop_loss = position['entry_price'] * (0.98 if position['side'] == 'buy' else 1.02)
            last_update_time = time.time()
            update_interval = 60  # Update every 60 seconds
            
            while True:
                current_time = time.time()
                if current_time - last_update_time < update_interval:
                    await asyncio.sleep(1)
                    continue
                
                # Get fresh futures metrics
                futures_metrics = self.market_data.get_futures_metrics()
                if futures_metrics:
                    # Check funding rate changes
                    funding_rate = float(futures_metrics['funding_rate']['fundingRate'])
                    if abs(funding_rate) > 0.002:  # 0.2%
                        logging.warning(f"[{symbol}] High funding rate detected: {funding_rate:.4%}")
                        await self.emergency_exit(symbol, position)
                        break

                    # Check basis changes
                    mark_info = futures_metrics['mark_price']
                    basis_pct = (mark_info['basis'] / mark_info['mark_price']) * 100
                    if abs(basis_pct) > 1:  # 1%
                        logging.warning(f"[{symbol}] High basis spread: {basis_pct:.2f}%")
                        await self.emergency_exit(symbol, position)
                        break
                    
                    correlation_risk = self.risk_monitor.check_correlation_risk(symbol)
                    if correlation_risk and not correlation_risk['is_safe']:
                        logging.warning(f"[{symbol}] High correlation detected:")
                        for pair in correlation_risk['high_correlation_pairs']:
                            logging.warning(f"- {pair[0]} and {pair[1]}: {pair[2]:.2f}")

                    # Add exposure check
                    exposure = self.risk_monitor.calculate_position_exposure()
                    if exposure and not exposure['is_safe']:
                        logging.warning(f"[{symbol}] High exposure detected: {exposure['exposure_ratio']:.2f}x")

                    # Check position risk status
                    position_status = await self.risk_monitor.monitor_position(symbol, position)
                    
                    if position_status and position_status['warnings']:
                        logging.warning(f"[{symbol}] Position risk warnings:")
                        for warning in position_status['warnings']:
                            logging.warning(f"- {warning}")
                        
                        # If close to liquidation, attempt emergency exit
                        if "Close to liquidation price" in position_status['warnings']:
                            logging.warning(f"[{symbol}] Emergency exit triggered due to liquidation risk")
                            await self.emergency_exit(symbol, position)
                            break

                # Regular monitoring logic
                current_data = self.market_data.fetch_data(min(self.timeframes))
                if current_data is None:
                    await asyncio.sleep(update_interval)
                    continue

                position_info = await self.order_manager.get_position_info(symbol)
                if not position_info or position_info['quantity'] == 0:
                    logging.info(f"[{symbol}] Position closed, stopping monitoring")
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]
                    break

                last_update_time = current_time
                await asyncio.sleep(1)

        except Exception as e:
            logging.error(f"[{symbol}] Error in position monitoring: {str(e)}")
            logging.error(traceback.format_exc())
    
    def validate_market_conditions(self, symbol, setup_type='long'):
        """Improved market conditions validation with adjusted funding thresholds"""
        try:
            # Get funding rate
            funding_data = self.client.futures_funding_rate(symbol=symbol)[-1]
            funding_rate = float(funding_data['fundingRate'])
            
            # Check if we're too close to funding
            funding_time = pd.to_datetime(funding_data['fundingTime'], unit='ms')
            current_time = pd.Timestamp.now()
            time_to_funding = (funding_time - current_time).total_seconds() / 3600  # hours
            
            # Analyze market depth
            depth = self.market_data.analyze_market_depth()
            
            conditions = {
                'is_valid': True,
                'reasons': []
            }
            
            # Revised funding rate checks based on trade direction
            if setup_type == 'long' and funding_rate > 0.005:  # 0.5% for longs
                conditions['is_valid'] = False
                conditions['reasons'].append(f"High funding rate for long: {funding_rate:.4%}")
            elif setup_type == 'short' and funding_rate < -0.005:  # -0.5% for shorts
                conditions['is_valid'] = False
                conditions['reasons'].append(f"High negative funding rate for short: {funding_rate:.4%}")
            
            # For shorts, high positive funding is actually good
            if setup_type == 'short' and funding_rate > 0:
                # Don't invalidate the trade, high funding favors shorts
                pass
                
            # For longs, high negative funding is good
            if setup_type == 'long' and funding_rate < 0:
                # Don't invalidate the trade, negative funding favors longs
                pass
            
            # Time to funding check remains the same
            if 0 <= time_to_funding <= 1:  # Within 1 hour of funding
                conditions['is_valid'] = False
                conditions['reasons'].append(f"Too close to funding time: {time_to_funding:.1f}h")
                
            # Liquidity checks
            if depth and depth['spread'] > 0.1:  # 0.1% spread
                conditions['is_valid'] = False
                conditions['reasons'].append(f"Wide spread: {depth['spread']:.3f}%")
                
            # Check if trying to long with negative imbalance or short with positive
            if depth and ((setup_type == 'long' and depth['imbalance'] < -0.2) or 
                        (setup_type == 'short' and depth['imbalance'] > 0.2)):
                conditions['is_valid'] = False
                conditions['reasons'].append(f"Unfavorable order book imbalance: {depth['imbalance']:.2f}")
            
            return conditions
            
        except Exception as e:
            logging.error(f"Error validating market conditions: {str(e)}")
            return {'is_valid': False, 'reasons': [str(e)]}
    
    async def emergency_exit(self, symbol, position):
        """Emergency exit with market order"""
        try:
            position_info = await self.order_manager.get_position_info(symbol)
            if position_info and position_info['quantity'] > 0:
                exit_side = SIDE_SELL if position['side'] == 'buy' else SIDE_BUY
                
                # Market order for immediate exit
                exit_order = self.client.create_order(
                    symbol=symbol,
                    side=exit_side,
                    type=ORDER_TYPE_MARKET,
                    quantity=position_info['quantity'],
                    reduceOnly=True  # Ensure it only reduces position
                )
                
                if exit_order and exit_order.get('status') == 'FILLED':
                    logging.info(f"[{symbol}] Emergency exit successful")
                    # Clean up position data
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]
                    return True
                    
        except Exception as e:
            logging.error(f"[{symbol}] Emergency exit failed: {str(e)}")
        return False

    async def close_all_positions(self):
        """Close all active positions"""
        for symbol, position in list(self.active_positions.items()):
            try:
                # Verify position still exists
                position_info = await self.order_manager.get_position_info(symbol)
                if position_info and position_info['quantity'] > 0:
                    await self.emergency_exit(symbol, position)
                else:
                    # Position no longer exists, clean up
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]
            except Exception as e:
                logging.error(f"Error closing position for {symbol}: {str(e)}")
    
    def validate_trade(self):
        """Enhanced futures trade validation"""
        try:
            # Get futures metrics
            futures_metrics = self.market_data.get_futures_metrics()
            if not futures_metrics:
                return False, "Could not get futures metrics"
                
            # Focus on futures-specific checks
            funding_rate = float(futures_metrics['funding_rate']['fundingRate'])
            hours_to_funding = futures_metrics['next_funding_time']['hours_remaining']
            mark_info = futures_metrics['mark_price']
            
            # Validate based on key futures metrics
            if abs(funding_rate) > 0.001:
                return False, f"High funding rate: {funding_rate:.4%}"
                
            if hours_to_funding < 1:
                return False, f"Too close to funding: {hours_to_funding:.1f}h"
                
            basis_pct = (mark_info['basis'] / mark_info['mark_price']) * 100
            if abs(basis_pct) > 0.5:
                return False, f"High basis spread: {basis_pct:.2f}%"
                
            return True, "Trade validated with futures metrics"
            
        except Exception as e:
            logging.error(f"Error in trade validation: {str(e)}")
            return False, f"Validation error: {str(e)}"
            
    async def execute_trade(self, setup, indicators):
        """Execute trade with position sizing based on confidence"""
        try:
            symbol = setup['metrics'].get('symbol', self.market_data.symbol)
            entry_tf = min(self.timeframes)
            entry_data = indicators[entry_tf]
            current_price = entry_data['close'].iloc[-1]
            account_info = self.client.get_account_info()
            
            # Calculate base position size
            base_position_size = self.order_manager.calculate_position_size(
                symbol=symbol,
                account_info=account_info,
                price=current_price,
                setup_confidence=setup['confidence']
            )

            # Apply dynamic adjustments
            adjusted_position_size = self.order_manager.adjust_position_sizing(
                base_position_size,
                symbol,
                setup['confidence']
            )
                        
            # Execute the trade
            return await self.order_manager.place_orders(
                symbol=symbol,
                side=SIDE_BUY if setup['type'] == 'buy' else SIDE_SELL,
                quantity=adjusted_position_size,
                setup_confidence=setup['confidence']
            )
                
        except Exception as e:
            logging.error(f"Error executing trade: {str(e)}")
            traceback.print_exc()
            return None

class FuturesContractManager:
    def __init__(self, client):
        self.client = client
        self.perpetual_suffix = 'PERP'
        self.quarterly_expiry_days = 90
        self.warning_threshold_days = 5  # Days before expiry to warn
        self.liquidity_threshold = 0.7  # Minimum liquidity ratio compared to perpetual
        
        # Cache contract info
        self.contract_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 3600  # 1 hour cache
        
    def _refresh_contract_cache(self):
        """Refresh the contract cache if needed"""
        current_time = time.time()
        if (self.cache_timestamp is None or 
            current_time - self.cache_timestamp > self.cache_duration):
            try:
                exchange_info = self.client.futures_exchange_info()
                
                for symbol in exchange_info['symbols']:
                    if symbol['contractType'] in ['PERPETUAL', 'CURRENT_QUARTER', 'NEXT_QUARTER']:
                        self.contract_cache[symbol['symbol']] = {
                            'contractType': symbol['contractType'],
                            'deliveryDate': symbol.get('deliveryDate', None),
                            'onboardDate': symbol.get('onboardDate', None),
                            'status': symbol['status'],
                            'baseAsset': symbol['baseAsset']
                        }
                        
                self.cache_timestamp = current_time
                
            except Exception as e:
                logging.error(f"Error refreshing contract cache: {str(e)}")
    
    def check_contract_expiry(self, symbol):
        """Check if a contract is approaching expiry"""
        try:
            self._refresh_contract_cache()
            contract_info = self.contract_cache.get(symbol)
            
            if not contract_info:
                return {
                    'is_valid': False,
                    'days_to_expiry': None,
                    'warning': f"Contract information not found for {symbol}"
                }
                
            # For perpetual contracts, always valid
            if contract_info['contractType'] == 'PERPETUAL':
                return {
                    'is_valid': True,
                    'days_to_expiry': None,
                    'warning': None
                }
                
            # For quarterly contracts
            if contract_info['deliveryDate']:
                delivery_date = pd.to_datetime(contract_info['deliveryDate'], unit='ms')
                current_date = pd.Timestamp.now()
                days_to_expiry = (delivery_date - current_date).days
                
                is_valid = days_to_expiry > self.warning_threshold_days
                warning = None
                
                if not is_valid:
                    warning = f"Contract expiring in {days_to_expiry} days"
                
                return {
                    'is_valid': is_valid,
                    'days_to_expiry': days_to_expiry,
                    'warning': warning
                }
                
            return {
                'is_valid': True,
                'days_to_expiry': None,
                'warning': None
            }
            
        except Exception as e:
            logging.error(f"Error checking contract expiry: {str(e)}")
            return {
                'is_valid': False,
                'days_to_expiry': None,
                'warning': str(e)
            }
            
    def analyze_contract_liquidity(self, base_asset):
        """Analyze liquidity across different contract types for a base asset"""
        try:
            self._refresh_contract_cache()
            
            # Get all contracts for the base asset
            asset_contracts = {
                symbol: info for symbol, info in self.contract_cache.items()
                if info['baseAsset'] == base_asset
            }
            
            if not asset_contracts:
                return {
                    'is_valid': False,
                    'metrics': {},
                    'warning': f"No contracts found for {base_asset}"
                }
                
            liquidity_metrics = {}
            
            for symbol, info in asset_contracts.items():
                try:
                    # Get 24h trading metrics
                    ticker = self.client.futures_ticker(symbol=symbol)
                    if not ticker:
                        continue
                        
                    # Calculate liquidity score based on volume and number of trades
                    volume = float(ticker.get('volume', 0))
                    trades = float(ticker.get('count', 0))
                    avg_trade_size = volume / trades if trades > 0 else 0
                    
                    # Get order book depth with error handling
                    try:
                        depth = self.client.get_order_book(symbol=symbol, limit=20)
                        if depth and depth.get('bids') and depth.get('asks'):
                            bid_depth = sum(float(bid[1]) for bid in depth['bids'])
                            ask_depth = sum(float(ask[1]) for ask in depth['asks'])
                            total_depth = bid_depth + ask_depth
                            
                            # Calculate spread
                            best_bid = float(depth['bids'][0][0])
                            best_ask = float(depth['asks'][0][0])
                            spread = (best_ask - best_bid) / best_bid
                        else:
                            total_depth = 0
                            spread = 1  # Default to high spread if no depth
                    except:
                        total_depth = 0
                        spread = 1
                    
                    # Combine metrics into a liquidity score
                    liquidity_score = (
                        volume * 0.4 +          # Volume weight
                        trades * 0.3 +          # Number of trades weight
                        total_depth * 0.2 +     # Order book depth weight
                        (1/(1 + spread)) * 0.1  # Spread weight (inverse)
                    )
                    
                    liquidity_metrics[symbol] = {
                        'volume_24h': volume,
                        'trades_24h': trades,
                        'avg_trade_size': avg_trade_size,
                        'order_book_depth': total_depth,
                        'spread': spread,
                        'liquidity_score': liquidity_score,
                        'contract_type': info['contractType']
                    }
                    
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {str(e)}")
                    continue
            
            if not liquidity_metrics:
                return {
                    'is_valid': False,
                    'metrics': {},
                    'warning': f"Could not get liquidity metrics for {base_asset}"
                }
            
            # Find perpetual contract liquidity as benchmark
            perp_contracts = [
                (symbol, metrics) for symbol, metrics in liquidity_metrics.items()
                if metrics['contract_type'] == 'PERPETUAL'
            ]
            
            if not perp_contracts:
                return {
                    'is_valid': True,  # Still valid, just use absolute metrics
                    'metrics': liquidity_metrics,
                    'most_liquid': max(
                        liquidity_metrics.items(),
                        key=lambda x: x[1]['liquidity_score']
                    )[0]
                }
                
            perp_symbol, perp_metrics = max(
                perp_contracts,
                key=lambda x: x[1]['liquidity_score']
            )
            perp_liquidity = perp_metrics['liquidity_score']
            
            # Compare each contract's liquidity to perpetual
            for symbol in liquidity_metrics:
                relative_liquidity = liquidity_metrics[symbol]['liquidity_score'] / perp_liquidity
                liquidity_metrics[symbol]['relative_liquidity'] = relative_liquidity
                liquidity_metrics[symbol]['is_liquid'] = relative_liquidity >= self.liquidity_threshold
            
            return {
                'is_valid': True,
                'metrics': liquidity_metrics,
                'most_liquid': max(
                    liquidity_metrics.items(),
                    key=lambda x: x[1]['liquidity_score']
                )[0]
            }
            
        except Exception as e:
            logging.error(f"Error analyzing contract liquidity: {str(e)}")
            return {
                'is_valid': False,
                'metrics': {},
                'warning': str(e)
            }

    def get_recommended_contract(self, base_asset):
        """Get the recommended contract for trading based on liquidity and expiry"""
        try:
            liquidity_analysis = self.analyze_contract_liquidity(base_asset)
            
            if not liquidity_analysis['is_valid']:
                return None
                
            valid_contracts = []
            
            for symbol, metrics in liquidity_analysis['metrics'].items():
                if metrics['is_liquid']:
                    expiry_check = self.check_contract_expiry(symbol)
                    if expiry_check['is_valid']:
                        valid_contracts.append((
                            symbol,
                            metrics['liquidity_score'],
                            metrics['contract_type']
                        ))
            
            if not valid_contracts:
                return None
                
            # Prefer perpetual contracts if liquid
            perp_contracts = [c for c in valid_contracts if c[2] == 'PERPETUAL']
            if perp_contracts:
                return max(perp_contracts, key=lambda x: x[1])[0]
                
            # Otherwise return most liquid valid contract
            return max(valid_contracts, key=lambda x: x[1])[0]
            
        except Exception as e:
            logging.error(f"Error getting recommended contract: {str(e)}")
            return None
                   
class OrderManager:
    def __init__(self, client, market_data):
        self.client = client
        self.market_data = market_data
        self.min_trade_amount = 10  # Minimum USDT value
        self.max_confidence_leverage = 1.5  # Maximum position size multiplier
        self.default_leverage = 3   # Default leverage for all trades
        self.max_allowed_leverage = 20  # Maximum allowed leverage
        self.margin_type = 'ISOLATED'  # Default to isolated margin
        
    def initialize_trading_parameters(self, symbol):
        """Initialize trading parameters for a symbol"""
        try:
            # Set margin type first - handle the "no need to change" error gracefully
            try:
                self.client.set_margin_type(symbol, self.margin_type)
            except BinanceAPIException as e:
                if e.code == -4046:  # "No need to change margin type"
                    logging.info(f"Margin type already set to {self.margin_type} for {symbol}")
                else:
                    logging.error(f"Error setting margin type: {str(e)}")
                    return False
            
            # Get leverage brackets
            brackets = self.client.get_leverage_brackets(symbol)
            if not brackets:
                logging.error(f"Could not get leverage brackets for {symbol}")
                return False
                
            # Calculate safe leverage based on position size
            max_leverage = brackets[0]['initialLeverage']
            safe_leverage = min(self.default_leverage, max_leverage, self.max_allowed_leverage)
            
            # Set leverage
            leverage_set = self.client.set_leverage(symbol, safe_leverage)
            if not leverage_set:
                logging.error(f"Could not set leverage for {symbol}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error initializing trading parameters: {str(e)}")
            return False
        
    def get_quantity_precision(self, symbol):
        """Get quantity precision with proper error handling"""
        try:
            # Handle case where symbol_info dict is passed instead of symbol string
            if isinstance(symbol, dict):
                if 'quantityPrecision' in symbol:
                    return int(symbol['quantityPrecision'])
                elif 'filters' in symbol:
                    # Extract from LOT_SIZE filter if available
                    lot_size = next((f for f in symbol['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                    if lot_size:
                        step_size = float(lot_size['stepSize'])
                        return len(str(step_size).rstrip('0').split('.')[-1])
                symbol = symbol.get('symbol', '')

            # If we have market data and symbol info
            if self.market_data and self.market_data.symbol_info:
                if 'quantityPrecision' in self.market_data.symbol_info:
                    return int(self.market_data.symbol_info['quantityPrecision'])

            # Get fresh from exchange if needed
            symbol_info = self.client.get_symbol_info(symbol)
            if symbol_info and 'quantityPrecision' in symbol_info:
                return int(symbol_info['quantityPrecision'])

            return 0  # Default to 0 for futures
                
        except Exception as e:
            logging.error(f"Error getting quantity precision: {str(e)}")
            return 0  # Default to 0 for futures

    def format_quantity(self, symbol, quantity):
        """Format quantity with proper error handling"""
        try:
            # Handle case where full symbol info is passed
            symbol_info = symbol if isinstance(symbol, dict) else None
            symbol_str = symbol_info['symbol'] if symbol_info else symbol
            
            # Get precision
            precision = self.get_quantity_precision(symbol if symbol_info else symbol_str)
            
            # Get step size
            step_size = 1.0  # Default step size
            if symbol_info and 'filters' in symbol_info:
                lot_size = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if lot_size:
                    step_size = float(lot_size['stepSize'])
            
            # Round to step size
            quantity = math.floor(float(quantity) / step_size) * step_size
            
            # Format to precision
            formatted_qty = f"{{:.{precision}f}}".format(quantity)
            
            # Validate against min quantity if available
            if symbol_info and 'minQty' in symbol_info:
                min_qty = float(symbol_info['minQty'])
                if float(formatted_qty) < min_qty:
                    formatted_qty = f"{{:.{precision}f}}".format(min_qty)
            
            return formatted_qty
            
        except Exception as e:
            logging.error(f"Error formatting quantity: {str(e)}")
            return str(int(quantity))  # Safe default for futures
    

    def calculate_position_size(self, symbol, account_info, price, setup_confidence=0):
        """Calculate position size with enhanced error handling"""
        try:
            # Extract wallet balance safely
            wallet_balance = 0
            if isinstance(account_info, dict):
                if 'assets' in account_info:
                    for asset in account_info['assets']:
                        if asset.get('asset') == 'USDT':
                            wallet_balance = float(asset.get('walletBalance', 0))
                            break
                else:
                    wallet_balance = float(account_info.get('totalWalletBalance', 0))
            
            logging.info(f"Available wallet balance: {wallet_balance} USDT")
            
            if wallet_balance <= 0:
                logging.error("Invalid wallet balance")
                return 0
                
            if not isinstance(price, (int, float)) or price <= 0:
                logging.error(f"Invalid price: {price}")
                return 0

            # Calculate base position size (1% of balance)
            risk_amount = wallet_balance * 0.01
            
            # Adjust based on confidence (0.5x to 1.5x)
            confidence_multiplier = max(0.5, min(1.5, setup_confidence / 100))
            risk_amount *= confidence_multiplier
            
            # Calculate quantity in base asset
            position_value = risk_amount * self.default_leverage
            quantity = position_value / float(price)
            
            # Get symbol info for minimum notional
            min_notional = 5.0  # Default minimum notional
            if self.market_data and self.market_data.symbol_info:
                min_notional = float(self.market_data.symbol_info.get('minNotional', 5.0))
            
            # Ensure minimum notional is met
            if quantity * price < min_notional:
                quantity = min_notional / price
                logging.info(f"Adjusted quantity to meet minimum notional: {quantity}")
            
            # Format quantity according to symbol precision
            formatted_qty = self.format_quantity(symbol, quantity)
            final_quantity = float(formatted_qty)
            
            # Final validation
            if final_quantity <= 0:
                logging.error("Calculated quantity is zero or negative")
                return 0
                
            logging.info(f"Calculated position size: {final_quantity} (Value: {final_quantity * price:.2f} USDT)")
            return final_quantity
            
        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            logging.error(traceback.format_exc())
            return 0
        
    async def place_orders(self, symbol, side, quantity, setup_confidence=0):
        """Place orders with dynamic price deviation checks based on volatility"""
        try:
            # Basic quantity validation
            quantity = float(quantity)
            if quantity <= 0:
                logging.error(f"Invalid quantity: {quantity}")
                return None
            
            # Get current market data with multiple sources
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            order_book = self.client.get_order_book(symbol=symbol, limit=5)
            mark_price = self.client.futures_mark_price(symbol=symbol)
            
            if not all([ticker, order_book, mark_price]):
                logging.error("Could not get complete market data")
                return None
                
            # Get different price references
            last_price = float(ticker['price'])
            mark_price = float(mark_price['markPrice'])
            best_bid = float(order_book['bids'][0][0])
            best_ask = float(order_book['asks'][0][0])
            
            # Calculate recent volatility
            volatility = self.calculate_recent_volatility(symbol)
            
            # Dynamic price range based on volatility
            base_range = 0.10  # 10% base range
            volatility_adjustment = min(0.15, volatility * 2)  # Cap at 15% additional
            price_range = base_range + volatility_adjustment
            
            logging.info(f"Price analysis for {symbol}:")
            logging.info(f"Last price: {last_price}")
            logging.info(f"Mark price: {mark_price}")
            logging.info(f"Best bid: {best_bid}")
            logging.info(f"Best ask: {best_ask}")
            logging.info(f"Volatility: {volatility:.4f}")
            logging.info(f"Allowed price range: {price_range:.2%}")
            
            # Use mark price as reference for deviation checks
            min_price = mark_price * (1 - price_range)
            max_price = mark_price * (1 + price_range)
            
            # Check if prices are within allowed range
            if side == SIDE_BUY:
                execution_price = min(best_ask, mark_price * 1.003)  # Cap at 0.3% above mark price
                if best_ask > max_price:
                    logging.error(f"Best ask price {best_ask} exceeds adjusted maximum price {max_price}")
                    return None
            else:  # SELL
                execution_price = max(best_bid, mark_price * 0.997)  # Cap at 0.3% below mark price
                if best_bid < min_price:
                    logging.error(f"Best bid price {best_bid} below adjusted minimum price {min_price}")
                    return None
                
            formatted_quantity = self.format_quantity(symbol, quantity)
            order_value = float(formatted_quantity) * execution_price
            
            # Check minimum order value
            if order_value < self.min_trade_amount:
                logging.error(f"Order value {order_value:.2f} USDT below minimum {self.min_trade_amount} USDT")
                return None
            
            # Get symbol info for price filters
            symbol_info = self.client.get_symbol_info(symbol)
            
            logging.info(f"Placing {side} order for {formatted_quantity} {symbol}")
            logging.info(f"Execution price: {execution_price:.8f}")
            logging.info(f"Order value: {order_value:.2f} USDT")
            
            # Place order with a small slippage buffer
            try:
                entry_order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=formatted_quantity,
                    price=self.format_price(symbol_info, execution_price)
                )
            except BinanceAPIException as e:
                if e.code == -4131:  # PERCENT_PRICE filter error
                    logging.warning("Retrying with market order due to price deviation...")
                    entry_order = self.client.create_order(
                        symbol=symbol,
                        side=side,
                        type=ORDER_TYPE_MARKET,
                        quantity=formatted_quantity
                    )
                else:
                    raise

            if not entry_order:
                logging.error("Order creation failed")
                return None
                
            # Check if order was filled
            if entry_order.get('status') == 'FILLED':
                logging.info(f"Order filled at average price: {entry_order['fills'][0]['price']}")
                return {'entry_order': entry_order}
                
            # Monitor order filling
            order_id = entry_order['orderId']
            filled = False
            retry_count = 0
            
            while not filled and retry_count < 3:
                try:
                    order_status = self.client.query_futures_order(
                        symbol=symbol,
                        orderId=order_id
                    )
                    
                    if order_status['status'] == 'FILLED':
                        filled = True
                        logging.info(f"Order filled at average price: {order_status['avgPrice']}")
                        return {'entry_order': order_status}
                    elif order_status['status'] == 'REJECTED':
                        logging.error(f"Order rejected: {order_status.get('rejectReason', 'Unknown reason')}")
                        break
                        
                    retry_count += 1
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logging.error(f"Error checking order status: {str(e)}")
                    break
            
            # Cancel unfilled order and try market order as fallback
            if not filled:
                try:
                    self.client.cancel_order(symbol=symbol, orderId=order_id)
                    logging.warning("Attempting market order as fallback...")
                    market_order = self.client.create_order(
                        symbol=symbol,
                        side=side,
                        type=ORDER_TYPE_MARKET,
                        quantity=formatted_quantity
                    )
                    
                    if market_order and market_order.get('status') == 'FILLED':
                        logging.info(f"Market order filled at average price: {market_order['fills'][0]['price']}")
                        return {'entry_order': market_order}
                        
                except Exception as e:
                    logging.error(f"Fallback market order failed: {str(e)}")
                    
            return None

        except Exception as e:
            logging.error(f"Error placing orders: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def calculate_recent_volatility(self, symbol, periods=20):
        """Calculate recent price volatility"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                limit=periods
            )
            
            if not klines:
                return 0.02  # Default volatility
                
            # Calculate price changes
            prices = [float(k[4]) for k in klines]
            price_changes = [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
            
            # Calculate volatility as standard deviation
            volatility = np.std(price_changes)
            return volatility
            
        except Exception as e:
            logging.error(f"Error calculating volatility: {str(e)}")
            return 0.02  # Default volatility
    
    async def get_position_info(self, symbol):
        """Get current futures position information"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if positions:
                for position in positions:
                    if position['symbol'] == symbol:
                        return {
                            'quantity': abs(float(position['positionAmt'])),
                            'entryPrice': float(position['entryPrice']),
                            'unrealizedProfit': float(position['unRealizedProfit']),
                            'liquidationPrice': float(position['liquidationPrice']),
                            'leverage': int(position['leverage']),
                            'marginType': position['marginType']
                        }
            return None
        except Exception as e:
            logging.error(f"Error getting position info: {str(e)}")
            return None

    async def execute_exit_order(self, symbol, side, quantity, exit_price, stop_price):
        """Execute exit order with OCO"""
        try:
            # Format prices according to symbol precision
            symbol_info = self.client.get_symbol_info(symbol)
            formatted_quantity = self.format_quantity(symbol_info, quantity)
            formatted_exit_price = self.format_price(symbol_info, exit_price)
            formatted_stop_price = self.format_price(symbol_info, stop_price)
            stop_limit_price = self.format_price(
                symbol_info, 
                stop_price * 0.99 if side == SIDE_SELL else stop_price * 1.01
            )

            # Place take profit order
            tp_order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=formatted_quantity,
                price=formatted_exit_price,
                reduceOnly=True
            )

            # Place stop loss order
            sl_order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='STOP_MARKET',
                timeInForce='GTC',
                quantity=formatted_quantity,
                stopPrice=formatted_stop_price,
                reduceOnly=True
            )

            if tp_order and sl_order:
                logging.info(f"Created exit orders for {symbol}:")
                logging.info(f"Take Profit: {formatted_exit_price}")
                logging.info(f"Stop Loss: {formatted_stop_price}")
                return {'tp_order': tp_order, 'sl_order': sl_order}
            else:
                logging.error(f"Failed to create exit orders for {symbol}")
                return None

        except Exception as e:
            logging.error(f"Error in exit order execution: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def format_price(self, symbol_info, price):
        """Format price according to symbol's precision rules"""
        try:
            price_filter = next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', symbol_info['filters']))
            tick_size = float(price_filter['tickSize'])
            precision = int(round(-math.log10(tick_size)))
            return '{:.{}f}'.format(round(price / tick_size) * tick_size, precision)
        except Exception as e:
            logging.error(f"Error formatting price: {str(e)}")
            return '{:.8f}'.format(price)

    async def cancel_open_orders(self, symbol):
        """Cancel all open orders for a symbol"""
        try:
            result = self.client.cancel_all_orders(symbol=symbol)
            logging.info(f"Cancelled open orders for {symbol}: {result}")
            return True
        except BinanceAPIException as e:
            logging.error(f"Error cancelling orders: {str(e)}")
            return False
        
    def cancel_all_orders(self, symbol):
        """Cancel all futures orders for a symbol"""
        try:
            return self.client.futures_cancel_all_open_orders(symbol=symbol)
        except BinanceAPIException as e:
            logging.error(f"Error canceling all orders: {str(e)}")
            return None    
        
    def adjust_position_sizing(self, base_size, symbol, setup_confidence):
        try:
            # Get market conditions
            depth = self.market_data.analyze_market_depth()
            
            # Start with the base size
            adjusted_size = base_size
            
            # Adjust for liquidity
            if depth:
                liquidity_factor = min(1, depth['liquidity_score'] / 1000)
                adjusted_size *= max(0.5, liquidity_factor)
            
            # Adjust for volatility
            volatility = self.calculate_volatility(symbol)
            if volatility > 0.02:  # High volatility
                adjusted_size *= 0.8
            
            # Adjust for setup confidence
            confidence_factor = min(1.2, max(0.5, setup_confidence / 100))
            adjusted_size *= confidence_factor
            
            # Get position exposure
            risk_monitor = RiskMonitor(self.client)
            exposure = risk_monitor.calculate_position_exposure()
            
            if exposure and exposure['exposure_ratio'] > 1.5:
                adjusted_size *= 0.7
            
            # Round to symbol precision
            symbol_info = self.client.get_symbol_info(symbol)
            return self.format_quantity(symbol_info, adjusted_size)
            
        except Exception as e:
            logging.error(f"Error adjusting position size: {str(e)}")
            return base_size
            
    def calculate_volatility(self, symbol, periods=20):
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_15MINUTE,
                limit=periods
            )
            closes = pd.Series([float(k[4]) for k in klines])
            return closes.pct_change().std()
        except Exception as e:
            logging.error(f"Error calculating volatility: {str(e)}")
            return 0.01  # Default to 1% volatility

class RiskMonitor:
    def __init__(self, client):
        self.client = client
        self.max_margin_ratio = 0.8  # 80% maximum margin usage
        self.min_margin_balance = 5  # Minimum USDT balance
        self.position_alerts = {}
        self.default_leverage = 3

    async def check_account_risk(self):
        """Check overall account risk metrics"""
        try:
            account = self.client.get_account_info()
            
            total_margin_ratio = float(account.get('totalMarginRatio', 0))
            wallet_balance = float(account.get('totalWalletBalance', 0))
            
            risk_status = {
                'margin_ratio': total_margin_ratio,
                'wallet_balance': wallet_balance,
                'is_safe': True,
                'warnings': []
            }
            
            # Check margin ratio
            if total_margin_ratio > self.max_margin_ratio:
                risk_status['is_safe'] = False
                risk_status['warnings'].append(f"High margin ratio: {total_margin_ratio:.2%}")
            
            # Check minimum balance
            if wallet_balance < self.min_margin_balance:
                risk_status['is_safe'] = False
                risk_status['warnings'].append(f"Low wallet balance: {wallet_balance} USDT")
            
            return risk_status
            
        except Exception as e:
            logging.error(f"Error checking account risk: {str(e)}")
            return None
        
    def calculate_position_exposure(self):
        try:
            # Get all positions
            positions = self.client.futures_position_information()
            if not positions:
                return {'is_safe': True, 'exposure_ratio': 0}
            
            total_exposure = 0
            position_details = {}
            
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    symbol = pos['symbol']
                    position_size = abs(float(pos['positionAmt']))
                    entry_price = float(pos['entryPrice'])
                    leverage = float(pos.get('leverage', self.default_leverage))
                    
                    # Calculate position value
                    position_value = position_size * entry_price
                    
                    # Calculate exposure (position value / leverage)
                    exposure = position_value / leverage
                    total_exposure += exposure
                    
                    position_details[symbol] = {
                        'exposure': exposure,
                        'leverage': leverage,
                        'size': position_size,
                        'value': position_value
                    }
            
            # Get account balance
            account = self.client.get_account_info()
            wallet_balance = float(account['totalWalletBalance'])
            
            # Calculate exposure ratio
            exposure_ratio = total_exposure / wallet_balance if wallet_balance > 0 else float('inf')
            
            return {
                'total_exposure': total_exposure,
                'exposure_ratio': exposure_ratio,
                'wallet_balance': wallet_balance,
                'positions': position_details,
                'is_safe': exposure_ratio < 2  # Max 2x total account exposure
            }
        except Exception as e:
            logging.error(f"Error calculating position exposure: {str(e)}")
            return None

    def check_correlation_risk(self, new_symbol=None, lookback_periods=30):
        try:
            # Get active positions
            positions = self.client.futures_position_information()
            active_symbols = [p['symbol'] for p in positions if float(p['positionAmt']) != 0]
            
            if new_symbol:
                active_symbols.append(new_symbol)
                
            if len(active_symbols) < 2:
                return {'is_safe': True, 'correlations': {}}
                
            # Fetch price data for all symbols
            price_data = {}
            for symbol in active_symbols:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_15MINUTE,
                    limit=lookback_periods
                )
                prices = [float(k[4]) for k in klines]  # Close prices
                price_data[symbol] = pd.Series(prices)
                
            # Calculate returns
            returns_data = {
                symbol: prices.pct_change().dropna()
                for symbol, prices in price_data.items()
            }
            
            # Calculate correlation matrix
            correlations = {}
            high_correlation_pairs = []
            
            for i, sym1 in enumerate(active_symbols):
                for sym2 in active_symbols[i+1:]:
                    if len(returns_data[sym1]) == len(returns_data[sym2]):
                        corr = returns_data[sym1].corr(returns_data[sym2])
                        correlations[f"{sym1}-{sym2}"] = corr
                        
                        if abs(corr) > 0.7:  # High correlation threshold
                            high_correlation_pairs.append((sym1, sym2, corr))
            
            return {
                'is_safe': len(high_correlation_pairs) == 0,
                'correlations': correlations,
                'high_correlation_pairs': high_correlation_pairs
            }
            
        except Exception as e:
            logging.error(f"Error checking correlation risk: {str(e)}")
            return None
                
async def run_trading_loop(client, symbols, timeframes):    
    # Initialize data and trade managers
    market_data_dict = {
        symbol: MarketData(symbol, timeframes, client.client) 
        for symbol in symbols
    }
    trade_managers = {
        symbol: TradeManager(client, market_data_dict[symbol]) 
        for symbol in symbols
    }
    
    contract_manager = FuturesContractManager(client)
    
    # Define shutdown handler outside the main loop
    async def shutdown(trade_managers):
        """Graceful shutdown handler"""
        logging.info("Initiating shutdown...")
        for manager in trade_managers.values():
            await manager.close_all_positions()
        logging.info("All positions closed")
    
    logging.info("\n=== Trading Bot Started ===")
    logging.info(f"Monitoring symbols: {', '.join(symbols)}")
    logging.info(f"Using timeframes: {timeframes} minutes")
    
    try:
        while True:
            loop_start_time = time.time()
            current_time = pd.Timestamp.now()
            
            # Check account risk first
            risk_monitor = RiskMonitor(client)
            account_risk = await risk_monitor.check_account_risk()
            
            if not account_risk or not account_risk['is_safe']:
                logging.warning("Account risk check failed, pausing trading")
                for warning in account_risk.get('warnings', []):
                    logging.warning(f"- {warning}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
                continue
            
            logging.info("\n" + "="*50)
            logging.info(f"Starting new iteration at {current_time}")
            
            # Process each symbol
            for symbol in symbols:
                try:
                    logging.info(f"\n Analyzing {symbol}...")
                    # Get recommended contract if needed
                    base_asset = symbol.replace('USDT', '')
                    contract_info = contract_manager.get_recommended_contract(base_asset)
                    
                    if not contract_info:
                        logging.warning(f"[{symbol}] No suitable contract found, skipping")
                        continue
                        
                    trading_symbol = contract_info if contract_info != symbol else symbol
                    
                    # Check if we're already trading this symbol
                    if trading_symbol in trade_managers and \
                    trading_symbol in trade_managers[trading_symbol].active_positions:
                        logging.info(f"[{trading_symbol}] Already have an active position, monitoring only")
                        continue
                    
                    await trade_managers[symbol].check_for_signals(trading_symbol)
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {str(e)}")
                    logging.error(traceback.format_exc())
                    continue
            
            # Calculate time until next interval
            loop_duration = time.time() - loop_start_time
            next_interval = 15 * 60  # 15 minutes in seconds
            sleep_time = max(0, next_interval - loop_duration)
            
            logging.info("\n--- Iteration Summary ---")
            logging.info(f"Duration: {loop_duration:.2f} seconds")
            logging.info(f"Next check in: {sleep_time/60:.1f} minutes")
            logging.info("="*50 + "\n")
            
            await asyncio.sleep(sleep_time)
            
    except asyncio.CancelledError:
        logging.info("Trading loop cancelled, initiating shutdown...")
        await shutdown(trade_managers)
    except Exception as e:
        logging.error(f"Critical error in trading loop: {str(e)}")
        logging.error(traceback.format_exc())
        # Attempt to close positions on critical error
        await shutdown(trade_managers)
        await asyncio.sleep(60)  # Wait before retrying
        
async def main():
    """Program entry point and initialization"""
    try:
        # Load configuration
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        trading_mode = os.getenv('TRADING_MODE', 'paper')

        if not api_key or not api_secret:
            raise ValueError("Binance API key and secret must be set in environment variables")

        # Initialize client based on mode
        if trading_mode == 'testnet':
            api_key = os.getenv('BINANCE_TESTNET_API_KEY')
            api_secret = os.getenv('BINANCE_TESTNET_SECRET_KEY')

            client = BinanceClient(api_key, api_secret, testnet=True)
        else:
            client = BinanceClient(api_key, api_secret)
            
        if not client.is_initialized():
            raise ConnectionError("Failed to connect to Binance")

        # Trading configuration
        symbols = [
            'BTCUSDT',    # Bitcoin/USDT
            'ETHUSDT',    # Ethereum/USDT
            'BNBUSDT',    # Binance Coin/USDT
            'SOLUSDT',    # Solana/USDT
            'XRPUSDT',    # Ripple/USDT
            'ADAUSDT',    # Cardano/USDT
            'DOGEUSDT',   # Dogecoin/USDT
            'LTCUSDT',    # Litecoin/USDT
            'DOTUSDT',     # Polkadot/USDT
        ]

        
        timeframes = (15, 30)

        # Start trading loop
        await run_trading_loop(client, symbols, timeframes)
        
    except asyncio.CancelledError:
        logging.info("Trading bot shutdown initiated...")
    except Exception as e:
        logging.error(f"Critical error in main: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        logging.info("Trading bot terminated")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script interrupted by user. Shutting down...")
    except Exception as e:
        logging.error(f"Runtime error: {str(e)}")