import pandas as pd
import numpy as np
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndicatorCalculator:
    def __init__(self, client=None):
        self.client = client
        self.default_params = {
            'atr_period': 14,
            'ema_short_period': 12,
            'ema_long_period': 26,
            'volume_sma': 20,
            'volatility_lookback': 30
        }
    
    def EMA(self, data, timeperiod=20):
        """
        Calculate Exponential Moving Average
        
        Parameters:
        data (pandas.Series): Price data
        timeperiod (int): Period for EMA calculation
        
        Returns:
        pandas.Series: EMA values
        """
        if len(data) < timeperiod:
            return pd.Series([np.nan] * len(data))
            
        # Calculate multiplier
        multiplier = 2 / (timeperiod + 1)
        
        # Calculate initial SMA
        sma = data.rolling(window=timeperiod).mean()
        
        # Initialize EMA with SMA
        ema = pd.Series(index=data.index)
        ema.iloc[:timeperiod] = sma.iloc[:timeperiod]
        
        # Calculate EMA
        for i in range(timeperiod, len(data)):
            ema.iloc[i] = (data.iloc[i] * multiplier) + (ema.iloc[i-1] * (1 - multiplier))
        
        return ema

    def ATR(self, high, low, close, timeperiod=14):
        """
        Calculate Average True Range
        
        Parameters:
        high (pandas.Series): High prices
        low (pandas.Series): Low prices
        close (pandas.Series): Close prices
        timeperiod (int): Period for ATR calculation
        
        Returns:
        pandas.Series: ATR values
        """
        if len(close) < timeperiod:
            return pd.Series([np.nan] * len(close))
        
        # Calculate True Range
        tr1 = high - low  # Current high - current low
        tr2 = abs(high - close.shift())  # Current high - previous close
        tr3 = abs(low - close.shift())  # Current low - previous close
        
        # True Range is the maximum of these three values
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using EMA of True Range
        atr = self.EMA(tr, timeperiod=timeperiod)
        
        return atr    
        
    def calculate_indicators(self, data, symbol=None, **kwargs):
        try:
            df = data.copy()
            
            # Core indicators only
            df['atr'] = self.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['ema_short'] = self.EMA(df['close'], timeperiod=12)
            df['ema_long'] = self.EMA(df['close'], timeperiod=26)
                    
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {str(e)}")
            return data
            