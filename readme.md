# Binance Futures Trading Bot

A simple python bot for Binance Futures that implements automated trading strategies with risk management, market analysis, and position monitoring capabilities.

## Features

- **Multiple Asset Support**: Trades major cryptocurrencies including BTC, ETH, BNB, SOL, XRP, ADA, DOGE, LTC, and DOT
- **Advanced Market Analysis**:
  - Real-time market depth analysis
  - Funding rate monitoring
  - Volatility calculations
  - Price correlation checks
- **Risk Management**:
  - Dynamic position sizing
  - Account exposure monitoring
  - Correlation risk assessment
  - Liquidation prevention
  - Emergency position exit capability
- **Smart Order Management**:
  - Supports both limit and market orders
  - Dynamic price deviation checks
  - Automatic order adjustment based on market conditions
- **Position Monitoring**:
  - Real-time position tracking
  - Automated stop-loss and take-profit management
  - Continuous risk assessment

## Prerequisites

- Python 3.7+
- Binance account with API access
- Required Python packages:
  ```
  python-binance
  pandas
  numpy
  python-dotenv
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd crypto-trading-bot
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your Binance API credentials:
   ```
   BINANCE_API_KEY=your_api_key
   BINANCE_API_SECRET=your_api_secret
   TRADING_MODE=paper  # Options: testnet, live
   ```


## Usage

Run the bot:
```bash
python trading_bot.py
```

The bot will:
1. Initialize connections and verify API access
2. Load trading configurations
3. Start monitoring configured trading pairs
4. Execute trades based on strategy conditions
5. Continuously monitor positions and manage risk

## Architecture

### Key Components

1. **BinanceClient**: Handles all Binance API interactions
2. **MarketData**: Manages market data collection and analysis
3. **TradeManager**: Coordinates trading operations and position management
4. **OrderManager**: Handles order execution and management
5. **RiskMonitor**: Monitors and manages trading risks
6. **FuturesContractManager**: Manages futures contract selection and analysis


## Testing

The bot supports two modes:
- `testnet`: Trading on Binance's test network
- `live`: Live trading with real funds

It's recommended to thoroughly test the bot in paper and testnet modes before using it with real funds.
