CONNECTOR NOTES - 01_connect.py STATUS: ✅ COMPLETED

* Production Databento connector that filters out irrelevant low volume contracts
* Current implementation (01_connect.py):
    1. ✅ Connects to Databento using API key from .env file
    2. ✅ Rolling 12-hour volume analysis that runs WEEKLY to find most liquid contract from:
        {
            'ES JUN25': 'ESM6',
            'ES SEP25': 'ESU5',
            'ES DEC25': 'ESZ5',
            'ES MAR26': 'ESH6'
        }
    3. ✅ Monitors live 1-minute OHLCV data for the most liquid contract
    4. ✅ Automatically switches contracts when weekly analysis detects liquidity changes
    5. ✅ Print statements REMOVED - now uses process_market_data() method as interface for next script
    6. ✅ Production-ready with proper error handling and continuous monitoring

* Ready for integration: The process_market_data() method is the interface point where next script will receive real-time market data. 

SIGNAL NOTES - 02_Signal.py
✅ STATUS: INTEGRATION COMPLETE
* Data pipeline is now fully operational and connected
* 01_connect.py successfully sends real Databento data to 02_signal.py
* All synthetic/test data has been removed - only real market data flows through
* Data structure is correctly implemented and validated

* The signal system receives the following data from 01_connect.py:
    {
    "timestamp": datetime,         # exact 1-min timestamp from Databento
    "symbol": str,                 # e.g., "ESM6" (most liquid ES contract)
    "open": float,                 # open price for the minute (scaled from Databento)
    "high": float,                 # high price for the minute (scaled from Databento)
    "low": float,                  # low price for the minute (scaled from Databento)
    "close": float,                # close price for the minute (scaled from Databento)
    "volume": float                # volume for the minute (from Databento)
    }

📋 NEXT STEPS: Add signal calculation logic to handle_bar() function
* The following calculations need to be implemented:
    1. 