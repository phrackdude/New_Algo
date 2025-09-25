#!/usr/bin/env python3
"""
Databento API Connector for V6 Real-Time Trading System
Handles real-time market data streaming and historical data requests
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any
import logging
import os

# Note: You'll need to install databento: pip install databento
try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    logging.warning("Databento not installed. Using simulation mode.")

logger = logging.getLogger(__name__)

class DatabentoConnector:
    """Handles Databento API connections and data streaming"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        self.live_client = None
        self.is_connected = False
        self.data_callback = None
        
        # ES futures contract mapping - use specific contract symbols
        self.contract_mapping = {
            'ES JUN25': 'ESM6',  # ES Jun 2026
            'ES SEP25': 'ESU5',  # ES Sep 2025 (current front month)
            'ES DEC25': 'ESZ5',  # ES Dec 2025
            'ES MAR26': 'ESH6',  # ES Mar 2026
            'ES.FUT': 'ESZ5',    # Default to Dec 2025
        }
        
    async def initialize(self, api_key: str = None):
        """Initialize Databento clients"""
        if not DATABENTO_AVAILABLE:
            logger.warning("⚠️  Databento not available - running in simulation mode")
            return False
            
        try:
            if api_key:
                self.api_key = api_key
                
            if not self.api_key:
                logger.error("❌ Databento API key required")
                return False
                
            # Initialize historical data client
            self.client = db.Historical(key=self.api_key)
            
            # Initialize live data client  
            self.live_client = db.Live(key=self.api_key)
            
            logger.info("✅ Databento clients initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Databento: {e}")
            return False
    
    def set_data_callback(self, callback: Callable[[pd.DataFrame], None]):
        """Set callback function for receiving real-time data"""
        self.data_callback = callback
    
    async def get_historical_data(self, contract: str, start_date: datetime, 
                                 end_date: datetime = None) -> pd.DataFrame:
        """Get historical OHLCV data for backtesting/initialization"""
        if not DATABENTO_AVAILABLE:
            raise RuntimeError("❌ DATABENTO NOT AVAILABLE - Install databento package!")
        if not self.client:
            raise RuntimeError("❌ DATABENTO CLIENT NOT INITIALIZED - Check API key!")
        
        try:
            # Map contract to Databento symbol
            symbol = self.contract_mapping.get(contract, 'ESU5')
            
            if end_date is None:
                end_date = datetime.now()
            
            logger.info(f"📈 Fetching historical data: {symbol} from {start_date} to {end_date}")
            
            # Request OHLCV data
            data = self.client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=[symbol],
                schema="ohlcv-1m",  # 1-minute bars
                start=start_date,
                end=end_date,
                stype_in="raw_symbol",  # Use "raw_symbol" for specific contract symbols
                stype_out="instrument_id"
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            # Databento returns: ['rtype', 'publisher_id', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'symbol']
            # We only need the OHLCV columns - timestamp is already the index
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            logger.info(f"✅ Historical data retrieved: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to get historical data: {e}")
            raise RuntimeError(f"❌ DATABENTO HISTORICAL DATA FAILED: {e}")
    
    def start_live_stream(self, contract: str):
        """Start live data streaming for specified contract using synchronous approach"""
        if not DATABENTO_AVAILABLE:
            raise RuntimeError("❌ DATABENTO NOT AVAILABLE - Install databento package!")
        if not self.live_client:
            raise RuntimeError("❌ DATABENTO LIVE CLIENT NOT INITIALIZED - Check API key!")
        
        try:
            # Use ES.FUT with parent stype for live data
            symbol = "ES.FUT"
            
            logger.info(f"📡 Starting live stream for {symbol}")
            
            # Subscribe to live data using the working approach
            self.live_client.subscribe(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",
                stype_in="parent",
                symbols=[symbol]
            )
            
            # Add callback to process data
            def process_record(record):
                logger.info(f"🔔 CALLBACK TRIGGERED: {type(record).__name__}")
                
                # Check for OHLCV records (they don't have schema attribute)
                if hasattr(record, 'close') and hasattr(record, 'volume'):
                    logger.info(f"📊 Processing OHLCV record: {type(record).__name__}")
                    
                    # Apply Databento price scaling
                    scaled_close = record.close / 1e9
                    logger.info(f"📈 OHLCV data: Close={scaled_close:.2f}, Volume={record.volume}")
                    
                    # CRITICAL FIX: Filter out wrong ES contracts by price range
                    # Main ES futures should be in range $3000-$8000 (typical range)
                    # Filter out mini contracts, spreads, and other instruments
                    if not (3000 <= scaled_close <= 8000):
                        logger.info(f"🚫 FILTERED OUT: Price ${scaled_close:.2f} outside main ES range (3000-8000)")
                        return  # Skip this record
                    
                    logger.info(f"✅ ACCEPTED: Main ES contract data - Close=${scaled_close:.2f}")
                    
                    # Convert to DataFrame format
                    data_row = pd.DataFrame({
                        'timestamp': [pd.to_datetime(record.ts_event, unit='ns')],
                        'open': [record.open / 1e9],  # Databento price scaling
                        'high': [record.high / 1e9],
                        'low': [record.low / 1e9],
                        'close': [record.close / 1e9],
                        'volume': [record.volume]
                    })
                    data_row.set_index('timestamp', inplace=True)
                    
                    # Send to callback
                    if self.data_callback:
                        logger.info(f"📤 Sending OHLCV data to callback: {data_row.iloc[0]['close']:.2f}")
                        try:
                            self.data_callback(data_row)
                            logger.info(f"✅ Callback executed successfully")
                        except Exception as e:
                            logger.error(f"❌ Callback error: {e}")
                    else:
                        logger.warning("⚠️ No data callback set")
                    return
                
                # Skip system messages and other non-data records
                if not hasattr(record, 'schema'):
                    logger.info(f"🔔 Skipping record without schema: {type(record).__name__}")
                    return
                    
                logger.info(f"📊 Received record: {record.schema} - {type(record).__name__}")
                
                if record.schema == "ohlcv-1m":
                    logger.info(f"📈 Processing OHLCV data: {record.close / 1e9:.2f}")
                    # Convert to DataFrame format
                    data_row = pd.DataFrame({
                        'timestamp': [pd.to_datetime(record.ts_event, unit='ns')],
                        'open': [record.open / 1e9],  # Databento price scaling
                        'high': [record.high / 1e9],
                        'low': [record.low / 1e9],
                        'close': [record.close / 1e9],
                        'volume': [record.volume]
                    })
                    data_row.set_index('timestamp', inplace=True)
                    
                    # Send to callback
                    if self.data_callback:
                        logger.info(f"📤 Sending data to callback: {data_row.iloc[0]['close']:.2f}")
                        self.data_callback(data_row)
                    else:
                        logger.warning("⚠️ No data callback set")
            
            logger.info("📞 Registering callback function")
            self.live_client.add_callback(process_record)
            
            logger.info("🚀 Starting live client")
            self.live_client.start()
            
            self.is_connected = True
            logger.info("✅ Live data stream started successfully")
                        
        except Exception as e:
            logger.error(f"❌ Live stream error: {e}")
            self.is_connected = False
    
    # SIMULATION FUNCTIONS DELETED - REAL DATA ONLY!
    
    # LIVE SIMULATION DELETED - REAL DATA ONLY!
    
    async def stop_stream(self):
        """Stop live data streaming"""
        self.is_connected = False
        if self.live_client:
            try:
                await self.live_client.stop()
            except Exception as e:
                logger.error(f"⚠️  Error stopping live client: {e}")
        logger.info("⏹️  Live stream stopped")
    
    def get_contract_info(self, contract: str) -> Dict[str, Any]:
        """Get contract specifications"""
        return {
            'contract': contract,
            'symbol': self.contract_mapping.get(contract, 'ESZ5'),
            'tick_size': 0.25,
            'tick_value': 12.50,
            'contract_size': 50,  # $50 per point
            'currency': 'USD',
            'exchange': 'CME'
        }

# Example usage and testing
async def test_databento_connector():
    """Test the Databento connector"""
    print("🧪 Testing Databento Connector")
    print("=" * 40)
    
    # Get API key from environment
    api_key = os.getenv('DATABENTO_API_KEY')
    
    connector = DatabentoConnector(api_key=api_key)
    
    # Test initialization
    initialized = await connector.initialize()
    print(f"Initialization: {'✅ Success' if initialized else '⚠️  Simulation Mode'}")
    
    # Test historical data
    start_date = datetime.now() - timedelta(days=1)
    hist_data = await connector.get_historical_data('ES JUN25', start_date)
    print(f"Historical data: {len(hist_data)} records")
    print(f"Price range: ${hist_data['low'].min():.2f} - ${hist_data['high'].max():.2f}")
    
    # Set up callback for live data
    def data_received(data):
        latest = data.iloc[-1]
        print(f"📊 Live data: {latest.name} | "
              f"${latest['close']:.2f} | "
              f"Vol: {latest['volume']:,}")
    
    connector.set_data_callback(data_received)
    
    # Test live stream (run for 5 minutes)
    print("\n📡 Starting live stream test (30 seconds)...")
    stream_task = asyncio.create_task(connector.start_live_stream('ES JUN25'))
    
    # Let it run for 30 seconds
    await asyncio.sleep(30)
    
    # Stop the stream
    await connector.stop_stream()
    stream_task.cancel()
    
    print("✅ Databento connector test completed")

if __name__ == "__main__":
    asyncio.run(test_databento_connector()) 