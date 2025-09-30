#!/usr/bin/env python3
"""
01_connect.py - Production Databento Connector
Data connector for ES futures with integrated signal processing pipeline.

This script:
1. Connects to Databento API using credentials from .env
2. Analyzes last 12 hours to find most liquid ES contract
3. Fetches 24-hour historical OHLCV data for the selected contract
4. Streams live 1-minute OHLCV bars as they arrive
5. Sends all data (historical and live) to 02_signal.py via handle_bar() function

Data format sent to signal processor:
{
    "timestamp": datetime,
    "symbol": str,
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": float
}
"""

import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from typing import Optional, Dict, Any

# Import signal processing module
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("signal_module", "02_signal.py")
    signal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signal_module)
    handle_bar = signal_module.handle_bar
    SIGNAL_MODULE_AVAILABLE = True
except Exception as e:
    SIGNAL_MODULE_AVAILABLE = False
    # Logger will be initialized below

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Databento
try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    logger.error("❌ Databento not installed. Please install with: pip install databento")
    exit(1)


class ProductionDatabentoConnector:
    """Production Databento connector for ES futures live data streaming"""
    
    def __init__(self):
        self.api_key = None
        self.historical_client = None
        self.live_client = None
        self.most_liquid_symbol = None
        self.is_streaming = False
        self.last_analysis_time = None
        self.analysis_interval = timedelta(weeks=1)  # Weekly analysis
        
        # Known ES futures contracts
        self.es_contracts = {
            'ES JUN25': 'ESM6',
            'ES SEP25': 'ESU5', 
            'ES DEC25': 'ESZ5',
            'ES MAR26': 'ESH6'
        }
        
    async def initialize(self) -> bool:
        """Initialize Databento clients with API credentials"""
        logger.info("🔌 Initializing Databento connector...")
        
        # Check signal module availability
        if not SIGNAL_MODULE_AVAILABLE:
            logger.warning("⚠️ Signal processing module (02_signal.py) not available")
            logger.warning("    Data will be logged instead of processed")
        else:
            logger.info("✅ Signal processing module loaded successfully")
        
        # Get API key from environment
        self.api_key = os.getenv('DATABENTO_API_KEY')
        if not self.api_key:
            logger.error("❌ DATABENTO_API_KEY not found in environment variables")
            logger.info("Please create a .env file with your Databento API key:")
            logger.info("DATABENTO_API_KEY=your_api_key_here")
            return False
        
        try:
            # Initialize clients
            self.historical_client = db.Historical(key=self.api_key)
            self.live_client = db.Live(key=self.api_key)
            
            logger.info("✅ Databento clients initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Databento clients: {e}")
            return False
    
    async def analyze_12h_volume(self, force_analysis: bool = False) -> Optional[str]:
        """
        STEP 2: Rolling 12-hour volume analysis (runs weekly)
        Returns the symbol with highest volume, or None if analysis fails
        """
        current_time = datetime.now()
        
        # Check if we need to run analysis (weekly or forced)
        if not force_analysis and self.last_analysis_time:
            time_since_analysis = current_time - self.last_analysis_time
            if time_since_analysis < self.analysis_interval:
                logger.info(f"📊 Skipping analysis - last run {time_since_analysis.days} days ago")
                return self.most_liquid_symbol
        
        logger.info("📊 Starting rolling 12-hour volume analysis...")
        
        try:
            # Set time range for last 12 hours
            end_time = current_time - timedelta(hours=6)  # Safe buffer for data availability
            start_time = end_time - timedelta(hours=12)
            
            logger.info(f"📅 Analyzing volume from {start_time} to {end_time}")
            
            # Fetch historical data for all ES contracts
            data = self.historical_client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["ES.FUT"],  # Parent symbol gets all ES contracts
                schema="ohlcv-1m",
                start=start_time,
                end=end_time,
                stype_in="parent",
                stype_out="instrument_id"
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                logger.warning("⚠️ No historical data returned")
                return None
            
            logger.info(f"📈 Retrieved {len(df)} historical records")
            
            # Filter data: exclude spreads and keep only known ES contracts
            logger.info("🔍 Filtering historical data...")
            
            # Remove spread contracts (containing '-')
            original_count = len(df)
            df = df[~df['symbol'].str.contains('-', na=False)]
            logger.info(f"📉 Removed {original_count - len(df)} spread contracts")
            
            # Keep only known ES contracts
            known_symbols = list(self.es_contracts.values())
            df = df[df['symbol'].isin(known_symbols)]
            logger.info(f"🎯 Filtered to known ES contracts: {known_symbols}")
            
            if df.empty:
                logger.warning("⚠️ No data remaining after filtering")
                return None
            
            # Calculate total volume by symbol over 12 hours
            volume_analysis = df.groupby('symbol')['volume'].sum().sort_values(ascending=False)
            
            logger.info("📊 12-hour volume analysis results:")
            for symbol, total_volume in volume_analysis.items():
                # Find readable name for symbol
                readable_name = next(
                    (name for name, sym in self.es_contracts.items() if sym == symbol), 
                    symbol
                )
                logger.info(f"  {symbol} ({readable_name}): {total_volume:,} contracts")
            
            # Get most liquid symbol
            most_liquid = volume_analysis.index[0]
            total_volume = volume_analysis.iloc[0]
            
            readable_name = next(
                (name for name, sym in self.es_contracts.items() if sym == most_liquid), 
                most_liquid
            )
            
            logger.info(f"🏆 Most liquid contract: {most_liquid} ({readable_name})")
            logger.info(f"📈 Total 12h volume: {total_volume:,} contracts")
            
            # Update tracking variables
            previous_symbol = self.most_liquid_symbol
            self.most_liquid_symbol = most_liquid
            self.last_analysis_time = current_time
            
            # Check if we need to switch contracts
            if previous_symbol and previous_symbol != most_liquid:
                logger.info(f"🔄 Contract change detected: {previous_symbol} → {most_liquid}")
                return "SWITCH_REQUIRED"
            
            return most_liquid
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze 12h volume: {e}")
            return None
    
    async def fetch_24h_historical_data(self) -> bool:
        """
        Fetch and process 24-hour historical OHLCV data for the selected contract
        """
        if not self.most_liquid_symbol:
            logger.error("❌ No most liquid symbol identified for historical data fetch")
            return False
        
        logger.info(f"📊 Fetching 24-hour historical data for {self.most_liquid_symbol}...")
        
        try:
            # Set time range for last 24 hours
            end_time = datetime.now() - timedelta(hours=1)  # Buffer for data availability
            start_time = end_time - timedelta(hours=24)
            
            logger.info(f"📅 Fetching historical data from {start_time} to {end_time}")
            
            # Fetch historical OHLCV data using parent symbol (same as backtest)
            data = self.historical_client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["ES.FUT"],  # Parent symbol (same as backtest)
                schema="ohlcv-1m",
                start=start_time,
                end=end_time,
                stype_in="parent",  # Parent symbol input
                stype_out="instrument_id"  # Get instrument_id output
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                logger.warning("⚠️ No historical data returned for 24h period")
                return False
            
            # Filter for our specific symbol
            df = df[df['symbol'] == self.most_liquid_symbol]
            
            if df.empty:
                logger.warning(f"⚠️ No historical data for symbol {self.most_liquid_symbol}")
                return False
            
            logger.info(f"📈 Processing {len(df)} historical 1-minute bars")
            
            # Process each historical bar (same as backtest)
            for idx, row in df.iterrows():
                # Prices appear to already be scaled correctly (same as backtest)
                # Create timestamp from DataFrame index (which should be the timestamp)
                timestamp = idx
                
                # Create standardized data structure
                historical_bar = {
                    'timestamp': timestamp,
                    'symbol': row['symbol'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                
                # Send to signal processing
                self.process_market_data(historical_bar)
            
            logger.info(f"✅ Processed {len(df)} historical bars successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch 24h historical data: {e}")
            return False
    
    def setup_live_data_callback(self):
        """Set up callback function for processing live data"""
        
        def process_live_data(record):
            """
            Process incoming live data records and pass to next processing stage
            """
            try:
                # Check if this is OHLCV data we care about
                if hasattr(record, 'close') and hasattr(record, 'volume'):
                    
                    # Apply Databento price scaling (prices are in fixed-point format)
                    scaled_open = record.open / 1e9
                    scaled_high = record.high / 1e9
                    scaled_low = record.low / 1e9
                    scaled_close = record.close / 1e9
                    
                    # Convert timestamp
                    timestamp = pd.to_datetime(record.ts_event, unit='ns')
                    
                    # Create standardized data structure
                    live_data = {
                        'timestamp': timestamp,
                        'symbol': getattr(record, 'symbol', 'Unknown'),
                        'open': scaled_open,
                        'high': scaled_high,
                        'low': scaled_low,
                        'close': scaled_close,
                        'volume': record.volume
                    }
                    
                    # Only process data for our monitored symbol
                    if live_data['symbol'] == self.most_liquid_symbol:
                        # Pass data to next processing stage
                        self.process_market_data(live_data)
                    
            except Exception as e:
                logger.error(f"❌ Error processing live data: {e}")
        
        return process_live_data
    
    def process_market_data(self, data: Dict[str, Any]):
        """
        Process market data - INTERFACE FOR SIGNAL PROCESSING
        This method will be called with each new market data point
        """
        try:
            if SIGNAL_MODULE_AVAILABLE:
                # Send data to signal processing module
                handle_bar(data)
            else:
                # Fallback: just log the data
                logger.info(f"📊 {data['symbol']} @ {data['timestamp']}: "
                           f"O={data['open']:.2f}, H={data['high']:.2f}, "
                           f"L={data['low']:.2f}, C={data['close']:.2f}, V={data['volume']:.0f}")
        except Exception as e:
            logger.error(f"❌ Error processing market data: {e}")
            logger.error(f"    Data: {data}")
    
    async def restart_live_stream(self):
        """Restart live stream with new most liquid contract"""
        logger.info("🔄 Restarting live stream with new contract...")
        
        # Stop current stream
        if self.is_streaming:
            await self.stop_stream()
            await asyncio.sleep(2)  # Give it time to properly disconnect
        
        # Start new stream
        await self.start_live_stream()
    
    async def start_live_stream(self) -> bool:
        """
        STEP 3: Subscribe to live data for the most liquid symbol
        """
        if not self.most_liquid_symbol:
            logger.error("❌ No most liquid symbol identified")
            return False
        
        try:
            logger.info(f"📡 Starting live stream for {self.most_liquid_symbol}...")
            
            # Subscribe to live OHLCV data
            self.live_client.subscribe(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",  # 1-minute bars
                stype_in="parent",
                symbols=["ES.FUT"]  # Subscribe to parent, filter in callback
            )
            
            # Set up data processing callback
            callback = self.setup_live_data_callback()
            self.live_client.add_callback(callback)
            
            # Start the live client
            logger.info("🚀 Starting live data client...")
            self.live_client.start()
            
            self.is_streaming = True
            logger.info("✅ Live stream started successfully")
            logger.info("📊 Monitoring real-time Databento data stream...")
            logger.info("⚠️  Press Ctrl+C to stop")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start live stream: {e}")
            return False
    
    async def stop_stream(self):
        """Stop the live data stream"""
        logger.info("⏹️  Stopping live stream...")
        
        self.is_streaming = False
        
        if self.live_client:
            try:
                self.live_client.stop()
                logger.info("✅ Live stream stopped")
            except Exception as e:
                logger.error(f"⚠️  Error stopping live client: {e}")
    
    async def run(self):
        """Main execution flow"""
        logger.info("🚀 Starting Production Databento Connector")
        logger.info("=" * 60)
        
        # STEP 1: Initialize connection
        if not await self.initialize():
            logger.error("❌ Failed to initialize connector")
            return
        
        # STEP 2: Initial 12h volume analysis
        most_liquid = await self.analyze_12h_volume(force_analysis=True)
        if not most_liquid:
            logger.error("❌ Failed to identify most liquid contract")
            return
        
        # STEP 3: Fetch and process 24-hour historical data
        logger.info("📊 Fetching 24-hour historical data...")
        if not await self.fetch_24h_historical_data():
            logger.warning("⚠️ Failed to fetch historical data, continuing with live stream only")
        
        # STEP 4: Start live monitoring
        if not await self.start_live_stream():
            logger.error("❌ Failed to start live stream")
            return
        
        # Main monitoring loop with weekly analysis checks
        logger.info("🔄 Starting continuous monitoring with weekly analysis...")
        try:
            analysis_check_interval = 3600  # Check every hour if we need to run analysis
            
            while self.is_streaming:
                await asyncio.sleep(analysis_check_interval)
                
                # Check if we need to run weekly analysis
                analysis_result = await self.analyze_12h_volume()
                
                if analysis_result == "SWITCH_REQUIRED":
                    logger.info("🔄 Most liquid contract changed - switching stream...")
                    await self.restart_live_stream()
                elif analysis_result is None:
                    logger.warning("⚠️ Analysis failed - continuing with current contract")
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            await self.stop_stream()
            logger.info("👋 Connector shutdown complete")


async def main():
    """Main entry point"""
    connector = ProductionDatabentoConnector()
    await connector.run()


if __name__ == "__main__":
    if not DATABENTO_AVAILABLE:
        logger.error("❌ Cannot run without Databento package")
        exit(1)
    
    # Run the connector
    asyncio.run(main())
