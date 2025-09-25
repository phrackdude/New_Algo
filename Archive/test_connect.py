#!/usr/bin/env python3
"""
Simple Databento API Test Script
Retrieves real market data from the past 24 hours and saves to CSV
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    logger.error("❌ Databento not installed. Please install with: pip install databento")
    exit(1)


def fetch_market_data():
    """Fetch real market data from the past 24 hours and filter for most liquid ES contract"""
    
    # Known ES futures contracts mapping
    known_es_contracts = {
        'ES JUN25': 'ESM6',
        'ES SEP25': 'ESU5', 
        'ES DEC25': 'ESZ5',
        'ES MAR26': 'ESH6'
    }
    
    # Get API key from environment
    api_key = os.getenv('DATABENTO_API_KEY')
    if not api_key:
        logger.error("❌ DATABENTO_API_KEY not found in environment variables")
        logger.info("Please create a .env file with your Databento API key:")
        logger.info("DATABENTO_API_KEY=your_api_key_here")
        return None
    
    try:
        # Initialize Databento client
        logger.info("🔌 Initializing Databento client...")
        client = db.Historical(key=api_key)
        
        # Set date range for past 24 hours, but ensure we don't exceed available data
        # Databento typically has data available up to a few hours ago
        end_date = datetime.now() - timedelta(hours=6)  # Use data from 6 hours ago to be safe
        start_date = end_date - timedelta(days=1)
        
        logger.info(f"📅 Fetching data from {start_date} to {end_date}")
        
        # Fetch data - using ES.FUT for E-mini S&P 500 futures
        # Using all available schema types to get comprehensive data
        symbol = "ES.FUT"
        dataset = "GLBX.MDP3"  # CME Globex MDP 3.0
        
        logger.info(f"📈 Requesting data for {symbol} from {dataset}")
        
        # Get OHLCV data (1-minute bars)
        data = client.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol],
            schema="ohlcv-1m",  # 1-minute OHLCV bars
            start=start_date,
            end=end_date,
            stype_in="parent",  # Use parent symbol type
            stype_out="instrument_id"
        )
        
        # Convert to DataFrame
        df = data.to_df()
        
        if df.empty:
            logger.warning("⚠️ No data returned from API")
            return None
        
        logger.info(f"✅ Retrieved {len(df)} records")
        logger.info(f"📊 Columns available: {list(df.columns)}")
        
        # Filter the dataset
        logger.info("🔍 Filtering dataset...")
        
        # Step 1: Exclude symbols with dashes (spreads)
        original_count = len(df)
        df = df[~df['symbol'].str.contains('-', na=False)]
        logger.info(f"📉 Removed {original_count - len(df)} spread contracts (containing '-')")
        
        # Step 2: Only keep known ES futures contracts
        known_symbols = list(known_es_contracts.values())  # ['ESM6', 'ESU5', 'ESZ5', 'ESH6']
        df = df[df['symbol'].isin(known_symbols)]
        logger.info(f"🎯 Filtered to known ES contracts: {known_symbols}")
        logger.info(f"📊 Remaining records: {len(df)}")
        
        if df.empty:
            logger.warning("⚠️ No data remaining after filtering")
            return None
        
        # Step 3: Find the most liquid symbol (highest total volume)
        volume_by_symbol = df.groupby('symbol')['volume'].sum().sort_values(ascending=False)
        logger.info("📈 Volume by symbol over 24 hours:")
        for symbol, total_volume in volume_by_symbol.items():
            logger.info(f"  {symbol}: {total_volume:,} contracts")
        
        most_liquid_symbol = volume_by_symbol.index[0]
        logger.info(f"🏆 Most liquid symbol: {most_liquid_symbol} with {volume_by_symbol.iloc[0]:,} total volume")
        
        # Step 4: Filter to only the most liquid symbol
        df = df[df['symbol'] == most_liquid_symbol]
        logger.info(f"✅ Final filtered dataset: {len(df)} records for {most_liquid_symbol}")
        
        # Display sample data info
        if not df.empty:
            logger.info(f"📈 Price range: ${df['low'].min()/1e9:.2f} - ${df['high'].max()/1e9:.2f}")
            logger.info(f"📊 Volume range: {df['volume'].min():,} - {df['volume'].max():,}")
            logger.info(f"⏰ Time range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch data: {e}")
        return None


def save_to_csv(df, filename="latest_data.csv"):
    """Save DataFrame to CSV file"""
    if df is None or df.empty:
        logger.error("❌ No data to save")
        return False
    
    try:
        # Get the archive directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, filename)
        
        # Save to CSV with all columns
        df.to_csv(csv_path)
        
        logger.info(f"💾 Data saved to: {csv_path}")
        logger.info(f"📁 File size: {os.path.getsize(csv_path):,} bytes")
        
        # Display first few rows as preview
        logger.info("📋 Data preview (first 5 rows):")
        print(df.head())
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save CSV: {e}")
        return False


def main():
    """Main execution function"""
    logger.info("🚀 Starting Databento data retrieval...")
    logger.info("=" * 50)
    
    # Check if Databento is available
    if not DATABENTO_AVAILABLE:
        logger.error("❌ Databento package not available")
        return
    
    # Fetch market data
    logger.info("1️⃣ Fetching market data...")
    df = fetch_market_data()
    
    if df is None:
        logger.error("❌ Failed to retrieve data")
        return
    
    # Save to CSV
    logger.info("2️⃣ Saving data to CSV...")
    success = save_to_csv(df)
    
    if success:
        logger.info("✅ Script completed successfully!")
        logger.info(f"📄 Filtered data file ready for backtesting: latest_data.csv")
        logger.info("🎯 Contains only the most liquid ES futures contract from the past 24 hours")
    else:
        logger.error("❌ Script failed to save data")


if __name__ == "__main__":
    main()
