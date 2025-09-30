#!/usr/bin/env python3
"""
cluster_test.py - Volume Cluster Detection with Direction Determination Backtest
Tests volume cluster detection, modal position analysis, and direction determination on historical data.

This script:
1. Connects to Databento API using same logic as 01_connect.py
2. Fetches 7 days of 1-minute OHLCV data for the currently selected ES contract
3. Aggregates data into 15-minute bars
4. Calculates daily average 1-minute volume
5. Identifies volume clusters using: 15min_cluster_volume / daily_avg_1min_volume >= 4.0x
6. Applies rolling volume ranking (top-1 per day)
7. Calculates modal position analysis for each cluster using 14-minute price action windows
8. Determines signal direction based on modal position (LONG ≤ 0.15, NO_SIGNAL > 0.15)
9. Reports cluster count, modal position distribution, and direction determination results
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from typing import Dict, Any, List
import asyncio
import statistics

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


class VolumeClusterTester:
    """Volume cluster detection backtest for ES futures"""
    
    def __init__(self):
        self.api_key = None
        self.historical_client = None
        self.most_liquid_symbol = None
        
        # Volume cluster parameters
        self.VOLUME_MULTIPLIER = 4.0
        
        # Rolling volume ranking parameters (bias-free)
        self.ROLLING_WINDOW_HOURS = 2.0
        self.TOP_N_CLUSTERS_PER_DAY = 1
        self.MIN_CLUSTERS_FOR_RANKING = 2
        
        # Direction determination parameters
        self.TIGHT_LONG_THRESHOLD = 0.25  # Modal position threshold for long signals (optimized from 0.15)
        self.ELIMINATE_SHORTS = True  # Disable short signals due to market bias
        
        # Signal Strength Parameters
        self.MIN_SIGNAL_STRENGTH = 0.25  # Optimized threshold from 30-day analysis (was 0.45)
        
        # Retest Parameters
        self.RETEST_TOLERANCE = 0.75  # points tolerance for retest confirmation
        self.RETEST_TIMEOUT = 30  # minutes timeout for retest confirmation
        
        # Known ES futures contracts (same as 01_connect.py)
        self.es_contracts = {
            'ES JUN25': 'ESM6',
            'ES SEP25': 'ESU5', 
            'ES DEC25': 'ESZ5',
            'ES MAR26': 'ESH6'
        }
        
    async def initialize(self) -> bool:
        """Initialize Databento client with API credentials"""
        logger.info("🔌 Initializing Databento client for cluster testing...")
        
        # Get API key from environment
        self.api_key = os.getenv('DATABENTO_API_KEY')
        if not self.api_key:
            logger.error("❌ DATABENTO_API_KEY not found in environment variables")
            logger.info("Please create a .env file with your Databento API key:")
            logger.info("DATABENTO_API_KEY=your_api_key_here")
            return False
        
        try:
            # Initialize historical client
            self.historical_client = db.Historical(key=self.api_key)
            logger.info("✅ Databento historical client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Databento client: {e}")
            return False
    
    async def identify_most_liquid_contract(self) -> str:
        """
        Identify the most liquid ES contract using same logic as 01_connect.py
        """
        logger.info("📊 Identifying most liquid ES contract...")
        
        try:
            # Set time range for last 12 hours (same as 01_connect.py)
            current_time = datetime.now()
            end_time = current_time - timedelta(hours=6)
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
            
            # Debug: Show actual symbols returned
            unique_symbols = df['symbol'].unique()
            logger.debug(f"🔍 Actual symbols returned: {list(unique_symbols)}")
            
            # Filter data: exclude spreads and keep only known ES contracts
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
            
            self.most_liquid_symbol = most_liquid
            return most_liquid
            
        except Exception as e:
            logger.error(f"❌ Failed to identify most liquid contract: {e}")
            return None
    
    async def fetch_10_day_data(self) -> List[Dict[str, Any]]:
        """
        Fetch 10 days of 1-minute OHLCV data for the selected contract
        Returns data in the same format as handle_bar() expects
        """
        if not self.most_liquid_symbol:
            logger.error("❌ No most liquid symbol identified")
            return []
        
        logger.info(f"📊 Fetching 10-day historical data for {self.most_liquid_symbol}...")
        
        try:
            # Set time range for last 10 days
            end_time = datetime.now() - timedelta(hours=3)  # Larger buffer for data availability
            start_time = end_time - timedelta(days=10)
            
            logger.info(f"📅 Fetching data from {start_time} to {end_time}")
            logger.debug(f"🔍 Using symbol: {self.most_liquid_symbol}")
            
            # Try fetching with parent symbol first, then filter
            logger.debug("🔄 Attempting fetch with parent symbol ES.FUT...")
            data = self.historical_client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["ES.FUT"],  # Use parent symbol
                schema="ohlcv-1m",
                start=start_time,
                end=end_time,
                stype_in="parent",
                stype_out="instrument_id"
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                logger.warning("⚠️ No historical data returned for 7-day period")
                return []
            
            # Filter for our specific symbol
            df = df[df['symbol'] == self.most_liquid_symbol]
            
            if df.empty:
                logger.warning(f"⚠️ No historical data for symbol {self.most_liquid_symbol}")
                return []
            
            logger.info(f"📈 Retrieved {len(df)} 1-minute bars")
            
            # Debug: Show DataFrame columns and first few rows
            logger.debug(f"🔍 DataFrame columns: {list(df.columns)}")
            logger.debug(f"🔍 DataFrame shape: {df.shape}")
            if not df.empty:
                logger.debug(f"🔍 First row sample: {df.iloc[0].to_dict()}")
            
            # Convert to handle_bar() format
            bars = []
            for idx, row in df.iterrows():
                # Prices appear to already be scaled correctly
                # Create timestamp from DataFrame index (which should be the timestamp)
                timestamp = idx
                
                # Create standardized data structure
                bar = {
                    'timestamp': timestamp,
                    'symbol': row['symbol'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                
                bars.append(bar)
            
            logger.info(f"✅ Processed {len(bars)} 1-minute bars")
            return bars
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch 7-day data: {e}")
            return []
    
    def aggregate_to_15min_bars(self, bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate 1-minute bars into 15-minute bars
        """
        logger.info("🔄 Aggregating 1-minute bars into 15-minute bars...")
        
        if not bars:
            return []
        
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to 15-minute bars
        agg_dict = {
            'symbol': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample with 15-minute frequency
        df_15min = df.resample('15min').agg(agg_dict)
        
        # Remove rows with NaN values (incomplete bars)
        df_15min = df_15min.dropna()
        
        # Convert back to list of dictionaries
        bars_15min = []
        for timestamp, row in df_15min.iterrows():
            bar = {
                'timestamp': timestamp,
                'symbol': row['symbol'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }
            bars_15min.append(bar)
        
        logger.info(f"✅ Created {len(bars_15min)} 15-minute bars")
        return bars_15min
    
    def get_rolling_volume_rank(self, cluster_time, cluster_volume_ratio, past_clusters):
        """
        BIAS-FREE VOLUME RANKING: Only uses clusters that occurred BEFORE current cluster
        Returns the rank of current cluster among recent clusters (1 = highest volume)
        """
        from datetime import timedelta
        
        # Define rolling window - only look at clusters from past N hours
        lookback_start = cluster_time - timedelta(hours=self.ROLLING_WINDOW_HOURS)
        
        # Filter to only past clusters within the rolling window
        relevant_clusters = []
        for past_cluster in past_clusters:
            if lookback_start <= past_cluster['timestamp'] < cluster_time:
                relevant_clusters.append(past_cluster)
        
        # Add current cluster for ranking
        current_cluster = {
            'timestamp': cluster_time,
            'volume_ratio': cluster_volume_ratio
        }
        all_clusters = relevant_clusters + [current_cluster]
        
        # Require minimum clusters for meaningful ranking
        if len(all_clusters) < self.MIN_CLUSTERS_FOR_RANKING:
            return 1  # Default to rank 1 if insufficient history
        
        # Sort by volume ratio (descending) and find current cluster's rank
        sorted_clusters = sorted(all_clusters, key=lambda x: x['volume_ratio'], reverse=True)
        
        for rank, cluster in enumerate(sorted_clusters, 1):
            if cluster['timestamp'] == cluster_time:
                return rank
        
        return len(sorted_clusters)  # Fallback
    
    def calculate_daily_thresholds(self, bars_1min: List[Dict[str, Any]], 
                                   bars_15min: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate daily volume thresholds for each day
        Returns: {date_str: {'daily_avg_15min_volume': float, 'daily_avg_1min_volume': float, 'volume_threshold': float}}
        """
        logger.info("📊 Calculating daily volume thresholds...")
        
        # Group bars by date
        daily_thresholds = {}
        
        # Convert to DataFrames
        df_1min = pd.DataFrame(bars_1min)
        df_15min = pd.DataFrame(bars_15min)
        
        if df_1min.empty or df_15min.empty:
            return daily_thresholds
        
        df_1min['date'] = pd.to_datetime(df_1min['timestamp']).dt.date
        df_15min['date'] = pd.to_datetime(df_15min['timestamp']).dt.date
        
        # Get unique dates
        dates = sorted(set(df_1min['date'].unique()) & set(df_15min['date'].unique()))
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            
            # Calculate daily averages
            daily_1min_volumes = df_1min[df_1min['date'] == date]['volume']
            daily_15min_volumes = df_15min[df_15min['date'] == date]['volume']
            
            if len(daily_1min_volumes) > 0 and len(daily_15min_volumes) > 0:
                daily_avg_1min_volume = daily_1min_volumes.mean()
                daily_avg_15min_volume = daily_15min_volumes.mean()
                volume_threshold = daily_avg_15min_volume * self.VOLUME_MULTIPLIER
                
                daily_thresholds[date_str] = {
                    'daily_avg_15min_volume': daily_avg_15min_volume,
                    'daily_avg_1min_volume': daily_avg_1min_volume,
                    'volume_threshold': volume_threshold
                }
                
                logger.info(f"📅 {date_str}: Avg 15min vol: {daily_avg_15min_volume:.0f}, "
                           f"Avg 1min vol: {daily_avg_1min_volume:.0f}, "
                           f"Threshold: {volume_threshold:.0f}")
                logger.info(f"     15min bars: {len(daily_15min_volumes)}, 1min bars: {len(daily_1min_volumes)}")
        
        return daily_thresholds
    
    def detect_volume_clusters_02_signal_method(self, bars_15min: List[Dict[str, Any]], 
                                                daily_thresholds: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Detect volume clusters using EXACT 02_signal.py methodology
        CRITICAL: Uses cluster_volume / daily_avg_1min_volume with 4.0x threshold (matching backtest line 431)
        This matches the VolumeClusterProcessor.process_15min_cluster() method
        """
        logger.info("🔍 Detecting volume clusters using 02_signal.py methodology...")
        
        clusters = []
        
        for bar in bars_15min:
            date_str = bar['timestamp'].strftime('%Y-%m-%d')
            
            if date_str not in daily_thresholds:
                continue
            
            thresholds = daily_thresholds[date_str]
            daily_avg_1min_volume = thresholds['daily_avg_1min_volume']
            
            # EXACT 02_signal.py methodology:
            # Calculate volume ratio: 15min_cluster_volume / daily_avg_1min_volume
            cluster_volume = bar['volume']  # This is already 15-minute aggregated volume
            volume_ratio = cluster_volume / daily_avg_1min_volume
            
            # Check if this qualifies as a volume cluster (4x threshold - same as 02_signal.py)
            if volume_ratio >= self.VOLUME_MULTIPLIER:  # VOLUME_MULTIPLIER = 4.0
                cluster_bar = bar.copy()
                cluster_bar['volume_ratio'] = volume_ratio
                cluster_bar['daily_avg_1min_volume'] = daily_avg_1min_volume
                cluster_bar['date'] = date_str
                
                clusters.append(cluster_bar)
                
                logger.info(f"🎯 Volume cluster detected: {bar['timestamp']} - "
                           f"15min Volume: {cluster_volume:.0f}, Daily avg 1min: {daily_avg_1min_volume:.0f}, "
                           f"Ratio: {volume_ratio:.2f}x (threshold: {self.VOLUME_MULTIPLIER}x)")
        
        logger.info(f"✅ Found {len(clusters)} volume clusters using 02_signal.py method")
        return clusters
    
    def process_clusters_with_ranking_02_signal_method(self, bars_15min: List[Dict[str, Any]], 
                                                       daily_thresholds: Dict[str, Dict[str, float]]) -> tuple:
        """
        Process clusters chronologically using EXACT 02_signal.py methodology with rolling ranking
        Returns: (all_clusters, tradeable_clusters)
        """
        logger.info("🔍 Processing clusters chronologically with 02_signal.py methodology...")
        
        all_clusters = []
        tradeable_clusters = []
        processed_clusters = []  # Track past clusters for ranking
        
        # Sort bars chronologically
        sorted_bars = sorted(bars_15min, key=lambda x: x['timestamp'])
        
        for bar in sorted_bars:
            date_str = bar['timestamp'].strftime('%Y-%m-%d')
            
            if date_str not in daily_thresholds:
                continue
            
            thresholds = daily_thresholds[date_str]
            daily_avg_1min_volume = thresholds['daily_avg_1min_volume']
            
            # EXACT 02_signal.py methodology:
            # Calculate volume ratio: 15min_cluster_volume / daily_avg_1min_volume
            cluster_volume = bar['volume']  # This is already 15-minute aggregated volume
            volume_ratio = cluster_volume / daily_avg_1min_volume
            
            # Check if this qualifies as a volume cluster (4x threshold)
            if volume_ratio >= self.VOLUME_MULTIPLIER:  # VOLUME_MULTIPLIER = 4.0
                
                # Get rolling volume rank (only using past clusters - bias-free)
                volume_rank = self.get_rolling_volume_rank(
                    bar['timestamp'], volume_ratio, processed_clusters
                )
                
                # Create cluster with ranking info
                cluster_bar = bar.copy()
                cluster_bar['volume_ratio'] = volume_ratio
                cluster_bar['daily_avg_1min_volume'] = daily_avg_1min_volume
                cluster_bar['volume_rank'] = volume_rank
                cluster_bar['date'] = date_str
                cluster_bar['is_tradeable'] = volume_rank <= self.TOP_N_CLUSTERS_PER_DAY
                
                all_clusters.append(cluster_bar)
                
                # Check if this cluster is tradeable (top-N)
                if volume_rank <= self.TOP_N_CLUSTERS_PER_DAY:
                    tradeable_clusters.append(cluster_bar)
                    logger.info(f"🎯 TRADEABLE Cluster: {bar['timestamp']} - "
                               f"15min Volume: {cluster_volume:.0f}, Daily avg 1min: {daily_avg_1min_volume:.0f}, "
                               f"Ratio: {volume_ratio:.2f}x, Rank: #{volume_rank}")
                else:
                    logger.info(f"📊 Cluster (not tradeable): {bar['timestamp']} - "
                               f"15min Volume: {cluster_volume:.0f}, Daily avg 1min: {daily_avg_1min_volume:.0f}, "
                               f"Ratio: {volume_ratio:.2f}x, Rank: #{volume_rank}")
                
                # Add to processed clusters for future ranking decisions
                processed_clusters.append({
                    'timestamp': bar['timestamp'],
                    'volume_ratio': volume_ratio
                })
            
        logger.info(f"✅ Found {len(all_clusters)} total clusters, {len(tradeable_clusters)} tradeable using 02_signal.py method")
        return all_clusters, tradeable_clusters
    
    def print_cluster_summary(self, clusters: List[Dict[str, Any]], 
                              daily_thresholds: Dict[str, Dict[str, float]]):
        """Print detailed summary of cluster detection results"""
        logger.info("=" * 60)
        logger.info("📊 VOLUME CLUSTER DETECTION RESULTS")
        logger.info("=" * 60)
        
        total_clusters = len(clusters)
        logger.info(f"🎯 Total Volume Clusters Detected: {total_clusters}")
        
        if total_clusters == 0:
            logger.info("❌ No volume clusters found in the 7-day period")
            return
        
        # Group clusters by date
        clusters_by_date = {}
        for cluster in clusters:
            date = cluster['date']
            if date not in clusters_by_date:
                clusters_by_date[date] = []
            clusters_by_date[date].append(cluster)
        
        logger.info("\n📅 Daily Cluster Breakdown:")
        for date in sorted(clusters_by_date.keys()):
            daily_clusters = clusters_by_date[date]
            logger.info(f"  {date}: {len(daily_clusters)} clusters")
            
            # Show top 3 clusters for this date
            sorted_clusters = sorted(daily_clusters, key=lambda x: x['volume_ratio'], reverse=True)
            for i, cluster in enumerate(sorted_clusters[:3]):
                logger.info(f"    #{i+1}: {cluster['timestamp'].strftime('%H:%M')} - "
                           f"Volume: {cluster['volume']:.0f}, Ratio: {cluster['volume_ratio']:.2f}x")
        
        # Overall statistics
        volume_ratios = [c['volume_ratio'] for c in clusters]
        logger.info(f"\n📈 Cluster Statistics:")
        logger.info(f"  Average volume ratio: {sum(volume_ratios)/len(volume_ratios):.2f}x")
        logger.info(f"  Maximum volume ratio: {max(volume_ratios):.2f}x")
        logger.info(f"  Minimum volume ratio: {min(volume_ratios):.2f}x")
        
        logger.info("=" * 60)
    
    def print_cluster_summary_with_ranking(self, all_clusters: List[Dict[str, Any]], 
                                           tradeable_clusters: List[Dict[str, Any]],
                                           daily_thresholds: Dict[str, Dict[str, float]]):
        """Print detailed summary with ranking information"""
        logger.info("=" * 60)
        logger.info("📊 VOLUME CLUSTER DETECTION WITH RANKING RESULTS")
        logger.info("=" * 60)
        
        total_clusters = len(all_clusters)
        total_tradeable = len(tradeable_clusters)
        
        logger.info(f"🎯 Total Volume Clusters Detected: {total_clusters}")
        logger.info(f"💰 Tradeable Clusters (Top-{self.TOP_N_CLUSTERS_PER_DAY}): {total_tradeable}")
        logger.info(f"📈 Trade Rate: {total_tradeable/total_clusters*100:.1f}% of clusters are tradeable")
        
        if total_clusters == 0:
            logger.info("❌ No volume clusters found in the 7-day period")
            return
        
        # Group all clusters by date
        clusters_by_date = {}
        tradeable_by_date = {}
        
        for cluster in all_clusters:
            date = cluster['date']
            if date not in clusters_by_date:
                clusters_by_date[date] = []
            clusters_by_date[date].append(cluster)
        
        for cluster in tradeable_clusters:
            date = cluster['date']
            if date not in tradeable_by_date:
                tradeable_by_date[date] = []
            tradeable_by_date[date].append(cluster)
        
        logger.info("\n📅 Daily Breakdown (All Clusters):")
        for date in sorted(clusters_by_date.keys()):
            daily_clusters = clusters_by_date[date]
            daily_tradeable = tradeable_by_date.get(date, [])
            logger.info(f"  {date}: {len(daily_clusters)} clusters ({len(daily_tradeable)} tradeable)")
            
            # Show all clusters for this date with ranking
            sorted_clusters = sorted(daily_clusters, key=lambda x: x['volume_ratio'], reverse=True)
            for i, cluster in enumerate(sorted_clusters):
                tradeable_mark = "💰 TRADE" if cluster['is_tradeable'] else "📊 skip"
                logger.info(f"    #{i+1}: {cluster['timestamp'].strftime('%H:%M')} - "
                           f"Volume: {cluster['volume']:.0f}, Ratio: {cluster['volume_ratio']:.1f}x, "
                           f"Rank: #{cluster['volume_rank']} - {tradeable_mark}")
        
        # Tradeable clusters summary
        if total_tradeable > 0:
            logger.info("\n💰 TRADEABLE CLUSTERS SUMMARY:")
            tradeable_ratios = [c['volume_ratio'] for c in tradeable_clusters]
            logger.info(f"  Average volume ratio: {sum(tradeable_ratios)/len(tradeable_ratios):.1f}x")
            logger.info(f"  Maximum volume ratio: {max(tradeable_ratios):.1f}x")
            logger.info(f"  Minimum volume ratio: {min(tradeable_ratios):.1f}x")
            logger.info(f"  Trades per day: {total_tradeable/7:.1f}")
        
        # All clusters summary
        logger.info(f"\n📊 ALL CLUSTERS SUMMARY:")
        all_ratios = [c['volume_ratio'] for c in all_clusters]
        logger.info(f"  Average volume ratio: {sum(all_ratios)/len(all_ratios):.1f}x")
        logger.info(f"  Maximum volume ratio: {max(all_ratios):.1f}x")
        logger.info(f"  Minimum volume ratio: {min(all_ratios):.1f}x")
        logger.info(f"  Clusters per day: {total_clusters/7:.1f}")
        
        # Ranking parameters
        logger.info(f"\n⚙️  RANKING PARAMETERS:")
        logger.info(f"  Rolling window: {self.ROLLING_WINDOW_HOURS} hours")
        logger.info(f"  Top-N filter: {self.TOP_N_CLUSTERS_PER_DAY}")
        logger.info(f"  Min clusters for ranking: {self.MIN_CLUSTERS_FOR_RANKING}")
        
        logger.info("=" * 60)
    
    def calculate_modal_position(self, cluster_timestamp: datetime, 
                                bars_1min: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate modal position for a volume cluster using 14-minute price action window
        
        Args:
            cluster_timestamp: Timestamp of the volume cluster (15-minute bar end)
            bars_1min: List of all 1-minute bars
        
        Returns:
            Dictionary containing modal analysis results
        """
        # Convert bars to DataFrame for easier manipulation
        df = pd.DataFrame(bars_1min)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Find the cluster start time (15 minutes before cluster end)
        cluster_start = cluster_timestamp - timedelta(minutes=15)
        
        # Get 14-minute window following cluster start (as per specification)
        window_end = cluster_start + timedelta(minutes=14)
        
        # Filter data to the 14-minute analysis window
        cluster_slice = df[
            (df['timestamp'] >= cluster_start) & 
            (df['timestamp'] < window_end)
        ].copy()
        
        if len(cluster_slice) == 0:
            logger.warning(f"⚠️ No price data found for modal analysis window: {cluster_start} to {window_end}")
            return {
                'cluster_timestamp': cluster_timestamp,
                'window_start': cluster_start,
                'window_end': window_end,
                'modal_price': None,
                'modal_position': None,
                'price_high': None,
                'price_low': None,
                'price_range': None,
                'data_points': 0,
                'error': 'No data in window'
            }
        
        logger.debug(f"📊 Modal analysis window: {cluster_start} to {window_end} ({len(cluster_slice)} bars)")
        
        # Calculate modal price (most frequently traded price level)
        # Round to ES tick size (0.25) and find mode
        cluster_slice['rounded_close'] = (cluster_slice['close'] / 0.25).round() * 0.25
        
        try:
            # Calculate mode of rounded closing prices
            modal_price = statistics.mode(cluster_slice['rounded_close'])
        except statistics.StatisticsError:
            # If no unique mode, use the most common price (first occurrence in case of tie)
            price_counts = cluster_slice['rounded_close'].value_counts()
            modal_price = price_counts.index[0]
            logger.debug(f"📊 No unique mode found, using most frequent price: {modal_price}")
        
        # Calculate price range for the window
        price_high = cluster_slice['high'].max()
        price_low = cluster_slice['low'].min()
        
        # Calculate modal position (where modal price sits in the range)
        price_range = price_high - price_low
        if price_range > 1e-9:  # Avoid division by zero
            modal_position = (modal_price - price_low) / price_range
        else:
            modal_position = 0.5  # Default to middle if no range
        
        modal_analysis = {
            'cluster_timestamp': cluster_timestamp,
            'window_start': cluster_start,
            'window_end': window_end,
            'modal_price': modal_price,
            'modal_position': modal_position,
            'price_high': price_high,
            'price_low': price_low,
            'price_range': price_range,
            'data_points': len(cluster_slice),
            'error': None
        }
        
        # Log modal analysis results
        logger.info(f"🎯 Modal Analysis - Cluster @ {cluster_timestamp.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"    Window: {cluster_start.strftime('%H:%M')} to {window_end.strftime('%H:%M')} ({len(cluster_slice)} bars)")
        logger.info(f"    Price Range: {price_low:.2f} - {price_high:.2f} (range: {price_range:.2f})")
        logger.info(f"    Modal Price: {modal_price:.2f}")
        logger.info(f"    Modal Position: {modal_position:.3f} ({'Bullish' if modal_position < 0.5 else 'Bearish'})")
        
        return modal_analysis
    
    def calculate_pre_cluster_momentum(self, cluster_timestamp: datetime, 
                                      bars_1min: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate pre-cluster momentum using 30-minute lookback period
        
        Args:
            cluster_timestamp: Timestamp of the volume cluster (15-minute bar end)
            bars_1min: List of all 1-minute bars
        
        Returns:
            Dictionary containing momentum analysis results
        """
        # Convert bars to DataFrame for easier manipulation
        df = pd.DataFrame(bars_1min)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Find the cluster start time (15 minutes before cluster end)
        cluster_start = cluster_timestamp - timedelta(minutes=15)
        
        # Get 30-minute lookback window (before cluster start)
        momentum_start = cluster_start - timedelta(minutes=30)
        
        # Filter data to the 30-minute momentum window
        momentum_slice = df[
            (df['timestamp'] >= momentum_start) & 
            (df['timestamp'] < cluster_start)
        ].copy()
        
        if len(momentum_slice) == 0:
            logger.warning(f"⚠️ No price data found for momentum analysis window: {momentum_start} to {cluster_start}")
            return {
                'cluster_timestamp': cluster_timestamp,
                'momentum_start': momentum_start,
                'momentum_end': cluster_start,
                'momentum': None,
                'start_price': None,
                'end_price': None,
                'price_change': None,
                'data_points': 0,
                'error': 'No data in momentum window'
            }
        
        logger.debug(f"📊 Momentum analysis window: {momentum_start} to {cluster_start} ({len(momentum_slice)} bars)")
        
        # Calculate momentum: (end_price - start_price) / start_price
        start_price = momentum_slice.iloc[0]['close']
        end_price = momentum_slice.iloc[-1]['close']
        price_change = end_price - start_price
        
        if start_price > 1e-9:  # Avoid division by zero
            momentum = price_change / start_price
        else:
            momentum = 0.0
        
        momentum_analysis = {
            'cluster_timestamp': cluster_timestamp,
            'momentum_start': momentum_start,
            'momentum_end': cluster_start,
            'momentum': momentum,
            'start_price': start_price,
            'end_price': end_price,
            'price_change': price_change,
            'data_points': len(momentum_slice),
            'error': None
        }
        
        # Log momentum analysis results
        direction = 'Positive' if momentum > 0 else 'Negative' if momentum < 0 else 'Neutral'
        logger.info(f"📈 Momentum Analysis - Cluster @ {cluster_timestamp.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"    Window: {momentum_start.strftime('%H:%M')} to {cluster_start.strftime('%H:%M')} ({len(momentum_slice)} bars)")
        logger.info(f"    Price Change: {start_price:.2f} → {end_price:.2f} ({price_change:+.2f})")
        logger.info(f"    Momentum: {momentum:.4f} ({momentum*100:+.2f}%) - {direction}")
        
        return momentum_analysis

    def determine_signal_direction(self, modal_position: float) -> Dict[str, Any]:
        """
        Determine trading signal direction based on modal position
        
        Args:
            modal_position: Modal position value (0.0 to 1.0)
        
        Returns:
            Dictionary containing direction analysis results
        """
        if modal_position is None:
            return {
                'direction': None,
                'position_strength': 0.0,
                'signal_type': 'NO_DATA',
                'reason': 'No modal position data available'
            }
        
        # Long Signal Criteria
        if modal_position <= self.TIGHT_LONG_THRESHOLD:
            position_strength = 1.0 - (modal_position / self.TIGHT_LONG_THRESHOLD)
            return {
                'direction': 'long',
                'position_strength': position_strength,
                'signal_type': 'LONG',
                'reason': f'Modal position {modal_position:.3f} <= {self.TIGHT_LONG_THRESHOLD} (strong buying pressure)'
            }
        
        # Short Signal Criteria (Currently Disabled)
        elif modal_position >= 0.85 and not self.ELIMINATE_SHORTS:
            position_strength = (modal_position - 0.85) / 0.15
            return {
                'direction': 'short',
                'position_strength': position_strength,
                'signal_type': 'SHORT',
                'reason': f'Modal position {modal_position:.3f} >= 0.85 (selling pressure)'
            }
        
        # No-Trade Zone
        else:
            if self.ELIMINATE_SHORTS and modal_position >= 0.85:
                reason = f'Modal position {modal_position:.3f} >= 0.85 but shorts eliminated'
            else:
                reason = f'Modal position {modal_position:.3f} in no-trade zone (0.15 < mp < 0.85)'
            
            return {
                'direction': None,
                'position_strength': 0.0,
                'signal_type': 'NO_SIGNAL',
                'reason': reason
            }
    
    def calculate_signal_strength(self, modal_position: float, volume_ratio: float, 
                                momentum: float, direction: str) -> Dict[str, Any]:
        """
        Calculate signal strength using simplified two-component formula
        
        Args:
            modal_position: Modal position value (0.0 to 1.0)
            volume_ratio: Volume ratio (cluster_volume / daily_avg_1min_volume)
            momentum: Pre-cluster momentum value (kept for compatibility but not used)
            direction: Signal direction ('long', 'short', or None)
        
        Returns:
            Dictionary containing signal strength analysis
        """
        if modal_position is None or volume_ratio is None:
            return {
                'signal_strength': 0.0,
                'position_strength': 0.0,
                'volume_strength': 0.0,
                'momentum_strength': 0.0,
                'meets_threshold': False,
                'error': 'Missing required data for signal strength calculation'
            }
        
        # Position Strength (70% weight) - increased from 50%
        position_strength = 1.0 - (modal_position / self.TIGHT_LONG_THRESHOLD)
        position_strength = max(0.0, min(1.0, position_strength))  # Clamp to [0, 1]
        
        # Volume Strength (30% weight) - same as before
        volume_strength = min(volume_ratio / 150.0, 1.0)
        volume_strength = max(0.0, volume_strength)  # Ensure non-negative
        
        # Momentum Strength (0% weight) - removed to eliminate trade blocking
        momentum_strength = 0.0  # Always zero - momentum component removed
        
        # Simplified Two-Component Formula
        signal_strength = (0.7 * position_strength + 
                          0.3 * volume_strength)
        
        # Signal Threshold Check
        meets_threshold = signal_strength >= self.MIN_SIGNAL_STRENGTH
        
        return {
            'signal_strength': signal_strength,
            'position_strength': position_strength,
            'volume_strength': volume_strength,
            'momentum_strength': momentum_strength,
            'meets_threshold': meets_threshold,
            'error': None
        }
    
    def calculate_retest_confirmation(self, cluster_timestamp: datetime, modal_price: float,
                                    bars_1min: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate retest confirmation for a volume cluster using post-cluster price action
        
        Args:
            cluster_timestamp: Timestamp of the volume cluster (15-minute bar end)
            modal_price: Modal price level to test for retest
            bars_1min: List of all 1-minute bars
        
        Returns:
            Dictionary containing retest analysis results
        """
        if modal_price is None:
            return {
                'retest_confirmed': False,
                'retest_time': None,
                'retest_price': None,
                'time_to_retest': None,
                'min_distance': None,
                'timeout_occurred': False,
                'data_points': 0,
                'error': 'No modal price available'
            }
        
        # Convert bars to DataFrame for easier manipulation
        df = pd.DataFrame(bars_1min)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Find the cluster end time
        cluster_end = cluster_timestamp
        
        # Get retest window: 30 minutes after cluster end
        retest_window_end = cluster_end + timedelta(minutes=self.RETEST_TIMEOUT)
        
        # Filter data to the retest analysis window
        retest_slice = df[
            (df['timestamp'] > cluster_end) & 
            (df['timestamp'] <= retest_window_end)
        ].copy()
        
        if len(retest_slice) == 0:
            logger.warning(f"⚠️ No price data found for retest window: {cluster_end} to {retest_window_end}")
            return {
                'retest_confirmed': False,
                'retest_time': None,
                'retest_price': None,
                'time_to_retest': None,
                'min_distance': None,
                'timeout_occurred': True,
                'data_points': 0,
                'error': 'No data in retest window'
            }
        
        logger.debug(f"📊 Retest analysis window: {cluster_end} to {retest_window_end} ({len(retest_slice)} bars)")
        
        # Check each minute for retest confirmation
        retest_confirmed = False
        retest_time = None
        retest_price = None
        time_to_retest = None
        min_distance = float('inf')
        
        for idx, row in retest_slice.iterrows():
            current_time = row['timestamp']
            current_price = row['close']
            
            # Calculate distance from modal price
            distance = abs(current_price - modal_price)
            min_distance = min(min_distance, distance)
            
            # Check if within retest tolerance
            if distance <= self.RETEST_TOLERANCE:
                retest_confirmed = True
                retest_time = current_time
                retest_price = current_price
                time_to_retest = (current_time - cluster_end).total_seconds() / 60.0  # minutes
                break
        
        # Handle case where min_distance was never updated
        if min_distance == float('inf'):
            min_distance = None
        
        timeout_occurred = not retest_confirmed
        
        retest_analysis = {
            'retest_confirmed': retest_confirmed,
            'retest_time': retest_time,
            'retest_price': retest_price,
            'time_to_retest': time_to_retest,
            'min_distance': min_distance,
            'timeout_occurred': timeout_occurred,
            'data_points': len(retest_slice),
            'error': None
        }
        
        # Log retest analysis results
        if retest_confirmed:
            logger.info(f"✅ Retest CONFIRMED: Modal={modal_price:.2f}, Retest={retest_price:.2f} @ {retest_time.strftime('%H:%M')}")
            logger.info(f"    Distance: {abs(retest_price - modal_price):.2f} ≤ {self.RETEST_TOLERANCE}, Time: {time_to_retest:.1f}min")
        else:
            logger.info(f"❌ Retest FAILED: Modal={modal_price:.2f}, Min distance: {min_distance:.2f} > {self.RETEST_TOLERANCE}")
            logger.info(f"    Timeout after {self.RETEST_TIMEOUT}min with {len(retest_slice)} bars analyzed")
        
        return retest_analysis
    
    def process_clusters_with_modal_analysis(self, bars_15min: List[Dict[str, Any]], 
                                           bars_1min: List[Dict[str, Any]],
                                           daily_thresholds: Dict[str, Dict[str, float]]) -> tuple:
        """
        Process clusters chronologically with modal position analysis
        Returns: (all_clusters_with_modal, tradeable_clusters_with_modal)
        """
        logger.info("🔍 Processing clusters with modal position analysis...")
        
        all_clusters = []
        tradeable_clusters = []
        processed_clusters = []  # Track past clusters for ranking
        
        # Sort bars chronologically
        sorted_bars = sorted(bars_15min, key=lambda x: x['timestamp'])
        
        for bar in sorted_bars:
            date_str = bar['timestamp'].strftime('%Y-%m-%d')
            
            if date_str not in daily_thresholds:
                continue
            
            thresholds = daily_thresholds[date_str]
            daily_avg_1min_volume = thresholds['daily_avg_1min_volume']
            
            # Calculate volume ratio
            cluster_volume = bar['volume']
            volume_ratio = cluster_volume / daily_avg_1min_volume
            
            # Check if this qualifies as a volume cluster (4x threshold)
            if volume_ratio >= self.VOLUME_MULTIPLIER:
                
                # Get rolling volume rank (only using past clusters - bias-free)
                volume_rank = self.get_rolling_volume_rank(
                    bar['timestamp'], volume_ratio, processed_clusters
                )
                
                # Calculate modal position analysis
                modal_analysis = self.calculate_modal_position(bar['timestamp'], bars_1min)
                
                # Determine signal direction based on modal position
                direction_analysis = self.determine_signal_direction(modal_analysis['modal_position'])
                
                # Create cluster with ranking and modal info
                cluster_bar = bar.copy()
                cluster_bar['volume_ratio'] = volume_ratio
                cluster_bar['daily_avg_1min_volume'] = daily_avg_1min_volume
                cluster_bar['volume_rank'] = volume_rank
                cluster_bar['date'] = date_str
                cluster_bar['is_tradeable'] = volume_rank <= self.TOP_N_CLUSTERS_PER_DAY
                
                # Add modal analysis results
                cluster_bar.update({
                    'modal_price': modal_analysis['modal_price'],
                    'modal_position': modal_analysis['modal_position'],
                    'price_high': modal_analysis['price_high'],
                    'price_low': modal_analysis['price_low'],
                    'price_range': modal_analysis['price_range'],
                    'modal_data_points': modal_analysis['data_points'],
                    'modal_error': modal_analysis['error']
                })
                
                # Add direction analysis results
                cluster_bar.update({
                    'signal_direction': direction_analysis['direction'],
                    'position_strength': direction_analysis['position_strength'],
                    'signal_type': direction_analysis['signal_type'],
                    'signal_reason': direction_analysis['reason']
                })
                
                all_clusters.append(cluster_bar)
                
                # Check if this cluster is tradeable (top-N)
                if volume_rank <= self.TOP_N_CLUSTERS_PER_DAY:
                    tradeable_clusters.append(cluster_bar)
                    
                    signal_info = f"Signal: {direction_analysis['signal_type']}"
                    if direction_analysis['direction']:
                        signal_info += f" ({direction_analysis['direction'].upper()}, strength: {direction_analysis['position_strength']:.3f})"
                    
                    modal_str = f"{modal_analysis['modal_position']:.3f}" if modal_analysis['modal_position'] is not None else "N/A"
                    logger.info(f"🎯 TRADEABLE Cluster: {bar['timestamp']} - "
                               f"Volume: {cluster_volume:.0f}, Ratio: {volume_ratio:.2f}x, "
                               f"Rank: #{volume_rank}, Modal: {modal_str}, "
                               f"{signal_info}")
                    
                    if direction_analysis['signal_type'] != 'NO_DATA':
                        logger.info(f"    📋 {direction_analysis['reason']}")
                else:
                    signal_info = f"Signal: {direction_analysis['signal_type']}"
                    if direction_analysis['direction']:
                        signal_info += f" ({direction_analysis['direction'].upper()})"
                    
                    modal_str = f"{modal_analysis['modal_position']:.3f}" if modal_analysis['modal_position'] is not None else "N/A"
                    logger.info(f"📊 Cluster (not tradeable): {bar['timestamp']} - "
                               f"Volume: {cluster_volume:.0f}, Ratio: {volume_ratio:.2f}x, "
                               f"Rank: #{volume_rank}, Modal: {modal_str}, "
                               f"{signal_info}")
                
                # Add to processed clusters for future ranking decisions
                processed_clusters.append({
                    'timestamp': bar['timestamp'],
                    'volume_ratio': volume_ratio
                })
        
        logger.info(f"✅ Processed {len(all_clusters)} clusters with modal analysis, {len(tradeable_clusters)} tradeable")
        return all_clusters, tradeable_clusters
    
    def process_clusters_with_momentum_analysis(self, bars_15min: List[Dict[str, Any]], 
                                              bars_1min: List[Dict[str, Any]],
                                              daily_thresholds: Dict[str, Dict[str, float]]) -> tuple:
        """
        Process clusters chronologically with modal position AND momentum analysis
        Returns: (all_clusters_with_momentum, tradeable_clusters_with_momentum)
        """
        logger.info("🔍 Processing clusters with modal position and momentum analysis...")
        
        all_clusters = []
        tradeable_clusters = []
        processed_clusters = []  # Track past clusters for ranking
        
        # Sort bars chronologically
        sorted_bars = sorted(bars_15min, key=lambda x: x['timestamp'])
        
        for bar in sorted_bars:
            date_str = bar['timestamp'].strftime('%Y-%m-%d')
            
            if date_str not in daily_thresholds:
                continue
            
            thresholds = daily_thresholds[date_str]
            daily_avg_1min_volume = thresholds['daily_avg_1min_volume']
            
            # Calculate volume ratio
            cluster_volume = bar['volume']
            volume_ratio = cluster_volume / daily_avg_1min_volume
            
            # Check if this qualifies as a volume cluster (4x threshold)
            if volume_ratio >= self.VOLUME_MULTIPLIER:
                
                # Get rolling volume rank (only using past clusters - bias-free)
                volume_rank = self.get_rolling_volume_rank(
                    bar['timestamp'], volume_ratio, processed_clusters
                )
                
                # Calculate modal position analysis
                modal_analysis = self.calculate_modal_position(bar['timestamp'], bars_1min)
                
                # Calculate pre-cluster momentum analysis
                momentum_analysis = self.calculate_pre_cluster_momentum(bar['timestamp'], bars_1min)
                
                # Determine signal direction based on modal position
                direction_analysis = self.determine_signal_direction(modal_analysis['modal_position'])
                
                # Create cluster with ranking, modal, and momentum info
                cluster_bar = bar.copy()
                cluster_bar['volume_ratio'] = volume_ratio
                cluster_bar['daily_avg_1min_volume'] = daily_avg_1min_volume
                cluster_bar['volume_rank'] = volume_rank
                cluster_bar['date'] = date_str
                cluster_bar['is_tradeable'] = volume_rank <= self.TOP_N_CLUSTERS_PER_DAY
                
                # Add modal analysis results
                cluster_bar.update({
                    'modal_price': modal_analysis['modal_price'],
                    'modal_position': modal_analysis['modal_position'],
                    'price_high': modal_analysis['price_high'],
                    'price_low': modal_analysis['price_low'],
                    'price_range': modal_analysis['price_range'],
                    'modal_data_points': modal_analysis['data_points'],
                    'modal_error': modal_analysis['error']
                })
                
                # Add momentum analysis results
                cluster_bar.update({
                    'momentum': momentum_analysis['momentum'],
                    'momentum_start_price': momentum_analysis['start_price'],
                    'momentum_end_price': momentum_analysis['end_price'],
                    'momentum_price_change': momentum_analysis['price_change'],
                    'momentum_data_points': momentum_analysis['data_points'],
                    'momentum_error': momentum_analysis['error']
                })
                
                # Add direction analysis results
                cluster_bar.update({
                    'signal_direction': direction_analysis['direction'],
                    'position_strength': direction_analysis['position_strength'],
                    'signal_type': direction_analysis['signal_type'],
                    'signal_reason': direction_analysis['reason']
                })
                
                all_clusters.append(cluster_bar)
                
                # Check if this cluster is tradeable (top-N)
                if volume_rank <= self.TOP_N_CLUSTERS_PER_DAY:
                    tradeable_clusters.append(cluster_bar)
                    
                    signal_info = f"Signal: {direction_analysis['signal_type']}"
                    if direction_analysis['direction']:
                        signal_info += f" ({direction_analysis['direction'].upper()}, strength: {direction_analysis['position_strength']:.3f})"
                    
                    modal_str = f"{modal_analysis['modal_position']:.3f}" if modal_analysis['modal_position'] is not None else "N/A"
                    momentum_str = f"{momentum_analysis['momentum']:.4f}" if momentum_analysis['momentum'] is not None else "N/A"
                    momentum_pct = f"({momentum_analysis['momentum']*100:+.2f}%)" if momentum_analysis['momentum'] is not None else ""
                    
                    logger.info(f"🎯 TRADEABLE Cluster: {bar['timestamp']} - "
                               f"Volume: {cluster_volume:.0f}, Ratio: {volume_ratio:.2f}x, "
                               f"Rank: #{volume_rank}, Modal: {modal_str}, "
                               f"Momentum: {momentum_str} {momentum_pct}, "
                               f"{signal_info}")
                    
                    if direction_analysis['signal_type'] != 'NO_DATA':
                        logger.info(f"    📋 {direction_analysis['reason']}")
                else:
                    signal_info = f"Signal: {direction_analysis['signal_type']}"
                    if direction_analysis['direction']:
                        signal_info += f" ({direction_analysis['direction'].upper()})"
                    
                    modal_str = f"{modal_analysis['modal_position']:.3f}" if modal_analysis['modal_position'] is not None else "N/A"
                    momentum_str = f"{momentum_analysis['momentum']:.4f}" if momentum_analysis['momentum'] is not None else "N/A"
                    
                    logger.info(f"📊 Cluster (not tradeable): {bar['timestamp']} - "
                               f"Volume: {cluster_volume:.0f}, Ratio: {volume_ratio:.2f}x, "
                               f"Rank: #{volume_rank}, Modal: {modal_str}, "
                               f"Momentum: {momentum_str}, {signal_info}")
                
                # Add to processed clusters for future ranking decisions
                processed_clusters.append({
                    'timestamp': bar['timestamp'],
                    'volume_ratio': volume_ratio
                })
        
        logger.info(f"✅ Processed {len(all_clusters)} clusters with momentum analysis, {len(tradeable_clusters)} tradeable")
        return all_clusters, tradeable_clusters
    
    def process_clusters_with_signal_strength_analysis(self, bars_15min: List[Dict[str, Any]], 
                                                      bars_1min: List[Dict[str, Any]],
                                                      daily_thresholds: Dict[str, Dict[str, float]]) -> tuple:
        """
        Process clusters chronologically with modal position, momentum, AND signal strength analysis
        Returns: (all_clusters_with_signal_strength, tradeable_clusters_with_signal_strength)
        """
        logger.info("🔍 Processing clusters with modal position, momentum, and signal strength analysis...")
        
        all_clusters = []
        tradeable_clusters = []
        processed_clusters = []  # Track past clusters for ranking
        
        # Sort bars chronologically
        sorted_bars = sorted(bars_15min, key=lambda x: x['timestamp'])
        
        for bar in sorted_bars:
            date_str = bar['timestamp'].strftime('%Y-%m-%d')
            
            if date_str not in daily_thresholds:
                continue
            
            thresholds = daily_thresholds[date_str]
            daily_avg_1min_volume = thresholds['daily_avg_1min_volume']
            
            # Calculate volume ratio
            cluster_volume = bar['volume']
            volume_ratio = cluster_volume / daily_avg_1min_volume
            
            # Check if this qualifies as a volume cluster (4x threshold)
            if volume_ratio >= self.VOLUME_MULTIPLIER:
                
                # Get rolling volume rank (only using past clusters - bias-free)
                volume_rank = self.get_rolling_volume_rank(
                    bar['timestamp'], volume_ratio, processed_clusters
                )
                
                # Calculate modal position analysis
                modal_analysis = self.calculate_modal_position(bar['timestamp'], bars_1min)
                
                # Calculate pre-cluster momentum analysis
                momentum_analysis = self.calculate_pre_cluster_momentum(bar['timestamp'], bars_1min)
                
                # Determine signal direction based on modal position
                direction_analysis = self.determine_signal_direction(modal_analysis['modal_position'])
                
                # Calculate signal strength
                strength_analysis = self.calculate_signal_strength(
                    modal_analysis['modal_position'],
                    volume_ratio,
                    momentum_analysis['momentum'],
                    direction_analysis['direction']
                )
                
                # Create cluster with ranking, modal, momentum, and signal strength info
                cluster_bar = bar.copy()
                cluster_bar['volume_ratio'] = volume_ratio
                cluster_bar['daily_avg_1min_volume'] = daily_avg_1min_volume
                cluster_bar['volume_rank'] = volume_rank
                cluster_bar['date'] = date_str
                cluster_bar['is_tradeable'] = volume_rank <= self.TOP_N_CLUSTERS_PER_DAY
                
                # Add modal analysis results
                cluster_bar.update({
                    'modal_price': modal_analysis['modal_price'],
                    'modal_position': modal_analysis['modal_position'],
                    'price_high': modal_analysis['price_high'],
                    'price_low': modal_analysis['price_low'],
                    'price_range': modal_analysis['price_range'],
                    'modal_data_points': modal_analysis['data_points'],
                    'modal_error': modal_analysis['error']
                })
                
                # Add momentum analysis results
                cluster_bar.update({
                    'momentum': momentum_analysis['momentum'],
                    'momentum_start_price': momentum_analysis['start_price'],
                    'momentum_end_price': momentum_analysis['end_price'],
                    'momentum_price_change': momentum_analysis['price_change'],
                    'momentum_data_points': momentum_analysis['data_points'],
                    'momentum_error': momentum_analysis['error']
                })
                
                # Add direction analysis results
                cluster_bar.update({
                    'signal_direction': direction_analysis['direction'],
                    'position_strength': direction_analysis['position_strength'],
                    'signal_type': direction_analysis['signal_type'],
                    'signal_reason': direction_analysis['reason']
                })
                
                # Add signal strength analysis results
                cluster_bar.update({
                    'signal_strength': strength_analysis['signal_strength'],
                    'signal_position_strength': strength_analysis['position_strength'],
                    'signal_volume_strength': strength_analysis['volume_strength'],
                    'signal_momentum_strength': strength_analysis['momentum_strength'],
                    'meets_strength_threshold': strength_analysis['meets_threshold'],
                    'strength_error': strength_analysis['error']
                })
                
                all_clusters.append(cluster_bar)
                
                # Check if this cluster is tradeable (top-N)
                if volume_rank <= self.TOP_N_CLUSTERS_PER_DAY:
                    tradeable_clusters.append(cluster_bar)
                    
                    signal_info = f"Signal: {direction_analysis['signal_type']}"
                    if direction_analysis['direction']:
                        signal_info += f" ({direction_analysis['direction'].upper()}, strength: {direction_analysis['position_strength']:.3f})"
                    
                    modal_str = f"{modal_analysis['modal_position']:.3f}" if modal_analysis['modal_position'] is not None else "N/A"
                    momentum_str = f"{momentum_analysis['momentum']:.4f}" if momentum_analysis['momentum'] is not None else "N/A"
                    momentum_pct = f"({momentum_analysis['momentum']*100:+.2f}%)" if momentum_analysis['momentum'] is not None else ""
                    
                    # Signal strength info
                    strength_str = f"{strength_analysis['signal_strength']:.3f}"
                    threshold_status = "✅ PASS" if strength_analysis['meets_threshold'] else "❌ FAIL"
                    
                    logger.info(f"🎯 TRADEABLE Cluster: {bar['timestamp']} - "
                               f"Volume: {cluster_volume:.0f}, Ratio: {volume_ratio:.2f}x, "
                               f"Rank: #{volume_rank}, Modal: {modal_str}, "
                               f"Momentum: {momentum_str} {momentum_pct}, "
                               f"Strength: {strength_str} ({threshold_status}), "
                               f"{signal_info}")
                    
                    if direction_analysis['signal_type'] != 'NO_DATA':
                        logger.info(f"    📋 {direction_analysis['reason']}")
                        if strength_analysis['meets_threshold']:
                            logger.info(f"    💪 Signal Strength Components: Pos={strength_analysis['position_strength']:.3f}, "
                                       f"Vol={strength_analysis['volume_strength']:.3f}, Mom={strength_analysis['momentum_strength']:.3f}")
                else:
                    signal_info = f"Signal: {direction_analysis['signal_type']}"
                    if direction_analysis['direction']:
                        signal_info += f" ({direction_analysis['direction'].upper()})"
                    
                    modal_str = f"{modal_analysis['modal_position']:.3f}" if modal_analysis['modal_position'] is not None else "N/A"
                    momentum_str = f"{momentum_analysis['momentum']:.4f}" if momentum_analysis['momentum'] is not None else "N/A"
                    strength_str = f"{strength_analysis['signal_strength']:.3f}"
                    
                    logger.info(f"📊 Cluster (not tradeable): {bar['timestamp']} - "
                               f"Volume: {cluster_volume:.0f}, Ratio: {volume_ratio:.2f}x, "
                               f"Rank: #{volume_rank}, Modal: {modal_str}, "
                               f"Momentum: {momentum_str}, Strength: {strength_str}, {signal_info}")
                
                # Add to processed clusters for future ranking decisions
                processed_clusters.append({
                    'timestamp': bar['timestamp'],
                    'volume_ratio': volume_ratio
                })
        
        logger.info(f"✅ Processed {len(all_clusters)} clusters with signal strength analysis, {len(tradeable_clusters)} tradeable")
        return all_clusters, tradeable_clusters
    
    def process_clusters_with_retest_analysis(self, bars_15min: List[Dict[str, Any]], 
                                            bars_1min: List[Dict[str, Any]],
                                            daily_thresholds: Dict[str, Dict[str, float]]) -> tuple:
        """
        Process clusters chronologically with modal position, momentum, signal strength, AND retest analysis
        Returns: (all_clusters_with_retest, tradeable_clusters_with_retest)
        """
        logger.info("🔍 Processing clusters with modal position, momentum, signal strength, and retest analysis...")
        
        all_clusters = []
        tradeable_clusters = []
        processed_clusters = []  # Track past clusters for ranking
        
        # Sort bars chronologically
        sorted_bars = sorted(bars_15min, key=lambda x: x['timestamp'])
        
        for bar in sorted_bars:
            date_str = bar['timestamp'].strftime('%Y-%m-%d')
            
            if date_str not in daily_thresholds:
                continue
            
            thresholds = daily_thresholds[date_str]
            daily_avg_1min_volume = thresholds['daily_avg_1min_volume']
            
            # Calculate volume ratio
            cluster_volume = bar['volume']
            volume_ratio = cluster_volume / daily_avg_1min_volume
            
            # Check if this qualifies as a volume cluster (4x threshold)
            if volume_ratio >= self.VOLUME_MULTIPLIER:
                
                # Get rolling volume rank (only using past clusters - bias-free)
                volume_rank = self.get_rolling_volume_rank(
                    bar['timestamp'], volume_ratio, processed_clusters
                )
                
                # Calculate modal position analysis
                modal_analysis = self.calculate_modal_position(bar['timestamp'], bars_1min)
                
                # Calculate pre-cluster momentum analysis
                momentum_analysis = self.calculate_pre_cluster_momentum(bar['timestamp'], bars_1min)
                
                # Determine signal direction based on modal position
                direction_analysis = self.determine_signal_direction(modal_analysis['modal_position'])
                
                # Calculate signal strength
                strength_analysis = self.calculate_signal_strength(
                    modal_analysis['modal_position'],
                    volume_ratio,
                    momentum_analysis['momentum'],
                    direction_analysis['direction']
                )
                
                # Calculate retest confirmation
                retest_analysis = self.calculate_retest_confirmation(
                    bar['timestamp'], 
                    modal_analysis['modal_price'], 
                    bars_1min
                )
                
                # Create cluster with all analysis components
                cluster_bar = bar.copy()
                cluster_bar['volume_ratio'] = volume_ratio
                cluster_bar['daily_avg_1min_volume'] = daily_avg_1min_volume
                cluster_bar['volume_rank'] = volume_rank
                cluster_bar['date'] = date_str
                cluster_bar['is_tradeable'] = volume_rank <= self.TOP_N_CLUSTERS_PER_DAY
                
                # Add modal analysis results
                cluster_bar.update({
                    'modal_price': modal_analysis['modal_price'],
                    'modal_position': modal_analysis['modal_position'],
                    'price_high': modal_analysis['price_high'],
                    'price_low': modal_analysis['price_low'],
                    'price_range': modal_analysis['price_range'],
                    'modal_data_points': modal_analysis['data_points'],
                    'modal_error': modal_analysis['error']
                })
                
                # Add momentum analysis results
                cluster_bar.update({
                    'momentum': momentum_analysis['momentum'],
                    'momentum_start_price': momentum_analysis['start_price'],
                    'momentum_end_price': momentum_analysis['end_price'],
                    'momentum_price_change': momentum_analysis['price_change'],
                    'momentum_data_points': momentum_analysis['data_points'],
                    'momentum_error': momentum_analysis['error']
                })
                
                # Add direction analysis results
                cluster_bar.update({
                    'signal_direction': direction_analysis['direction'],
                    'position_strength': direction_analysis['position_strength'],
                    'signal_type': direction_analysis['signal_type'],
                    'signal_reason': direction_analysis['reason']
                })
                
                # Add signal strength analysis results
                cluster_bar.update({
                    'signal_strength': strength_analysis['signal_strength'],
                    'signal_position_strength': strength_analysis['position_strength'],
                    'signal_volume_strength': strength_analysis['volume_strength'],
                    'signal_momentum_strength': strength_analysis['momentum_strength'],
                    'meets_strength_threshold': strength_analysis['meets_threshold'],
                    'strength_error': strength_analysis['error']
                })
                
                # Add retest analysis results
                cluster_bar.update({
                    'retest_confirmed': retest_analysis['retest_confirmed'],
                    'retest_time': retest_analysis['retest_time'],
                    'retest_price': retest_analysis['retest_price'],
                    'time_to_retest': retest_analysis['time_to_retest'],
                    'min_distance': retest_analysis['min_distance'],
                    'timeout_occurred': retest_analysis['timeout_occurred'],
                    'retest_data_points': retest_analysis['data_points'],
                    'retest_error': retest_analysis['error']
                })
                
                all_clusters.append(cluster_bar)
                
                # Check if this cluster is tradeable (top-N)
                if volume_rank <= self.TOP_N_CLUSTERS_PER_DAY:
                    tradeable_clusters.append(cluster_bar)
                    
                    signal_info = f"Signal: {direction_analysis['signal_type']}"
                    if direction_analysis['direction']:
                        signal_info += f" ({direction_analysis['direction'].upper()}, strength: {direction_analysis['position_strength']:.3f})"
                    
                    modal_str = f"{modal_analysis['modal_position']:.3f}" if modal_analysis['modal_position'] is not None else "N/A"
                    momentum_str = f"{momentum_analysis['momentum']:.4f}" if momentum_analysis['momentum'] is not None else "N/A"
                    momentum_pct = f"({momentum_analysis['momentum']*100:+.2f}%)" if momentum_analysis['momentum'] is not None else ""
                    
                    # Signal strength and retest info
                    strength_str = f"{strength_analysis['signal_strength']:.3f}"
                    threshold_status = "✅ PASS" if strength_analysis['meets_threshold'] else "❌ FAIL"
                    retest_status = "✅ RETEST" if retest_analysis['retest_confirmed'] else "❌ NO_RETEST"
                    
                    logger.info(f"🎯 TRADEABLE Cluster: {bar['timestamp']} - "
                               f"Volume: {cluster_volume:.0f}, Ratio: {volume_ratio:.2f}x, "
                               f"Rank: #{volume_rank}, Modal: {modal_str}, "
                               f"Momentum: {momentum_str} {momentum_pct}, "
                               f"Strength: {strength_str} ({threshold_status}), "
                               f"Retest: {retest_status}, {signal_info}")
                    
                    if direction_analysis['signal_type'] != 'NO_DATA':
                        logger.info(f"    📋 {direction_analysis['reason']}")
                        if strength_analysis['meets_threshold']:
                            logger.info(f"    💪 Signal Strength Components: Pos={strength_analysis['position_strength']:.3f}, "
                                       f"Vol={strength_analysis['volume_strength']:.3f}, Mom={strength_analysis['momentum_strength']:.3f}")
                        if retest_analysis['retest_confirmed']:
                            logger.info(f"    🔄 Retest Details: Time={retest_analysis['time_to_retest']:.1f}min, "
                                       f"Price={retest_analysis['retest_price']:.2f}, Distance={abs(retest_analysis['retest_price'] - modal_analysis['modal_price']):.2f}")
                else:
                    signal_info = f"Signal: {direction_analysis['signal_type']}"
                    if direction_analysis['direction']:
                        signal_info += f" ({direction_analysis['direction'].upper()})"
                    
                    modal_str = f"{modal_analysis['modal_position']:.3f}" if modal_analysis['modal_position'] is not None else "N/A"
                    momentum_str = f"{momentum_analysis['momentum']:.4f}" if momentum_analysis['momentum'] is not None else "N/A"
                    strength_str = f"{strength_analysis['signal_strength']:.3f}"
                    retest_status = "✅ RETEST" if retest_analysis['retest_confirmed'] else "❌ NO_RETEST"
                    
                    logger.info(f"📊 Cluster (not tradeable): {bar['timestamp']} - "
                               f"Volume: {cluster_volume:.0f}, Ratio: {volume_ratio:.2f}x, "
                               f"Rank: #{volume_rank}, Modal: {modal_str}, "
                               f"Momentum: {momentum_str}, Strength: {strength_str}, "
                               f"Retest: {retest_status}, {signal_info}")
                
                # Add to processed clusters for future ranking decisions
                processed_clusters.append({
                    'timestamp': bar['timestamp'],
                    'volume_ratio': volume_ratio
                })
        
        logger.info(f"✅ Processed {len(all_clusters)} clusters with retest analysis, {len(tradeable_clusters)} tradeable")
        return all_clusters, tradeable_clusters
    
    def print_modal_analysis_summary(self, all_clusters: List[Dict[str, Any]], 
                                   tradeable_clusters: List[Dict[str, Any]]):
        """Print detailed summary with modal position analysis"""
        logger.info("=" * 60)
        logger.info("📊 VOLUME CLUSTER DETECTION WITH MODAL ANALYSIS RESULTS")
        logger.info("=" * 60)
        
        total_clusters = len(all_clusters)
        total_tradeable = len(tradeable_clusters)
        
        logger.info(f"🎯 Total Volume Clusters Detected: {total_clusters}")
        logger.info(f"💰 Tradeable Clusters (Top-{self.TOP_N_CLUSTERS_PER_DAY}): {total_tradeable}")
        logger.info(f"📈 Trade Rate: {total_tradeable/total_clusters*100:.1f}% of clusters are tradeable")
        
        if total_clusters == 0:
            logger.info("❌ No volume clusters found in the 7-day period")
            return
        
        # Modal position analysis
        valid_modal_clusters = [c for c in all_clusters if c['modal_position'] is not None]
        valid_tradeable_modal = [c for c in tradeable_clusters if c['modal_position'] is not None]
        
        if len(valid_modal_clusters) > 0:
            logger.info(f"\n🎯 MODAL POSITION ANALYSIS (All {len(valid_modal_clusters)} clusters):")
            modal_positions = [c['modal_position'] for c in valid_modal_clusters]
            bullish_count = sum(1 for mp in modal_positions if mp < 0.5)
            bearish_count = sum(1 for mp in modal_positions if mp >= 0.5)
            
            logger.info(f"  Average Modal Position: {sum(modal_positions)/len(modal_positions):.3f}")
            logger.info(f"  Bullish Clusters (< 0.5): {bullish_count} ({bullish_count/len(modal_positions)*100:.1f}%)")
            logger.info(f"  Bearish Clusters (>= 0.5): {bearish_count} ({bearish_count/len(modal_positions)*100:.1f}%)")
            logger.info(f"  Modal Position Range: {min(modal_positions):.3f} - {max(modal_positions):.3f}")
        
        if len(valid_tradeable_modal) > 0:
            logger.info(f"\n💰 TRADEABLE MODAL ANALYSIS ({len(valid_tradeable_modal)} clusters):")
            tradeable_positions = [c['modal_position'] for c in valid_tradeable_modal]
            tradeable_bullish = sum(1 for mp in tradeable_positions if mp < 0.5)
            tradeable_bearish = sum(1 for mp in tradeable_positions if mp >= 0.5)
            
            logger.info(f"  Average Modal Position: {sum(tradeable_positions)/len(tradeable_positions):.3f}")
            logger.info(f"  Bullish Trades (< 0.5): {tradeable_bullish} ({tradeable_bullish/len(tradeable_positions)*100:.1f}%)")
            logger.info(f"  Bearish Trades (>= 0.5): {tradeable_bearish} ({tradeable_bearish/len(tradeable_positions)*100:.1f}%)")
        
        # Group clusters by date for detailed breakdown
        clusters_by_date = {}
        for cluster in all_clusters:
            date = cluster['date']
            if date not in clusters_by_date:
                clusters_by_date[date] = []
            clusters_by_date[date].append(cluster)
        
        logger.info("\n📅 Daily Breakdown with Modal Analysis:")
        for date in sorted(clusters_by_date.keys()):
            daily_clusters = clusters_by_date[date]
            daily_tradeable = [c for c in daily_clusters if c['is_tradeable']]
            logger.info(f"  {date}: {len(daily_clusters)} clusters ({len(daily_tradeable)} tradeable)")
            
            # Show clusters for this date with modal positions
            sorted_clusters = sorted(daily_clusters, key=lambda x: x['volume_ratio'], reverse=True)
            for i, cluster in enumerate(sorted_clusters):
                tradeable_mark = "💰 TRADE" if cluster['is_tradeable'] else "📊 skip"
                modal_info = ""
                if cluster['modal_position'] is not None:
                    sentiment = "BULL" if cluster['modal_position'] < 0.5 else "BEAR"
                    modal_info = f", Modal: {cluster['modal_position']:.3f} ({sentiment})"
                
                logger.info(f"    #{i+1}: {cluster['timestamp'].strftime('%H:%M')} - "
                           f"Vol: {cluster['volume']:.0f}, Ratio: {cluster['volume_ratio']:.1f}x, "
                           f"Rank: #{cluster['volume_rank']}{modal_info} - {tradeable_mark}")
        
        logger.info("=" * 60)
    
    def print_direction_analysis_summary(self, all_clusters: List[Dict[str, Any]], 
                                       tradeable_clusters: List[Dict[str, Any]]):
        """Print detailed summary with direction determination analysis"""
        logger.info("=" * 80)
        logger.info("📊 VOLUME CLUSTER DETECTION WITH DIRECTION DETERMINATION RESULTS")
        logger.info("=" * 80)
        
        total_clusters = len(all_clusters)
        total_tradeable = len(tradeable_clusters)
        
        logger.info(f"🎯 Total Volume Clusters Detected: {total_clusters}")
        logger.info(f"💰 Tradeable Clusters (Top-{self.TOP_N_CLUSTERS_PER_DAY}): {total_tradeable}")
        logger.info(f"📈 Trade Rate: {total_tradeable/total_clusters*100:.1f}% of clusters are tradeable")
        
        if total_clusters == 0:
            logger.info("❌ No volume clusters found in the 7-day period")
            return
        
        # Direction Analysis - All Clusters
        valid_clusters = [c for c in all_clusters if c['modal_position'] is not None]
        if len(valid_clusters) > 0:
            logger.info(f"\n🎯 DIRECTION ANALYSIS (All {len(valid_clusters)} clusters with modal data):")
            
            # Count signal types
            signal_counts = {}
            for cluster in valid_clusters:
                signal_type = cluster['signal_type']
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            
            for signal_type, count in signal_counts.items():
                percentage = count / len(valid_clusters) * 100
                logger.info(f"  {signal_type}: {count} clusters ({percentage:.1f}%)")
            
            # Long signal analysis
            long_clusters = [c for c in valid_clusters if c['signal_type'] == 'LONG']
            if long_clusters:
                long_strengths = [c['position_strength'] for c in long_clusters]
                logger.info(f"\n📈 LONG SIGNAL ANALYSIS ({len(long_clusters)} signals):")
                logger.info(f"  Average Position Strength: {sum(long_strengths)/len(long_strengths):.3f}")
                logger.info(f"  Strength Range: {min(long_strengths):.3f} - {max(long_strengths):.3f}")
                
                # Modal position distribution for long signals
                long_modals = [c['modal_position'] for c in long_clusters]
                logger.info(f"  Modal Position Range: {min(long_modals):.3f} - {max(long_modals):.3f}")
                logger.info(f"  Average Modal Position: {sum(long_modals)/len(long_modals):.3f}")
        
        # Direction Analysis - Tradeable Clusters
        valid_tradeable = [c for c in tradeable_clusters if c['modal_position'] is not None]
        if len(valid_tradeable) > 0:
            logger.info(f"\n💰 TRADEABLE DIRECTION ANALYSIS ({len(valid_tradeable)} clusters):")
            
            # Count signal types for tradeable
            tradeable_signal_counts = {}
            for cluster in valid_tradeable:
                signal_type = cluster['signal_type']
                tradeable_signal_counts[signal_type] = tradeable_signal_counts.get(signal_type, 0) + 1
            
            for signal_type, count in tradeable_signal_counts.items():
                percentage = count / len(valid_tradeable) * 100
                logger.info(f"  {signal_type}: {count} trades ({percentage:.1f}%)")
            
            # Tradeable long signal analysis
            tradeable_longs = [c for c in valid_tradeable if c['signal_type'] == 'LONG']
            if tradeable_longs:
                tradeable_long_strengths = [c['position_strength'] for c in tradeable_longs]
                logger.info(f"\n📈 TRADEABLE LONG SIGNALS ({len(tradeable_longs)} trades):")
                logger.info(f"  Average Position Strength: {sum(tradeable_long_strengths)/len(tradeable_long_strengths):.3f}")
                logger.info(f"  Strength Range: {min(tradeable_long_strengths):.3f} - {max(tradeable_long_strengths):.3f}")
                
                # Daily distribution of tradeable long signals
                long_dates = {}
                for cluster in tradeable_longs:
                    date = cluster['date']
                    long_dates[date] = long_dates.get(date, 0) + 1
                
                logger.info(f"  Daily Distribution: {len(long_dates)} days with long signals")
                logger.info(f"  Average Long Signals per Day: {len(tradeable_longs)/7:.1f}")
        
        # Daily breakdown with direction analysis
        clusters_by_date = {}
        for cluster in all_clusters:
            date = cluster['date']
            if date not in clusters_by_date:
                clusters_by_date[date] = []
            clusters_by_date[date].append(cluster)
        
        logger.info("\n📅 Daily Breakdown with Direction Analysis:")
        for date in sorted(clusters_by_date.keys()):
            daily_clusters = clusters_by_date[date]
            daily_tradeable = [c for c in daily_clusters if c['is_tradeable']]
            
            # Count signals by type for this day
            daily_longs = len([c for c in daily_clusters if c['signal_type'] == 'LONG'])
            daily_no_signals = len([c for c in daily_clusters if c['signal_type'] == 'NO_SIGNAL'])
            
            tradeable_longs = len([c for c in daily_tradeable if c['signal_type'] == 'LONG'])
            tradeable_no_signals = len([c for c in daily_tradeable if c['signal_type'] == 'NO_SIGNAL'])
            
            logger.info(f"  {date}: {len(daily_clusters)} clusters ({len(daily_tradeable)} tradeable)")
            logger.info(f"    All: {daily_longs} LONG, {daily_no_signals} NO_SIGNAL")
            logger.info(f"    Tradeable: {tradeable_longs} LONG, {tradeable_no_signals} NO_SIGNAL")
        
        # Parameter summary
        logger.info(f"\n⚙️  DIRECTION DETERMINATION PARAMETERS:")
        logger.info(f"  Long Threshold: ≤ {self.TIGHT_LONG_THRESHOLD}")
        logger.info(f"  Short Threshold: ≥ 0.85 (DISABLED: {self.ELIMINATE_SHORTS})")
        logger.info(f"  No-Trade Zone: {self.TIGHT_LONG_THRESHOLD} < modal_position < 0.85")
        
        logger.info("=" * 80)
    
    def print_momentum_analysis_summary(self, all_clusters: List[Dict[str, Any]], 
                                       tradeable_clusters: List[Dict[str, Any]]):
        """Print detailed summary with momentum analysis"""
        logger.info("=" * 80)
        logger.info("📊 VOLUME CLUSTER DETECTION WITH MOMENTUM ANALYSIS RESULTS")
        logger.info("=" * 80)
        
        total_clusters = len(all_clusters)
        total_tradeable = len(tradeable_clusters)
        
        logger.info(f"🎯 Total Volume Clusters Detected: {total_clusters}")
        logger.info(f"💰 Tradeable Clusters (Top-{self.TOP_N_CLUSTERS_PER_DAY}): {total_tradeable}")
        logger.info(f"📈 Trade Rate: {total_tradeable/total_clusters*100:.1f}% of clusters are tradeable")
        
        if total_clusters == 0:
            logger.info("❌ No volume clusters found in the 10-day period")
            return
        
        # Momentum Analysis - All Clusters
        valid_momentum_clusters = [c for c in all_clusters if c['momentum'] is not None]
        if len(valid_momentum_clusters) > 0:
            logger.info(f"\n📈 MOMENTUM ANALYSIS (All {len(valid_momentum_clusters)} clusters with momentum data):")
            
            momentums = [c['momentum'] for c in valid_momentum_clusters]
            positive_momentum = [m for m in momentums if m > 0]
            negative_momentum = [m for m in momentums if m < 0]
            neutral_momentum = [m for m in momentums if m == 0]
            
            logger.info(f"  Average Momentum: {sum(momentums)/len(momentums):.4f} ({sum(momentums)/len(momentums)*100:+.2f}%)")
            logger.info(f"  Positive Momentum: {len(positive_momentum)} clusters ({len(positive_momentum)/len(momentums)*100:.1f}%)")
            logger.info(f"  Negative Momentum: {len(negative_momentum)} clusters ({len(negative_momentum)/len(momentums)*100:.1f}%)")
            logger.info(f"  Neutral Momentum: {len(neutral_momentum)} clusters ({len(neutral_momentum)/len(momentums)*100:.1f}%)")
            logger.info(f"  Momentum Range: {min(momentums):.4f} to {max(momentums):.4f}")
            
            if positive_momentum:
                logger.info(f"  Avg Positive Momentum: {sum(positive_momentum)/len(positive_momentum):.4f} ({sum(positive_momentum)/len(positive_momentum)*100:+.2f}%)")
            if negative_momentum:
                logger.info(f"  Avg Negative Momentum: {sum(negative_momentum)/len(negative_momentum):.4f} ({sum(negative_momentum)/len(negative_momentum)*100:+.2f}%)")
        
        # Momentum Analysis - Tradeable Clusters
        valid_tradeable_momentum = [c for c in tradeable_clusters if c['momentum'] is not None]
        if len(valid_tradeable_momentum) > 0:
            logger.info(f"\n💰 TRADEABLE MOMENTUM ANALYSIS ({len(valid_tradeable_momentum)} clusters):")
            
            tradeable_momentums = [c['momentum'] for c in valid_tradeable_momentum]
            tradeable_positive = [m for m in tradeable_momentums if m > 0]
            tradeable_negative = [m for m in tradeable_momentums if m < 0]
            
            logger.info(f"  Average Momentum: {sum(tradeable_momentums)/len(tradeable_momentums):.4f} ({sum(tradeable_momentums)/len(tradeable_momentums)*100:+.2f}%)")
            logger.info(f"  Positive Momentum: {len(tradeable_positive)} trades ({len(tradeable_positive)/len(tradeable_momentums)*100:.1f}%)")
            logger.info(f"  Negative Momentum: {len(tradeable_negative)} trades ({len(tradeable_negative)/len(tradeable_momentums)*100:.1f}%)")
            logger.info(f"  Momentum Range: {min(tradeable_momentums):.4f} to {max(tradeable_momentums):.4f}")
        
        # Direction + Momentum Analysis for LONG signals
        long_clusters = [c for c in valid_momentum_clusters if c['signal_type'] == 'LONG']
        if long_clusters:
            logger.info(f"\n📈 LONG SIGNAL MOMENTUM ANALYSIS ({len(long_clusters)} signals):")
            long_momentums = [c['momentum'] for c in long_clusters]
            long_positive_momentum = [m for m in long_momentums if m > 0]
            long_negative_momentum = [m for m in long_momentums if m < 0]
            
            logger.info(f"  Average Momentum: {sum(long_momentums)/len(long_momentums):.4f} ({sum(long_momentums)/len(long_momentums)*100:+.2f}%)")
            logger.info(f"  Positive Momentum: {len(long_positive_momentum)} ({len(long_positive_momentum)/len(long_momentums)*100:.1f}%)")
            logger.info(f"  Negative Momentum: {len(long_negative_momentum)} ({len(long_negative_momentum)/len(long_momentums)*100:.1f}%)")
            
            # Tradeable long signals with momentum
            tradeable_longs = [c for c in long_clusters if c['is_tradeable']]
            if tradeable_longs:
                tradeable_long_momentums = [c['momentum'] for c in tradeable_longs]
                logger.info(f"  TRADEABLE LONG SIGNALS: {len(tradeable_longs)} trades")
                logger.info(f"    Average Momentum: {sum(tradeable_long_momentums)/len(tradeable_long_momentums):.4f} ({sum(tradeable_long_momentums)/len(tradeable_long_momentums)*100:+.2f}%)")
        
        # Daily breakdown with momentum
        clusters_by_date = {}
        for cluster in all_clusters:
            date = cluster['date']
            if date not in clusters_by_date:
                clusters_by_date[date] = []
            clusters_by_date[date].append(cluster)
        
        logger.info("\n📅 Daily Breakdown with Momentum Analysis:")
        for date in sorted(clusters_by_date.keys()):
            daily_clusters = clusters_by_date[date]
            daily_tradeable = [c for c in daily_clusters if c['is_tradeable']]
            
            # Calculate daily momentum stats
            daily_valid_momentum = [c for c in daily_clusters if c['momentum'] is not None]
            if daily_valid_momentum:
                daily_momentums = [c['momentum'] for c in daily_valid_momentum]
                avg_momentum = sum(daily_momentums) / len(daily_momentums)
                positive_count = len([m for m in daily_momentums if m > 0])
                
                logger.info(f"  {date}: {len(daily_clusters)} clusters ({len(daily_tradeable)} tradeable)")
                logger.info(f"    Avg Momentum: {avg_momentum:.4f} ({avg_momentum*100:+.2f}%), "
                           f"Positive: {positive_count}/{len(daily_momentums)} ({positive_count/len(daily_momentums)*100:.0f}%)")
        
        # Parameter summary
        logger.info(f"\n⚙️  MOMENTUM ANALYSIS PARAMETERS:")
        logger.info(f"  Lookback Period: 30 minutes before cluster start")
        logger.info(f"  Calculation: (end_price - start_price) / start_price")
        logger.info(f"  Combined with Modal Position Analysis")
        
        logger.info("=" * 80)
    
    def print_signal_strength_analysis_summary(self, all_clusters: List[Dict[str, Any]], 
                                             tradeable_clusters: List[Dict[str, Any]]):
        """Print detailed summary with signal strength analysis"""
        logger.info("=" * 90)
        logger.info("📊 VOLUME CLUSTER DETECTION WITH SIGNAL STRENGTH ANALYSIS RESULTS")
        logger.info("=" * 90)
        
        total_clusters = len(all_clusters)
        total_tradeable = len(tradeable_clusters)
        
        logger.info(f"🎯 Total Volume Clusters Detected: {total_clusters}")
        logger.info(f"💰 Tradeable Clusters (Top-{self.TOP_N_CLUSTERS_PER_DAY}): {total_tradeable}")
        logger.info(f"📈 Trade Rate: {total_tradeable/total_clusters*100:.1f}% of clusters are tradeable")
        
        if total_clusters == 0:
            logger.info("❌ No volume clusters found in the 10-day period")
            return
        
        # Signal Strength Analysis - All Clusters
        valid_strength_clusters = [c for c in all_clusters if c.get('signal_strength') is not None]
        if len(valid_strength_clusters) > 0:
            logger.info(f"\n💪 SIGNAL STRENGTH ANALYSIS (All {len(valid_strength_clusters)} clusters with strength data):")
            
            strengths = [c['signal_strength'] for c in valid_strength_clusters]
            passing_threshold = [c for c in valid_strength_clusters if c['meets_strength_threshold']]
            
            logger.info(f"  Average Signal Strength: {sum(strengths)/len(strengths):.3f}")
            logger.info(f"  Signal Strength Range: {min(strengths):.3f} - {max(strengths):.3f}")
            logger.info(f"  Clusters Meeting Threshold (≥{self.MIN_SIGNAL_STRENGTH}): {len(passing_threshold)} ({len(passing_threshold)/len(strengths)*100:.1f}%)")
            
            # Component analysis
            position_strengths = [c['signal_position_strength'] for c in valid_strength_clusters]
            volume_strengths = [c['signal_volume_strength'] for c in valid_strength_clusters]
            momentum_strengths = [c['signal_momentum_strength'] for c in valid_strength_clusters]
            
            logger.info(f"\n  📊 STRENGTH COMPONENTS BREAKDOWN:")
            logger.info(f"    Position Strength (50%): Avg={sum(position_strengths)/len(position_strengths):.3f}, Range={min(position_strengths):.3f}-{max(position_strengths):.3f}")
            logger.info(f"    Volume Strength (30%): Avg={sum(volume_strengths)/len(volume_strengths):.3f}, Range={min(volume_strengths):.3f}-{max(volume_strengths):.3f}")
            logger.info(f"    Momentum Strength (20%): Avg={sum(momentum_strengths)/len(momentum_strengths):.3f}, Range={min(momentum_strengths):.3f}-{max(momentum_strengths):.3f}")
        
        # Signal Strength Analysis - Tradeable Clusters
        valid_tradeable_strength = [c for c in tradeable_clusters if c.get('signal_strength') is not None]
        if len(valid_tradeable_strength) > 0:
            logger.info(f"\n💰 TRADEABLE SIGNAL STRENGTH ANALYSIS ({len(valid_tradeable_strength)} clusters):")
            
            tradeable_strengths = [c['signal_strength'] for c in valid_tradeable_strength]
            tradeable_passing = [c for c in valid_tradeable_strength if c['meets_strength_threshold']]
            
            logger.info(f"  Average Signal Strength: {sum(tradeable_strengths)/len(tradeable_strengths):.3f}")
            logger.info(f"  Signal Strength Range: {min(tradeable_strengths):.3f} - {max(tradeable_strengths):.3f}")
            logger.info(f"  Tradeable Meeting Threshold: {len(tradeable_passing)} ({len(tradeable_passing)/len(tradeable_strengths)*100:.1f}%)")
            
            # Tradeable component analysis
            tradeable_pos_strengths = [c['signal_position_strength'] for c in valid_tradeable_strength]
            tradeable_vol_strengths = [c['signal_volume_strength'] for c in valid_tradeable_strength]
            tradeable_mom_strengths = [c['signal_momentum_strength'] for c in valid_tradeable_strength]
            
            logger.info(f"  📊 TRADEABLE STRENGTH COMPONENTS:")
            logger.info(f"    Position Strength: Avg={sum(tradeable_pos_strengths)/len(tradeable_pos_strengths):.3f}")
            logger.info(f"    Volume Strength: Avg={sum(tradeable_vol_strengths)/len(tradeable_vol_strengths):.3f}")
            logger.info(f"    Momentum Strength: Avg={sum(tradeable_mom_strengths)/len(tradeable_mom_strengths):.3f}")
        
        # High-Quality Signal Analysis (LONG + Meets Strength Threshold)
        high_quality_signals = [c for c in all_clusters if 
                               c.get('signal_type') == 'LONG' and 
                               c.get('meets_strength_threshold') == True]
        
        if high_quality_signals:
            logger.info(f"\n🎯 HIGH-QUALITY SIGNALS ANALYSIS (LONG + Strength ≥{self.MIN_SIGNAL_STRENGTH}):")
            logger.info(f"  Total High-Quality Signals: {len(high_quality_signals)} out of {len(all_clusters)} clusters ({len(high_quality_signals)/len(all_clusters)*100:.1f}%)")
            
            # Tradeable high-quality signals
            tradeable_high_quality = [c for c in high_quality_signals if c['is_tradeable']]
            logger.info(f"  Tradeable High-Quality Signals: {len(tradeable_high_quality)} ({len(tradeable_high_quality)/len(high_quality_signals)*100:.1f}% of high-quality)")
            
            if tradeable_high_quality:
                hq_strengths = [c['signal_strength'] for c in tradeable_high_quality]
                hq_modal_positions = [c['modal_position'] for c in tradeable_high_quality]
                hq_momentums = [c['momentum'] for c in tradeable_high_quality if c['momentum'] is not None]
                
                logger.info(f"  📊 TRADEABLE HIGH-QUALITY SIGNAL CHARACTERISTICS:")
                logger.info(f"    Average Signal Strength: {sum(hq_strengths)/len(hq_strengths):.3f}")
                logger.info(f"    Average Modal Position: {sum(hq_modal_positions)/len(hq_modal_positions):.3f}")
                if hq_momentums:
                    logger.info(f"    Average Momentum: {sum(hq_momentums)/len(hq_momentums):.4f} ({sum(hq_momentums)/len(hq_momentums)*100:+.2f}%)")
                
                # Daily distribution
                hq_dates = {}
                for cluster in tradeable_high_quality:
                    date = cluster['date']
                    hq_dates[date] = hq_dates.get(date, 0) + 1
                
                logger.info(f"    Daily Distribution: {len(hq_dates)} days with high-quality signals")
                logger.info(f"    Average High-Quality Signals per Day: {len(tradeable_high_quality)/10:.1f}")
        
        # Daily breakdown with signal strength
        clusters_by_date = {}
        for cluster in all_clusters:
            date = cluster['date']
            if date not in clusters_by_date:
                clusters_by_date[date] = []
            clusters_by_date[date].append(cluster)
        
        logger.info("\n📅 Daily Breakdown with Signal Strength Analysis:")
        for date in sorted(clusters_by_date.keys()):
            daily_clusters = clusters_by_date[date]
            daily_tradeable = [c for c in daily_clusters if c['is_tradeable']]
            
            # Count high-quality signals for this day
            daily_hq = len([c for c in daily_clusters if c.get('signal_type') == 'LONG' and c.get('meets_strength_threshold') == True])
            tradeable_hq = len([c for c in daily_tradeable if c.get('signal_type') == 'LONG' and c.get('meets_strength_threshold') == True])
            
            # Calculate daily strength stats
            daily_valid_strength = [c for c in daily_clusters if c.get('signal_strength') is not None]
            if daily_valid_strength:
                daily_strengths = [c['signal_strength'] for c in daily_valid_strength]
                avg_strength = sum(daily_strengths) / len(daily_strengths)
                passing_count = len([c for c in daily_valid_strength if c['meets_strength_threshold']])
                
                logger.info(f"  {date}: {len(daily_clusters)} clusters ({len(daily_tradeable)} tradeable)")
                logger.info(f"    Avg Strength: {avg_strength:.3f}, Passing Threshold: {passing_count}/{len(daily_valid_strength)} ({passing_count/len(daily_valid_strength)*100:.0f}%)")
                logger.info(f"    High-Quality Signals: {daily_hq} total, {tradeable_hq} tradeable")
        
        # Parameter summary
        logger.info(f"\n⚙️  SIGNAL STRENGTH PARAMETERS:")
        logger.info(f"  Position Strength Weight: 70% (1.0 - modal_position / {self.TIGHT_LONG_THRESHOLD})")
        logger.info(f"  Volume Strength Weight: 30% (min(volume_ratio / 150.0, 1.0))")
        logger.info(f"  Momentum Strength Weight: 0% (removed - was blocking trades)")
        logger.info(f"  Signal Strength Threshold: ≥ {self.MIN_SIGNAL_STRENGTH}")
        logger.info(f"  Combined Formula: 0.7×Position + 0.3×Volume")
        
        logger.info("=" * 90)
    
    def print_retest_analysis_summary(self, all_clusters: List[Dict[str, Any]], 
                                    tradeable_clusters: List[Dict[str, Any]]):
        """Print detailed summary with retest analysis"""
        logger.info("=" * 100)
        logger.info("📊 VOLUME CLUSTER DETECTION WITH RETEST ANALYSIS RESULTS")
        logger.info("=" * 100)
        
        total_clusters = len(all_clusters)
        total_tradeable = len(tradeable_clusters)
        
        logger.info(f"🎯 Total Volume Clusters Detected: {total_clusters}")
        logger.info(f"💰 Tradeable Clusters (Top-{self.TOP_N_CLUSTERS_PER_DAY}): {total_tradeable}")
        logger.info(f"📈 Trade Rate: {total_tradeable/total_clusters*100:.1f}% of clusters are tradeable")
        
        if total_clusters == 0:
            logger.info("❌ No volume clusters found in the 10-day period")
            return
        
        # Retest Analysis - All Clusters
        valid_retest_clusters = [c for c in all_clusters if c.get('retest_confirmed') is not None]
        if len(valid_retest_clusters) > 0:
            logger.info(f"\n🔄 RETEST ANALYSIS (All {len(valid_retest_clusters)} clusters with retest data):")
            
            retest_confirmed = [c for c in valid_retest_clusters if c['retest_confirmed']]
            retest_failed = [c for c in valid_retest_clusters if not c['retest_confirmed']]
            
            logger.info(f"  Retest Confirmed: {len(retest_confirmed)} clusters ({len(retest_confirmed)/len(valid_retest_clusters)*100:.1f}%)")
            logger.info(f"  Retest Failed: {len(retest_failed)} clusters ({len(retest_failed)/len(valid_retest_clusters)*100:.1f}%)")
            
            if retest_confirmed:
                retest_times = [c['time_to_retest'] for c in retest_confirmed if c['time_to_retest'] is not None]
                retest_distances = [abs(c['retest_price'] - c['modal_price']) for c in retest_confirmed 
                                  if c['retest_price'] is not None and c['modal_price'] is not None]
                
                if retest_times:
                    logger.info(f"  Average Time to Retest: {sum(retest_times)/len(retest_times):.1f} minutes")
                    logger.info(f"  Retest Time Range: {min(retest_times):.1f} - {max(retest_times):.1f} minutes")
                
                if retest_distances:
                    logger.info(f"  Average Retest Distance: {sum(retest_distances)/len(retest_distances):.2f} points")
                    logger.info(f"  Retest Distance Range: {min(retest_distances):.2f} - {max(retest_distances):.2f} points")
            
            if retest_failed:
                failed_distances = [c['min_distance'] for c in retest_failed if c['min_distance'] is not None]
                if failed_distances:
                    logger.info(f"  Failed Retest Min Distance: {sum(failed_distances)/len(failed_distances):.2f} points (avg)")
        
        # Retest Analysis - Tradeable Clusters
        valid_tradeable_retest = [c for c in tradeable_clusters if c.get('retest_confirmed') is not None]
        if len(valid_tradeable_retest) > 0:
            logger.info(f"\n💰 TRADEABLE RETEST ANALYSIS ({len(valid_tradeable_retest)} clusters):")
            
            tradeable_retest_confirmed = [c for c in valid_tradeable_retest if c['retest_confirmed']]
            tradeable_retest_failed = [c for c in valid_tradeable_retest if not c['retest_confirmed']]
            
            logger.info(f"  Tradeable Retest Confirmed: {len(tradeable_retest_confirmed)} ({len(tradeable_retest_confirmed)/len(valid_tradeable_retest)*100:.1f}%)")
            logger.info(f"  Tradeable Retest Failed: {len(tradeable_retest_failed)} ({len(tradeable_retest_failed)/len(valid_tradeable_retest)*100:.1f}%)")
            
            if tradeable_retest_confirmed:
                tradeable_times = [c['time_to_retest'] for c in tradeable_retest_confirmed if c['time_to_retest'] is not None]
                if tradeable_times:
                    logger.info(f"  Avg Tradeable Retest Time: {sum(tradeable_times)/len(tradeable_times):.1f} minutes")
        
        # High-Quality Signal Analysis (LONG + Meets Strength Threshold + Retest Confirmed)
        high_quality_signals = [c for c in all_clusters if 
                               c.get('signal_type') == 'LONG' and 
                               c.get('meets_strength_threshold') == True and
                               c.get('retest_confirmed') == True]
        
        if high_quality_signals:
            logger.info(f"\n🎯 ULTRA-HIGH-QUALITY SIGNALS (LONG + Strength ≥{self.MIN_SIGNAL_STRENGTH} + Retest):")
            logger.info(f"  Total Ultra-High-Quality Signals: {len(high_quality_signals)} out of {len(all_clusters)} clusters ({len(high_quality_signals)/len(all_clusters)*100:.1f}%)")
            
            # Tradeable ultra-high-quality signals
            tradeable_ultra_hq = [c for c in high_quality_signals if c['is_tradeable']]
            logger.info(f"  Tradeable Ultra-High-Quality Signals: {len(tradeable_ultra_hq)} ({len(tradeable_ultra_hq)/len(high_quality_signals)*100:.1f}% of ultra-high-quality)")
            
            if tradeable_ultra_hq:
                uhq_strengths = [c['signal_strength'] for c in tradeable_ultra_hq]
                uhq_modal_positions = [c['modal_position'] for c in tradeable_ultra_hq]
                uhq_retest_times = [c['time_to_retest'] for c in tradeable_ultra_hq if c['time_to_retest'] is not None]
                
                logger.info(f"  📊 TRADEABLE ULTRA-HIGH-QUALITY SIGNAL CHARACTERISTICS:")
                logger.info(f"    Average Signal Strength: {sum(uhq_strengths)/len(uhq_strengths):.3f}")
                logger.info(f"    Average Modal Position: {sum(uhq_modal_positions)/len(uhq_modal_positions):.3f}")
                if uhq_retest_times:
                    logger.info(f"    Average Retest Time: {sum(uhq_retest_times)/len(uhq_retest_times):.1f} minutes")
                
                # Daily distribution
                uhq_dates = {}
                for cluster in tradeable_ultra_hq:
                    date = cluster['date']
                    uhq_dates[date] = uhq_dates.get(date, 0) + 1
                
                logger.info(f"    Daily Distribution: {len(uhq_dates)} days with ultra-high-quality signals")
                logger.info(f"    Average Ultra-High-Quality Signals per Day: {len(tradeable_ultra_hq)/10:.1f}")
        
        # Daily breakdown with retest analysis
        clusters_by_date = {}
        for cluster in all_clusters:
            date = cluster['date']
            if date not in clusters_by_date:
                clusters_by_date[date] = []
            clusters_by_date[date].append(cluster)
        
        logger.info("\n📅 Daily Breakdown with Retest Analysis:")
        for date in sorted(clusters_by_date.keys()):
            daily_clusters = clusters_by_date[date]
            daily_tradeable = [c for c in daily_clusters if c['is_tradeable']]
            
            # Count retest confirmations for this day
            daily_retests = len([c for c in daily_clusters if c.get('retest_confirmed') == True])
            tradeable_retests = len([c for c in daily_tradeable if c.get('retest_confirmed') == True])
            
            # Count ultra-high-quality signals for this day
            daily_uhq = len([c for c in daily_clusters if c.get('signal_type') == 'LONG' and 
                           c.get('meets_strength_threshold') == True and c.get('retest_confirmed') == True])
            tradeable_uhq = len([c for c in daily_tradeable if c.get('signal_type') == 'LONG' and 
                               c.get('meets_strength_threshold') == True and c.get('retest_confirmed') == True])
            
            logger.info(f"  {date}: {len(daily_clusters)} clusters ({len(daily_tradeable)} tradeable)")
            logger.info(f"    Retests: {daily_retests} total, {tradeable_retests} tradeable")
            logger.info(f"    Ultra-HQ Signals: {daily_uhq} total, {tradeable_uhq} tradeable")
        
        # Parameter summary
        logger.info(f"\n⚙️  RETEST ANALYSIS PARAMETERS:")
        logger.info(f"  Retest Tolerance: ±{self.RETEST_TOLERANCE} points")
        logger.info(f"  Retest Timeout: {self.RETEST_TIMEOUT} minutes")
        logger.info(f"  Retest Window: Post-cluster formation")
        logger.info(f"  Ultra-HQ Criteria: LONG + Strength ≥{self.MIN_SIGNAL_STRENGTH} + Retest Confirmed")
        
        logger.info("=" * 100)
    
    async def run_cluster_test(self):
        """Main execution flow for cluster testing"""
        logger.info("🚀 Starting Volume Cluster Detection Backtest")
        logger.info("=" * 60)
        
        # Step 1: Initialize connection
        if not await self.initialize():
            logger.error("❌ Failed to initialize connector")
            return
        
        # Step 2: Identify most liquid contract
        if not await self.identify_most_liquid_contract():
            logger.error("❌ Failed to identify most liquid contract")
            return
        
        # Step 3: Fetch 10-day historical data
        bars_1min = await self.fetch_10_day_data()
        if not bars_1min:
            logger.error("❌ Failed to fetch historical data")
            return
        
        # Step 4: Aggregate to 15-minute bars
        bars_15min = self.aggregate_to_15min_bars(bars_1min)
        if not bars_15min:
            logger.error("❌ Failed to aggregate 15-minute bars")
            return
        
        # Step 5: Calculate daily thresholds
        daily_thresholds = self.calculate_daily_thresholds(bars_1min, bars_15min)
        if not daily_thresholds:
            logger.error("❌ Failed to calculate daily thresholds")
            return
        
        # Step 6: Process clusters chronologically with retest analysis
        all_clusters, tradeable_clusters = self.process_clusters_with_retest_analysis(bars_15min, bars_1min, daily_thresholds)
        
        # Step 7: Print results with retest analysis
        self.print_retest_analysis_summary(all_clusters, tradeable_clusters)


async def main():
    """Main entry point"""
    tester = VolumeClusterTester()
    await tester.run_cluster_test()


if __name__ == "__main__":
    if not DATABENTO_AVAILABLE:
        logger.error("❌ Cannot run without Databento package")
        exit(1)
    
    # Run the cluster test
    asyncio.run(main())
