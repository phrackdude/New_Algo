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
        self.TIGHT_LONG_THRESHOLD = 0.15  # Modal position threshold for long signals
        self.ELIMINATE_SHORTS = True  # Disable short signals due to market bias
        
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
    
    async def fetch_7_day_data(self) -> List[Dict[str, Any]]:
        """
        Fetch 7 days of 1-minute OHLCV data for the selected contract
        Returns data in the same format as handle_bar() expects
        """
        if not self.most_liquid_symbol:
            logger.error("❌ No most liquid symbol identified")
            return []
        
        logger.info(f"📊 Fetching 7-day historical data for {self.most_liquid_symbol}...")
        
        try:
            # Set time range for last 7 days
            end_time = datetime.now() - timedelta(hours=3)  # Larger buffer for data availability
            start_time = end_time - timedelta(days=7)
            
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
        logger.info(f"  No-Trade Zone: 0.15 < modal_position < 0.85")
        
        logger.info("=" * 80)
    
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
        
        # Step 3: Fetch 7-day historical data
        bars_1min = await self.fetch_7_day_data()
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
        
        # Step 6: Process clusters chronologically with modal analysis and direction determination
        all_clusters, tradeable_clusters = self.process_clusters_with_modal_analysis(bars_15min, bars_1min, daily_thresholds)
        
        # Step 7: Print results with direction determination analysis
        self.print_direction_analysis_summary(all_clusters, tradeable_clusters)


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
