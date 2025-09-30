#!/usr/bin/env python3
"""
signal_optimization_test.py - Compare Signal Generation Strategies
Tests different approaches to increase trade frequency while maintaining quality.

This script compares:
1. Current settings (baseline): TIGHT_LONG_THRESHOLD=0.15, ELIMINATE_SHORTS=True
2. Option A: Relaxed modal position (TIGHT_LONG_THRESHOLD=0.25, ELIMINATE_SHORTS=True)  
3. Option B: Enable shorts (TIGHT_LONG_THRESHOLD=0.15, ELIMINATE_SHORTS=False)

Analyzes 10-day performance to determine optimal configuration.
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


class SignalOptimizationTester:
    """Compare different signal generation strategies"""
    
    def __init__(self):
        self.api_key = None
        self.historical_client = None
        self.most_liquid_symbol = None
        
        # Base parameters (same for all tests)
        self.VOLUME_MULTIPLIER = 4.0
        self.ROLLING_WINDOW_HOURS = 2.0
        self.TOP_N_CLUSTERS_PER_DAY = 1
        self.MIN_CLUSTERS_FOR_RANKING = 2
        self.MIN_SIGNAL_STRENGTH = 0.25
        self.RETEST_TOLERANCE = 0.75
        self.RETEST_TIMEOUT = 30
        
        # Test configurations
        self.test_configs = {
            'current': {
                'name': 'Current (Baseline)',
                'TIGHT_LONG_THRESHOLD': 0.15,
                'ELIMINATE_SHORTS': True
            },
            'relaxed_modal': {
                'name': 'Option A: Relaxed Modal Position',
                'TIGHT_LONG_THRESHOLD': 0.25,
                'ELIMINATE_SHORTS': True
            },
            'enable_shorts': {
                'name': 'Option B: Enable Short Signals',
                'TIGHT_LONG_THRESHOLD': 0.15,
                'ELIMINATE_SHORTS': False
            }
        }
        
        # Known ES futures contracts
        self.es_contracts = {
            'ES JUN25': 'ESM6',
            'ES SEP25': 'ESU5', 
            'ES DEC25': 'ESZ5',
            'ES MAR26': 'ESH6'
        }
        
    async def initialize(self) -> bool:
        """Initialize Databento client with API credentials"""
        logger.info("🔌 Initializing Databento client for signal optimization testing...")
        
        # Get API key from environment
        self.api_key = os.getenv('DATABENTO_API_KEY')
        if not self.api_key:
            logger.error("❌ DATABENTO_API_KEY not found in environment variables")
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
        """Identify the most liquid ES contract (same logic as cluster_test.py)"""
        logger.info("📊 Identifying most liquid ES contract...")
        
        try:
            # Set time range for last 12 hours
            current_time = datetime.now()
            end_time = current_time - timedelta(hours=6)
            start_time = end_time - timedelta(hours=12)
            
            # Fetch historical data for all ES contracts
            data = self.historical_client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["ES.FUT"],
                schema="ohlcv-1m",
                start=start_time,
                end=end_time,
                stype_in="parent",
                stype_out="instrument_id"
            )
            
            # Convert to DataFrame and filter
            df = data.to_df()
            if df.empty:
                logger.warning("⚠️ No historical data returned")
                return None
            
            # Filter out spreads and keep only known ES contracts
            df = df[~df['symbol'].str.contains('-', na=False)]
            known_symbols = list(self.es_contracts.values())
            df = df[df['symbol'].isin(known_symbols)]
            
            if df.empty:
                logger.warning("⚠️ No data remaining after filtering")
                return None
            
            # Calculate total volume by symbol
            volume_analysis = df.groupby('symbol')['volume'].sum().sort_values(ascending=False)
            most_liquid = volume_analysis.index[0]
            
            readable_name = next(
                (name for name, sym in self.es_contracts.items() if sym == most_liquid), 
                most_liquid
            )
            
            logger.info(f"🏆 Most liquid contract: {most_liquid} ({readable_name})")
            self.most_liquid_symbol = most_liquid
            return most_liquid
            
        except Exception as e:
            logger.error(f"❌ Failed to identify most liquid contract: {e}")
            return None
    
    async def fetch_10_day_data(self) -> List[Dict[str, Any]]:
        """Fetch 10 days of 1-minute OHLCV data"""
        if not self.most_liquid_symbol:
            logger.error("❌ No most liquid symbol identified")
            return []
        
        logger.info(f"📊 Fetching 10-day historical data for {self.most_liquid_symbol}...")
        
        try:
            # Set time range for last 10 days
            end_time = datetime.now() - timedelta(hours=3)
            start_time = end_time - timedelta(days=10)
            
            # Fetch data
            data = self.historical_client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["ES.FUT"],
                schema="ohlcv-1m",
                start=start_time,
                end=end_time,
                stype_in="parent",
                stype_out="instrument_id"
            )
            
            # Convert to DataFrame and filter for our symbol
            df = data.to_df()
            df = df[df['symbol'] == self.most_liquid_symbol]
            
            if df.empty:
                logger.warning(f"⚠️ No historical data for symbol {self.most_liquid_symbol}")
                return []
            
            logger.info(f"📈 Retrieved {len(df)} 1-minute bars")
            
            # Convert to handle_bar() format
            bars = []
            for idx, row in df.iterrows():
                bar = {
                    'timestamp': idx,
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
            logger.error(f"❌ Failed to fetch 10-day data: {e}")
            return []
    
    def aggregate_to_15min_bars(self, bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate 1-minute bars into 15-minute bars"""
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
        
        df_15min = df.resample('15min').agg(agg_dict).dropna()
        
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
        
        return bars_15min
    
    def calculate_daily_thresholds(self, bars_1min: List[Dict[str, Any]], 
                                   bars_15min: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate daily volume thresholds for each day"""
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
        
        return daily_thresholds
    
    def get_rolling_volume_rank(self, cluster_time, cluster_volume_ratio, past_clusters):
        """BIAS-FREE VOLUME RANKING: Only uses clusters that occurred BEFORE current cluster"""
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
    
    def calculate_modal_position(self, cluster_timestamp: datetime, bars_1min: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate modal position for a volume cluster using 14-minute price action window"""
        # Convert bars to DataFrame for easier manipulation
        df = pd.DataFrame(bars_1min)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Find the cluster start time (15 minutes before cluster end)
        cluster_start = cluster_timestamp - timedelta(minutes=15)
        
        # Get 14-minute window following cluster start
        window_end = cluster_start + timedelta(minutes=14)
        
        # Filter data to the 14-minute analysis window
        cluster_slice = df[
            (df['timestamp'] >= cluster_start) & 
            (df['timestamp'] < window_end)
        ].copy()
        
        if len(cluster_slice) == 0:
            return {
                'modal_price': None,
                'modal_position': None,
                'price_high': None,
                'price_low': None,
                'price_range': None,
                'data_points': 0,
                'error': 'No data in window'
            }
        
        # Calculate modal price (most frequently traded price level)
        # Round to ES tick size (0.25) and find mode
        cluster_slice['rounded_close'] = (cluster_slice['close'] / 0.25).round() * 0.25
        
        try:
            # Calculate mode of rounded closing prices
            modal_price = statistics.mode(cluster_slice['rounded_close'])
        except statistics.StatisticsError:
            # If no unique mode, use the most common price
            price_counts = cluster_slice['rounded_close'].value_counts()
            modal_price = price_counts.index[0]
        
        # Calculate price range for the window
        price_high = cluster_slice['high'].max()
        price_low = cluster_slice['low'].min()
        
        # Calculate modal position (where modal price sits in the range)
        price_range = price_high - price_low
        if price_range > 1e-9:  # Avoid division by zero
            modal_position = (modal_price - price_low) / price_range
        else:
            modal_position = 0.5  # Default to middle if no range
        
        return {
            'modal_price': modal_price,
            'modal_position': modal_position,
            'price_high': price_high,
            'price_low': price_low,
            'price_range': price_range,
            'data_points': len(cluster_slice),
            'error': None
        }
    
    def calculate_pre_cluster_momentum(self, cluster_timestamp: datetime, bars_1min: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate pre-cluster momentum using 30-minute lookback period"""
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
            return {
                'momentum': None,
                'start_price': None,
                'end_price': None,
                'price_change': None,
                'data_points': 0,
                'error': 'No data in momentum window'
            }
        
        # Calculate momentum: (end_price - start_price) / start_price
        start_price = momentum_slice.iloc[0]['close']
        end_price = momentum_slice.iloc[-1]['close']
        price_change = end_price - start_price
        
        if start_price > 1e-9:  # Avoid division by zero
            momentum = price_change / start_price
        else:
            momentum = 0.0
        
        return {
            'momentum': momentum,
            'start_price': start_price,
            'end_price': end_price,
            'price_change': price_change,
            'data_points': len(momentum_slice),
            'error': None
        }
    
    def determine_signal_direction(self, modal_position: float, config: Dict[str, Any]) -> Dict[str, Any]:
        """Determine trading signal direction based on modal position and config"""
        if modal_position is None:
            return {
                'direction': None,
                'position_strength': 0.0,
                'signal_type': 'NO_DATA',
                'reason': 'No modal position data available'
            }
        
        tight_long_threshold = config['TIGHT_LONG_THRESHOLD']
        eliminate_shorts = config['ELIMINATE_SHORTS']
        
        # Long Signal Criteria
        if modal_position <= tight_long_threshold:
            position_strength = 1.0 - (modal_position / tight_long_threshold)
            return {
                'direction': 'long',
                'position_strength': position_strength,
                'signal_type': 'LONG',
                'reason': f'Modal position {modal_position:.3f} <= {tight_long_threshold} (strong buying pressure)'
            }
        
        # Short Signal Criteria
        elif modal_position >= 0.85 and not eliminate_shorts:
            position_strength = (modal_position - 0.85) / 0.15
            return {
                'direction': 'short',
                'position_strength': position_strength,
                'signal_type': 'SHORT',
                'reason': f'Modal position {modal_position:.3f} >= 0.85 (selling pressure)'
            }
        
        # No-Trade Zone
        else:
            if eliminate_shorts and modal_position >= 0.85:
                reason = f'Modal position {modal_position:.3f} >= 0.85 but shorts eliminated'
            else:
                reason = f'Modal position {modal_position:.3f} in no-trade zone ({tight_long_threshold} < mp < 0.85)'
            
            return {
                'direction': None,
                'position_strength': 0.0,
                'signal_type': 'NO_SIGNAL',
                'reason': reason
            }
    
    def calculate_signal_strength(self, modal_position: float, volume_ratio: float, 
                                momentum: float, direction: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate signal strength using simplified two-component formula"""
        if modal_position is None or volume_ratio is None:
            return {
                'signal_strength': 0.0,
                'position_strength': 0.0,
                'volume_strength': 0.0,
                'momentum_strength': 0.0,
                'meets_threshold': False,
                'error': 'Missing required data for signal strength calculation'
            }
        
        tight_long_threshold = config['TIGHT_LONG_THRESHOLD']
        
        # Position Strength (70% weight)
        if direction == 'long':
            position_strength = 1.0 - (modal_position / tight_long_threshold)
        elif direction == 'short':
            position_strength = (modal_position - 0.85) / 0.15
        else:
            position_strength = 0.0
            
        position_strength = max(0.0, min(1.0, position_strength))  # Clamp to [0, 1]
        
        # Volume Strength (30% weight)
        volume_strength = min(volume_ratio / 150.0, 1.0)
        volume_strength = max(0.0, volume_strength)  # Ensure non-negative
        
        # Momentum Strength (0% weight) - removed
        momentum_strength = 0.0
        
        # Simplified Two-Component Formula
        signal_strength = (0.7 * position_strength + 0.3 * volume_strength)
        
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
        """Calculate retest confirmation for a volume cluster using post-cluster price action"""
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
        
        return {
            'retest_confirmed': retest_confirmed,
            'retest_time': retest_time,
            'retest_price': retest_price,
            'time_to_retest': time_to_retest,
            'min_distance': min_distance,
            'timeout_occurred': timeout_occurred,
            'data_points': len(retest_slice),
            'error': None
        }
    
    def process_clusters_with_config(self, bars_15min: List[Dict[str, Any]], 
                                   bars_1min: List[Dict[str, Any]],
                                   daily_thresholds: Dict[str, Dict[str, float]],
                                   config: Dict[str, Any]) -> tuple:
        """Process clusters chronologically with specific configuration"""
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
                
                # Determine signal direction based on modal position and config
                direction_analysis = self.determine_signal_direction(modal_analysis['modal_position'], config)
                
                # Calculate signal strength
                strength_analysis = self.calculate_signal_strength(
                    modal_analysis['modal_position'],
                    volume_ratio,
                    momentum_analysis['momentum'],
                    direction_analysis['direction'],
                    config
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
                
                # Add all analysis results
                cluster_bar.update({
                    'modal_price': modal_analysis['modal_price'],
                    'modal_position': modal_analysis['modal_position'],
                    'price_high': modal_analysis['price_high'],
                    'price_low': modal_analysis['price_low'],
                    'price_range': modal_analysis['price_range'],
                    'modal_data_points': modal_analysis['data_points'],
                    'modal_error': modal_analysis['error'],
                    
                    'momentum': momentum_analysis['momentum'],
                    'momentum_start_price': momentum_analysis['start_price'],
                    'momentum_end_price': momentum_analysis['end_price'],
                    'momentum_price_change': momentum_analysis['price_change'],
                    'momentum_data_points': momentum_analysis['data_points'],
                    'momentum_error': momentum_analysis['error'],
                    
                    'signal_direction': direction_analysis['direction'],
                    'position_strength': direction_analysis['position_strength'],
                    'signal_type': direction_analysis['signal_type'],
                    'signal_reason': direction_analysis['reason'],
                    
                    'signal_strength': strength_analysis['signal_strength'],
                    'signal_position_strength': strength_analysis['position_strength'],
                    'signal_volume_strength': strength_analysis['volume_strength'],
                    'signal_momentum_strength': strength_analysis['momentum_strength'],
                    'meets_strength_threshold': strength_analysis['meets_threshold'],
                    'strength_error': strength_analysis['error'],
                    
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
                
                # Add to processed clusters for future ranking decisions
                processed_clusters.append({
                    'timestamp': bar['timestamp'],
                    'volume_ratio': volume_ratio
                })
        
        return all_clusters, tradeable_clusters
    
    def analyze_config_performance(self, config_name: str, all_clusters: List[Dict[str, Any]], 
                                 tradeable_clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance metrics for a specific configuration"""
        total_clusters = len(all_clusters)
        total_tradeable = len(tradeable_clusters)
        
        if total_clusters == 0:
            return {
                'config_name': config_name,
                'total_clusters': 0,
                'error': 'No clusters found'
            }
        
        # Signal direction analysis
        long_signals = [c for c in all_clusters if c.get('signal_type') == 'LONG']
        short_signals = [c for c in all_clusters if c.get('signal_type') == 'SHORT']
        no_signals = [c for c in all_clusters if c.get('signal_type') == 'NO_SIGNAL']
        
        # Tradeable signal analysis
        tradeable_long = [c for c in tradeable_clusters if c.get('signal_type') == 'LONG']
        tradeable_short = [c for c in tradeable_clusters if c.get('signal_type') == 'SHORT']
        
        # High-quality signal analysis (meets strength threshold)
        hq_signals = [c for c in all_clusters if 
                     c.get('signal_direction') is not None and 
                     c.get('meets_strength_threshold') == True]
        
        tradeable_hq = [c for c in tradeable_clusters if 
                       c.get('signal_direction') is not None and 
                       c.get('meets_strength_threshold') == True]
        
        # Ultra-high-quality signals (HQ + retest confirmed)
        ultra_hq = [c for c in all_clusters if 
                   c.get('signal_direction') is not None and 
                   c.get('meets_strength_threshold') == True and
                   c.get('retest_confirmed') == True]
        
        tradeable_ultra_hq = [c for c in tradeable_clusters if 
                             c.get('signal_direction') is not None and 
                             c.get('meets_strength_threshold') == True and
                             c.get('retest_confirmed') == True]
        
        # Retest analysis
        valid_retest = [c for c in all_clusters if c.get('retest_confirmed') is not None]
        retest_confirmed = [c for c in valid_retest if c['retest_confirmed']]
        
        # Calculate trades per day
        trades_per_day = len(tradeable_ultra_hq) / 10.0  # 10-day analysis
        
        return {
            'config_name': config_name,
            'total_clusters': total_clusters,
            'tradeable_clusters': total_tradeable,
            'trade_rate': total_tradeable / total_clusters * 100,
            
            # Signal direction breakdown
            'long_signals': len(long_signals),
            'short_signals': len(short_signals),
            'no_signals': len(no_signals),
            'long_percentage': len(long_signals) / total_clusters * 100,
            'short_percentage': len(short_signals) / total_clusters * 100,
            
            # Tradeable signal breakdown
            'tradeable_long': len(tradeable_long),
            'tradeable_short': len(tradeable_short),
            
            # High-quality signals
            'hq_signals': len(hq_signals),
            'tradeable_hq': len(tradeable_hq),
            'hq_percentage': len(hq_signals) / total_clusters * 100,
            
            # Ultra-high-quality signals (final trades)
            'ultra_hq_signals': len(ultra_hq),
            'tradeable_ultra_hq': len(tradeable_ultra_hq),
            'ultra_hq_percentage': len(ultra_hq) / total_clusters * 100,
            
            # Performance metrics
            'trades_per_day': trades_per_day,
            'trades_per_10_days': len(tradeable_ultra_hq),
            
            # Retest performance
            'retest_success_rate': len(retest_confirmed) / len(valid_retest) * 100 if valid_retest else 0,
            'retest_confirmed': len(retest_confirmed),
            'retest_failed': len(valid_retest) - len(retest_confirmed) if valid_retest else 0
        }
    
    def print_comparison_results(self, results: Dict[str, Dict[str, Any]]):
        """Print detailed comparison of all configurations"""
        logger.info("=" * 120)
        logger.info("📊 SIGNAL OPTIMIZATION COMPARISON RESULTS")
        logger.info("=" * 120)
        
        # Summary table
        logger.info("\n📋 PERFORMANCE SUMMARY:")
        logger.info(f"{'Configuration':<30} {'Trades/Day':<12} {'Total Trades':<12} {'HQ Signals':<12} {'Retest Rate':<12}")
        logger.info("-" * 78)
        
        for config_key, result in results.items():
            if 'error' in result:
                logger.info(f"{result['config_name']:<30} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
                continue
                
            logger.info(f"{result['config_name']:<30} "
                       f"{result['trades_per_day']:<12.2f} "
                       f"{result['trades_per_10_days']:<12} "
                       f"{result['hq_signals']:<12} "
                       f"{result['retest_success_rate']:<12.1f}%")
        
        # Detailed breakdown for each configuration
        for config_key, result in results.items():
            if 'error' in result:
                continue
                
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 {result['config_name'].upper()}")
            logger.info(f"{'='*60}")
            
            logger.info(f"🎯 CLUSTER ANALYSIS:")
            logger.info(f"  Total Clusters: {result['total_clusters']}")
            logger.info(f"  Tradeable Clusters: {result['tradeable_clusters']} ({result['trade_rate']:.1f}%)")
            
            logger.info(f"\n📈 SIGNAL DIRECTION BREAKDOWN:")
            logger.info(f"  LONG Signals: {result['long_signals']} ({result['long_percentage']:.1f}%)")
            logger.info(f"  SHORT Signals: {result['short_signals']} ({result['short_percentage']:.1f}%)")
            logger.info(f"  NO_SIGNAL: {result['no_signals']} ({100 - result['long_percentage'] - result['short_percentage']:.1f}%)")
            
            logger.info(f"\n💰 TRADEABLE SIGNALS:")
            logger.info(f"  Tradeable LONG: {result['tradeable_long']}")
            logger.info(f"  Tradeable SHORT: {result['tradeable_short']}")
            
            logger.info(f"\n🔥 HIGH-QUALITY SIGNALS (Strength ≥{self.MIN_SIGNAL_STRENGTH}):")
            logger.info(f"  Total HQ Signals: {result['hq_signals']} ({result['hq_percentage']:.1f}%)")
            logger.info(f"  Tradeable HQ: {result['tradeable_hq']}")
            
            logger.info(f"\n🚀 ULTRA-HIGH-QUALITY SIGNALS (HQ + Retest):")
            logger.info(f"  Total Ultra-HQ: {result['ultra_hq_signals']} ({result['ultra_hq_percentage']:.1f}%)")
            logger.info(f"  Tradeable Ultra-HQ: {result['tradeable_ultra_hq']}")
            
            logger.info(f"\n📊 PERFORMANCE METRICS:")
            logger.info(f"  Trades per Day: {result['trades_per_day']:.2f}")
            logger.info(f"  Trades per 10 Days: {result['trades_per_10_days']}")
            logger.info(f"  Retest Success Rate: {result['retest_success_rate']:.1f}%")
        
        # Recommendation
        logger.info(f"\n{'='*60}")
        logger.info("🎯 RECOMMENDATION:")
        logger.info(f"{'='*60}")
        
        # Find best performing configuration by trades per day
        best_config = max(results.values(), 
                         key=lambda x: x.get('trades_per_day', 0) if 'error' not in x else 0)
        
        if 'error' not in best_config:
            logger.info(f"🏆 WINNER: {best_config['config_name']}")
            logger.info(f"  📈 Trades per Day: {best_config['trades_per_day']:.2f}")
            logger.info(f"  🎯 Total Tradeable Ultra-HQ Signals: {best_config['tradeable_ultra_hq']}")
            logger.info(f"  ✅ Retest Success Rate: {best_config['retest_success_rate']:.1f}%")
            
            if best_config['trades_per_day'] > 0.15:  # More than 1.5 trades per 10 days
                logger.info(f"  ✅ EXCELLENT: This configuration provides good trade frequency!")
            elif best_config['trades_per_day'] > 0.05:  # More than 0.5 trades per 10 days
                logger.info(f"  ⚠️  MODERATE: This configuration provides moderate trade frequency.")
            else:
                logger.info(f"  ❌ LOW: This configuration is still quite conservative.")
                
        logger.info("=" * 120)
    
    async def run_optimization_test(self):
        """Main execution flow for optimization testing"""
        logger.info("🚀 Starting Signal Optimization Analysis")
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
        
        # Step 6: Test all configurations
        results = {}
        
        for config_key, config in self.test_configs.items():
            logger.info(f"\n🔄 Testing {config['name']}...")
            logger.info(f"   TIGHT_LONG_THRESHOLD: {config['TIGHT_LONG_THRESHOLD']}")
            logger.info(f"   ELIMINATE_SHORTS: {config['ELIMINATE_SHORTS']}")
            
            all_clusters, tradeable_clusters = self.process_clusters_with_config(
                bars_15min, bars_1min, daily_thresholds, config
            )
            
            result = self.analyze_config_performance(config['name'], all_clusters, tradeable_clusters)
            results[config_key] = result
            
            logger.info(f"   ✅ Found {result.get('trades_per_10_days', 0)} tradeable ultra-HQ signals")
        
        # Step 7: Print comparison results
        self.print_comparison_results(results)
        
        # Step 8: Return best configuration for implementation
        best_config_key = max(results.keys(), 
                             key=lambda x: results[x].get('trades_per_day', 0) if 'error' not in results[x] else 0)
        
        return {
            'best_config_key': best_config_key,
            'best_config': self.test_configs[best_config_key],
            'results': results
        }


async def main():
    """Main entry point"""
    tester = SignalOptimizationTester()
    optimization_results = await tester.run_optimization_test()
    
    if optimization_results:
        logger.info(f"\n🎯 RECOMMENDED CONFIGURATION: {optimization_results['best_config']['name']}")
        logger.info(f"   Parameters: {optimization_results['best_config']}")
        
        return optimization_results
    else:
        logger.error("❌ Optimization test failed")
        return None


if __name__ == "__main__":
    if not DATABENTO_AVAILABLE:
        logger.error("❌ Cannot run without Databento package")
        exit(1)
    
    # Run the optimization test
    results = asyncio.run(main())
