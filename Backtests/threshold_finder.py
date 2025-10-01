#!/usr/bin/env python3
"""
threshold_finder.py - Signal Strength Threshold Optimization
Analyzes 30 days of real Databento data to find optimal signal strength threshold.

This script:
1. Connects to Databento API using same logic as cluster_test.py
2. Fetches 30 days of 1-minute OHLCV data for the currently selected ES contract
3. Processes all volume clusters with signal strength analysis
4. Tests multiple threshold values (0.10 to 0.50 in 0.05 increments)
5. Analyzes trade frequency, signal quality distribution, and component contributions
6. Recommends optimal threshold based on trade frequency targets and signal quality
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from typing import Dict, Any, List
import asyncio
import statistics
import numpy as np

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


class ThresholdFinder:
    """Signal strength threshold optimization for ES futures"""
    
    def __init__(self):
        self.api_key = None
        self.historical_client = None
        self.most_liquid_symbol = None
        
        # Volume cluster parameters (matching cluster_test.py)
        self.VOLUME_MULTIPLIER = 4.0
        
        # Rolling volume ranking parameters
        self.ROLLING_WINDOW_HOURS = 2.0
        self.TOP_N_CLUSTERS_PER_DAY = 1
        self.MIN_CLUSTERS_FOR_RANKING = 2
        
        # Direction determination parameters
        self.TIGHT_LONG_THRESHOLD = 0.15
        self.ELIMINATE_SHORTS = True
        
        # Threshold testing parameters
        self.TEST_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        self.TARGET_TRADES_PER_DAY = [0.5, 1.0, 1.5, 2.0]  # Target trade frequencies
        
        # Known ES futures contracts
        self.es_contracts = {
            'ES JUN25': 'ESM6',
            'ES SEP25': 'ESU5', 
            'ES DEC25': 'ESZ5',
            'ES MAR26': 'ESH6'
        }
        
    async def initialize(self) -> bool:
        """Initialize Databento client with API credentials"""
        logger.info("🔌 Initializing Databento client for threshold optimization...")
        
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
        """Identify the most liquid ES contract using same logic as cluster_test.py"""
        logger.info("📊 Identifying most liquid ES contract...")
        
        try:
            # Set time range for last 12 hours
            current_time = datetime.now()
            end_time = current_time - timedelta(hours=6)
            start_time = end_time - timedelta(hours=12)
            
            logger.info(f"📅 Analyzing volume from {start_time} to {end_time}")
            
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
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                logger.warning("⚠️ No historical data returned")
                return None
            
            # Filter data: exclude spreads and keep only known ES contracts
            df = df[~df['symbol'].str.contains('-', na=False)]
            known_symbols = list(self.es_contracts.values())
            df = df[df['symbol'].isin(known_symbols)]
            
            if df.empty:
                logger.warning("⚠️ No data remaining after filtering")
                return None
            
            # Calculate total volume by symbol over 12 hours
            volume_analysis = df.groupby('symbol')['volume'].sum().sort_values(ascending=False)
            
            logger.info("📊 12-hour volume analysis results:")
            for symbol, total_volume in volume_analysis.items():
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
    
    async def fetch_30_day_data(self) -> List[Dict[str, Any]]:
        """Fetch 30 days of 1-minute OHLCV data for the selected contract"""
        if not self.most_liquid_symbol:
            logger.error("❌ No most liquid symbol identified")
            return []
        
        logger.info(f"📊 Fetching 30-day historical data for {self.most_liquid_symbol}...")
        
        try:
            # Set time range for last 30 days
            end_time = datetime.now() - timedelta(hours=3)
            start_time = end_time - timedelta(days=30)
            
            logger.info(f"📅 Fetching data from {start_time} to {end_time}")
            
            # Fetch with parent symbol and filter
            data = self.historical_client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["ES.FUT"],
                schema="ohlcv-1m",
                start=start_time,
                end=end_time,
                stype_in="parent",
                stype_out="instrument_id"
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                logger.warning("⚠️ No historical data returned for 30-day period")
                return []
            
            # Filter for our specific symbol
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
            logger.error(f"❌ Failed to fetch 30-day data: {e}")
            return []
    
    def aggregate_to_15min_bars(self, bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate 1-minute bars into 15-minute bars"""
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
        
        df_15min = df.resample('15min').agg(agg_dict)
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
    
    def calculate_daily_thresholds(self, bars_1min: List[Dict[str, Any]], 
                                   bars_15min: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate daily volume thresholds for each day"""
        logger.info("📊 Calculating daily volume thresholds...")
        
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
        
        logger.info(f"✅ Calculated thresholds for {len(daily_thresholds)} days")
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
            return 1
        
        # Sort by volume ratio (descending) and find current cluster's rank
        sorted_clusters = sorted(all_clusters, key=lambda x: x['volume_ratio'], reverse=True)
        
        for rank, cluster in enumerate(sorted_clusters, 1):
            if cluster['timestamp'] == cluster_time:
                return rank
        
        return len(sorted_clusters)
    
    def calculate_modal_position(self, cluster_timestamp: datetime, 
                                bars_1min: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        cluster_slice['rounded_close'] = (cluster_slice['close'] / 0.25).round() * 0.25
        
        try:
            modal_price = statistics.mode(cluster_slice['rounded_close'])
        except statistics.StatisticsError:
            price_counts = cluster_slice['rounded_close'].value_counts()
            modal_price = price_counts.index[0]
        
        # Calculate price range for the window
        price_high = cluster_slice['high'].max()
        price_low = cluster_slice['low'].min()
        
        # Calculate modal position
        price_range = price_high - price_low
        if price_range > 1e-9:
            modal_position = (modal_price - price_low) / price_range
        else:
            modal_position = 0.5
        
        return {
            'modal_price': modal_price,
            'modal_position': modal_position,
            'price_high': price_high,
            'price_low': price_low,
            'price_range': price_range,
            'data_points': len(cluster_slice),
            'error': None
        }
    
    def calculate_pre_cluster_momentum(self, cluster_timestamp: datetime, 
                                      bars_1min: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        
        if start_price > 1e-9:
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

    def determine_signal_direction(self, modal_position: float) -> Dict[str, Any]:
        """Determine trading signal direction based on modal position"""
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
        """Calculate signal strength using three-component formula"""
        if modal_position is None or volume_ratio is None or momentum is None:
            return {
                'signal_strength': 0.0,
                'position_strength': 0.0,
                'volume_strength': 0.0,
                'momentum_strength': 0.0,
                'error': 'Missing required data for signal strength calculation'
            }
        
        # Position Strength (50% weight)
        position_strength = 1.0 - (modal_position / self.TIGHT_LONG_THRESHOLD)
        position_strength = max(0.0, min(1.0, position_strength))
        
        # Volume Strength (30% weight) - capped at 150x volume ratio
        volume_strength = min(volume_ratio / 150.0, 1.0)
        volume_strength = max(0.0, volume_strength)
        
        # Momentum Strength (20% weight)
        if direction == "long":
            momentum_strength = max(0, momentum * 8)
        elif direction == "short":
            momentum_strength = max(0, -momentum * 8)
        else:
            momentum_strength = 0.0
        
        momentum_strength = min(momentum_strength, 1.0)
        
        # Three-Component Formula
        signal_strength = (0.5 * position_strength + 
                          0.3 * volume_strength + 
                          0.2 * momentum_strength)
        
        return {
            'signal_strength': signal_strength,
            'position_strength': position_strength,
            'volume_strength': volume_strength,
            'momentum_strength': momentum_strength,
            'error': None
        }
    
    def process_clusters_with_signal_strength_analysis(self, bars_15min: List[Dict[str, Any]], 
                                                      bars_1min: List[Dict[str, Any]],
                                                      daily_thresholds: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Process clusters chronologically with complete signal strength analysis"""
        logger.info("🔍 Processing clusters with signal strength analysis for threshold optimization...")
        
        all_clusters = []
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
                
                # Get rolling volume rank
                volume_rank = self.get_rolling_volume_rank(
                    bar['timestamp'], volume_ratio, processed_clusters
                )
                
                # Calculate modal position analysis
                modal_analysis = self.calculate_modal_position(bar['timestamp'], bars_1min)
                
                # Calculate pre-cluster momentum analysis
                momentum_analysis = self.calculate_pre_cluster_momentum(bar['timestamp'], bars_1min)
                
                # Determine signal direction
                direction_analysis = self.determine_signal_direction(modal_analysis['modal_position'])
                
                # Calculate signal strength
                strength_analysis = self.calculate_signal_strength(
                    modal_analysis['modal_position'],
                    volume_ratio,
                    momentum_analysis['momentum'],
                    direction_analysis['direction']
                )
                
                # Create comprehensive cluster data
                cluster_bar = bar.copy()
                cluster_bar.update({
                    'volume_ratio': volume_ratio,
                    'daily_avg_1min_volume': daily_avg_1min_volume,
                    'volume_rank': volume_rank,
                    'date': date_str,
                    'is_tradeable': volume_rank <= self.TOP_N_CLUSTERS_PER_DAY,
                    
                    # Modal analysis
                    'modal_price': modal_analysis['modal_price'],
                    'modal_position': modal_analysis['modal_position'],
                    'price_high': modal_analysis['price_high'],
                    'price_low': modal_analysis['price_low'],
                    'price_range': modal_analysis['price_range'],
                    'modal_data_points': modal_analysis['data_points'],
                    'modal_error': modal_analysis['error'],
                    
                    # Momentum analysis
                    'momentum': momentum_analysis['momentum'],
                    'momentum_start_price': momentum_analysis['start_price'],
                    'momentum_end_price': momentum_analysis['end_price'],
                    'momentum_price_change': momentum_analysis['price_change'],
                    'momentum_data_points': momentum_analysis['data_points'],
                    'momentum_error': momentum_analysis['error'],
                    
                    # Direction analysis
                    'signal_direction': direction_analysis['direction'],
                    'position_strength': direction_analysis['position_strength'],
                    'signal_type': direction_analysis['signal_type'],
                    'signal_reason': direction_analysis['reason'],
                    
                    # Signal strength analysis
                    'signal_strength': strength_analysis['signal_strength'],
                    'signal_position_strength': strength_analysis['position_strength'],
                    'signal_volume_strength': strength_analysis['volume_strength'],
                    'signal_momentum_strength': strength_analysis['momentum_strength'],
                    'strength_error': strength_analysis['error']
                })
                
                all_clusters.append(cluster_bar)
                
                # Add to processed clusters for future ranking decisions
                processed_clusters.append({
                    'timestamp': bar['timestamp'],
                    'volume_ratio': volume_ratio
                })
        
        logger.info(f"✅ Processed {len(all_clusters)} clusters with signal strength analysis")
        return all_clusters
    
    def analyze_threshold_performance(self, clusters: List[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
        """Analyze performance metrics for different threshold values"""
        logger.info("📊 Analyzing threshold performance across multiple values...")
        
        results = {}
        
        # Filter to tradeable clusters with valid signal strength
        tradeable_clusters = [c for c in clusters if 
                             c['is_tradeable'] and 
                             c['signal_strength'] is not None and
                             c['signal_direction'] is not None]
        
        total_tradeable = len(tradeable_clusters)
        total_days = len(set(c['date'] for c in clusters))
        
        logger.info(f"📈 Analyzing {total_tradeable} tradeable clusters over {total_days} days")
        
        for threshold in self.TEST_THRESHOLDS:
            # Count signals meeting this threshold
            qualifying_signals = [c for c in tradeable_clusters if c['signal_strength'] >= threshold]
            
            # Count by signal type
            long_signals = [c for c in qualifying_signals if c['signal_type'] == 'LONG']
            no_signals = [c for c in qualifying_signals if c['signal_type'] == 'NO_SIGNAL']
            
            # Calculate metrics
            total_qualifying = len(qualifying_signals)
            trades_per_day = total_qualifying / total_days if total_days > 0 else 0
            pass_rate = total_qualifying / total_tradeable * 100 if total_tradeable > 0 else 0
            
            # Signal strength statistics for qualifying signals
            if qualifying_signals:
                strengths = [c['signal_strength'] for c in qualifying_signals]
                avg_strength = sum(strengths) / len(strengths)
                min_strength = min(strengths)
                max_strength = max(strengths)
                
                # Component analysis
                pos_strengths = [c['signal_position_strength'] for c in qualifying_signals]
                vol_strengths = [c['signal_volume_strength'] for c in qualifying_signals]
                mom_strengths = [c['signal_momentum_strength'] for c in qualifying_signals]
                
                avg_pos_strength = sum(pos_strengths) / len(pos_strengths)
                avg_vol_strength = sum(vol_strengths) / len(vol_strengths)
                avg_mom_strength = sum(mom_strengths) / len(mom_strengths)
            else:
                avg_strength = min_strength = max_strength = 0.0
                avg_pos_strength = avg_vol_strength = avg_mom_strength = 0.0
            
            results[threshold] = {
                'threshold': threshold,
                'total_qualifying': total_qualifying,
                'long_signals': len(long_signals),
                'no_signals': len(no_signals),
                'trades_per_day': trades_per_day,
                'pass_rate': pass_rate,
                'avg_strength': avg_strength,
                'min_strength': min_strength,
                'max_strength': max_strength,
                'avg_pos_strength': avg_pos_strength,
                'avg_vol_strength': avg_vol_strength,
                'avg_mom_strength': avg_mom_strength
            }
        
        return results
    
    def find_optimal_thresholds(self, threshold_results: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
        """Find optimal thresholds based on different trade frequency targets"""
        logger.info("🎯 Finding optimal thresholds for different trade frequency targets...")
        
        recommendations = {}
        
        for target_trades_per_day in self.TARGET_TRADES_PER_DAY:
            best_threshold = None
            best_diff = float('inf')
            
            for threshold, results in threshold_results.items():
                diff = abs(results['trades_per_day'] - target_trades_per_day)
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = threshold
            
            if best_threshold is not None:
                recommendations[f'{target_trades_per_day}_trades_per_day'] = {
                    'target_trades_per_day': target_trades_per_day,
                    'recommended_threshold': best_threshold,
                    'actual_trades_per_day': threshold_results[best_threshold]['trades_per_day'],
                    'pass_rate': threshold_results[best_threshold]['pass_rate'],
                    'avg_strength': threshold_results[best_threshold]['avg_strength'],
                    'long_signals': threshold_results[best_threshold]['long_signals']
                }
        
        return recommendations
    
    def print_threshold_analysis_results(self, threshold_results: Dict[float, Dict[str, Any]], 
                                        recommendations: Dict[str, Any]):
        """Print comprehensive threshold analysis results"""
        logger.info("=" * 100)
        logger.info("📊 SIGNAL STRENGTH THRESHOLD OPTIMIZATION RESULTS (30-Day Analysis)")
        logger.info("=" * 100)
        
        # Overall statistics
        total_clusters = sum(results['total_qualifying'] for results in threshold_results.values() if results['threshold'] == min(self.TEST_THRESHOLDS))
        
        logger.info(f"📈 Analysis Overview:")
        logger.info(f"  Test Period: 30 days")
        logger.info(f"  Thresholds Tested: {', '.join(f'{t:.2f}' for t in self.TEST_THRESHOLDS)}")
        logger.info(f"  Total Tradeable Clusters: {total_clusters}")
        
        # Detailed threshold performance
        logger.info(f"\n📊 THRESHOLD PERFORMANCE BREAKDOWN:")
        logger.info(f"{'Threshold':<10} {'Qualifying':<11} {'Long Sig':<9} {'Trades/Day':<11} {'Pass Rate':<10} {'Avg Strength':<12} {'Pos/Vol/Mom':<15}")
        logger.info("-" * 90)
        
        for threshold in self.TEST_THRESHOLDS:
            results = threshold_results[threshold]
            components = f"{results['avg_pos_strength']:.2f}/{results['avg_vol_strength']:.2f}/{results['avg_mom_strength']:.2f}"
            
            logger.info(f"{threshold:<10.2f} {results['total_qualifying']:<11} {results['long_signals']:<9} "
                       f"{results['trades_per_day']:<11.2f} {results['pass_rate']:<10.1f}% "
                       f"{results['avg_strength']:<12.3f} {components:<15}")
        
        # Recommendations
        logger.info(f"\n🎯 THRESHOLD RECOMMENDATIONS:")
        for target, rec in recommendations.items():
            logger.info(f"\n  Target: {rec['target_trades_per_day']} trades/day")
            logger.info(f"    Recommended Threshold: {rec['recommended_threshold']:.2f}")
            logger.info(f"    Actual Trades/Day: {rec['actual_trades_per_day']:.2f}")
            logger.info(f"    Pass Rate: {rec['pass_rate']:.1f}%")
            logger.info(f"    Average Signal Strength: {rec['avg_strength']:.3f}")
            logger.info(f"    Long Signals: {rec['long_signals']}")
        
        # Key insights
        logger.info(f"\n💡 KEY INSIGHTS:")
        
        # Find threshold with best balance (around 1 trade per day)
        best_balance = recommendations.get('1.0_trades_per_day')
        if best_balance:
            logger.info(f"  • For balanced trading (1 trade/day): Use threshold {best_balance['recommended_threshold']:.2f}")
        
        # Find most permissive threshold with reasonable quality
        min_threshold_results = threshold_results[min(self.TEST_THRESHOLDS)]
        max_threshold_results = threshold_results[max(self.TEST_THRESHOLDS)]
        
        logger.info(f"  • Most permissive ({min(self.TEST_THRESHOLDS):.2f}): {min_threshold_results['trades_per_day']:.2f} trades/day, {min_threshold_results['pass_rate']:.1f}% pass rate")
        logger.info(f"  • Most restrictive ({max(self.TEST_THRESHOLDS):.2f}): {max_threshold_results['trades_per_day']:.2f} trades/day, {max_threshold_results['pass_rate']:.1f}% pass rate")
        
        # Component analysis insights
        avg_pos_across_thresholds = np.mean([r['avg_pos_strength'] for r in threshold_results.values() if r['total_qualifying'] > 0])
        avg_vol_across_thresholds = np.mean([r['avg_vol_strength'] for r in threshold_results.values() if r['total_qualifying'] > 0])
        avg_mom_across_thresholds = np.mean([r['avg_mom_strength'] for r in threshold_results.values() if r['total_qualifying'] > 0])
        
        logger.info(f"\n🔧 COMPONENT ANALYSIS:")
        logger.info(f"  • Position Strength: {avg_pos_across_thresholds:.3f} avg (50% weight)")
        logger.info(f"  • Volume Strength: {avg_vol_across_thresholds:.3f} avg (30% weight)")
        logger.info(f"  • Momentum Strength: {avg_mom_across_thresholds:.3f} avg (20% weight)")
        
        if avg_pos_across_thresholds < 0.1:
            logger.info(f"  ⚠️  Position strength is very low - consider relaxing TIGHT_LONG_THRESHOLD")
        if avg_mom_across_thresholds < 0.01:
            logger.info(f"  ⚠️  Momentum strength is minimal - consider increasing momentum multiplier")
        
        logger.info("=" * 100)
    
    async def run_threshold_optimization(self):
        """Main execution flow for threshold optimization"""
        logger.info("🚀 Starting Signal Strength Threshold Optimization (30-Day Analysis)")
        logger.info("=" * 100)
        
        # Step 1: Initialize connection
        if not await self.initialize():
            logger.error("❌ Failed to initialize connector")
            return
        
        # Step 2: Identify most liquid contract
        if not await self.identify_most_liquid_contract():
            logger.error("❌ Failed to identify most liquid contract")
            return
        
        # Step 3: Fetch 30-day historical data
        bars_1min = await self.fetch_30_day_data()
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
        
        # Step 6: Process clusters with signal strength analysis
        all_clusters = self.process_clusters_with_signal_strength_analysis(bars_15min, bars_1min, daily_thresholds)
        
        # Step 7: Analyze threshold performance
        threshold_results = self.analyze_threshold_performance(all_clusters)
        
        # Step 8: Find optimal thresholds
        recommendations = self.find_optimal_thresholds(threshold_results)
        
        # Step 9: Print comprehensive results
        self.print_threshold_analysis_results(threshold_results, recommendations)


async def main():
    """Main entry point"""
    finder = ThresholdFinder()
    await finder.run_threshold_optimization()


if __name__ == "__main__":
    if not DATABENTO_AVAILABLE:
        logger.error("❌ Cannot run without Databento package")
        exit(1)
    
    # Run the threshold optimization
    asyncio.run(main())


