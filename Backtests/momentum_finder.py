#!/usr/bin/env python3
"""
momentum_finder.py - Momentum Multiplier Optimization
Analyzes 30 days of real Databento data to find optimal momentum multiplier for signal strength calculation.

This script:
1. Connects to Databento API using same logic as threshold_finder.py
2. Fetches 30 days of 1-minute OHLCV data for the currently selected ES contract
3. Processes all volume clusters with signal strength analysis
4. Tests multiple momentum multipliers (4x to 32x in 4x increments)
5. Analyzes impact on signal strength, trade frequency, and component balance
6. Recommends optimal momentum multiplier for balanced three-component formula
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


class MomentumFinder:
    """Momentum multiplier optimization for ES futures signal strength calculation"""
    
    def __init__(self):
        self.api_key = None
        self.historical_client = None
        self.most_liquid_symbol = None
        
        # Volume cluster parameters (matching production)
        self.VOLUME_MULTIPLIER = 4.0
        self.ROLLING_WINDOW_HOURS = 2.0
        self.TOP_N_CLUSTERS_PER_DAY = 1
        self.MIN_CLUSTERS_FOR_RANKING = 2
        
        # Direction determination parameters
        self.TIGHT_LONG_THRESHOLD = 0.15
        self.ELIMINATE_SHORTS = True
        
        # Signal strength parameters
        self.MIN_SIGNAL_STRENGTH = 0.25  # Current optimized threshold
        
        # Momentum testing parameters
        self.TEST_MULTIPLIERS = [4, 8, 12, 16, 20, 24, 28, 32]  # Current is 8x
        self.CURRENT_MULTIPLIER = 8  # For comparison
        
        # Known ES futures contracts
        self.es_contracts = {
            'ES JUN25': 'ESM6',
            'ES SEP25': 'ESU5', 
            'ES DEC25': 'ESZ5',
            'ES MAR26': 'ESH6'
        }
        
    async def initialize(self) -> bool:
        """Initialize Databento client with API credentials"""
        logger.info("🔌 Initializing Databento client for momentum optimization...")
        
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
        """Identify the most liquid ES contract"""
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
        """BIAS-FREE VOLUME RANKING"""
        from datetime import timedelta
        
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
        
        if len(all_clusters) < self.MIN_CLUSTERS_FOR_RANKING:
            return 1
        
        # Sort by volume ratio and find rank
        sorted_clusters = sorted(all_clusters, key=lambda x: x['volume_ratio'], reverse=True)
        
        for rank, cluster in enumerate(sorted_clusters, 1):
            if cluster['timestamp'] == cluster_time:
                return rank
        
        return len(sorted_clusters)
    
    def calculate_modal_position(self, cluster_timestamp: datetime, 
                                bars_1min: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate modal position for a volume cluster"""
        # Convert bars to DataFrame
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
        
        # Calculate modal price
        cluster_slice['rounded_close'] = (cluster_slice['close'] / 0.25).round() * 0.25
        
        try:
            modal_price = statistics.mode(cluster_slice['rounded_close'])
        except statistics.StatisticsError:
            price_counts = cluster_slice['rounded_close'].value_counts()
            modal_price = price_counts.index[0]
        
        # Calculate price range and modal position
        price_high = cluster_slice['high'].max()
        price_low = cluster_slice['low'].min()
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
        # Convert bars to DataFrame
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
                'reason': f'Modal position {modal_position:.3f} <= {self.TIGHT_LONG_THRESHOLD}'
            }
        
        # Short Signal Criteria (Currently Disabled)
        elif modal_position >= 0.85 and not self.ELIMINATE_SHORTS:
            position_strength = (modal_position - 0.85) / 0.15
            return {
                'direction': 'short',
                'position_strength': position_strength,
                'signal_type': 'SHORT',
                'reason': f'Modal position {modal_position:.3f} >= 0.85'
            }
        
        # No-Trade Zone
        else:
            if self.ELIMINATE_SHORTS and modal_position >= 0.85:
                reason = f'Modal position {modal_position:.3f} >= 0.85 but shorts eliminated'
            else:
                reason = f'Modal position {modal_position:.3f} in no-trade zone'
            
            return {
                'direction': None,
                'position_strength': 0.0,
                'signal_type': 'NO_SIGNAL',
                'reason': reason
            }
    
    def calculate_signal_strength_with_multiplier(self, modal_position: float, volume_ratio: float, 
                                                momentum: float, direction: str, momentum_multiplier: int) -> Dict[str, Any]:
        """Calculate signal strength using specified momentum multiplier"""
        if modal_position is None or volume_ratio is None or momentum is None:
            return {
                'signal_strength': 0.0,
                'position_strength': 0.0,
                'volume_strength': 0.0,
                'momentum_strength': 0.0,
                'error': 'Missing required data'
            }
        
        # Position Strength (50% weight)
        position_strength = 1.0 - (modal_position / self.TIGHT_LONG_THRESHOLD)
        position_strength = max(0.0, min(1.0, position_strength))
        
        # Volume Strength (30% weight) - capped at 150x volume ratio
        volume_strength = min(volume_ratio / 150.0, 1.0)
        volume_strength = max(0.0, volume_strength)
        
        # Momentum Strength (20% weight) - with variable multiplier
        if direction == "long":
            momentum_strength = max(0, momentum * momentum_multiplier)
        elif direction == "short":
            momentum_strength = max(0, -momentum * momentum_multiplier)
        else:
            momentum_strength = 0.0
        
        momentum_strength = min(momentum_strength, 1.0)  # Cap at 1.0
        
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
    
    def process_clusters_for_momentum_analysis(self, bars_15min: List[Dict[str, Any]], 
                                             bars_1min: List[Dict[str, Any]],
                                             daily_thresholds: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Process clusters with comprehensive analysis for momentum optimization"""
        logger.info("🔍 Processing clusters for momentum multiplier analysis...")
        
        all_clusters = []
        processed_clusters = []
        
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
            
            # Check if this qualifies as a volume cluster
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
                
                # Create comprehensive cluster data
                cluster_bar = bar.copy()
                cluster_bar.update({
                    'volume_ratio': volume_ratio,
                    'daily_avg_1min_volume': daily_avg_1min_volume,
                    'volume_rank': volume_rank,
                    'date': date_str,
                    'is_tradeable': volume_rank <= self.TOP_N_CLUSTERS_PER_DAY,
                    
                    # Modal analysis
                    'modal_position': modal_analysis['modal_position'],
                    'modal_error': modal_analysis['error'],
                    
                    # Momentum analysis
                    'momentum': momentum_analysis['momentum'],
                    'momentum_error': momentum_analysis['error'],
                    
                    # Direction analysis
                    'signal_direction': direction_analysis['direction'],
                    'signal_type': direction_analysis['signal_type'],
                })
                
                all_clusters.append(cluster_bar)
                
                # Add to processed clusters for future ranking decisions
                processed_clusters.append({
                    'timestamp': bar['timestamp'],
                    'volume_ratio': volume_ratio
                })
        
        logger.info(f"✅ Processed {len(all_clusters)} clusters for momentum analysis")
        return all_clusters
    
    def analyze_momentum_multiplier_performance(self, clusters: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Analyze performance metrics for different momentum multipliers"""
        logger.info("📊 Analyzing momentum multiplier performance...")
        
        results = {}
        
        # Filter to tradeable clusters with valid data
        tradeable_clusters = [c for c in clusters if 
                             c['is_tradeable'] and 
                             c['modal_position'] is not None and
                             c['momentum'] is not None and
                             c['signal_direction'] is not None]
        
        total_tradeable = len(tradeable_clusters)
        total_days = len(set(c['date'] for c in clusters))
        
        logger.info(f"📈 Analyzing {total_tradeable} tradeable clusters over {total_days} days")
        
        for multiplier in self.TEST_MULTIPLIERS:
            # Calculate signal strengths with this multiplier
            qualifying_signals = []
            all_strengths = []
            all_momentum_strengths = []
            
            for cluster in tradeable_clusters:
                strength_result = self.calculate_signal_strength_with_multiplier(
                    cluster['modal_position'],
                    cluster['volume_ratio'],
                    cluster['momentum'],
                    cluster['signal_direction'],
                    multiplier
                )
                
                all_strengths.append(strength_result['signal_strength'])
                all_momentum_strengths.append(strength_result['momentum_strength'])
                
                # Check if meets threshold
                if strength_result['signal_strength'] >= self.MIN_SIGNAL_STRENGTH:
                    qualifying_signals.append({
                        **cluster,
                        **strength_result
                    })
            
            # Count by signal type
            long_signals = [c for c in qualifying_signals if c['signal_type'] == 'LONG']
            
            # Calculate metrics
            total_qualifying = len(qualifying_signals)
            trades_per_day = total_qualifying / total_days if total_days > 0 else 0
            pass_rate = total_qualifying / total_tradeable * 100 if total_tradeable > 0 else 0
            
            # Signal strength statistics
            if qualifying_signals:
                qualifying_strengths = [c['signal_strength'] for c in qualifying_signals]
                avg_strength = sum(qualifying_strengths) / len(qualifying_strengths)
                min_strength = min(qualifying_strengths)
                max_strength = max(qualifying_strengths)
                
                # Component analysis for qualifying signals
                pos_strengths = [c['position_strength'] for c in qualifying_signals]
                vol_strengths = [c['volume_strength'] for c in qualifying_signals]
                mom_strengths = [c['momentum_strength'] for c in qualifying_signals]
                
                avg_pos_strength = sum(pos_strengths) / len(pos_strengths)
                avg_vol_strength = sum(vol_strengths) / len(vol_strengths)
                avg_mom_strength = sum(mom_strengths) / len(mom_strengths)
            else:
                avg_strength = min_strength = max_strength = 0.0
                avg_pos_strength = avg_vol_strength = avg_mom_strength = 0.0
            
            # Overall momentum strength statistics (all tradeable clusters)
            overall_avg_mom_strength = sum(all_momentum_strengths) / len(all_momentum_strengths) if all_momentum_strengths else 0.0
            overall_avg_strength = sum(all_strengths) / len(all_strengths) if all_strengths else 0.0
            
            # Momentum contribution analysis
            momentum_impact = 0.2 * overall_avg_mom_strength  # 20% weight
            
            results[multiplier] = {
                'multiplier': multiplier,
                'total_qualifying': total_qualifying,
                'long_signals': len(long_signals),
                'trades_per_day': trades_per_day,
                'pass_rate': pass_rate,
                'avg_strength': avg_strength,
                'min_strength': min_strength,
                'max_strength': max_strength,
                'avg_pos_strength': avg_pos_strength,
                'avg_vol_strength': avg_vol_strength,
                'avg_mom_strength': avg_mom_strength,
                'overall_avg_mom_strength': overall_avg_mom_strength,
                'overall_avg_strength': overall_avg_strength,
                'momentum_impact': momentum_impact,
                'is_current': multiplier == self.CURRENT_MULTIPLIER
            }
        
        return results
    
    def find_optimal_momentum_multiplier(self, multiplier_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Find optimal momentum multiplier based on various criteria"""
        logger.info("🎯 Finding optimal momentum multiplier...")
        
        recommendations = {}
        
        # Criteria 1: Maximum momentum contribution (highest momentum impact)
        max_momentum_impact = max(results['momentum_impact'] for results in multiplier_results.values())
        best_momentum_impact = None
        for multiplier, results in multiplier_results.items():
            if results['momentum_impact'] == max_momentum_impact:
                best_momentum_impact = multiplier
                break
        
        # Criteria 2: Balanced performance (good momentum + decent trade frequency)
        best_balanced = None
        best_balanced_score = 0
        for multiplier, results in multiplier_results.items():
            # Balance score: momentum impact + normalized trades per day
            normalized_trades = min(results['trades_per_day'] / 1.0, 1.0)  # Cap at 1.0 trades/day
            balance_score = results['momentum_impact'] + 0.5 * normalized_trades
            
            if balance_score > best_balanced_score:
                best_balanced_score = balance_score
                best_balanced = multiplier
        
        # Criteria 3: Maximum trade frequency (highest qualifying signals)
        max_trades = max(results['trades_per_day'] for results in multiplier_results.values())
        best_trade_frequency = None
        for multiplier, results in multiplier_results.items():
            if results['trades_per_day'] == max_trades:
                best_trade_frequency = multiplier
                break
        
        # Current multiplier performance
        current_performance = multiplier_results[self.CURRENT_MULTIPLIER]
        
        recommendations = {
            'max_momentum_impact': {
                'multiplier': best_momentum_impact,
                'momentum_impact': multiplier_results[best_momentum_impact]['momentum_impact'],
                'trades_per_day': multiplier_results[best_momentum_impact]['trades_per_day'],
                'overall_avg_strength': multiplier_results[best_momentum_impact]['overall_avg_strength'],
                'reason': 'Maximizes momentum contribution to signal strength'
            },
            'balanced_performance': {
                'multiplier': best_balanced,
                'momentum_impact': multiplier_results[best_balanced]['momentum_impact'],
                'trades_per_day': multiplier_results[best_balanced]['trades_per_day'],
                'overall_avg_strength': multiplier_results[best_balanced]['overall_avg_strength'],
                'balance_score': best_balanced_score,
                'reason': 'Best balance of momentum impact and trade frequency'
            },
            'max_trade_frequency': {
                'multiplier': best_trade_frequency,
                'momentum_impact': multiplier_results[best_trade_frequency]['momentum_impact'],
                'trades_per_day': multiplier_results[best_trade_frequency]['trades_per_day'],
                'overall_avg_strength': multiplier_results[best_trade_frequency]['overall_avg_strength'],
                'reason': 'Maximizes trade frequency'
            },
            'current_performance': {
                'multiplier': self.CURRENT_MULTIPLIER,
                'momentum_impact': current_performance['momentum_impact'],
                'trades_per_day': current_performance['trades_per_day'],
                'overall_avg_strength': current_performance['overall_avg_strength'],
                'reason': 'Current production multiplier'
            }
        }
        
        return recommendations
    
    def print_momentum_analysis_results(self, multiplier_results: Dict[int, Dict[str, Any]], 
                                       recommendations: Dict[str, Any]):
        """Print comprehensive momentum multiplier analysis results"""
        logger.info("=" * 100)
        logger.info("📊 MOMENTUM MULTIPLIER OPTIMIZATION RESULTS (30-Day Analysis)")
        logger.info("=" * 100)
        
        # Overall statistics
        logger.info(f"📈 Analysis Overview:")
        logger.info(f"  Test Period: 30 days")
        logger.info(f"  Multipliers Tested: {', '.join(f'{m}x' for m in self.TEST_MULTIPLIERS)}")
        logger.info(f"  Current Multiplier: {self.CURRENT_MULTIPLIER}x")
        logger.info(f"  Signal Strength Threshold: {self.MIN_SIGNAL_STRENGTH}")
        
        # Detailed multiplier performance
        logger.info(f"\n📊 MULTIPLIER PERFORMANCE BREAKDOWN:")
        logger.info(f"{'Multiplier':<10} {'Qualifying':<11} {'Trades/Day':<11} {'Pass Rate':<10} {'Mom Impact':<11} {'Avg Strength':<12} {'Current':<8}")
        logger.info("-" * 95)
        
        for multiplier in self.TEST_MULTIPLIERS:
            results = multiplier_results[multiplier]
            current_mark = "✅ CURR" if results['is_current'] else ""
            
            logger.info(f"{multiplier:<10}x {results['total_qualifying']:<11} {results['trades_per_day']:<11.2f} "
                       f"{results['pass_rate']:<10.1f}% {results['momentum_impact']:<11.3f} "
                       f"{results['overall_avg_strength']:<12.3f} {current_mark:<8}")
        
        # Detailed component analysis
        logger.info(f"\n📊 COMPONENT STRENGTH ANALYSIS (for qualifying signals):")
        logger.info(f"{'Multiplier':<10} {'Position':<10} {'Volume':<10} {'Momentum':<10} {'Combined':<10}")
        logger.info("-" * 55)
        
        for multiplier in self.TEST_MULTIPLIERS:
            results = multiplier_results[multiplier]
            if results['total_qualifying'] > 0:
                logger.info(f"{multiplier:<10}x {results['avg_pos_strength']:<10.3f} "
                           f"{results['avg_vol_strength']:<10.3f} {results['avg_mom_strength']:<10.3f} "
                           f"{results['avg_strength']:<10.3f}")
        
        # Recommendations
        logger.info(f"\n🎯 MOMENTUM MULTIPLIER RECOMMENDATIONS:")
        
        for criteria, rec in recommendations.items():
            logger.info(f"\n  {criteria.replace('_', ' ').title()}:")
            logger.info(f"    Recommended Multiplier: {rec['multiplier']}x")
            logger.info(f"    Momentum Impact: {rec['momentum_impact']:.3f}")
            logger.info(f"    Trades/Day: {rec['trades_per_day']:.2f}")
            logger.info(f"    Overall Avg Strength: {rec['overall_avg_strength']:.3f}")
            logger.info(f"    Reason: {rec['reason']}")
        
        # Key insights and comparison with current
        current_perf = recommendations['current_performance']
        best_balanced = recommendations['balanced_performance']
        
        logger.info(f"\n💡 KEY INSIGHTS:")
        
        if best_balanced['multiplier'] != self.CURRENT_MULTIPLIER:
            improvement_mom = best_balanced['momentum_impact'] - current_perf['momentum_impact']
            improvement_trades = best_balanced['trades_per_day'] - current_perf['trades_per_day']
            
            logger.info(f"  • Recommended change: {self.CURRENT_MULTIPLIER}x → {best_balanced['multiplier']}x")
            logger.info(f"  • Momentum impact improvement: {improvement_mom:+.3f}")
            logger.info(f"  • Trade frequency change: {improvement_trades:+.2f} trades/day")
            logger.info(f"  • Overall strength improvement: {best_balanced['overall_avg_strength'] - current_perf['overall_avg_strength']:+.3f}")
        else:
            logger.info(f"  • Current multiplier ({self.CURRENT_MULTIPLIER}x) is already optimal")
        
        # Component balance analysis
        best_results = multiplier_results[best_balanced['multiplier']]
        if best_results['total_qualifying'] > 0:
            pos_weight_actual = 0.5 * best_results['avg_pos_strength']
            vol_weight_actual = 0.3 * best_results['avg_vol_strength']
            mom_weight_actual = 0.2 * best_results['avg_mom_strength']
            
            logger.info(f"\n🔧 COMPONENT BALANCE ANALYSIS (Recommended {best_balanced['multiplier']}x):")
            logger.info(f"  • Position Component: {pos_weight_actual:.3f} (50% × {best_results['avg_pos_strength']:.3f})")
            logger.info(f"  • Volume Component: {vol_weight_actual:.3f} (30% × {best_results['avg_vol_strength']:.3f})")
            logger.info(f"  • Momentum Component: {mom_weight_actual:.3f} (20% × {best_results['avg_mom_strength']:.3f})")
            
            total_contribution = pos_weight_actual + vol_weight_actual + mom_weight_actual
            logger.info(f"  • Total Signal Strength: {total_contribution:.3f}")
            
            if best_results['avg_mom_strength'] < 0.1:
                logger.info(f"  ⚠️  Momentum strength still low - consider alternative approaches")
            elif best_results['avg_mom_strength'] > 0.8:
                logger.info(f"  ⚠️  Momentum strength very high - may be over-optimized")
            else:
                logger.info(f"  ✅ Momentum strength well-balanced")
        
        logger.info("=" * 100)
    
    async def run_momentum_optimization(self):
        """Main execution flow for momentum multiplier optimization"""
        logger.info("🚀 Starting Momentum Multiplier Optimization (30-Day Analysis)")
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
        
        # Step 6: Process clusters for momentum analysis
        all_clusters = self.process_clusters_for_momentum_analysis(bars_15min, bars_1min, daily_thresholds)
        
        # Step 7: Analyze momentum multiplier performance
        multiplier_results = self.analyze_momentum_multiplier_performance(all_clusters)
        
        # Step 8: Find optimal momentum multiplier
        recommendations = self.find_optimal_momentum_multiplier(multiplier_results)
        
        # Step 9: Print comprehensive results
        self.print_momentum_analysis_results(multiplier_results, recommendations)


async def main():
    """Main entry point"""
    finder = MomentumFinder()
    await finder.run_momentum_optimization()


if __name__ == "__main__":
    if not DATABENTO_AVAILABLE:
        logger.error("❌ Cannot run without Databento package")
        exit(1)
    
    # Run the momentum optimization
    asyncio.run(main())
