#!/usr/bin/env python3
"""
trade_profitability_analysis.py - Analyze Profitability of Ultra-High-Quality Trades
Extracts specific trade details and calculates profitability using different exit strategies.

This script:
1. Runs the optimized cluster test to identify ultra-HQ trades
2. Extracts specific trade entry details (price, time, signal strength)
3. Analyzes post-entry price action for different exit strategies
4. Calculates profit/loss for each trade and overall performance
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


class TradeProfitabilityAnalyzer:
    """Analyze profitability of ultra-high-quality trades"""
    
    def __init__(self):
        self.api_key = None
        self.historical_client = None
        self.most_liquid_symbol = None
        
        # Base parameters (same as optimized cluster_test.py)
        self.VOLUME_MULTIPLIER = 4.0
        self.ROLLING_WINDOW_HOURS = 2.0
        self.TOP_N_CLUSTERS_PER_DAY = 1
        self.MIN_CLUSTERS_FOR_RANKING = 2
        self.TIGHT_LONG_THRESHOLD = 0.25  # Optimized threshold
        self.ELIMINATE_SHORTS = True
        self.MIN_SIGNAL_STRENGTH = 0.25
        self.RETEST_TOLERANCE = 0.75
        self.RETEST_TIMEOUT = 30
        
        # Exit strategy parameters for testing
        self.exit_strategies = {
            'quick_profit': {'target': 2.0, 'stop_loss': 1.5, 'time_limit': 60},  # 2 point target, 1.5 stop, 1 hour
            'moderate': {'target': 4.0, 'stop_loss': 2.0, 'time_limit': 120},    # 4 point target, 2 stop, 2 hours
            'patient': {'target': 6.0, 'stop_loss': 3.0, 'time_limit': 240},     # 6 point target, 3 stop, 4 hours
            'eod_close': {'target': None, 'stop_loss': 3.0, 'time_limit': 480},  # End of day close, 3 stop, 8 hours
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
        logger.info("🔌 Initializing Databento client for trade profitability analysis...")
        
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
    
    def find_ultra_hq_trades(self, bars_15min: List[Dict[str, Any]], 
                           bars_1min: List[Dict[str, Any]],
                           daily_thresholds: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Find ultra-high-quality trades (same logic as cluster_test.py but focused on extraction)"""
        ultra_hq_trades = []
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
            
            # Check if this qualifies as a volume cluster (4x threshold)
            if volume_ratio >= self.VOLUME_MULTIPLIER:
                
                # Get rolling volume rank
                volume_rank = self.get_rolling_volume_rank(
                    bar['timestamp'], volume_ratio, processed_clusters
                )
                
                # Check if tradeable (top-1)
                if volume_rank <= self.TOP_N_CLUSTERS_PER_DAY:
                    
                    # Calculate modal position analysis
                    modal_analysis = self.calculate_modal_position(bar['timestamp'], bars_1min)
                    
                    # Calculate momentum analysis
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
                    
                    # Calculate retest confirmation
                    retest_analysis = self.calculate_retest_confirmation(
                        bar['timestamp'], 
                        modal_analysis['modal_price'], 
                        bars_1min
                    )
                    
                    # Check if this is ultra-high-quality (LONG + Strength + Retest)
                    if (direction_analysis['direction'] == 'long' and 
                        strength_analysis['meets_threshold'] and
                        retest_analysis['retest_confirmed']):
                        
                        trade = {
                            'cluster_timestamp': bar['timestamp'],
                            'retest_timestamp': retest_analysis['retest_time'],
                            'entry_price': retest_analysis['retest_price'],
                            'modal_price': modal_analysis['modal_price'],
                            'signal_strength': strength_analysis['signal_strength'],
                            'modal_position': modal_analysis['modal_position'],
                            'volume_ratio': volume_ratio,
                            'volume_rank': volume_rank,
                            'time_to_retest': retest_analysis['time_to_retest'],
                            'symbol': bar['symbol'],
                            'date': date_str
                        }
                        
                        ultra_hq_trades.append(trade)
                        logger.info(f"🎯 ULTRA-HQ TRADE FOUND:")
                        logger.info(f"    Date: {date_str}")
                        logger.info(f"    Entry: {trade['entry_price']:.2f} @ {trade['retest_timestamp']}")
                        logger.info(f"    Signal Strength: {trade['signal_strength']:.3f}")
                        logger.info(f"    Modal Position: {trade['modal_position']:.3f}")
                        logger.info(f"    Time to Retest: {trade['time_to_retest']:.1f} minutes")
                
                # Add to processed clusters for future ranking decisions
                processed_clusters.append({
                    'timestamp': bar['timestamp'],
                    'volume_ratio': volume_ratio
                })
        
        return ultra_hq_trades
    
    def analyze_trade_profitability(self, trade: Dict[str, Any], bars_1min: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze profitability of a trade using different exit strategies"""
        entry_time = trade['retest_timestamp']
        entry_price = trade['entry_price']
        
        # Convert bars to DataFrame for easier manipulation
        df = pd.DataFrame(bars_1min)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Get bars after entry
        post_entry_bars = df[df['timestamp'] > entry_time].copy()
        
        if len(post_entry_bars) == 0:
            return {strategy: {'error': 'No post-entry data'} for strategy in self.exit_strategies.keys()}
        
        trade_results = {}
        
        for strategy_name, strategy in self.exit_strategies.items():
            result = self.simulate_exit_strategy(
                entry_time, entry_price, post_entry_bars, strategy, strategy_name
            )
            trade_results[strategy_name] = result
        
        return trade_results
    
    def simulate_exit_strategy(self, entry_time: datetime, entry_price: float, 
                             post_entry_bars: pd.DataFrame, strategy: Dict[str, Any], 
                             strategy_name: str) -> Dict[str, Any]:
        """Simulate a specific exit strategy"""
        target = strategy['target']
        stop_loss = strategy['stop_loss']
        time_limit = strategy['time_limit']  # minutes
        
        exit_time_limit = entry_time + timedelta(minutes=time_limit)
        
        for idx, row in post_entry_bars.iterrows():
            current_time = row['timestamp']
            current_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            time_elapsed = (current_time - entry_time).total_seconds() / 60.0
            profit_loss = current_price - entry_price
            
            # Check for stop loss hit (using low of bar)
            if low_price <= entry_price - stop_loss:
                return {
                    'exit_reason': 'stop_loss',
                    'exit_time': current_time,
                    'exit_price': entry_price - stop_loss,
                    'profit_loss': -stop_loss,
                    'time_elapsed': time_elapsed,
                    'max_favorable': high_price - entry_price,
                    'max_adverse': entry_price - low_price
                }
            
            # Check for target hit (using high of bar)
            if target and high_price >= entry_price + target:
                return {
                    'exit_reason': 'target_hit',
                    'exit_time': current_time,
                    'exit_price': entry_price + target,
                    'profit_loss': target,
                    'time_elapsed': time_elapsed,
                    'max_favorable': high_price - entry_price,
                    'max_adverse': entry_price - low_price
                }
            
            # Check for time limit
            if current_time >= exit_time_limit:
                return {
                    'exit_reason': 'time_limit',
                    'exit_time': current_time,
                    'exit_price': current_price,
                    'profit_loss': profit_loss,
                    'time_elapsed': time_elapsed,
                    'max_favorable': post_entry_bars['high'].max() - entry_price,
                    'max_adverse': entry_price - post_entry_bars['low'].min()
                }
        
        # If we get here, we ran out of data
        last_row = post_entry_bars.iloc[-1]
        return {
            'exit_reason': 'end_of_data',
            'exit_time': last_row['timestamp'],
            'exit_price': last_row['close'],
            'profit_loss': last_row['close'] - entry_price,
            'time_elapsed': (last_row['timestamp'] - entry_time).total_seconds() / 60.0,
            'max_favorable': post_entry_bars['high'].max() - entry_price,
            'max_adverse': entry_price - post_entry_bars['low'].min()
        }
    
    # Include all the helper methods from cluster_test.py (simplified versions)
    def get_rolling_volume_rank(self, cluster_time, cluster_volume_ratio, past_clusters):
        """BIAS-FREE VOLUME RANKING: Only uses clusters that occurred BEFORE current cluster"""
        from datetime import timedelta
        
        lookback_start = cluster_time - timedelta(hours=self.ROLLING_WINDOW_HOURS)
        
        relevant_clusters = []
        for past_cluster in past_clusters:
            if lookback_start <= past_cluster['timestamp'] < cluster_time:
                relevant_clusters.append(past_cluster)
        
        current_cluster = {
            'timestamp': cluster_time,
            'volume_ratio': cluster_volume_ratio
        }
        all_clusters = relevant_clusters + [current_cluster]
        
        if len(all_clusters) < self.MIN_CLUSTERS_FOR_RANKING:
            return 1
        
        sorted_clusters = sorted(all_clusters, key=lambda x: x['volume_ratio'], reverse=True)
        
        for rank, cluster in enumerate(sorted_clusters, 1):
            if cluster['timestamp'] == cluster_time:
                return rank
        
        return len(sorted_clusters)
    
    def calculate_modal_position(self, cluster_timestamp: datetime, bars_1min: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate modal position for a volume cluster"""
        df = pd.DataFrame(bars_1min)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        cluster_start = cluster_timestamp - timedelta(minutes=15)
        window_end = cluster_start + timedelta(minutes=14)
        
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
        
        cluster_slice['rounded_close'] = (cluster_slice['close'] / 0.25).round() * 0.25
        
        try:
            modal_price = statistics.mode(cluster_slice['rounded_close'])
        except statistics.StatisticsError:
            price_counts = cluster_slice['rounded_close'].value_counts()
            modal_price = price_counts.index[0]
        
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
    
    def calculate_pre_cluster_momentum(self, cluster_timestamp: datetime, bars_1min: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate pre-cluster momentum"""
        df = pd.DataFrame(bars_1min)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        cluster_start = cluster_timestamp - timedelta(minutes=15)
        momentum_start = cluster_start - timedelta(minutes=30)
        
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
        """Determine signal direction"""
        if modal_position is None:
            return {
                'direction': None,
                'position_strength': 0.0,
                'signal_type': 'NO_DATA',
                'reason': 'No modal position data available'
            }
        
        if modal_position <= self.TIGHT_LONG_THRESHOLD:
            position_strength = 1.0 - (modal_position / self.TIGHT_LONG_THRESHOLD)
            return {
                'direction': 'long',
                'position_strength': position_strength,
                'signal_type': 'LONG',
                'reason': f'Modal position {modal_position:.3f} <= {self.TIGHT_LONG_THRESHOLD} (strong buying pressure)'
            }
        
        elif modal_position >= 0.85 and not self.ELIMINATE_SHORTS:
            position_strength = (modal_position - 0.85) / 0.15
            return {
                'direction': 'short',
                'position_strength': position_strength,
                'signal_type': 'SHORT',
                'reason': f'Modal position {modal_position:.3f} >= 0.85 (selling pressure)'
            }
        
        else:
            if self.ELIMINATE_SHORTS and modal_position >= 0.85:
                reason = f'Modal position {modal_position:.3f} >= 0.85 but shorts eliminated'
            else:
                reason = f'Modal position {modal_position:.3f} in no-trade zone ({self.TIGHT_LONG_THRESHOLD} < mp < 0.85)'
            
            return {
                'direction': None,
                'position_strength': 0.0,
                'signal_type': 'NO_SIGNAL',
                'reason': reason
            }
    
    def calculate_signal_strength(self, modal_position: float, volume_ratio: float, 
                                momentum: float, direction: str) -> Dict[str, Any]:
        """Calculate signal strength"""
        if modal_position is None or volume_ratio is None:
            return {
                'signal_strength': 0.0,
                'position_strength': 0.0,
                'volume_strength': 0.0,
                'momentum_strength': 0.0,
                'meets_threshold': False,
                'error': 'Missing required data for signal strength calculation'
            }
        
        if direction == 'long':
            position_strength = 1.0 - (modal_position / self.TIGHT_LONG_THRESHOLD)
        elif direction == 'short':
            position_strength = (modal_position - 0.85) / 0.15
        else:
            position_strength = 0.0
            
        position_strength = max(0.0, min(1.0, position_strength))
        
        volume_strength = min(volume_ratio / 150.0, 1.0)
        volume_strength = max(0.0, volume_strength)
        
        momentum_strength = 0.0  # Removed as per optimization
        
        signal_strength = (0.7 * position_strength + 0.3 * volume_strength)
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
        """Calculate retest confirmation"""
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
        
        df = pd.DataFrame(bars_1min)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        cluster_end = cluster_timestamp
        retest_window_end = cluster_end + timedelta(minutes=self.RETEST_TIMEOUT)
        
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
        
        retest_confirmed = False
        retest_time = None
        retest_price = None
        time_to_retest = None
        min_distance = float('inf')
        
        for idx, row in retest_slice.iterrows():
            current_time = row['timestamp']
            current_price = row['close']
            
            distance = abs(current_price - modal_price)
            min_distance = min(min_distance, distance)
            
            if distance <= self.RETEST_TOLERANCE:
                retest_confirmed = True
                retest_time = current_time
                retest_price = current_price
                time_to_retest = (current_time - cluster_end).total_seconds() / 60.0
                break
        
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
    
    def print_trade_analysis_summary(self, trades: List[Dict[str, Any]], 
                                   trade_results: Dict[str, Dict[str, Dict[str, Any]]]):
        """Print comprehensive trade analysis summary"""
        logger.info("=" * 100)
        logger.info("💰 ULTRA-HIGH-QUALITY TRADE PROFITABILITY ANALYSIS")
        logger.info("=" * 100)
        
        if not trades:
            logger.info("❌ No ultra-high-quality trades found in the 10-day period")
            return
        
        logger.info(f"🎯 TRADE SUMMARY:")
        logger.info(f"  Total Ultra-HQ Trades: {len(trades)}")
        logger.info(f"  Analysis Period: 10 days")
        logger.info(f"  Average Trades per Day: {len(trades)/10:.1f}")
        
        # Individual trade details
        logger.info(f"\n📋 INDIVIDUAL TRADE DETAILS:")
        for i, trade in enumerate(trades, 1):
            logger.info(f"\n  🎯 TRADE #{i}:")
            logger.info(f"    Date: {trade['date']}")
            logger.info(f"    Cluster Time: {trade['cluster_timestamp'].strftime('%H:%M')}")
            logger.info(f"    Entry Time: {trade['retest_timestamp'].strftime('%H:%M')}")
            logger.info(f"    Entry Price: {trade['entry_price']:.2f}")
            logger.info(f"    Signal Strength: {trade['signal_strength']:.3f}")
            logger.info(f"    Modal Position: {trade['modal_position']:.3f}")
            logger.info(f"    Time to Retest: {trade['time_to_retest']:.1f} minutes")
        
        # Profitability analysis by strategy
        logger.info(f"\n📊 PROFITABILITY ANALYSIS BY EXIT STRATEGY:")
        
        for strategy_name, strategy_params in self.exit_strategies.items():
            logger.info(f"\n  🎯 {strategy_name.upper()} STRATEGY:")
            logger.info(f"    Target: {strategy_params['target'] or 'EOD'} points")
            logger.info(f"    Stop Loss: {strategy_params['stop_loss']} points")
            logger.info(f"    Time Limit: {strategy_params['time_limit']} minutes")
            
            profitable_trades = 0
            total_pnl = 0.0
            max_win = 0.0
            max_loss = 0.0
            
            for i, trade in enumerate(trades, 1):
                trade_key = f"trade_{i}"
                if trade_key in trade_results and strategy_name in trade_results[trade_key]:
                    result = trade_results[trade_key][strategy_name]
                    
                    if 'error' not in result:
                        pnl = result['profit_loss']
                        total_pnl += pnl
                        
                        if pnl > 0:
                            profitable_trades += 1
                            max_win = max(max_win, pnl)
                        else:
                            max_loss = min(max_loss, pnl)
                        
                        logger.info(f"      Trade #{i}: {pnl:+.2f} pts ({result['exit_reason']}) in {result['time_elapsed']:.0f}min")
            
            win_rate = profitable_trades / len(trades) * 100 if trades else 0
            avg_pnl = total_pnl / len(trades) if trades else 0
            
            logger.info(f"    📈 STRATEGY PERFORMANCE:")
            logger.info(f"      Win Rate: {win_rate:.1f}% ({profitable_trades}/{len(trades)})")
            logger.info(f"      Total P&L: {total_pnl:+.2f} points")
            logger.info(f"      Average P&L: {avg_pnl:+.2f} points per trade")
            logger.info(f"      Best Win: {max_win:+.2f} points")
            logger.info(f"      Worst Loss: {max_loss:+.2f} points")
            
            # Calculate risk-adjusted metrics
            if trades:
                risk_reward = abs(max_win / max_loss) if max_loss != 0 else float('inf')
                expectancy = avg_pnl
                
                logger.info(f"      Risk/Reward: {risk_reward:.2f}:1")
                logger.info(f"      Expectancy: {expectancy:+.2f} points per trade")
        
        # Best strategy recommendation
        strategy_performance = {}
        for strategy_name in self.exit_strategies.keys():
            total_pnl = 0.0
            for i, trade in enumerate(trades, 1):
                trade_key = f"trade_{i}"
                if trade_key in trade_results and strategy_name in trade_results[trade_key]:
                    result = trade_results[trade_key][strategy_name]
                    if 'error' not in result:
                        total_pnl += result['profit_loss']
            strategy_performance[strategy_name] = total_pnl
        
        if strategy_performance:
            best_strategy = max(strategy_performance, key=strategy_performance.get)
            best_pnl = strategy_performance[best_strategy]
            
            logger.info(f"\n🏆 BEST PERFORMING STRATEGY:")
            logger.info(f"  {best_strategy.upper()}: {best_pnl:+.2f} total points")
            logger.info(f"  Average per trade: {best_pnl/len(trades):+.2f} points")
        
        logger.info("=" * 100)
    
    async def run_profitability_analysis(self):
        """Main execution flow for profitability analysis"""
        logger.info("🚀 Starting Ultra-High-Quality Trade Profitability Analysis")
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
        
        # Step 6: Find ultra-high-quality trades
        logger.info("🔍 Identifying ultra-high-quality trades...")
        ultra_hq_trades = self.find_ultra_hq_trades(bars_15min, bars_1min, daily_thresholds)
        
        if not ultra_hq_trades:
            logger.info("❌ No ultra-high-quality trades found in the 10-day period")
            return
        
        logger.info(f"✅ Found {len(ultra_hq_trades)} ultra-high-quality trades")
        
        # Step 7: Analyze profitability for each trade
        logger.info("📊 Analyzing trade profitability...")
        
        trade_results = {}
        for i, trade in enumerate(ultra_hq_trades, 1):
            logger.info(f"🔍 Analyzing Trade #{i}...")
            results = self.analyze_trade_profitability(trade, bars_1min)
            trade_results[f"trade_{i}"] = results
        
        # Step 8: Print comprehensive analysis
        self.print_trade_analysis_summary(ultra_hq_trades, trade_results)
        
        return {
            'trades': ultra_hq_trades,
            'results': trade_results
        }


async def main():
    """Main entry point"""
    analyzer = TradeProfitabilityAnalyzer()
    analysis_results = await analyzer.run_profitability_analysis()
    
    if analysis_results:
        logger.info(f"\n✅ ANALYSIS COMPLETE: {len(analysis_results['trades'])} trades analyzed")
        return analysis_results
    else:
        logger.error("❌ Profitability analysis failed")
        return None


if __name__ == "__main__":
    if not DATABENTO_AVAILABLE:
        logger.error("❌ Cannot run without Databento package")
        exit(1)
    
    # Run the profitability analysis
    results = asyncio.run(main())
