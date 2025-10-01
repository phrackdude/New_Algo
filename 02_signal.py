#!/usr/bin/env python3
"""
02_signal.py - Volume Cluster Signal Processing Module with Retest Confirmation
Implements 15-minute volume clustering methodology with retest-based signal confirmation.

This script:
1. Buffers 1-minute bars into 15-minute clusters
2. Calculates volume ratios: 15min_cluster_volume / daily_avg_1min_volume  
3. Identifies volume clusters using 4.0x threshold
4. Ranks clusters and only signals top 1-2 per day
5. Calculates modal position analysis for each cluster
6. Determines signal direction: LONG (≤ 0.25), NO_SIGNAL (0.25-0.85), SHORT disabled
7. Calculates signal strength using position + volume components
8. Adds high-quality signals to retest queue (awaiting modal price confirmation)
9. Executes ultra-high-quality trades only after retest confirmation within 30 minutes
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import deque
import statistics
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Volume Cluster Parameters (matching backtest)
VOLUME_THRESHOLD = 4.0  # 4x multiplier for cluster identification
TOP_N_CLUSTERS_PER_DAY = 1  # Only trade top 1 cluster per day
ROLLING_WINDOW_HOURS = 2.0  # Rolling window for volume ranking
MIN_CLUSTERS_FOR_RANKING = 2  # Minimum clusters needed for ranking

# Direction Determination Parameters
TIGHT_LONG_THRESHOLD = 0.25  # Modal position threshold for long signals (optimized from 0.15)
ELIMINATE_SHORTS = True  # Disable short signals due to market bias

# Signal Strength Parameters
MIN_SIGNAL_STRENGTH = 0.25  # Optimized threshold from 30-day analysis (was 0.45)

# Retest Parameters
RETEST_TOLERANCE = 0.75  # points tolerance for retest confirmation
RETEST_TIMEOUT = 30  # minutes timeout for retest confirmation


class VolumeClusterProcessor:
    """
    Processes 1-minute bars into 15-minute volume clusters
    Maintains exact methodology from successful backtest
    """
    
    def __init__(self):
        # 15-minute bar buffer (15 x 1-minute bars)
        self.minute_bars_buffer: deque = deque(maxlen=15)
        
        # Daily volume tracking
        self.daily_bars: List[Dict[str, Any]] = []
        self.daily_avg_1min_volume: Optional[float] = None
        self.current_date: Optional[str] = None
        
        # Volume cluster tracking
        self.processed_clusters: List[Dict[str, Any]] = []
        self.daily_cluster_count = 0
        
        # Historical data storage for modal analysis
        self.historical_bars: List[Dict[str, Any]] = []
        self.max_historical_days = 2  # Keep 2 days of data for modal analysis
        
        # Retest tracking for pending signals
        self.pending_retests: List[Dict[str, Any]] = []
        
        logger.info("🔧 Volume Cluster Processor initialized")
        logger.info(f"📊 Parameters: Threshold={VOLUME_THRESHOLD}x, Top-N={TOP_N_CLUSTERS_PER_DAY}")
    
    def reset_daily_state(self, new_date: str):
        """Reset state for new trading day"""
        logger.info(f"📅 New trading day: {new_date}")
        
        # Calculate daily average from yesterday's data if available
        if len(self.daily_bars) > 0:
            volumes = [bar['volume'] for bar in self.daily_bars]
            self.daily_avg_1min_volume = statistics.mean(volumes)
            logger.info(f"📊 Daily avg 1-min volume: {self.daily_avg_1min_volume:.0f}")
        
        # Reset for new day
        self.daily_bars = []
        self.current_date = new_date
        self.daily_cluster_count = 0
        self.minute_bars_buffer.clear()
        
        # Keep only recent clusters for ranking (within rolling window)
        cutoff_time = datetime.now() - timedelta(hours=ROLLING_WINDOW_HOURS)
        self.processed_clusters = [
            cluster for cluster in self.processed_clusters 
            if cluster['timestamp'] >= cutoff_time
        ]
    
    def add_minute_bar(self, bar_data: Dict[str, Any]):
        """Add 1-minute bar to buffer and daily tracking"""
        # Check pending retests first (before processing new clusters)
        self.check_pending_retests(bar_data)
        
        # Clean up expired retests
        self.cleanup_expired_retests(bar_data['timestamp'])
        
        # Track for daily average calculation
        self.daily_bars.append(bar_data.copy())
        
        # Add to historical data storage for modal analysis
        self.historical_bars.append(bar_data.copy())
        
        # Clean up old historical data (keep only last N days)
        if len(self.historical_bars) > self.max_historical_days * 24 * 60:  # N days * 24 hours * 60 minutes
            self.historical_bars = self.historical_bars[-self.max_historical_days * 24 * 60:]
        
        # Add to 15-minute buffer
        self.minute_bars_buffer.append(bar_data.copy())
        
        # Check if we have a complete 15-minute period
        if len(self.minute_bars_buffer) == 15:
            self.process_15min_cluster()
    
    def process_15min_cluster(self):
        """
        Process 15-minute cluster and check for volume signals
        Exact methodology from backtest line 431: cluster_volume / avg_volume
        """
        if len(self.minute_bars_buffer) < 15:
            return
        
        if self.daily_avg_1min_volume is None:
            logger.debug("⏳ Waiting for daily average volume calculation")
            return
        
        # Aggregate 15-minute OHLCV data
        bars = list(self.minute_bars_buffer)
        cluster_timestamp = bars[-1]['timestamp']  # Use last bar timestamp
        
        # Calculate 15-minute aggregated volume (sum of 15 x 1-minute volumes)
        cluster_volume = sum(bar['volume'] for bar in bars)
        
        # Calculate OHLCV for the 15-minute period
        cluster_open = bars[0]['open']
        cluster_high = max(bar['high'] for bar in bars)
        cluster_low = min(bar['low'] for bar in bars)
        cluster_close = bars[-1]['close']
        
        # CRITICAL: Volume ratio calculation matching backtest line 431
        volume_ratio = cluster_volume / self.daily_avg_1min_volume
        
        logger.debug(f"📊 15-min cluster: Volume={cluster_volume:.0f}, "
                    f"Daily avg={self.daily_avg_1min_volume:.0f}, "
                    f"Ratio={volume_ratio:.2f}x")
        
        # Check if this qualifies as a volume cluster (4x threshold)
        if volume_ratio >= VOLUME_THRESHOLD:
            self.handle_volume_cluster(
                timestamp=cluster_timestamp,
                volume=cluster_volume,
                volume_ratio=volume_ratio,
                open_price=cluster_open,
                high_price=cluster_high,
                low_price=cluster_low,
                close_price=cluster_close,
                symbol=bars[0]['symbol']
            )
    
    def handle_volume_cluster(self, timestamp: datetime, volume: float, volume_ratio: float,
                             open_price: float, high_price: float, low_price: float, 
                             close_price: float, symbol: str):
        """
        Handle detected volume cluster with rolling ranking
        Only signals if cluster ranks in top-N for the day
        """
        # Create cluster data
        cluster_data = {
            'timestamp': timestamp,
            'symbol': symbol,
            'volume': volume,
            'volume_ratio': volume_ratio,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        }
        
        # Calculate rolling volume rank (bias-free: only uses past clusters)
        volume_rank = self.get_rolling_volume_rank(timestamp, volume_ratio)
        cluster_data['volume_rank'] = volume_rank
        
        logger.info(f"🎯 Volume cluster detected: {symbol} @ {timestamp}")
        logger.info(f"    Volume: {volume:.0f}, Ratio: {volume_ratio:.2f}x, Rank: {volume_rank}")
        
        # Only signal if this cluster ranks in top-N
        if volume_rank <= TOP_N_CLUSTERS_PER_DAY:
            logger.info(f"🚨 TRADING SIGNAL: Top-{volume_rank} volume cluster!")
            
            # Calculate modal position analysis
            modal_analysis = self.calculate_modal_position(timestamp)
            cluster_data.update(modal_analysis)
            
            # Calculate pre-cluster momentum analysis
            momentum_analysis = self.calculate_pre_cluster_momentum(timestamp)
            cluster_data.update({
                'momentum': momentum_analysis['momentum'],
                'momentum_start_price': momentum_analysis['start_price'],
                'momentum_end_price': momentum_analysis['end_price'],
                'momentum_price_change': momentum_analysis['price_change'],
                'momentum_data_points': momentum_analysis['data_points'],
                'momentum_error': momentum_analysis['error']
            })
            
            # Determine signal direction based on modal position
            direction_analysis = self.determine_signal_direction(modal_analysis['modal_position'])
            cluster_data.update({
                'signal_direction': direction_analysis['direction'],
                'position_strength': direction_analysis['position_strength'],
                'signal_type': direction_analysis['signal_type'],
                'signal_reason': direction_analysis['reason']
            })
            
            # Calculate signal strength
            strength_analysis = self.calculate_signal_strength(
                modal_analysis['modal_position'],
                volume_ratio,
                momentum_analysis['momentum'],
                direction_analysis['direction']
            )
            cluster_data.update({
                'signal_strength': strength_analysis['signal_strength'],
                'signal_position_strength': strength_analysis['position_strength'],
                'signal_volume_strength': strength_analysis['volume_strength'],
                'signal_momentum_strength': strength_analysis['momentum_strength'],
                'meets_strength_threshold': strength_analysis['meets_threshold'],
                'strength_error': strength_analysis['error']
            })
            
            # Add high-quality signals to retest queue instead of immediate execution
            if direction_analysis['direction'] is not None and strength_analysis['meets_threshold']:
                logger.info(f"⏳ HIGH-QUALITY SIGNAL: {direction_analysis['signal_type']} signal with strength {strength_analysis['signal_strength']:.3f}")
                logger.info(f"📋 {direction_analysis['reason']}")
                logger.info(f"💪 Signal Components: Pos={strength_analysis['position_strength']:.3f}, "
                           f"Vol={strength_analysis['volume_strength']:.3f}, Mom={strength_analysis['momentum_strength']:.3f}")
                logger.info(f"🔄 Awaiting retest confirmation at modal price {modal_analysis['modal_price']:.2f}")
                
                # Add to retest queue instead of immediate execution
                self.add_pending_retest(cluster_data)
                self.daily_cluster_count += 1
            elif direction_analysis['direction'] is not None:
                logger.info(f"⚠️  WEAK SIGNAL: {direction_analysis['signal_type']} signal with strength {strength_analysis['signal_strength']:.3f} < {MIN_SIGNAL_STRENGTH}")
                logger.info(f"📋 {direction_analysis['reason']}")
                logger.info(f"💪 Signal Components: Pos={strength_analysis['position_strength']:.3f}, "
                           f"Vol={strength_analysis['volume_strength']:.3f}, Mom={strength_analysis['momentum_strength']:.3f}")
                logger.info(f"📊 Not added to retest queue (insufficient strength)")
            else:
                logger.info(f"⏸️  NO SIGNAL: {direction_analysis['signal_type']} - {direction_analysis['reason']} (Strength: {strength_analysis['signal_strength']:.3f})")
        else:
            logger.info(f"⏸️  Cluster rank {volume_rank} > {TOP_N_CLUSTERS_PER_DAY}, skipping")
        
        # Add to processed clusters for future ranking
        self.processed_clusters.append(cluster_data)
    
    def get_rolling_volume_rank(self, cluster_time: datetime, cluster_volume_ratio: float) -> int:
        """
        BIAS-FREE VOLUME RANKING: Only uses clusters that occurred BEFORE current cluster
        Returns the rank of current cluster among recent clusters (1 = highest volume)
        Exact methodology from backtest lines 276-308
        """
        # Define rolling window - only look at clusters from past N hours
        lookback_start = cluster_time - timedelta(hours=ROLLING_WINDOW_HOURS)
        
        # Filter to only past clusters within the rolling window
        relevant_clusters = []
        for past_cluster in self.processed_clusters:
            if lookback_start <= past_cluster['timestamp'] < cluster_time:
                relevant_clusters.append(past_cluster)
        
        # Add current cluster for ranking
        current_cluster = {
            'timestamp': cluster_time,
            'volume_ratio': cluster_volume_ratio
        }
        all_clusters = relevant_clusters + [current_cluster]
        
        # Require minimum clusters for meaningful ranking
        if len(all_clusters) < MIN_CLUSTERS_FOR_RANKING:
            return 1  # Default to rank 1 if insufficient history
        
        # Sort by volume ratio (descending) and find current cluster's rank
        sorted_clusters = sorted(all_clusters, key=lambda x: x['volume_ratio'], reverse=True)
        
        for rank, cluster in enumerate(sorted_clusters, 1):
            if cluster['timestamp'] == cluster_time:
                return rank
        
        return len(sorted_clusters)  # Fallback
    
    def calculate_modal_position(self, cluster_timestamp: datetime) -> Dict[str, Any]:
        """
        Calculate modal position for a volume cluster using 14-minute price action window
        
        Args:
            cluster_timestamp: Timestamp of the volume cluster (15-minute bar end)
        
        Returns:
            Dictionary containing modal analysis results
        """
        # Convert historical bars to DataFrame for easier manipulation
        if not self.historical_bars:
            return {
                'modal_price': None,
                'modal_position': None,
                'price_high': None,
                'price_low': None,
                'price_range': None,
                'data_points': 0,
                'error': 'No historical data available'
            }
        
        df = pd.DataFrame(self.historical_bars)
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
            'modal_price': modal_price,
            'modal_position': modal_position,
            'price_high': price_high,
            'price_low': price_low,
            'price_range': price_range,
            'data_points': len(cluster_slice),
            'error': None
        }
        
        # Log modal analysis results
        sentiment = 'Bullish' if modal_position < 0.5 else 'Bearish'
        logger.info(f"🎯 Modal Analysis: Price={modal_price:.2f}, Position={modal_position:.3f} ({sentiment})")
        
        return modal_analysis
    
    def calculate_pre_cluster_momentum(self, cluster_timestamp: datetime) -> Dict[str, Any]:
        """
        Calculate pre-cluster momentum using 30-minute lookback period
        
        Args:
            cluster_timestamp: Timestamp of the volume cluster (15-minute bar end)
        
        Returns:
            Dictionary containing momentum analysis results
        """
        # Convert historical bars to DataFrame for easier manipulation
        if not self.historical_bars:
            return {
                'momentum': None,
                'start_price': None,
                'end_price': None,
                'price_change': None,
                'data_points': 0,
                'error': 'No historical data available'
            }
        
        df = pd.DataFrame(self.historical_bars)
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
            'momentum': momentum,
            'start_price': start_price,
            'end_price': end_price,
            'price_change': price_change,
            'data_points': len(momentum_slice),
            'error': None
        }
        
        # Log momentum analysis results
        direction = 'Positive' if momentum > 0 else 'Negative' if momentum < 0 else 'Neutral'
        logger.info(f"📈 Momentum Analysis: {start_price:.2f} → {end_price:.2f} ({price_change:+.2f})")
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
        if modal_position <= TIGHT_LONG_THRESHOLD:
            position_strength = 1.0 - (modal_position / TIGHT_LONG_THRESHOLD)
            return {
                'direction': 'long',
                'position_strength': position_strength,
                'signal_type': 'LONG',
                'reason': f'Modal position {modal_position:.3f} <= {TIGHT_LONG_THRESHOLD} (strong buying pressure)'
            }
        
        # Short Signal Criteria (Currently Disabled)
        elif modal_position >= 0.85 and not ELIMINATE_SHORTS:
            position_strength = (modal_position - 0.85) / 0.15
            return {
                'direction': 'short',
                'position_strength': position_strength,
                'signal_type': 'SHORT',
                'reason': f'Modal position {modal_position:.3f} >= 0.85 (selling pressure)'
            }
        
        # No-Trade Zone
        else:
            if ELIMINATE_SHORTS and modal_position >= 0.85:
                reason = f'Modal position {modal_position:.3f} >= 0.85 but shorts eliminated'
            else:
                reason = f'Modal position {modal_position:.3f} in no-trade zone ({TIGHT_LONG_THRESHOLD} < mp < 0.85)'
            
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
        position_strength = 1.0 - (modal_position / TIGHT_LONG_THRESHOLD)
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
        meets_threshold = signal_strength >= MIN_SIGNAL_STRENGTH
        
        return {
            'signal_strength': signal_strength,
            'position_strength': position_strength,
            'volume_strength': volume_strength,
            'momentum_strength': momentum_strength,
            'meets_threshold': meets_threshold,
            'error': None
        }
    
    def check_pending_retests(self, current_bar: Dict[str, Any]):
        """
        Check if any pending retests are confirmed by the current bar
        
        Args:
            current_bar: Current 1-minute bar data
        """
        if not self.pending_retests:
            return
        
        current_time = current_bar['timestamp']
        current_price = current_bar['close']
        
        # Check each pending retest
        completed_retests = []
        
        for i, retest in enumerate(self.pending_retests):
            modal_price = retest['modal_price']
            cluster_time = retest['cluster_timestamp']
            
            # Check if retest window has expired
            time_elapsed = (current_time - cluster_time).total_seconds() / 60.0  # minutes
            if time_elapsed > RETEST_TIMEOUT:
                logger.info(f"⏰ Retest TIMEOUT: Modal={modal_price:.2f} after {RETEST_TIMEOUT}min")
                completed_retests.append(i)
                continue
            
            # Check if current price is within retest tolerance
            distance = abs(current_price - modal_price)
            if distance <= RETEST_TOLERANCE:
                logger.info(f"✅ Retest CONFIRMED: Modal={modal_price:.2f}, Current={current_price:.2f}")
                logger.info(f"    Distance: {distance:.2f} ≤ {RETEST_TOLERANCE}, Time: {time_elapsed:.1f}min")
                
                # Execute the trade signal
                self.execute_confirmed_signal(retest, current_bar, time_elapsed)
                completed_retests.append(i)
        
        # Remove completed retests (in reverse order to maintain indices)
        for i in reversed(completed_retests):
            del self.pending_retests[i]
    
    def execute_confirmed_signal(self, retest_data: Dict[str, Any], 
                                confirmation_bar: Dict[str, Any], 
                                time_to_retest: float):
        """
        Execute a trading signal that has been confirmed by retest
        
        Args:
            retest_data: Original cluster data with signal information
            confirmation_bar: Bar that confirmed the retest
            time_to_retest: Time elapsed from cluster to retest confirmation
        """
        logger.info("🎯 EXECUTING CONFIRMED TRADE SIGNAL")
        logger.info(f"    Symbol: {retest_data['symbol']}")
        logger.info(f"    Original Cluster: {retest_data['cluster_timestamp']}")
        logger.info(f"    Retest Confirmation: {confirmation_bar['timestamp']}")
        logger.info(f"    Time to Confirmation: {time_to_retest:.1f} minutes")
        logger.info(f"    Modal Price: {retest_data['modal_price']:.2f}")
        logger.info(f"    Confirmation Price: {confirmation_bar['close']:.2f}")
        logger.info(f"    Signal Direction: {retest_data['signal_direction'].upper()}")
        logger.info(f"    Signal Strength: {retest_data['signal_strength']:.3f}")
        
        # ✅ TRADE EXECUTION: Send confirmed signal to 03_trader.py
        try:
            # Import 03_trader module for trade execution (relative import to use venv)
            import sys
            import os
            
            # Add current directory to path if not already there
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            # Simple import that respects venv
            import importlib
            trader_module = importlib.import_module('03_trader')
            
            # Prepare signal data for trader
            signal_data = {
                'symbol': retest_data['symbol'],
                'signal_direction': retest_data['signal_direction'],
                'signal_strength': retest_data['signal_strength'],
                'modal_price': retest_data['modal_price'],
                'modal_position': retest_data.get('modal_position'),
                'volume_ratio': retest_data['volume_ratio'],
                'cluster_timestamp': retest_data['cluster_timestamp'],
                'confirmation_price': confirmation_bar['close'],
                'retest_time': confirmation_bar['timestamp'],
                'time_to_retest': time_to_retest
            }
            
            # Execute trade through 03_trader.py
            trader_module.handle_confirmed_signal(signal_data)
            
        except Exception as e:
            logger.error(f"❌ Error executing trade through 03_trader.py: {e}")
            logger.info("🚀 ULTRA-HIGH-QUALITY SIGNAL PROCESSED (execution failed)")
    
    def add_pending_retest(self, cluster_data: Dict[str, Any]):
        """
        Add a cluster to pending retest queue if it meets quality criteria
        
        Args:
            cluster_data: Cluster data with signal information
        """
        # Only add to retest queue if it's a high-quality signal
        if (cluster_data.get('signal_direction') is not None and 
            cluster_data.get('meets_strength_threshold') and
            cluster_data.get('modal_price') is not None):
            
            retest_entry = {
                'cluster_timestamp': cluster_data['timestamp'],
                'symbol': cluster_data['symbol'],
                'modal_price': cluster_data['modal_price'],
                'signal_direction': cluster_data['signal_direction'],
                'signal_strength': cluster_data['signal_strength'],
                'volume_ratio': cluster_data['volume_ratio'],
                'volume_rank': cluster_data['volume_rank']
            }
            
            self.pending_retests.append(retest_entry)
            logger.info(f"⏳ Added to retest queue: {cluster_data['signal_direction'].upper()} signal at {cluster_data['modal_price']:.2f}")
            logger.info(f"    Pending retests: {len(self.pending_retests)}")
        else:
            logger.debug("📊 Signal not added to retest queue (insufficient quality)")
    
    def cleanup_expired_retests(self, current_time: datetime):
        """
        Remove expired retests from the pending queue
        
        Args:
            current_time: Current timestamp
        """
        if not self.pending_retests:
            return
        
        expired_indices = []
        for i, retest in enumerate(self.pending_retests):
            time_elapsed = (current_time - retest['cluster_timestamp']).total_seconds() / 60.0
            if time_elapsed > RETEST_TIMEOUT:
                expired_indices.append(i)
        
        # Remove expired retests
        for i in reversed(expired_indices):
            expired_retest = self.pending_retests[i]
            logger.info(f"⏰ Retest EXPIRED: {expired_retest['signal_direction'].upper()} signal at {expired_retest['modal_price']:.2f}")
            del self.pending_retests[i]
    
    def generate_trading_signal(self, cluster_data: Dict[str, Any]):
        """
        Generate trading signal for qualified volume cluster with direction determination
        This is where the actual trading logic would be implemented
        """
        logger.info("🎯 GENERATING TRADING SIGNAL")
        logger.info(f"    Symbol: {cluster_data['symbol']}")
        logger.info(f"    Timestamp: {cluster_data['timestamp']}")
        logger.info(f"    Volume Ratio: {cluster_data['volume_ratio']:.2f}x")
        logger.info(f"    Volume Rank: {cluster_data['volume_rank']}")
        logger.info(f"    OHLC: O={cluster_data['open']:.2f}, H={cluster_data['high']:.2f}, "
                   f"L={cluster_data['low']:.2f}, C={cluster_data['close']:.2f}")
        
        # Direction Determination Analysis
        signal_type = cluster_data.get('signal_type', 'UNKNOWN')
        signal_direction = cluster_data.get('signal_direction')
        position_strength = cluster_data.get('position_strength', 0.0)
        
        logger.info(f"    📈 DIRECTION ANALYSIS:")
        logger.info(f"        Signal Type: {signal_type}")
        logger.info(f"        Direction: {signal_direction.upper() if signal_direction else 'NONE'}")
        logger.info(f"        Position Strength: {position_strength:.3f}")
        logger.info(f"        Reason: {cluster_data.get('signal_reason', 'Unknown')}")
        
        # Modal Position Analysis
        if cluster_data.get('modal_position') is not None:
            modal_position = cluster_data['modal_position']
            modal_price = cluster_data['modal_price']
            
            logger.info(f"    📊 MODAL ANALYSIS:")
            logger.info(f"        Modal Price: {modal_price:.2f}")
            logger.info(f"        Modal Position: {modal_position:.3f}")
            logger.info(f"        Price Range: {cluster_data['price_low']:.2f} - {cluster_data['price_high']:.2f}")
            logger.info(f"        Data Points: {cluster_data['data_points']}")
            
            # Signal strength classification
            if signal_direction == 'long':
                if position_strength > 0.7:
                    logger.info(f"    🔥 VERY STRONG LONG: Exceptional buying pressure")
                elif position_strength > 0.4:
                    logger.info(f"    💪 STRONG LONG: Clear buying pressure")
                else:
                    logger.info(f"    📈 MODERATE LONG: Moderate buying pressure")
            elif signal_direction == 'short':
                if position_strength > 0.7:
                    logger.info(f"    🔥 VERY STRONG SHORT: Exceptional selling pressure")
                elif position_strength > 0.4:
                    logger.info(f"    💪 STRONG SHORT: Clear selling pressure")
                else:
                    logger.info(f"    📉 MODERATE SHORT: Moderate selling pressure")
        else:
            logger.warning(f"    ❌ No modal analysis available: {cluster_data.get('error', 'Unknown error')}")
        
        # Momentum Analysis
        if cluster_data.get('momentum') is not None:
            momentum = cluster_data['momentum']
            momentum_start_price = cluster_data.get('momentum_start_price')
            momentum_end_price = cluster_data.get('momentum_end_price')
            momentum_price_change = cluster_data.get('momentum_price_change')
            
            logger.info(f"    📈 MOMENTUM ANALYSIS:")
            logger.info(f"        Price Change: {momentum_start_price:.2f} → {momentum_end_price:.2f} ({momentum_price_change:+.2f})")
            logger.info(f"        Momentum: {momentum:.4f} ({momentum*100:+.2f}%)")
            logger.info(f"        Data Points: {cluster_data.get('momentum_data_points', 0)}")
            
            # Momentum classification
            if momentum > 0.001:  # > 0.1%
                logger.info(f"    🚀 STRONG POSITIVE MOMENTUM: Excellent trend continuation setup")
            elif momentum > 0:
                logger.info(f"    📈 POSITIVE MOMENTUM: Favorable trend direction")
            elif momentum < -0.001:  # < -0.1%
                logger.info(f"    📉 STRONG NEGATIVE MOMENTUM: Potential reversal setup")
            else:
                logger.info(f"    ⚖️  NEUTRAL MOMENTUM: Sideways price action")
        else:
            logger.warning(f"    ❌ No momentum analysis available: {cluster_data.get('momentum_error', 'Unknown error')}")
        
        # ✅ Direction determination implemented
        # ✅ Modal position analysis implemented
        # ✅ Momentum calculation implemented
        # ✅ Signal strength scoring implemented (position + volume components)
        # ✅ Retest confirmation system implemented
        # ✅ Position sizing algorithm implemented (03_trader.py)
        # ✅ Risk management implemented (stop loss, take profit in 03_trader.py)
        # ✅ Paper trading execution implemented (03_trader.py with SQLite database)
        # ✅ Bayesian learning system implemented (03_trader.py)


# Global processor instance
volume_processor = VolumeClusterProcessor()


def handle_bar(bar_data: Dict[str, Any]) -> None:
    """
    Handle incoming bar data from the data connector.
    Processes 1-minute bars into 15-minute volume clusters using exact backtest methodology.
    
    Args:
        bar_data (Dict[str, Any]): Bar data containing:
            - timestamp: datetime object
            - symbol: str (e.g., 'ESM6')
            - open: float
            - high: float
            - low: float
            - close: float
            - volume: float
    """
    global volume_processor
    
    try:
        # Validate required fields
        required_fields = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_fields = [field for field in required_fields if field not in bar_data]
        
        if missing_fields:
            logger.error(f"❌ Missing required fields in bar data: {missing_fields}")
            return
        
        # Extract timestamp and check for new trading day
        timestamp = bar_data['timestamp']
        current_date = timestamp.strftime('%Y-%m-%d')
        
        # Reset daily state if new day
        if volume_processor.current_date != current_date:
            volume_processor.reset_daily_state(current_date)
        
        # Log individual bar (debug level to reduce noise)
        logger.debug(f"📊 Processing 1-min bar: {bar_data['symbol']} @ {timestamp}")
        logger.debug(f"    OHLCV: O={bar_data['open']:.2f}, H={bar_data['high']:.2f}, "
                    f"L={bar_data['low']:.2f}, C={bar_data['close']:.2f}, V={bar_data['volume']:.0f}")
        
        # Add bar to volume cluster processor
        # This will automatically:
        # 1. Buffer into 15-minute periods
        # 2. Calculate volume ratios against daily average
        # 3. Identify clusters using 4x threshold  
        # 4. Rank clusters and only signal top-N per day
        volume_processor.add_minute_bar(bar_data)
        
        # ✅ TRADE MONITORING: Send market data to 03_trader.py for open position monitoring
        try:
            # Import 03_trader module for market data handling (relative import to use venv)
            import sys
            import os
            
            # Add current directory to path if not already there
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            # Simple import that respects venv
            import importlib
            trader_module = importlib.import_module('03_trader')
            
            # Send market data for trade monitoring
            trader_module.handle_market_data(bar_data)
            
        except Exception as e:
            logger.debug(f"📊 Trade monitoring update failed: {e}")  # Debug level to reduce noise
        
    except Exception as e:
        logger.error(f"❌ Error processing bar data: {e}")
        logger.error(f"    Bar data: {bar_data}")


def initialize_signal_processor():
    """
    Initialize the signal processor (placeholder for future use).
    This function can be called to set up any required state or configuration.
    """
    logger.info("🔧 Signal processor initialized")
    # TODO: Initialize indicators, load configuration, etc.


if __name__ == "__main__":
    # This module is designed to be imported by 01_connect.py
    # It does not generate or use any synthetic data
    logger.info("🔧 02_signal.py - Signal processing module")
    logger.info("📋 This module only processes real market data from Databento")
    logger.info("⚠️  Run 01_connect.py to start the data pipeline")
