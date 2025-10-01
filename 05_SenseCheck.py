#!/usr/bin/env python3
"""
05_SenseCheck.py - Signal Pipeline Sense Check
Analyzes past N hours of market data to validate signal generation logic.

This script:
1. Fetches historical data from Databento (past 24 hours, adjustable)
2. Replays data through signal processing pipeline
3. Tracks signal generation statistics:
   - Clusters exceeding 4.0x threshold
   - Clusters passing all checks
   - Points of failure for rejected clusters
   - Long trades that would have been executed
4. Outputs summary analysis to terminal
"""

import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from typing import Dict, Any, List
from collections import defaultdict
import statistics

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress most logs for cleaner output
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Databento
try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    print("❌ Databento not installed. Please install with: pip install databento")
    exit(1)

# Import signal processing components (we'll use the actual logic)
from datetime import timedelta as td
import math

# Signal parameters (from 02_signal.py)
VOLUME_THRESHOLD = 4.0
TOP_N_CLUSTERS_PER_DAY = 1
ROLLING_WINDOW_HOURS = 2.0
MIN_CLUSTERS_FOR_RANKING = 2
TIGHT_LONG_THRESHOLD = 0.25
ELIMINATE_SHORTS = True
MIN_SIGNAL_STRENGTH = 0.25
SIGNAL_SCALING_FACTOR = 0.3
RETEST_TOLERANCE = 0.75  # points tolerance for retest
RETEST_TIMEOUT = 30  # minutes timeout for retest


class SenseCheckAnalyzer:
    """Analyzes historical data through signal pipeline"""
    
    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours
        self.historical_client = None
        
        # Tracking statistics
        self.total_clusters_above_threshold = 0
        self.clusters_passed_checks = 0
        self.total_long_trades = 0
        self.retest_confirmed = 0
        self.retest_timeout = 0
        
        # Failure point tracking
        self.failure_reasons = defaultdict(int)
        
        # Detailed cluster tracking
        self.all_clusters = []
        self.qualifying_signals = []
        self.confirmed_trades = []
        
        # Data buffers (mimicking 02_signal.py)
        self.minute_bars_buffer = []
        self.daily_bars = []
        self.daily_avg_1min_volume = None
        self.processed_clusters = []
        self.historical_bars = []
        
    def initialize(self) -> bool:
        """Initialize Databento client"""
        api_key = os.getenv('DATABENTO_API_KEY')
        if not api_key:
            print("❌ DATABENTO_API_KEY not found in environment variables")
            return False
        
        try:
            self.historical_client = db.Historical(key=api_key)
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Databento client: {e}")
            return False
    
    def fetch_historical_data(self, symbol: str = None) -> pd.DataFrame:
        """Fetch historical OHLCV data from Databento"""
        print(f"\n📊 Fetching {self.lookback_hours}-hour historical data...")
        
        try:
            # Set time range (use safe buffer - Databento data has delay)
            end_time = datetime.now() - timedelta(hours=6)
            start_time = end_time - timedelta(hours=self.lookback_hours)
            
            print(f"📅 Time range: {start_time} to {end_time}")
            
            # Fetch data for all ES contracts
            data = self.historical_client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["ES.FUT"],
                schema="ohlcv-1m",
                start=start_time,
                end=end_time,
                stype_in="parent",
                stype_out="instrument_id"
            )
            
            df = data.to_df()
            
            if df.empty:
                print("⚠️  No data returned")
                return df
            
            print(f"✅ Retrieved {len(df)} total records")
            
            # Find most liquid symbol if not specified
            if symbol is None:
                # Filter out spreads
                df_filtered = df[~df['symbol'].str.contains('-', na=False)]
                
                # Get volume by symbol
                volume_by_symbol = df_filtered.groupby('symbol')['volume'].sum().sort_values(ascending=False)
                
                if len(volume_by_symbol) > 0:
                    symbol = volume_by_symbol.index[0]
                    print(f"\n📊 Most liquid contract: {symbol} ({volume_by_symbol.iloc[0]:,.0f} total volume)")
                    print(f"   Available contracts: {list(volume_by_symbol.head(3).index)}")
                else:
                    print("⚠️  No valid symbols found")
                    return pd.DataFrame()
            
            # Filter for specific symbol
            df = df[df['symbol'] == symbol]
            
            print(f"✅ Using {len(df)} 1-minute bars for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()
    
    def calculate_modal_position(self, cluster_start_idx: int, cluster_end_idx: int, 
                                 df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate modal position for a cluster (simplified from 02_signal.py)"""
        # Get 14-minute window from cluster start
        window_end_idx = min(cluster_start_idx + 14, len(df))
        window_slice = df.iloc[cluster_start_idx:window_end_idx]
        
        if len(window_slice) == 0:
            return {'modal_position': None, 'modal_price': None, 'error': 'No data in window'}
        
        # Round to ES tick size and find mode
        rounded_closes = (window_slice['close'] / 0.25).round() * 0.25
        
        try:
            modal_price = statistics.mode(rounded_closes)
        except statistics.StatisticsError:
            modal_price = rounded_closes.value_counts().index[0]
        
        # Calculate modal position
        price_high = window_slice['high'].max()
        price_low = window_slice['low'].min()
        price_range = price_high - price_low
        
        if price_range > 1e-9:
            modal_position = (modal_price - price_low) / price_range
        else:
            modal_position = 0.5
        
        return {
            'modal_position': modal_position,
            'modal_price': modal_price,
            'price_range': price_range,
            'error': None
        }
    
    def calculate_pre_cluster_momentum(self, cluster_start_idx: int, df: pd.DataFrame) -> float:
        """Calculate 30-minute pre-cluster momentum"""
        # Get 30 bars before cluster
        momentum_start_idx = max(0, cluster_start_idx - 30)
        momentum_slice = df.iloc[momentum_start_idx:cluster_start_idx]
        
        if len(momentum_slice) < 2:
            return 0.0
        
        start_price = momentum_slice.iloc[0]['close']
        end_price = momentum_slice.iloc[-1]['close']
        
        if start_price > 1e-9:
            momentum = (end_price - start_price) / start_price
        else:
            momentum = 0.0
        
        return momentum
    
    def determine_signal_direction(self, modal_position: float) -> Dict[str, Any]:
        """Determine signal direction based on modal position"""
        if modal_position is None:
            return {
                'direction': None,
                'position_strength': 0.0,
                'reason': 'No modal position data'
            }
        
        # Long signal
        if modal_position <= TIGHT_LONG_THRESHOLD:
            position_strength = 1.0 - (modal_position / TIGHT_LONG_THRESHOLD)
            return {
                'direction': 'long',
                'position_strength': position_strength,
                'reason': f'Long signal (modal_pos={modal_position:.3f} <= {TIGHT_LONG_THRESHOLD})'
            }
        
        # Short signal (disabled)
        elif modal_position >= 0.85 and not ELIMINATE_SHORTS:
            position_strength = (modal_position - 0.85) / 0.15
            return {
                'direction': 'short',
                'position_strength': position_strength,
                'reason': f'Short signal (modal_pos={modal_position:.3f} >= 0.85)'
            }
        
        # No-trade zone
        else:
            if ELIMINATE_SHORTS and modal_position >= 0.85:
                reason = f'Shorts eliminated (modal_pos={modal_position:.3f} >= 0.85)'
            else:
                reason = f'No-trade zone (modal_pos={modal_position:.3f} in {TIGHT_LONG_THRESHOLD}-0.85)'
            
            return {
                'direction': None,
                'position_strength': 0.0,
                'reason': reason
            }
    
    def calculate_signal_strength(self, modal_position: float, volume_ratio: float) -> float:
        """Calculate signal strength (simplified two-component formula)"""
        if modal_position is None or volume_ratio is None:
            return 0.0
        
        # Position strength (70% weight)
        position_strength = 1.0 - (modal_position / TIGHT_LONG_THRESHOLD)
        position_strength = max(0.0, min(1.0, position_strength))
        
        # Volume strength (30% weight)
        volume_strength = min(volume_ratio / 150.0, 1.0)
        volume_strength = max(0.0, volume_strength)
        
        # Combined signal strength
        signal_strength = (0.7 * position_strength + 0.3 * volume_strength)
        
        return signal_strength
    
    def get_rolling_volume_rank(self, cluster_time: datetime, cluster_volume_ratio: float,
                                lookback_hours: float = ROLLING_WINDOW_HOURS) -> int:
        """Calculate rolling volume rank (bias-free)"""
        lookback_start = cluster_time - timedelta(hours=lookback_hours)
        
        # Filter to only past clusters within rolling window
        relevant_clusters = [
            c for c in self.processed_clusters
            if lookback_start <= c['timestamp'] < cluster_time
        ]
        
        # Add current cluster
        all_clusters = relevant_clusters + [{'timestamp': cluster_time, 'volume_ratio': cluster_volume_ratio}]
        
        # Require minimum clusters for ranking
        if len(all_clusters) < MIN_CLUSTERS_FOR_RANKING:
            return 1
        
        # Sort by volume ratio and find rank
        sorted_clusters = sorted(all_clusters, key=lambda x: x['volume_ratio'], reverse=True)
        
        for rank, cluster in enumerate(sorted_clusters, 1):
            if cluster['timestamp'] == cluster_time:
                return rank
        
        return len(sorted_clusters)
    
    def analyze_data(self, df: pd.DataFrame):
        """Main analysis loop - replay data through signal pipeline"""
        print(f"\n🔄 Processing {len(df)} bars through signal pipeline...\n")
        
        if df.empty:
            print("❌ No data to analyze")
            return
        
        # Calculate daily average volume first (use all data for this analysis)
        self.daily_avg_1min_volume = df['volume'].mean()
        print(f"📊 Daily average 1-min volume: {self.daily_avg_1min_volume:.0f}\n")
        
        # Process bars in 15-minute windows
        for i in range(14, len(df)):  # Start at bar 15 (need 14 previous bars for modal analysis)
            # Build 15-minute cluster
            cluster_start_idx = i - 14
            cluster_end_idx = i + 1
            cluster_bars = df.iloc[cluster_start_idx:cluster_end_idx]
            
            # Calculate cluster volume
            cluster_volume = cluster_bars['volume'].sum()
            volume_ratio = cluster_volume / self.daily_avg_1min_volume
            
            # Check if cluster exceeds threshold
            if volume_ratio >= VOLUME_THRESHOLD:
                self.total_clusters_above_threshold += 1
                
                cluster_timestamp = cluster_bars.index[-1]
                
                # Calculate rolling rank
                volume_rank = self.get_rolling_volume_rank(cluster_timestamp, volume_ratio)
                
                # Store cluster info
                cluster_info = {
                    'timestamp': cluster_timestamp,
                    'volume_ratio': volume_ratio,
                    'volume_rank': volume_rank,
                    'cluster_volume': cluster_volume
                }
                
                # Check if cluster ranks in top-N
                if volume_rank <= TOP_N_CLUSTERS_PER_DAY:
                    # Calculate modal analysis
                    modal_analysis = self.calculate_modal_position(cluster_start_idx, cluster_end_idx, df)
                    modal_position = modal_analysis['modal_position']
                    
                    cluster_info['modal_position'] = modal_position
                    cluster_info['modal_price'] = modal_analysis['modal_price']
                    
                    # Check for failure points
                    if modal_position is None:
                        self.failure_reasons['modal_calculation_failed'] += 1
                        cluster_info['failure_reason'] = 'Modal calculation failed'
                    else:
                        # Determine signal direction
                        direction_analysis = self.determine_signal_direction(modal_position)
                        cluster_info['direction'] = direction_analysis['direction']
                        cluster_info['direction_reason'] = direction_analysis['reason']
                        
                        if direction_analysis['direction'] is None:
                            self.failure_reasons['no_signal_direction'] += 1
                            cluster_info['failure_reason'] = 'No signal direction (no-trade zone or shorts disabled)'
                        else:
                            # Calculate signal strength
                            signal_strength = self.calculate_signal_strength(modal_position, volume_ratio)
                            cluster_info['signal_strength'] = signal_strength
                            
                            if signal_strength < MIN_SIGNAL_STRENGTH:
                                self.failure_reasons['weak_signal_strength'] += 1
                                cluster_info['failure_reason'] = f'Weak signal strength ({signal_strength:.3f} < {MIN_SIGNAL_STRENGTH})'
                            else:
                                # SUCCESS - This would generate a trade signal
                                self.clusters_passed_checks += 1
                                if direction_analysis['direction'] == 'long':
                                    self.total_long_trades += 1
                                
                                cluster_info['passed'] = True
                                cluster_info['failure_reason'] = None
                                self.qualifying_signals.append(cluster_info)
                else:
                    self.failure_reasons['volume_rank_too_low'] += 1
                    cluster_info['failure_reason'] = f'Volume rank too low (rank {volume_rank} > {TOP_N_CLUSTERS_PER_DAY})'
                
                # Store cluster for analysis
                self.all_clusters.append(cluster_info)
                
                # Add to processed clusters for future ranking
                self.processed_clusters.append({
                    'timestamp': cluster_timestamp,
                    'volume_ratio': volume_ratio
                })
    
    def simulate_retest_confirmations(self, df: pd.DataFrame):
        """Simulate retest confirmation for qualifying signals"""
        print(f"\n🔄 Simulating retest confirmations for {len(self.qualifying_signals)} signals...\n")
        
        for signal in self.qualifying_signals:
            signal_time = signal['timestamp']
            modal_price = signal['modal_price']
            
            # Find the index of this signal in the dataframe
            signal_idx = df.index.get_loc(signal_time)
            
            # Look forward up to 30 minutes
            retest_window_end = signal_time + timedelta(minutes=RETEST_TIMEOUT)
            future_bars = df.loc[signal_time:retest_window_end]
            
            # Check if price retests modal price within tolerance
            retest_found = False
            retest_time = None
            retest_price = None
            
            for idx, bar in future_bars.iterrows():
                if idx == signal_time:
                    continue  # Skip the signal bar itself
                
                distance = abs(bar['close'] - modal_price)
                if distance <= RETEST_TOLERANCE:
                    retest_found = True
                    retest_time = idx
                    retest_price = bar['close']
                    break
            
            if retest_found:
                self.retest_confirmed += 1
                signal['retest_confirmed'] = True
                signal['retest_time'] = retest_time
                signal['retest_price'] = retest_price
                signal['time_to_retest'] = (retest_time - signal_time).total_seconds() / 60.0
                self.confirmed_trades.append(signal)
            else:
                self.retest_timeout += 1
                signal['retest_confirmed'] = False
                signal['retest_time'] = None
    
    def print_summary(self):
        """Print detailed summary analysis"""
        print("\n" + "="*70)
        print("📊 SIGNAL PIPELINE SENSE CHECK SUMMARY")
        print("="*70)
        
        print(f"\n⏰ Analysis Period: Past {self.lookback_hours} hours")
        print(f"📅 Time: {datetime.now()}")
        
        print(f"\n{'='*70}")
        print("🎯 CLUSTER STATISTICS")
        print(f"{'='*70}")
        print(f"Clusters above {VOLUME_THRESHOLD}x threshold:        {self.total_clusters_above_threshold}")
        print(f"Clusters passing all checks:          {self.clusters_passed_checks}")
        print(f"Signals awaiting retest:               {self.total_long_trades}")
        
        print(f"\n{'='*70}")
        print("🔄 RETEST CONFIRMATION STATISTICS")
        print(f"{'='*70}")
        print(f"Retest confirmed (trade executed):     {self.retest_confirmed}")
        print(f"Retest timeout (no execution):         {self.retest_timeout}")
        
        if self.total_long_trades > 0:
            retest_confirmation_rate = (self.retest_confirmed / self.total_long_trades) * 100
            print(f"\n✅ Retest Confirmation Rate:           {retest_confirmation_rate:.1f}%")
        
        # Calculate pass rate
        if self.total_clusters_above_threshold > 0:
            pass_rate = (self.clusters_passed_checks / self.total_clusters_above_threshold) * 100
            print(f"\n✅ Signal Pass Rate:                   {pass_rate:.1f}%")
        
        # Failure analysis
        print(f"\n{'='*70}")
        print("❌ FAILURE POINT ANALYSIS")
        print(f"{'='*70}")
        
        if self.failure_reasons:
            # Sort by frequency
            sorted_failures = sorted(self.failure_reasons.items(), key=lambda x: x[-1], reverse=True)
            
            total_failures = sum(self.failure_reasons.values())
            
            print(f"\nTotal Rejected Clusters: {total_failures}")
            print(f"\nMost Common Failure Points:")
            
            for i, (reason, count) in enumerate(sorted_failures, 1):
                percentage = (count / total_failures) * 100 if total_failures > 0 else 0
                
                # Clean up reason names
                reason_display = reason.replace('_', ' ').title()
                
                print(f"  {i}. {reason_display:40s} {count:4d} ({percentage:5.1f}%)")
        else:
            print("No failures recorded (all clusters passed!)")
        
        # Confirmed trades details
        if self.confirmed_trades:
            print(f"\n{'='*70}")
            print("✅ CONFIRMED TRADES (Retest Successful)")
            print(f"{'='*70}")
            
            for i, trade in enumerate(self.confirmed_trades, 1):
                print(f"\nTrade #{i}:")
                print(f"  Signal Time:     {trade['timestamp']}")
                print(f"  Retest Time:     {trade['retest_time']}")
                print(f"  Time to Retest:  {trade['time_to_retest']:.1f} min")
                print(f"  Modal Price:     {trade['modal_price']:.2f}")
                print(f"  Retest Price:    {trade['retest_price']:.2f}")
                print(f"  Distance:        {abs(trade['retest_price'] - trade['modal_price']):.2f} pts")
                print(f"  Volume Ratio:    {trade['volume_ratio']:.2f}x")
                print(f"  Modal Position:  {trade['modal_position']:.3f}")
                print(f"  Direction:       {trade['direction'].upper()}")
                print(f"  Signal Strength: {trade['signal_strength']:.3f}")
        
        # Qualifying signals details (all, including timeouts)
        if self.qualifying_signals:
            print(f"\n{'='*70}")
            print("📋 ALL QUALIFYING SIGNALS (Including Timeouts)")
            print(f"{'='*70}")
            
            for i, signal in enumerate(self.qualifying_signals, 1):
                status = "✅ CONFIRMED" if signal.get('retest_confirmed') else "⏰ TIMEOUT"
                print(f"\nSignal #{i}: {status}")
                print(f"  Time:            {signal['timestamp']}")
                print(f"  Volume Ratio:    {signal['volume_ratio']:.2f}x")
                print(f"  Modal Position:  {signal['modal_position']:.3f}")
                print(f"  Modal Price:     {signal.get('modal_price', 'N/A'):.2f}")
                print(f"  Direction:       {signal['direction'].upper()}")
                print(f"  Signal Strength: {signal['signal_strength']:.3f}")
                if signal.get('retest_confirmed'):
                    print(f"  Retest Time:     {signal['retest_time']}")
                    print(f"  Time to Retest:  {signal['time_to_retest']:.1f} min")
        
        # Volume ratio distribution
        if self.all_clusters:
            print(f"\n{'='*70}")
            print("📈 VOLUME RATIO DISTRIBUTION (All Clusters ≥ 4.0x)")
            print(f"{'='*70}")
            
            volume_ratios = [c['volume_ratio'] for c in self.all_clusters]
            
            print(f"  Minimum:         {min(volume_ratios):.2f}x")
            print(f"  Maximum:         {max(volume_ratios):.2f}x")
            print(f"  Average:         {statistics.mean(volume_ratios):.2f}x")
            print(f"  Median:          {statistics.median(volume_ratios):.2f}x")
        
        print(f"\n{'='*70}")
        print("✅ Analysis Complete")
        print(f"{'='*70}\n")


def main():
    """Main entry point"""
    print("🔍 SIGNAL PIPELINE SENSE CHECK")
    print("="*70)
    
    # Parse lookback hours (default 24, easily adjustable)
    lookback_hours = 24  # ← ADJUST THIS VALUE TO CHANGE LOOKBACK PERIOD
    
    print(f"\n⚙️  Configuration:")
    print(f"   Lookback Period: {lookback_hours} hours")
    print(f"   Volume Threshold: {VOLUME_THRESHOLD}x")
    print(f"   Top-N Clusters: {TOP_N_CLUSTERS_PER_DAY}")
    print(f"   Min Signal Strength: {MIN_SIGNAL_STRENGTH}")
    
    # Initialize analyzer
    analyzer = SenseCheckAnalyzer(lookback_hours=lookback_hours)
    
    if not analyzer.initialize():
        print("\n❌ Failed to initialize analyzer")
        return
    
    # Fetch data
    df = analyzer.fetch_historical_data()
    
    if df.empty:
        print("\n❌ No data available for analysis")
        return
    
    # Analyze
    analyzer.analyze_data(df)
    
    # Simulate retest confirmations
    analyzer.simulate_retest_confirmations(df)
    
    # Print summary
    analyzer.print_summary()


if __name__ == "__main__":
    main()

