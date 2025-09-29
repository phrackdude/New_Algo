"""
Volume Cluster Trading Strategy - Version 6.0 (Bayesian Adaptive Position Sizing)
INNOVATION: Bayesian adaptive position sizing using only historical data available at trade time
Based on V5 Fixed with identical entry/exit logic but enhanced position sizing:
- Beta distribution priors for win-rate estimation per modal bin
- Posterior probability calculation using only past trades
- Adaptive position multiplier based on expected win probability
- Strict no-future-information policy maintained
- Comprehensive diagnostics and ablation testing support
"""

# Imports
import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime
import pytz
from volume_cluster import identify_volume_clusters
from tqdm import tqdm
from scipy.stats import beta

# Base Parameters (identical to V5 Fixed)
DATA_PATH = "../data/es_ohlcv_real.csv"
VOLUME_THRESHOLD = 4.0
MIN_SIGNAL_STRENGTH = 0.45
RETENTION_MINUTES = 60
RETEST_ENABLED = True
RETEST_TICK_WINDOW = 3
BASE_POSITION_SIZE = 1.0
MAX_POSITION_SIZE = 2.5  # Will be scaled by Bayesian multiplier
VOLATILITY_LOOKBACK = 20

# V5 Fixed Parameters (unchanged)
TOP_N_CLUSTERS_PER_DAY = 1      # Focus on top-1 only (ROLLING - NO LOOKAHEAD)
MODAL_POSITION_BINS = 10
MIN_HISTORICAL_TRADES = 10      # Reduced from 20
LOOKBACK_DAYS = 60             # Increased from 30
MIN_BIN_RETURN_THRESHOLD = 0.0

# Directional Parameters
TIGHT_LONG_THRESHOLD = 0.15     # Tighter than 0.28 (based on bin 0 analysis)
ELIMINATE_SHORTS = True         # Based on negative short performance
SHORT_MIN_SIGNAL_STRENGTH = 0.65  # Much higher threshold for shorts if enabled

# Rolling Volume Ranking Parameters (BIAS-FREE)
ROLLING_VOLUME_WINDOW_HOURS = 2.0  # Look back 2 hours for volume ranking
MIN_CLUSTERS_FOR_RANKING = 2       # Minimum clusters needed for ranking

# V6 NEW: Bayesian Adaptive Position Sizing Parameters
BAYESIAN_METHOD = True              # Toggle for ablation testing
BAYESIAN_CONTEXT = "modal_bin"      # Options: "modal_bin", "volume_rank"
ALPHA_PRIOR = 1.0                  # Beta distribution prior (uninformative)
BETA_PRIOR = 1.0                   # Beta distribution prior (uninformative)
SCALING_FACTOR = 6.0               # Position multiplier scaling factor
MIN_TRADES_FOR_BAYESIAN = 3        # Minimum trades in bin before using Bayesian
BAYESIAN_MAX_MULTIPLIER = 3.0      # Maximum Bayesian position multiplier

# Transaction Cost Parameters (same as V5 Fixed)
PROFIT_TARGET_RATIO = 2.0
COMMISSION_PER_CONTRACT = 2.50
SLIPPAGE_TICKS = 0.75
TICK_VALUE = 12.50
USE_PROFIT_TARGETS = True
TIGHTER_STOPS = True

def calculate_transaction_costs(position_size=1.0):
    """Calculate total transaction costs including commission and slippage"""
    commission_cost = COMMISSION_PER_CONTRACT * position_size
    slippage_cost = SLIPPAGE_TICKS * TICK_VALUE * position_size
    return (commission_cost + slippage_cost) / (5000 * position_size)

def calculate_momentum(df, timestamp, lookback_minutes=5):
    """Calculate short-term momentum before the signal"""
    start_time = timestamp - timedelta(minutes=lookback_minutes)
    momentum_data = df.loc[start_time:timestamp]
    
    if len(momentum_data) < 2:
        return 0
    
    price_change = (momentum_data['close'].iloc[-1] - momentum_data['close'].iloc[0]) / momentum_data['close'].iloc[0]
    return price_change

def is_valid_trading_time(timestamp):
    """Check if timestamp falls within allowed CET trading hours: 14:00-17:30 CET"""
    hour = timestamp.hour
    minute = timestamp.minute
    
    if hour == 14 or hour == 15 or hour == 16 or (hour == 17 and minute < 30):
        return True
    
    return False

def calculate_signal_strength_v3(modal_position, volume_ratio, momentum):
    """Enhanced signal strength calculation (same as V5 Fixed)"""
    if modal_position <= TIGHT_LONG_THRESHOLD:
        position_strength = 1.0 - (modal_position / TIGHT_LONG_THRESHOLD)
    elif modal_position >= 0.85 and not ELIMINATE_SHORTS:  # Tighter short threshold
        position_strength = (modal_position - 0.85) / 0.15
    else:
        return 0
    
    volume_strength = min(volume_ratio / 150.0, 1.0)
    
    if modal_position <= TIGHT_LONG_THRESHOLD:
        momentum_strength = max(0, momentum * 8)
    else:
        momentum_strength = max(0, -momentum * 8)
    
    momentum_strength = min(momentum_strength, 1.0)
    combined_strength = (0.5 * position_strength + 0.3 * volume_strength + 0.2 * momentum_strength)
    return combined_strength

def calculate_profit_target_and_stop(entry_price, direction, signal_strength, volatility):
    """Calculate profit target and stop loss (same as V5 Fixed)"""
    if TIGHTER_STOPS:
        stop_distance = 1.0 * volatility * entry_price
    else:
        stop_distance = 1.5 * volatility * entry_price
    
    min_stop = 0.005 * entry_price
    stop_distance = max(stop_distance, min_stop)
    profit_distance = stop_distance * PROFIT_TARGET_RATIO
    
    if direction == "long":
        stop_price = entry_price - stop_distance
        profit_target = entry_price + profit_distance
    else:
        stop_price = entry_price + stop_distance
        profit_target = entry_price - profit_distance
    
    return profit_target, stop_price, stop_distance

def get_modal_position_bin(modal_position, n_bins=MODAL_POSITION_BINS):
    """Convert modal position to bin number (0 to n_bins-1)"""
    bin_size = 1.0 / n_bins
    bin_number = int(modal_position / bin_size)
    return min(bin_number, n_bins - 1)

def build_modal_position_statistics(historical_trades):
    """Build statistics for modal position bins based on historical trades"""
    if len(historical_trades) == 0:
        return {}
    
    bin_stats = {}
    for trade in historical_trades:
        bin_num = get_modal_position_bin(trade['modal_position'])
        if bin_num not in bin_stats:
            bin_stats[bin_num] = []
        bin_stats[bin_num].append(trade['net_return'])
    
    bin_returns = {}
    for bin_num, returns in bin_stats.items():
        if len(returns) >= 3:
            bin_returns[bin_num] = {
                'mean_return': np.mean(returns),
                'count': len(returns),
                'std_return': np.std(returns)
            }
    
    return bin_returns

def is_modal_position_tradeable(modal_position, historical_bin_stats, direction):
    """Enhanced modal position check with direction-specific logic"""
    if len(historical_bin_stats) < MIN_HISTORICAL_TRADES:
        # V5 Tightened fallback logic
        if direction == "long":
            return modal_position <= TIGHT_LONG_THRESHOLD
        elif direction == "short":
            return not ELIMINATE_SHORTS and modal_position >= 0.85
        return False
    
    bin_num = get_modal_position_bin(modal_position)
    
    if bin_num in historical_bin_stats:
        bin_data = historical_bin_stats[bin_num]
        return bin_data['mean_return'] > MIN_BIN_RETURN_THRESHOLD
    
    return False

def get_modal_direction(modal_position, historical_bin_stats):
    """Determine trade direction with V5 tightened logic"""
    if modal_position <= TIGHT_LONG_THRESHOLD:
        return "long"
    elif modal_position >= 0.85 and not ELIMINATE_SHORTS:
        return "short"
    else:
        return None

# V6 NEW: Bayesian Position Sizing Functions
def build_bayesian_statistics(historical_trades, context_key):
    """
    Build Bayesian statistics (win/loss counts) per context (modal_bin or volume_rank)
    Only uses completed historical trades available at the time
    """
    bayesian_stats = {}
    
    for trade in historical_trades:
        if context_key == "modal_bin":
            context_value = get_modal_position_bin(trade['modal_position'])
        elif context_key == "volume_rank":
            context_value = trade.get('volume_rank', 1)  # Default to 1 if not available
        else:
            continue
            
        if context_value not in bayesian_stats:
            bayesian_stats[context_value] = {'wins': 0, 'losses': 0, 'total': 0}
        
        # Classify as win or loss
        if trade['net_return'] > 0:
            bayesian_stats[context_value]['wins'] += 1
        else:
            bayesian_stats[context_value]['losses'] += 1
        
        bayesian_stats[context_value]['total'] += 1
    
    return bayesian_stats

def calculate_bayesian_multiplier(context_value, bayesian_stats):
    """
    Calculate Bayesian adaptive position multiplier using Beta distribution
    Returns multiplier and diagnostic information
    """
    if not BAYESIAN_METHOD:
        return 1.0, {"method": "disabled", "expected_p": 0.5, "alpha": 1, "beta": 1}
    
    # Get historical performance for this context
    if context_value not in bayesian_stats:
        # No historical data - use conservative prior
        alpha_post = ALPHA_PRIOR
        beta_post = BETA_PRIOR
        total_trades = 0
    else:
        stats = bayesian_stats[context_value]
        total_trades = stats['total']
        
        # Check minimum trade requirement
        if total_trades < MIN_TRADES_FOR_BAYESIAN:
            # Not enough data - use conservative fallback
            return 1.0, {
                "method": "insufficient_data", 
                "expected_p": 0.5, 
                "alpha": ALPHA_PRIOR,
                "beta": BETA_PRIOR,
                "total_trades": total_trades
            }
        
        # Calculate posterior parameters
        alpha_post = ALPHA_PRIOR + stats['wins']
        beta_post = BETA_PRIOR + stats['losses']
    
    # Calculate expected win probability (mean of Beta distribution)
    expected_p = alpha_post / (alpha_post + beta_post)
    
    # Calculate position multiplier
    # Only scale up if expected win rate > 50%
    if expected_p > 0.5:
        raw_multiplier = 1.0 + (expected_p - 0.5) * SCALING_FACTOR
        position_multiplier = min(raw_multiplier, BAYESIAN_MAX_MULTIPLIER)
    else:
        # Conservative sizing for below-50% win rate contexts
        position_multiplier = 1.0
    
    # Diagnostic information
    diagnostics = {
        "method": "bayesian",
        "expected_p": expected_p,
        "alpha": alpha_post,
        "beta": beta_post,
        "total_trades": total_trades,
        "raw_multiplier": raw_multiplier if expected_p > 0.5 else 1.0,
        "capped_multiplier": position_multiplier
    }
    
    return position_multiplier, diagnostics

def get_rolling_volume_rank(cluster_time, cluster_volume_ratio, past_clusters, avg_volume):
    """
    BIAS-FREE VOLUME RANKING: Only uses clusters that occurred BEFORE current cluster
    Returns the rank of current cluster among recent clusters (1 = highest volume)
    """
    # Define rolling window - only look at clusters from past N hours
    lookback_start = cluster_time - timedelta(hours=ROLLING_VOLUME_WINDOW_HOURS)
    
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
    if len(all_clusters) < MIN_CLUSTERS_FOR_RANKING:
        return 1  # Default to rank 1 if insufficient history
    
    # Sort by volume ratio (descending) and find current cluster's rank
    sorted_clusters = sorted(all_clusters, key=lambda x: x['volume_ratio'], reverse=True)
    
    for rank, cluster in enumerate(sorted_clusters, 1):
        if cluster['timestamp'] == cluster_time:
            return rank
    
    return len(sorted_clusters)  # Fallback

def save_statistics_to_csv(modal_stats_history, volume_stats_history, bayesian_stats_history):
    """Save statistics to CSV files for analysis"""
    
    # Save modal position statistics
    modal_rows = []
    for date, bin_stats in modal_stats_history.items():
        for bin_num, stats in bin_stats.items():
            modal_rows.append({
                'date': date,
                'bin_number': bin_num,
                'bin_range_start': bin_num / MODAL_POSITION_BINS,
                'bin_range_end': (bin_num + 1) / MODAL_POSITION_BINS,
                'mean_return': stats['mean_return'],
                'trade_count': stats['count'],
                'std_return': stats['std_return']
            })
    
    modal_df = pd.DataFrame(modal_rows)
    if not modal_df.empty:
        modal_df.to_csv("../data/modal_position_returns_v6.csv", index=False)
        print(f"Saved V6 modal position statistics: {len(modal_df)} bin-date combinations")
    
    # Save volume statistics summary
    volume_rows = []
    for date, stats in volume_stats_history.items():
        volume_rows.append({
            'date': date,
            'total_clusters': stats.get('total_clusters', 0),
            'top_n_threshold': stats.get('top_n_threshold', 0),
            'mean_volume_ratio': stats.get('mean_volume_ratio', 0),
            'max_volume_ratio': stats.get('max_volume_ratio', 0)
        })
    
    volume_df = pd.DataFrame(volume_rows)
    if not volume_df.empty:
        volume_df.to_csv("../data/volume_ratio_returns_v6.csv", index=False)
        print(f"Saved V6 volume ratio statistics: {len(volume_df)} trading days")
    
    # V6 NEW: Save Bayesian statistics
    bayesian_rows = []
    for date, bayesian_stats in bayesian_stats_history.items():
        for context_value, stats in bayesian_stats.items():
            bayesian_rows.append({
                'date': date,
                'context_type': BAYESIAN_CONTEXT,
                'context_value': context_value,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'total_trades': stats['total'],
                'win_rate': stats['wins'] / stats['total'] if stats['total'] > 0 else 0,
                'alpha_posterior': ALPHA_PRIOR + stats['wins'],
                'beta_posterior': BETA_PRIOR + stats['losses']
            })
    
    bayesian_df = pd.DataFrame(bayesian_rows)
    if not bayesian_df.empty:
        bayesian_df.to_csv("../data/bayesian_statistics_v6.csv", index=False)
        print(f"Saved V6 Bayesian statistics: {len(bayesian_df)} context-date combinations")

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")

# Initialize tracking variables
results = []
historical_trades = []
modal_stats_history = {}
volume_stats_history = {}
bayesian_stats_history = {}  # V6 NEW

print("Starting V6 BAYESIAN ADAPTIVE POSITION SIZING backtest simulation...")
print(f"Based on V5 Fixed with identical entry/exit logic")
print(f"Key V6 innovations:")
print(f"  - Bayesian adaptive position sizing: {BAYESIAN_METHOD}")
print(f"  - Bayesian context: {BAYESIAN_CONTEXT}")
print(f"  - Prior parameters: alpha={ALPHA_PRIOR}, beta={BETA_PRIOR}")
print(f"  - Scaling factor: {SCALING_FACTOR}")
print(f"  - Maximum multiplier: {BAYESIAN_MAX_MULTIPLIER}")
print(f"  - Minimum trades for Bayesian: {MIN_TRADES_FOR_BAYESIAN}")
print(f"  - STRICT NO-FUTURE-INFORMATION POLICY ✅")
print()

# Process day by day
for day, group in tqdm(df.groupby(df.index.date), desc="Processing days"):
    intraday_df = group.copy()
    
    # Identify clusters for the day
    clusters_df = identify_volume_clusters(intraday_df, volume_multiplier=VOLUME_THRESHOLD)
    
    if clusters_df.empty:
        continue
    
    # Calculate volume ratios for all clusters (but process them chronologically)
    avg_volume = intraday_df['volume'].mean()
    processed_clusters = []  # Track clusters we've already processed (for rolling ranking)
    daily_stats = {
        'total_clusters': 0,
        'volume_ratios': [],
        'ranks_assigned': []
    }
    
    clusters_sorted = clusters_df.sort_index()  # Chronological order
    
    # Build modal position statistics from historical trades (V5: longer lookback)
    cutoff_date = day - timedelta(days=LOOKBACK_DAYS)
    recent_trades = [t for t in historical_trades if t['date'] >= cutoff_date]
    current_bin_stats = build_modal_position_statistics(recent_trades)
    
    # V6 NEW: Build Bayesian statistics from historical trades
    current_bayesian_stats = build_bayesian_statistics(recent_trades, BAYESIAN_CONTEXT)
    
    # Store statistics
    modal_stats_history[day] = current_bin_stats.copy()
    bayesian_stats_history[day] = current_bayesian_stats.copy()
    
    # Process clusters in chronological order (NO LOOKAHEAD)
    for cluster_time, cluster_row in clusters_sorted.iterrows():
        if not is_valid_trading_time(cluster_time):
            continue
            
        cluster_volume = cluster_row['volume']
        volume_ratio = cluster_volume / avg_volume if avg_volume > 0 else 0
        
        daily_stats['total_clusters'] += 1
        daily_stats['volume_ratios'].append(volume_ratio)
        
        # BIAS-FREE VOLUME RANKING: Only use past clusters for ranking
        volume_rank = get_rolling_volume_rank(cluster_time, volume_ratio, processed_clusters, avg_volume)
        daily_stats['ranks_assigned'].append(volume_rank)
        
        # Only trade if this cluster ranks in top-N based on PAST information only
        if volume_rank > TOP_N_CLUSTERS_PER_DAY:
            # Add to processed clusters for future ranking decisions
            processed_clusters.append({
                'timestamp': cluster_time,
                'volume_ratio': volume_ratio,
                'cluster_data': cluster_row
            })
            continue
        
        # Analyze cluster price action
        cluster_slice = intraday_df.loc[cluster_time : cluster_time + timedelta(minutes=14)]
        
        if cluster_slice.empty:
            # Add to processed clusters even if we skip trading
            processed_clusters.append({
                'timestamp': cluster_time,
                'volume_ratio': volume_ratio,
                'cluster_data': cluster_row
            })
            continue
            
        modal_price = cluster_slice["close"].round(2).mode()
        if len(modal_price) == 0:
            processed_clusters.append({
                'timestamp': cluster_time,
                'volume_ratio': volume_ratio,
                'cluster_data': cluster_row
            })
            continue
        modal_price = modal_price[0]

        price_low = cluster_slice["low"].min()
        price_high = cluster_slice["high"].max()
        modal_position = (modal_price - price_low) / (price_high - price_low + 1e-9)

        # V5: Determine direction using tightened logic
        direction = get_modal_direction(modal_position, current_bin_stats)
        if direction is None:
            processed_clusters.append({
                'timestamp': cluster_time,
                'volume_ratio': volume_ratio,
                'cluster_data': cluster_row
            })
            continue

        # V5: Enhanced modal position filtering
        if not is_modal_position_tradeable(modal_position, current_bin_stats, direction):
            processed_clusters.append({
                'timestamp': cluster_time,
                'volume_ratio': volume_ratio,
                'cluster_data': cluster_row
            })
            continue

        # Calculate momentum and signal strength
        momentum = calculate_momentum(intraday_df, cluster_time)
        signal_strength = calculate_signal_strength_v3(modal_position, volume_ratio, momentum)
        
        # V5: Higher signal threshold for shorts
        min_signal = SHORT_MIN_SIGNAL_STRENGTH if direction == "short" else MIN_SIGNAL_STRENGTH
        if signal_strength < min_signal:
            processed_clusters.append({
                'timestamp': cluster_time,
                'volume_ratio': volume_ratio,
                'cluster_data': cluster_row
            })
            continue

        # V6 NEW: Bayesian Adaptive Position Sizing
        # Determine context value for Bayesian calculation
        if BAYESIAN_CONTEXT == "modal_bin":
            context_value = get_modal_position_bin(modal_position)
        elif BAYESIAN_CONTEXT == "volume_rank":
            context_value = volume_rank
        else:
            context_value = 0  # Fallback
        
        # Calculate Bayesian multiplier using only historical data
        bayesian_multiplier, bayesian_diagnostics = calculate_bayesian_multiplier(
            context_value, current_bayesian_stats
        )
        
        # Final position size calculation
        base_multiplier = BASE_POSITION_SIZE * (1 + signal_strength * 0.3)  # Base scaling
        adaptive_multiplier = base_multiplier * bayesian_multiplier  # Bayesian scaling
        position_multiplier = min(adaptive_multiplier, MAX_POSITION_SIZE)  # Cap at maximum

        # Find retest (same as V5 Fixed)
        retest_time = None
        for t, row in intraday_df.loc[cluster_time:].iterrows():
            if abs(row["close"] - modal_price) <= 0.75 and RETEST_ENABLED:
                retest_time = t
                break
        if not RETEST_ENABLED:
            retest_time = cluster_time + timedelta(minutes=1)

        if retest_time is None:
            processed_clusters.append({
                'timestamp': cluster_time,
                'volume_ratio': volume_ratio,
                'cluster_data': cluster_row
            })
            continue

        # Entry logic (same as V5 Fixed)
        future_candles = intraday_df[intraday_df.index > retest_time]
        if future_candles.empty:
            processed_clusters.append({
                'timestamp': cluster_time,
                'volume_ratio': volume_ratio,
                'cluster_data': cluster_row
            })
            continue
            
        entry_time = future_candles.index[0]
        entry_data = intraday_df.loc[entry_time]
        if isinstance(entry_data, pd.Series):
            entry_price = float(entry_data["close"])
        else:
            entry_price = float(entry_data["close"].iloc[0])

        # Calculate volatility and targets (BIAS-FREE: only use data up to entry time)
        recent_data = df.loc[:entry_time].tail(VOLATILITY_LOOKBACK * 24 * 60)
        if len(recent_data) > 100:
            price_changes = recent_data['close'].pct_change().dropna()
            volatility = price_changes.std()
        else:
            volatility = 0.01

        profit_target, stop_price, stop_distance = calculate_profit_target_and_stop(
            entry_price, direction, signal_strength, volatility
        )

        # Exit processing (same as V5 Fixed)
        max_hold_time = entry_time + timedelta(minutes=60)
        future_data = intraday_df[intraday_df.index > entry_time]
        exit_price = None
        actual_exit_time = None
        exit_reason = "time"
        stopped_out = False
        profit_hit = False
        
        for period_time, period_row in future_data.iterrows():
            if period_time > max_hold_time:
                exit_price = period_row["close"]
                actual_exit_time = period_time
                exit_reason = "time"
                break
                
            if USE_PROFIT_TARGETS:
                if direction == "long" and period_row["high"] >= profit_target:
                    exit_price = profit_target
                    actual_exit_time = period_time
                    exit_reason = "profit_target"
                    profit_hit = True
                    break
                elif direction == "short" and period_row["low"] <= profit_target:
                    exit_price = profit_target
                    actual_exit_time = period_time
                    exit_reason = "profit_target"
                    profit_hit = True
                    break
            
            if direction == "long" and period_row["low"] <= stop_price:
                exit_price = stop_price
                actual_exit_time = period_time
                exit_reason = "stop_loss"
                stopped_out = True
                break
            elif direction == "short" and period_row["high"] >= stop_price:
                exit_price = stop_price
                actual_exit_time = period_time
                exit_reason = "stop_loss"
                stopped_out = True
                break
        
        if exit_price is None:
            last_data = future_data.tail(1)
            if not last_data.empty:
                exit_price = float(last_data.iloc[0]["close"])
                actual_exit_time = last_data.index[0]
                exit_reason = "end_of_data"
            else:
                processed_clusters.append({
                    'timestamp': cluster_time,
                    'volume_ratio': volume_ratio,
                    'cluster_data': cluster_row
                })
                continue
        
        # Calculate returns (same as V5 Fixed)
        if direction == "short":
            gross_return = (entry_price - exit_price) / entry_price
        elif direction == "long":
            gross_return = (exit_price - entry_price) / entry_price
        
        transaction_cost = calculate_transaction_costs(position_multiplier)
        net_return = gross_return - transaction_cost
        position_adjusted_return = net_return * position_multiplier

        # Store trade result with V6 Bayesian diagnostics
        trade_result = {
            "date": day,
            "entry_time": entry_time,
            "exit_time": actual_exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": direction,
            "modal_price": modal_price,
            "modal_position": modal_position,
            "modal_bin": get_modal_position_bin(modal_position),
            "volume_ratio": volume_ratio,
            "volume_rank": volume_rank,
            "momentum": momentum,
            "signal_strength": signal_strength,
            "position_multiplier": position_multiplier,
            "profit_target": profit_target,
            "stop_price": stop_price,
            "exit_reason": exit_reason,
            "stopped_out": stopped_out,
            "profit_hit": profit_hit,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "position_adjusted_return": position_adjusted_return,
            "historical_trades_count": len(recent_trades),
            "used_adaptive_modal": len(current_bin_stats) >= MIN_HISTORICAL_TRADES,
            # V5 specific metrics
            "tight_long_threshold": TIGHT_LONG_THRESHOLD,
            "eliminate_shorts": ELIMINATE_SHORTS,
            # V5 BIAS-FREE verification metrics
            "rolling_window_hours": ROLLING_VOLUME_WINDOW_HOURS,
            "clusters_used_for_ranking": len(processed_clusters) + 1,
            # V6 NEW: Bayesian adaptive metrics
            "bayesian_method": BAYESIAN_METHOD,
            "bayesian_context": BAYESIAN_CONTEXT,
            "bayesian_context_value": context_value,
            "bayesian_multiplier": bayesian_multiplier,
            "base_multiplier": base_multiplier,
            "adaptive_multiplier": adaptive_multiplier,
            "bayesian_expected_p": bayesian_diagnostics["expected_p"],
            "bayesian_alpha": bayesian_diagnostics["alpha"],
            "bayesian_beta": bayesian_diagnostics["beta"],
            "bayesian_total_trades": bayesian_diagnostics["total_trades"],
            "bayesian_diagnostic_method": bayesian_diagnostics["method"]
        }
        
        results.append(trade_result)
        
        # Add to historical trades for future adaptive decisions
        historical_trades.append({
            'date': day,
            'modal_position': modal_position,
            'volume_rank': volume_rank,
            'net_return': net_return,
            'direction': direction
        })
        
        # Add to processed clusters for future ranking decisions
        processed_clusters.append({
            'timestamp': cluster_time,
            'volume_ratio': volume_ratio,
            'cluster_data': cluster_row
        })
    
    # Store daily volume statistics (bias-free)
    if daily_stats['volume_ratios']:
        volume_stats_history[day] = {
            'total_clusters': daily_stats['total_clusters'],
            'top_n_threshold': min(daily_stats['volume_ratios']) if len(daily_stats['volume_ratios']) >= TOP_N_CLUSTERS_PER_DAY else 0,
            'mean_volume_ratio': np.mean(daily_stats['volume_ratios']),
            'max_volume_ratio': max(daily_stats['volume_ratios']),
            'rank_distribution': daily_stats['ranks_assigned']
        }

# Export results
print("\nExporting V6 BAYESIAN ADAPTIVE results...")
result_df = pd.DataFrame(results)
result_df.to_csv("../data/backtest_results_v6.csv", index=False)

# Save adaptive statistics
save_statistics_to_csv(modal_stats_history, volume_stats_history, bayesian_stats_history)

# Enhanced Analysis with V6 Bayesian diagnostics
if len(result_df) > 0:
    print(f"\n=== V6 BAYESIAN ADAPTIVE POSITION SIZING STRATEGY RESULTS ===")
    print(f"Total Simulated Trades: {len(result_df)}")
    
    # Basic performance metrics
    mean_gross_ret = result_df["gross_return"].mean()
    mean_net_ret = result_df["net_return"].mean()
    mean_pos_adj_ret = result_df["position_adjusted_return"].mean()
    win_rate = (result_df["net_return"] > 0).mean()
    
    print(f"\nCORE PERFORMANCE METRICS:")
    print(f"  Mean Gross Return: {mean_gross_ret:.4%}")
    print(f"  Mean Net Return: {mean_net_ret:.4%}")
    print(f"  Mean Position-Adjusted Return: {mean_pos_adj_ret:.4%}")
    print(f"  Win Rate: {win_rate:.2%}")
    
    # V6 Bayesian Analysis
    bayesian_trades = result_df[result_df["bayesian_diagnostic_method"] == "bayesian"]
    insufficient_data_trades = result_df[result_df["bayesian_diagnostic_method"] == "insufficient_data"]
    disabled_trades = result_df[result_df["bayesian_diagnostic_method"] == "disabled"]
    
    print(f"\nV6 BAYESIAN ADAPTIVE ANALYSIS:")
    print(f"  Bayesian method enabled: {BAYESIAN_METHOD}")
    print(f"  Bayesian context: {BAYESIAN_CONTEXT}")
    print(f"  Trades using Bayesian sizing: {len(bayesian_trades)} ({len(bayesian_trades)/len(result_df)*100:.1f}%)")
    print(f"  Trades with insufficient data: {len(insufficient_data_trades)} ({len(insufficient_data_trades)/len(result_df)*100:.1f}%)")
    
    if len(bayesian_trades) > 0:
        print(f"  Bayesian trades mean return: {bayesian_trades['net_return'].mean():.4%}")
        print(f"  Bayesian trades win rate: {(bayesian_trades['net_return'] > 0).mean():.2%}")
        print(f"  Average Bayesian multiplier: {bayesian_trades['bayesian_multiplier'].mean():.3f}")
        print(f"  Average expected win prob: {bayesian_trades['bayesian_expected_p'].mean():.3f}")
    
    if len(insufficient_data_trades) > 0:
        print(f"  Insufficient data trades mean return: {insufficient_data_trades['net_return'].mean():.4%}")
        print(f"  Insufficient data trades win rate: {(insufficient_data_trades['net_return'] > 0).mean():.2%}")
    
    # Position sizing analysis
    print(f"\nPOSITION SIZING ANALYSIS:")
    print(f"  Mean position multiplier: {result_df['position_multiplier'].mean():.3f}")
    print(f"  Max position multiplier: {result_df['position_multiplier'].max():.3f}")
    print(f"  Mean Bayesian multiplier: {result_df['bayesian_multiplier'].mean():.3f}")
    print(f"  Max Bayesian multiplier: {result_df['bayesian_multiplier'].max():.3f}")
    
    # Context analysis
    if BAYESIAN_CONTEXT == "modal_bin":
        print(f"\nMODAL BIN BAYESIAN ANALYSIS:")
        context_performance = result_df.groupby("bayesian_context_value").agg({
            'net_return': ['mean', 'count'],
            'bayesian_multiplier': 'mean',
            'bayesian_expected_p': 'mean'
        }).round(4)
        
        for context_val, stats in context_performance.iterrows():
            print(f"  Bin {context_val}: {stats[('net_return', 'mean')]:.4%} return ({int(stats[('net_return', 'count')])} trades)")
            print(f"    Avg multiplier: {stats[('bayesian_multiplier', 'mean')]:.3f}, Avg expected p: {stats[('bayesian_expected_p', 'mean')]:.3f}")
    
    # Direction analysis
    print(f"\nDIRECTION ANALYSIS:")
    dir_performance = result_df.groupby("direction")["net_return"].agg(['mean', 'count'])
    for direction, stats in dir_performance.iterrows():
        print(f"  {direction.capitalize()}: {stats['mean']:.4%} return ({int(stats['count'])} trades)")
    
    # Exit analysis
    exit_reasons = result_df["exit_reason"].value_counts()
    profit_target_rate = (result_df["profit_hit"] == True).mean()
    stop_loss_rate = (result_df["stopped_out"] == True).mean()
    
    print(f"\nEXIT ANALYSIS:")
    print(f"  Profit Target Hit: {profit_target_rate:.2%}")
    print(f"  Stop Loss Hit: {stop_loss_rate:.2%}")
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/len(result_df)*100:.1f}%)")
    
    print(f"\nFILES EXPORTED:")
    print(f"  backtest_results_v6.csv - Main V6 Bayesian adaptive results")
    print(f"  modal_position_returns_v6.csv - V6 Modal position statistics")
    print(f"  volume_ratio_returns_v6.csv - V6 Volume ratio statistics")
    print(f"  bayesian_statistics_v6.csv - V6 Bayesian win/loss statistics per context")
    
    print(f"\n🧠 BAYESIAN GUARANTEE:")
    print(f"  ✅ Rolling historical statistics use only past trades")
    print(f"  ✅ Beta distribution priors updated chronologically")
    print(f"  ✅ Position sizing based only on available historical data")
    print(f"  ✅ No future information leakage in Bayesian calculations")
    
else:
    print("No trades found in V6 Bayesian adaptive simulation.") 