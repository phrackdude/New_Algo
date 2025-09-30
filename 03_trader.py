#!/usr/bin/env python3
"""
03_trader.py - Bayesian Learning Position Sizing and Trade Management Module

This script implements:
1. Bayesian learning system with modal position binning
2. Multi-factor position sizing (signal strength + Bayesian multipliers)
3. SQLite database management for portfolio and trade tracking
4. Paper trading execution with risk management
5. Integration with 02_signal.py for ultra-high-quality trade signals

Database Schema:
- portfolio: Current account balance, equity, open positions count
- trades: Individual trade records with entry/exit details
- bayesian_data: Historical performance by modal position bins
"""

import logging
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import math
import statistics
from contextlib import contextmanager
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bayesian Learning Parameters
N_BINS = 20  # Modal position bins
MIN_TRADES_FOR_BAYESIAN = 3  # Minimum trades required for Bayesian adjustment
BAYESIAN_SCALING_FACTOR = 6.0  # Scaling factor for performance score
BAYESIAN_MAX_MULTIPLIER = 3.0  # Maximum Bayesian multiplier

# Position Sizing Parameters
BASE_POSITION_SIZE = 1  # Base position size in contracts
MAX_POSITION_SIZE = 3  # Maximum position size in contracts
SIGNAL_SCALING_FACTOR = 0.3  # Signal strength scaling factor (1.0 → 1.3x range)

# Risk Management Parameters
INITIAL_ACCOUNT_BALANCE = 50000.0  # Starting account balance
ES_POINT_VALUE = 50.0  # ES contract point value in USD
VOLATILITY_MULTIPLIER = 1.0  # Conservative volatility multiplier
MIN_STOP_DISTANCE_PCT = 0.005  # 0.5% minimum stop distance
REWARD_TO_RISK_RATIO = 2.0  # 2:1 reward-to-risk ratio
MAX_TRADE_DURATION_MINUTES = 60  # Maximum trade duration in minutes

# Transaction Cost Model
COMMISSION_PER_CONTRACT = 2.50  # Commission per contract, per side
SLIPPAGE_TICKS = 0.75  # ES ticks slippage
TICK_VALUE = 12.50  # Dollars per tick
SLIPPAGE_COST = SLIPPAGE_TICKS * TICK_VALUE  # $9.375 per contract
TOTAL_COST_PER_CONTRACT = COMMISSION_PER_CONTRACT + SLIPPAGE_COST  # $11.875 per contract

# Database Configuration
DATABASE_DIR = "/Users/albertbeccu/Library/CloudStorage/OneDrive-Personal/NordicOwl/Thoendel/New Algo Trader/New_Algo/Databases"
DATABASE_FILE = os.path.join(DATABASE_DIR, "trading_system.db")


class TradingDatabase:
    """
    SQLite database manager for portfolio, trades, and Bayesian learning data
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ensure_database_directory()
        self.initialize_database()
        logger.info(f"🗄️ Database initialized: {db_path}")
    
    def ensure_database_directory(self):
        """Ensure the database directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def initialize_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Portfolio table - tracks account balance and equity
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    account_balance REAL NOT NULL,
                    equity REAL NOT NULL,
                    open_positions_count INTEGER DEFAULT 0,
                    realized_pnl REAL DEFAULT 0.0,
                    unrealized_pnl REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Trades table - individual trade records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,  -- 'long' or 'short'
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time DATETIME NOT NULL,
                    exit_price REAL NULL,
                    exit_time DATETIME NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',  -- 'open', 'closed', 'cancelled'
                    pnl REAL NULL,
                    signal_strength REAL NOT NULL,
                    modal_position REAL NOT NULL,
                    modal_bin INTEGER NOT NULL,
                    volume_ratio REAL NOT NULL,
                    bayesian_multiplier REAL NOT NULL,
                    signal_multiplier REAL NOT NULL,
                    final_quantity INTEGER NOT NULL,
                    cluster_timestamp DATETIME NOT NULL,
                    retest_time DATETIME NOT NULL,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Bayesian data table - performance tracking by modal position bins
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bayesian_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    modal_bin INTEGER NOT NULL UNIQUE,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_return REAL DEFAULT 0.0,
                    avg_return REAL DEFAULT 0.0,
                    win_rate REAL DEFAULT 0.0,
                    performance_score REAL DEFAULT 0.0,
                    bayesian_multiplier REAL DEFAULT 1.0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Historical OHLC data table - for volatility calculations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_ohlc (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol)
                )
            """)
            
            # Create index for faster volatility queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_ohlc_timestamp 
                ON historical_ohlc(timestamp DESC, symbol)
            """)
            
            # Initialize portfolio if empty
            cursor.execute("SELECT COUNT(*) FROM portfolio")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO portfolio (account_balance, equity, open_positions_count)
                    VALUES (?, ?, 0)
                """, (INITIAL_ACCOUNT_BALANCE, INITIAL_ACCOUNT_BALANCE))
                logger.info(f"💰 Initialized portfolio with ${INITIAL_ACCOUNT_BALANCE:,.2f}")
            
            # Initialize Bayesian bins if empty
            cursor.execute("SELECT COUNT(*) FROM bayesian_data")
            if cursor.fetchone()[0] == 0:
                for bin_num in range(N_BINS):
                    cursor.execute("""
                        INSERT INTO bayesian_data (modal_bin, total_trades, winning_trades, 
                                                 total_return, avg_return, win_rate, 
                                                 performance_score, bayesian_multiplier)
                        VALUES (?, 0, 0, 0.0, 0.0, 0.0, 0.0, 1.0)
                    """, (bin_num,))
                logger.info(f"🧠 Initialized {N_BINS} Bayesian learning bins")
    
    def store_ohlc_data(self, bar_data: Dict[str, Any]):
        """Store OHLC data for volatility calculations"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Convert pandas Timestamp to string for SQLite compatibility
            timestamp_str = str(bar_data['timestamp'])
            
            cursor.execute("""
                INSERT OR REPLACE INTO historical_ohlc 
                (timestamp, symbol, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp_str, bar_data['symbol'], 
                bar_data['open'], bar_data['high'], bar_data['low'], 
                bar_data['close'], bar_data['volume']
            ))
            
            # Clean up old data (keep only last 24 hours for volatility calculations)
            # Convert timestamp for cutoff calculation
            if hasattr(bar_data['timestamp'], 'to_pydatetime'):
                # pandas Timestamp
                cutoff_time = bar_data['timestamp'].to_pydatetime() - timedelta(hours=24)
            else:
                # Standard datetime
                cutoff_time = bar_data['timestamp'] - timedelta(hours=24)
            
            cursor.execute("""
                DELETE FROM historical_ohlc 
                WHERE timestamp < ? AND symbol = ?
            """, (str(cutoff_time), bar_data['symbol']))


class VolatilityEstimator:
    """
    Multi-method volatility estimation with intelligent fallbacks
    """
    
    def __init__(self, database: TradingDatabase):
        self.db = database
        logger.info("📊 Volatility Estimator initialized")
    
    def get_historical_data(self, symbol: str, lookback_hours: int) -> List[Dict[str, Any]]:
        """Get historical OHLC data for volatility calculations"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            cursor.execute("""
                SELECT * FROM historical_ohlc 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (symbol, cutoff_time))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def calculate_garman_klass_volatility(self, symbol: str, lookback_hours: int = 8) -> Optional[float]:
        """
        Calculate Garman-Klass volatility estimator
        More efficient than close-to-close volatility as it uses OHLC data
        """
        try:
            data = self.get_historical_data(symbol, lookback_hours)
            if len(data) < 10:  # Need minimum data points
                return None
            
            gk_values = []
            for bar in data:
                if bar['high_price'] > 0 and bar['low_price'] > 0 and bar['open_price'] > 0 and bar['close_price'] > 0:
                    # Garman-Klass formula
                    ln_hl = math.log(bar['high_price'] / bar['low_price'])
                    ln_co = math.log(bar['close_price'] / bar['open_price'])
                    
                    gk_value = ln_hl ** 2 - (2 * math.log(2) - 1) * ln_co ** 2
                    gk_values.append(gk_value)
            
            if len(gk_values) < 5:
                return None
                
            # Annualized volatility (assuming 252 trading days, 6.5 hours per day)
            mean_gk = statistics.mean(gk_values)
            volatility = math.sqrt(mean_gk * 252 * 6.5)
            
            logger.debug(f"📊 Garman-Klass volatility ({lookback_hours}h): {volatility:.4f}")
            return volatility
            
        except Exception as e:
            logger.debug(f"⚠️ Garman-Klass volatility calculation failed: {e}")
            return None
    
    def calculate_atr_volatility(self, symbol: str, lookback_hours: int = 4) -> Optional[float]:
        """
        Calculate ATR-based volatility estimator
        """
        try:
            data = self.get_historical_data(symbol, lookback_hours)
            if len(data) < 5:
                return None
            
            true_ranges = []
            closes = []
            
            for i, bar in enumerate(data):
                if i == 0:
                    closes.append(bar['close_price'])
                    continue
                
                prev_close = data[i-1]['close_price']
                current_high = bar['high_price']
                current_low = bar['low_price']
                current_close = bar['close_price']
                
                # True Range calculation
                tr1 = current_high - current_low
                tr2 = abs(current_high - prev_close)
                tr3 = abs(current_low - prev_close)
                true_range = max(tr1, tr2, tr3)
                
                true_ranges.append(true_range)
                closes.append(current_close)
            
            if len(true_ranges) < 3:
                return None
            
            # ATR-based volatility
            mean_atr = statistics.mean(true_ranges)
            mean_close = statistics.mean(closes)
            
            if mean_close > 0:
                volatility = mean_atr / mean_close
                logger.debug(f"📊 ATR volatility ({lookback_hours}h): {volatility:.4f}")
                return volatility
            
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ ATR volatility calculation failed: {e}")
            return None
    
    def calculate_fallback_volatility(self, current_price: float) -> float:
        """
        Fallback volatility estimate based on price level
        """
        # Base volatility of 0.15% plus price scaling
        volatility = 0.0015 + (current_price / 1000000)
        logger.debug(f"📊 Fallback volatility: {volatility:.4f}")
        return volatility
    
    def estimate_volatility(self, symbol: str, current_price: float) -> float:
        """
        Estimate volatility using multiple methods with intelligent fallbacks
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Volatility estimate (annualized)
        """
        # Try Garman-Klass first (most accurate)
        volatility = self.calculate_garman_klass_volatility(symbol, 8)
        if volatility is not None and 0.001 <= volatility <= 1.0:  # Sanity check
            logger.debug("📊 Using Garman-Klass volatility")
            return volatility
        
        # Try ATR-based volatility
        volatility = self.calculate_atr_volatility(symbol, 4)
        if volatility is not None and 0.001 <= volatility <= 1.0:  # Sanity check
            logger.debug("📊 Using ATR volatility")
            return volatility
        
        # Use fallback
        volatility = self.calculate_fallback_volatility(current_price)
        logger.debug("📊 Using fallback volatility")
        return volatility


class BayesianLearningSystem:
    """
    Bayesian learning system for modal position performance tracking
    """
    
    def __init__(self, database: TradingDatabase):
        self.db = database
        logger.info("🧠 Bayesian Learning System initialized")
    
    def get_modal_bin(self, modal_position: float) -> int:
        """Convert modal position to bin number"""
        if modal_position is None:
            return 0  # Default bin for None values
        
        bin_size = 1.0 / N_BINS
        bin_number = int(math.floor(modal_position / bin_size))
        return min(bin_number, N_BINS - 1)
    
    def get_bayesian_multiplier(self, modal_position: float) -> float:
        """
        Calculate Bayesian multiplier for given modal position
        
        Args:
            modal_position: Modal position value (0.0 to 1.0)
            
        Returns:
            Bayesian multiplier (1.0 to BAYESIAN_MAX_MULTIPLIER)
        """
        modal_bin = self.get_modal_bin(modal_position)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT total_trades, win_rate, avg_return, bayesian_multiplier
                FROM bayesian_data 
                WHERE modal_bin = ?
            """, (modal_bin,))
            
            row = cursor.fetchone()
            if row and row['total_trades'] >= MIN_TRADES_FOR_BAYESIAN:
                multiplier = row['bayesian_multiplier']
                logger.debug(f"🧠 Bayesian multiplier for bin {modal_bin}: {multiplier:.3f} "
                           f"({row['total_trades']} trades, {row['win_rate']:.1%} win rate)")
                return multiplier
            else:
                logger.debug(f"🧠 Insufficient data for bin {modal_bin}, using 1.0x multiplier")
                return 1.0
    
    def update_bayesian_data(self, trade_data: Dict[str, Any]):
        """
        Update Bayesian learning data when a trade is closed
        
        Args:
            trade_data: Closed trade data including modal_bin and pnl
        """
        modal_bin = trade_data['modal_bin']
        pnl = trade_data['pnl']
        is_winner = pnl > 0
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current data for this bin
            cursor.execute("""
                SELECT total_trades, winning_trades, total_return
                FROM bayesian_data 
                WHERE modal_bin = ?
            """, (modal_bin,))
            
            row = cursor.fetchone()
            if row:
                new_total_trades = row['total_trades'] + 1
                new_winning_trades = row['winning_trades'] + (1 if is_winner else 0)
                new_total_return = row['total_return'] + pnl
                
                # Calculate updated statistics
                new_win_rate = new_winning_trades / new_total_trades
                new_avg_return = new_total_return / new_total_trades
                new_performance_score = new_win_rate * new_avg_return
                
                # Calculate new Bayesian multiplier
                if new_total_trades >= MIN_TRADES_FOR_BAYESIAN:
                    new_bayesian_multiplier = 1.0 + (new_performance_score * BAYESIAN_SCALING_FACTOR)
                    new_bayesian_multiplier = min(new_bayesian_multiplier, BAYESIAN_MAX_MULTIPLIER)
                else:
                    new_bayesian_multiplier = 1.0
                
                # Update the database
                cursor.execute("""
                    UPDATE bayesian_data 
                    SET total_trades = ?, winning_trades = ?, total_return = ?,
                        avg_return = ?, win_rate = ?, performance_score = ?,
                        bayesian_multiplier = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE modal_bin = ?
                """, (new_total_trades, new_winning_trades, new_total_return,
                      new_avg_return, new_win_rate, new_performance_score,
                      new_bayesian_multiplier, modal_bin))
                
                logger.info(f"🧠 Updated Bayesian data for bin {modal_bin}:")
                logger.info(f"    Trades: {new_total_trades}, Win Rate: {new_win_rate:.1%}")
                logger.info(f"    Avg Return: {new_avg_return:.2f}, Multiplier: {new_bayesian_multiplier:.3f}")


class PositionSizingEngine:
    """
    Multi-factor position sizing engine with signal strength and Bayesian multipliers
    """
    
    def __init__(self, database: TradingDatabase, bayesian_system: BayesianLearningSystem):
        self.db = database
        self.bayesian = bayesian_system
        logger.info("📏 Position Sizing Engine initialized")
    
    def calculate_position_size(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate position size using multi-factor approach
        
        Args:
            signal_data: Signal data from 02_signal.py
            
        Returns:
            Dictionary with position sizing details
        """
        signal_strength = signal_data.get('signal_strength', 0.0)
        modal_position = signal_data.get('modal_position', 0.5)
        
        # Step 1: Signal-based scaling
        signal_multiplier = 1.0 + (signal_strength * SIGNAL_SCALING_FACTOR)
        
        # Step 2: Base position with signal scaling
        base_quantity_scaled = BASE_POSITION_SIZE * signal_multiplier
        
        # Step 3: Apply Bayesian multiplier
        bayesian_multiplier = self.bayesian.get_bayesian_multiplier(modal_position)
        total_multiplier = base_quantity_scaled * bayesian_multiplier
        
        # Step 4: Apply constraints
        final_quantity = max(1, min(MAX_POSITION_SIZE, round(total_multiplier)))
        
        # Check available capital and open positions
        available_quantity = self.check_position_limits()
        final_quantity = min(final_quantity, available_quantity)
        
        position_info = {
            'signal_multiplier': signal_multiplier,
            'bayesian_multiplier': bayesian_multiplier,
            'base_quantity_scaled': base_quantity_scaled,
            'total_multiplier': total_multiplier,
            'final_quantity': final_quantity,
            'modal_bin': self.bayesian.get_modal_bin(modal_position),
            'reasoning': f"Signal: {signal_multiplier:.3f}x, Bayesian: {bayesian_multiplier:.3f}x, Final: {final_quantity}"
        }
        
        logger.info(f"📏 Position sizing calculated:")
        logger.info(f"    Signal Strength: {signal_strength:.3f} → {signal_multiplier:.3f}x multiplier")
        logger.info(f"    Modal Position: {modal_position:.3f} → {bayesian_multiplier:.3f}x Bayesian")
        logger.info(f"    Final Quantity: {final_quantity} contracts")
        
        return position_info
    
    def check_position_limits(self) -> int:
        """
        Check current open positions and available capital
        
        Returns:
            Maximum allowable position size
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current portfolio status
            cursor.execute("""
                SELECT account_balance, open_positions_count
                FROM portfolio 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                account_balance = row['account_balance']
                open_positions = row['open_positions_count']
                
                # Simple risk management: limit total open positions
                max_positions_by_count = max(0, MAX_POSITION_SIZE - open_positions)
                
                # Capital-based limit (ensure sufficient margin)
                required_margin = STOP_LOSS_POINTS * ES_POINT_VALUE  # Margin per contract
                max_positions_by_capital = int(account_balance / (required_margin * 10))  # 10x margin buffer
                
                available_quantity = min(max_positions_by_count, max_positions_by_capital, MAX_POSITION_SIZE)
                
                logger.debug(f"💰 Position limits: Balance=${account_balance:,.2f}, "
                           f"Open={open_positions}, Available={available_quantity}")
                
                return max(0, available_quantity)
            else:
                return 0


class PaperTrader:
    """
    Paper trading execution engine with risk management
    """
    
    def __init__(self, database: TradingDatabase, position_engine: PositionSizingEngine, volatility_estimator: VolatilityEstimator):
        self.db = database
        self.position_engine = position_engine
        self.volatility_estimator = volatility_estimator
        logger.info("📄 Paper Trader initialized")
    
    def generate_trade_id(self, symbol: str, timestamp: datetime) -> str:
        """Generate unique trade ID"""
        return f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    def calculate_volatility_based_stops(self, entry_price: float, symbol: str) -> Tuple[float, float]:
        """
        Calculate volatility-based stop loss and profit target
        
        Args:
            entry_price: Entry price for the trade
            symbol: Trading symbol
            
        Returns:
            Tuple of (stop_distance, profit_target_distance)
        """
        # Estimate current volatility
        volatility = self.volatility_estimator.estimate_volatility(symbol, entry_price)
        
        # Calculate volatility-based stop distance
        volatility_stop_distance = volatility * entry_price * VOLATILITY_MULTIPLIER
        
        # Apply minimum stop distance (0.5% of entry price)
        min_stop_distance = MIN_STOP_DISTANCE_PCT * entry_price
        stop_distance = max(volatility_stop_distance, min_stop_distance)
        
        # Profit target distance (2:1 reward-to-risk ratio)
        profit_target_distance = stop_distance * REWARD_TO_RISK_RATIO
        
        logger.debug(f"📊 Volatility-based risk management:")
        logger.debug(f"    Volatility: {volatility:.4f}")
        logger.debug(f"    Volatility Stop: ${volatility_stop_distance:.2f}")
        logger.debug(f"    Minimum Stop: ${min_stop_distance:.2f}")
        logger.debug(f"    Final Stop: ${stop_distance:.2f}")
        logger.debug(f"    Profit Target: ${profit_target_distance:.2f}")
        
        return stop_distance, profit_target_distance
    
    def calculate_transaction_costs(self, quantity: int, entry_price: float) -> Dict[str, float]:
        """
        Calculate realistic transaction costs including commission and slippage
        
        Args:
            quantity: Number of contracts
            entry_price: Entry price
            
        Returns:
            Dictionary with cost breakdown
        """
        # Round-trip costs (entry + exit)
        total_commission = COMMISSION_PER_CONTRACT * quantity * 2  # Both sides
        total_slippage = SLIPPAGE_COST * quantity * 2  # Both sides
        total_cost = total_commission + total_slippage
        
        # Calculate cost as fraction of position value
        contract_value = entry_price * ES_POINT_VALUE
        position_value = contract_value * quantity
        cost_fraction = total_cost / position_value if position_value > 0 else 0
        
        return {
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_cost': total_cost,
            'cost_fraction': cost_fraction,
            'cost_per_contract': TOTAL_COST_PER_CONTRACT * 2  # Round-trip
        }
    
    def execute_trade(self, signal_data: Dict[str, Any]) -> Optional[str]:
        """
        Execute a paper trade based on confirmed signal
        
        Args:
            signal_data: Confirmed signal data from 02_signal.py
            
        Returns:
            Trade ID if successful, None if failed
        """
        try:
            # Calculate position size
            position_info = self.position_engine.calculate_position_size(signal_data)
            
            if position_info['final_quantity'] <= 0:
                logger.warning("⚠️ Cannot execute trade: No available position size")
                return None
            
            # Extract trade parameters
            symbol = signal_data['symbol']
            direction = signal_data['signal_direction']
            entry_price = signal_data['confirmation_price']
            entry_time = signal_data['retest_time']
            cluster_timestamp = signal_data['cluster_timestamp']
            quantity = position_info['final_quantity']
            
            # Calculate volatility-based stop loss and take profit
            stop_distance, profit_target_distance = self.calculate_volatility_based_stops(entry_price, symbol)
            
            if direction == 'long':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + profit_target_distance
            else:  # short
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - profit_target_distance
            
            # Calculate transaction costs
            transaction_costs = self.calculate_transaction_costs(quantity, entry_price)
            
            # Generate trade ID
            trade_id = self.generate_trade_id(symbol, entry_time)
            
            # Record trade in database
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trades (
                        trade_id, symbol, direction, quantity, entry_price, entry_time,
                        stop_loss, take_profit, status, signal_strength, modal_position,
                        modal_bin, volume_ratio, bayesian_multiplier, signal_multiplier,
                        final_quantity, cluster_timestamp, retest_time, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id, symbol, direction, quantity, entry_price, entry_time,
                    stop_loss, take_profit, signal_data['signal_strength'],
                    signal_data['modal_position'], position_info['modal_bin'],
                    signal_data['volume_ratio'], position_info['bayesian_multiplier'],
                    position_info['signal_multiplier'], quantity, cluster_timestamp,
                    entry_time, f"Auto-executed from ultra-HQ signal. {position_info['reasoning']} | "
                               f"Transaction costs: ${transaction_costs['total_cost']:.2f} "
                               f"({transaction_costs['cost_fraction']:.2%} of position)"
                ))
                
                # Update portfolio
                cursor.execute("""
                    UPDATE portfolio 
                    SET open_positions_count = open_positions_count + ?,
                        timestamp = CURRENT_TIMESTAMP
                    WHERE id = (SELECT MAX(id) FROM portfolio)
                """, (quantity,))
            
            logger.info("🚀 PAPER TRADE EXECUTED:")
            logger.info(f"    Trade ID: {trade_id}")
            logger.info(f"    Symbol: {symbol}")
            logger.info(f"    Direction: {direction.upper()}")
            logger.info(f"    Quantity: {quantity} contracts")
            logger.info(f"    Entry: ${entry_price:.2f}")
            logger.info(f"    Stop Loss: ${stop_loss:.2f} (${stop_distance:.2f} distance)")
            logger.info(f"    Take Profit: ${take_profit:.2f} (${profit_target_distance:.2f} distance)")
            logger.info(f"    Risk: ${(stop_distance / entry_price * entry_price * ES_POINT_VALUE * quantity):,.2f}")
            logger.info(f"    Reward: ${(profit_target_distance / entry_price * entry_price * ES_POINT_VALUE * quantity):,.2f}")
            logger.info(f"    Transaction Costs: ${transaction_costs['total_cost']:.2f} ({transaction_costs['cost_fraction']:.2%})")
            logger.info(f"    R:R Ratio: {REWARD_TO_RISK_RATIO:.1f}:1")
            
            return trade_id
            
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            return None
    
    def check_open_trades(self, current_bar: Dict[str, Any]):
        """
        Check open trades for stop loss or take profit triggers
        
        Args:
            current_bar: Current market data bar
        """
        current_price = current_bar['close']
        current_time = current_bar['timestamp']
        symbol = current_bar['symbol']
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get open trades for this symbol
            cursor.execute("""
                SELECT * FROM trades 
                WHERE symbol = ? AND status = 'open'
            """, (symbol,))
            
            open_trades = cursor.fetchall()
            
            for trade in open_trades:
                trade_id = trade['trade_id']
                direction = trade['direction']
                entry_price = trade['entry_price']
                stop_loss = trade['stop_loss']
                take_profit = trade['take_profit']
                quantity = trade['quantity']
                
                # Check for exit conditions
                should_exit = False
                exit_reason = ""
                
                # Time-based exit check
                entry_time = trade['entry_time']
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                
                trade_duration_minutes = (current_time - entry_time).total_seconds() / 60.0
                
                if trade_duration_minutes > MAX_TRADE_DURATION_MINUTES:
                    should_exit = True
                    exit_reason = f"Time-based Exit ({trade_duration_minutes:.0f}min > {MAX_TRADE_DURATION_MINUTES}min)"
                elif direction == 'long':
                    if current_price <= stop_loss:
                        should_exit = True
                        exit_reason = "Stop Loss Hit"
                    elif current_price >= take_profit:
                        should_exit = True
                        exit_reason = "Take Profit Hit"
                else:  # short
                    if current_price >= stop_loss:
                        should_exit = True
                        exit_reason = "Stop Loss Hit"
                    elif current_price <= take_profit:
                        should_exit = True
                        exit_reason = "Take Profit Hit"
                
                if should_exit:
                    self.close_trade(trade_id, current_price, current_time, exit_reason)
    
    def close_trade(self, trade_id: str, exit_price: float, exit_time: datetime, reason: str):
        """
        Close an open trade and update portfolio
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Reason for closing
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get trade details
            cursor.execute("""
                SELECT * FROM trades WHERE trade_id = ? AND status = 'open'
            """, (trade_id,))
            
            trade = cursor.fetchone()
            if not trade:
                logger.warning(f"⚠️ Trade {trade_id} not found or already closed")
                return
            
            # Calculate P&L
            direction = trade['direction']
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            
            if direction == 'long':
                pnl_points = exit_price - entry_price
            else:  # short
                pnl_points = entry_price - exit_price
            
            # Calculate gross P&L
            gross_pnl_dollars = pnl_points * ES_POINT_VALUE * quantity
            
            # Calculate transaction costs (round-trip)
            transaction_costs = self.calculate_transaction_costs(quantity, entry_price)
            
            # Net P&L after transaction costs
            pnl_dollars = gross_pnl_dollars - transaction_costs['total_cost']
            
            # Update trade record
            cursor.execute("""
                UPDATE trades 
                SET exit_price = ?, exit_time = ?, status = 'closed', 
                    pnl = ?, notes = notes || ? || ?, updated_at = CURRENT_TIMESTAMP
                WHERE trade_id = ?
            """, (exit_price, exit_time, pnl_dollars, f" | {reason} at ${exit_price:.2f}", 
                  f" | Gross P&L: ${gross_pnl_dollars:+,.2f} | Costs: ${transaction_costs['total_cost']:.2f} | Net P&L: ${pnl_dollars:+,.2f}", trade_id))
            
            # Update portfolio
            cursor.execute("""
                SELECT account_balance FROM portfolio 
                ORDER BY timestamp DESC LIMIT 1
            """)
            current_balance = cursor.fetchone()['account_balance']
            new_balance = current_balance + pnl_dollars
            
            cursor.execute("""
                INSERT INTO portfolio (account_balance, equity, open_positions_count, realized_pnl)
                SELECT ?, ?, open_positions_count - ?, ?
                FROM portfolio 
                ORDER BY timestamp DESC LIMIT 1
            """, (new_balance, new_balance, quantity, pnl_dollars))
            
            logger.info(f"💰 TRADE CLOSED:")
            logger.info(f"    Trade ID: {trade_id}")
            logger.info(f"    Reason: {reason}")
            logger.info(f"    Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}")
            logger.info(f"    Gross P&L: {pnl_points:+.2f} points = ${gross_pnl_dollars:+,.2f}")
            logger.info(f"    Transaction Costs: ${transaction_costs['total_cost']:.2f}")
            logger.info(f"    Net P&L: ${pnl_dollars:+,.2f}")
            logger.info(f"    New Balance: ${new_balance:,.2f}")
            
            # Update Bayesian learning data
            # Note: bayesian_system is already available in this module scope
            bayesian_system.update_bayesian_data({
                'modal_bin': trade['modal_bin'],
                'pnl': pnl_points  # Use points for Bayesian learning
            })


# Global instances
database = TradingDatabase(DATABASE_FILE)
bayesian_system = BayesianLearningSystem(database)
volatility_estimator = VolatilityEstimator(database)
position_engine = PositionSizingEngine(database, bayesian_system)
paper_trader = PaperTrader(database, position_engine, volatility_estimator)


def handle_confirmed_signal(signal_data: Dict[str, Any]) -> None:
    """
    Handle confirmed ultra-high-quality signal from 02_signal.py
    
    Args:
        signal_data: Confirmed signal data with retest confirmation
    """
    global paper_trader
    
    try:
        logger.info("🎯 RECEIVED CONFIRMED SIGNAL FOR EXECUTION")
        logger.info(f"    Symbol: {signal_data['symbol']}")
        logger.info(f"    Direction: {signal_data['signal_direction'].upper()}")
        logger.info(f"    Signal Strength: {signal_data['signal_strength']:.3f}")
        logger.info(f"    Modal Position: {signal_data['modal_position']:.3f}")
        logger.info(f"    Confirmation Price: ${signal_data['confirmation_price']:.2f}")
        
        # Execute the paper trade
        trade_id = paper_trader.execute_trade(signal_data)
        
        if trade_id:
            logger.info(f"✅ Trade executed successfully: {trade_id}")
        else:
            logger.warning("⚠️ Trade execution failed")
            
    except Exception as e:
        logger.error(f"❌ Error handling confirmed signal: {e}")


def handle_market_data(bar_data: Dict[str, Any]) -> None:
    """
    Handle incoming market data for open trade monitoring and volatility calculation
    
    Args:
        bar_data: Current market data bar
    """
    global paper_trader, database
    
    try:
        # Store OHLC data for volatility calculations
        database.store_ohlc_data(bar_data)
        
        # Check open trades for exit conditions
        paper_trader.check_open_trades(bar_data)
        
    except Exception as e:
        logger.error(f"❌ Error handling market data: {e}")


def get_portfolio_status() -> Dict[str, Any]:
    """
    Get current portfolio status for monitoring
    
    Returns:
        Portfolio status dictionary
    """
    with database.get_connection() as conn:
        cursor = conn.cursor()
        
        # Get latest portfolio data
        cursor.execute("""
            SELECT * FROM portfolio 
            ORDER BY timestamp DESC LIMIT 1
        """)
        portfolio = cursor.fetchone()
        
        # Get open trades count and details
        cursor.execute("""
            SELECT COUNT(*) as count, SUM(quantity) as total_quantity
            FROM trades WHERE status = 'open'
        """)
        open_trades = cursor.fetchone()
        
        # Get recent trade performance
        cursor.execute("""
            SELECT COUNT(*) as total_trades, 
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                   AVG(pnl) as avg_pnl,
                   SUM(pnl) as total_pnl
            FROM trades WHERE status = 'closed'
        """)
        performance = cursor.fetchone()
        
        return {
            'account_balance': portfolio['account_balance'] if portfolio else INITIAL_ACCOUNT_BALANCE,
            'equity': portfolio['equity'] if portfolio else INITIAL_ACCOUNT_BALANCE,
            'open_positions': open_trades['count'] if open_trades else 0,
            'open_quantity': open_trades['total_quantity'] if open_trades and open_trades['total_quantity'] else 0,
            'total_trades': performance['total_trades'] if performance and performance['total_trades'] else 0,
            'winning_trades': performance['winning_trades'] if performance and performance['winning_trades'] else 0,
            'win_rate': (performance['winning_trades'] / performance['total_trades'] * 100) if performance and performance['total_trades'] and performance['total_trades'] > 0 else 0.0,
            'avg_pnl': performance['avg_pnl'] if performance and performance['avg_pnl'] else 0.0,
            'total_pnl': performance['total_pnl'] if performance and performance['total_pnl'] else 0.0
        }


if __name__ == "__main__":
    # This module is designed to be imported by 02_signal.py
    logger.info("🔧 03_trader.py - Enhanced Bayesian Learning Trading Module")
    logger.info("📋 This module handles ultra-high-quality signals from 02_signal.py")
    logger.info("🧠 Enhanced Features:")
    logger.info("    • Bayesian learning with 20-bin modal position tracking")
    logger.info("    • Multi-factor position sizing (signal + Bayesian multipliers)")
    logger.info("    • Volatility-based risk management (Garman-Klass + ATR)")
    logger.info("    • Realistic transaction costs ($11.88 per contract round-trip)")
    logger.info("    • Time-based exits (60-minute maximum)")
    logger.info("    • 2:1 reward-to-risk ratio targeting")
    
    # Display portfolio status
    status = get_portfolio_status()
    logger.info("💰 Current Portfolio Status:")
    logger.info(f"    Balance: ${status['account_balance']:,.2f}")
    logger.info(f"    Open Positions: {status['open_positions']} ({status['open_quantity']} contracts)")
    logger.info(f"    Total Trades: {status['total_trades']}")
    logger.info(f"    Win Rate: {status['win_rate']:.1f}%")
    logger.info(f"    Total P&L: ${status['total_pnl']:+,.2f} (after transaction costs)")
