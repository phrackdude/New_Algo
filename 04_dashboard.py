#!/usr/bin/env python3
"""
04_dashboard.py - Real-Time Trading System Dashboard
Comprehensive web dashboard for monitoring the algorithmic trading system.

This dashboard provides:
1. Portfolio Status - Account balance, equity, open positions
2. P&L Analysis - Real-time and historical performance
3. Latest Trades - Recent trade history with details
4. Bayesian Statistics - Modal position bin performance
5. Cluster Analysis - Latest volume cluster data and signal thresholds

Features:
- Real-time data updates via AJAX
- Interactive charts with Plotly
- Responsive design for desktop and mobile
- Local testing with production deployment capability
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
import logging

# Flask and web components
from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
import plotly.utils

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_DIR = "/Users/albertbeccu/Library/CloudStorage/OneDrive-Personal/NordicOwl/Thoendel/New Algo Trader/New_Algo/Databases"
DATABASE_FILE = os.path.join(DATABASE_DIR, "trading_system.db")

# Dashboard configuration
REFRESH_INTERVAL = 5000  # 5 seconds in milliseconds
MAX_TRADES_DISPLAY = 50  # Maximum trades to display in table
MAX_CHART_DAYS = 7  # Maximum days for P&L chart

# Flask app
app = Flask(__name__)
app.secret_key = 'algo_trading_dashboard_2024'


class DashboardDatabase:
    """Database interface for dashboard queries"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        logger.info(f"🗄️ Dashboard database initialized: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise e
        finally:
            conn.close()
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get latest portfolio data
                cursor.execute("""
                    SELECT * FROM portfolio 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                portfolio = cursor.fetchone()
                
                # Get open trades summary
                cursor.execute("""
                    SELECT 
                        COUNT(*) as open_count,
                        SUM(quantity) as total_quantity,
                        AVG(signal_strength) as avg_signal_strength
                    FROM trades WHERE status = 'open'
                """)
                open_trades = cursor.fetchone()
                
                # Get today's performance
                today = datetime.now().strftime('%Y-%m-%d')
                cursor.execute("""
                    SELECT 
                        COUNT(*) as today_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as today_winners,
                        SUM(pnl) as today_pnl
                    FROM trades 
                    WHERE status = 'closed' AND DATE(exit_time) = ?
                """, (today,))
                today_perf = cursor.fetchone()
                
                # Get all-time performance
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        AVG(pnl) as avg_pnl,
                        SUM(pnl) as total_pnl,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade
                    FROM trades WHERE status = 'closed'
                """)
                all_time = cursor.fetchone()
                
                # Calculate win rates
                today_win_rate = (today_perf['today_winners'] / today_perf['today_trades'] * 100) if today_perf and today_perf['today_trades'] > 0 else 0
                all_time_win_rate = (all_time['winning_trades'] / all_time['total_trades'] * 100) if all_time and all_time['total_trades'] > 0 else 0
                
                return {
                    'timestamp': portfolio['timestamp'] if portfolio else datetime.now().isoformat(),
                    'account_balance': portfolio['account_balance'] if portfolio else 50000.0,
                    'equity': portfolio['equity'] if portfolio else 50000.0,
                    'realized_pnl': portfolio['realized_pnl'] if portfolio else 0.0,
                    'unrealized_pnl': portfolio['unrealized_pnl'] if portfolio else 0.0,
                    'open_positions': open_trades['open_count'] if open_trades else 0,
                    'open_quantity': open_trades['total_quantity'] if open_trades and open_trades['total_quantity'] else 0,
                    'avg_signal_strength': open_trades['avg_signal_strength'] if open_trades and open_trades['avg_signal_strength'] else 0.0,
                    'today_trades': today_perf['today_trades'] if today_perf else 0,
                    'today_pnl': today_perf['today_pnl'] if today_perf and today_perf['today_pnl'] else 0.0,
                    'today_win_rate': today_win_rate,
                    'total_trades': all_time['total_trades'] if all_time else 0,
                    'total_pnl': all_time['total_pnl'] if all_time and all_time['total_pnl'] else 0.0,
                    'all_time_win_rate': all_time_win_rate,
                    'avg_pnl_per_trade': all_time['avg_pnl'] if all_time and all_time['avg_pnl'] else 0.0,
                    'best_trade': all_time['best_trade'] if all_time and all_time['best_trade'] else 0.0,
                    'worst_trade': all_time['worst_trade'] if all_time and all_time['worst_trade'] else 0.0
                }
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return self._get_default_portfolio()
    
    def _get_default_portfolio(self) -> Dict[str, Any]:
        """Return default portfolio status when database is empty"""
        return {
            'timestamp': datetime.now().isoformat(),
            'account_balance': 50000.0,
            'equity': 50000.0,
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'open_positions': 0,
            'open_quantity': 0,
            'avg_signal_strength': 0.0,
            'today_trades': 0,
            'today_pnl': 0.0,
            'today_win_rate': 0.0,
            'total_trades': 0,
            'total_pnl': 0.0,
            'all_time_win_rate': 0.0,
            'avg_pnl_per_trade': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
    
    def get_latest_trades(self, limit: int = MAX_TRADES_DISPLAY) -> List[Dict[str, Any]]:
        """Get latest trades with full details"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        trade_id,
                        symbol,
                        direction,
                        quantity,
                        entry_price,
                        entry_time,
                        exit_price,
                        exit_time,
                        stop_loss,
                        take_profit,
                        status,
                        pnl,
                        signal_strength,
                        modal_position,
                        modal_bin,
                        volume_ratio,
                        bayesian_multiplier,
                        signal_multiplier,
                        cluster_timestamp,
                        retest_time,
                        notes,
                        created_at
                    FROM trades 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                
                trades = []
                for row in cursor.fetchall():
                    trade = dict(row)
                    
                    # Calculate trade metrics
                    if trade['status'] == 'closed' and trade['entry_price'] and trade['exit_price']:
                        if trade['direction'] == 'long':
                            points = trade['exit_price'] - trade['entry_price']
                        else:
                            points = trade['entry_price'] - trade['exit_price']
                        trade['points'] = points
                        trade['points_per_contract'] = points / trade['quantity'] if trade['quantity'] > 0 else 0
                    else:
                        trade['points'] = None
                        trade['points_per_contract'] = None
                    
                    # Format timestamps
                    if trade['entry_time']:
                        trade['entry_time_formatted'] = datetime.fromisoformat(trade['entry_time'].replace('Z', '')).strftime('%m/%d %H:%M')
                    if trade['exit_time']:
                        trade['exit_time_formatted'] = datetime.fromisoformat(trade['exit_time'].replace('Z', '')).strftime('%m/%d %H:%M')
                    
                    # Calculate trade duration if closed
                    if trade['status'] == 'closed' and trade['entry_time'] and trade['exit_time']:
                        entry_dt = datetime.fromisoformat(trade['entry_time'].replace('Z', ''))
                        exit_dt = datetime.fromisoformat(trade['exit_time'].replace('Z', ''))
                        duration_minutes = (exit_dt - entry_dt).total_seconds() / 60
                        trade['duration_minutes'] = duration_minutes
                    else:
                        trade['duration_minutes'] = None
                    
                    trades.append(trade)
                
                return trades
        except Exception as e:
            logger.error(f"Error getting latest trades: {e}")
            return []
    
    def get_bayesian_stats(self) -> List[Dict[str, Any]]:
        """Get Bayesian learning statistics by modal position bin"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        modal_bin,
                        total_trades,
                        winning_trades,
                        total_return,
                        avg_return,
                        win_rate,
                        performance_score,
                        bayesian_multiplier,
                        last_updated
                    FROM bayesian_data 
                    ORDER BY modal_bin ASC
                """)
                
                stats = []
                for row in cursor.fetchall():
                    stat = dict(row)
                    
                    # Calculate bin range (0.0-0.05, 0.05-0.10, etc.)
                    bin_size = 1.0 / 20  # 20 bins
                    bin_start = stat['modal_bin'] * bin_size
                    bin_end = (stat['modal_bin'] + 1) * bin_size
                    stat['bin_range'] = f"{bin_start:.2f}-{bin_end:.2f}"
                    
                    # Calculate loss rate
                    losing_trades = stat['total_trades'] - stat['winning_trades']
                    stat['losing_trades'] = losing_trades
                    stat['loss_rate'] = (losing_trades / stat['total_trades'] * 100) if stat['total_trades'] > 0 else 0
                    
                    # Format percentages
                    stat['win_rate_pct'] = stat['win_rate'] * 100 if stat['win_rate'] else 0
                    
                    stats.append(stat)
                
                return stats
        except Exception as e:
            logger.error(f"Error getting Bayesian stats: {e}")
            return []
    
    def get_cluster_analysis(self, limit: int = 10) -> Dict[str, Any]:
        """Get latest cluster analysis data and thresholds"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get recent trades with cluster data
                cursor.execute("""
                    SELECT 
                        cluster_timestamp,
                        symbol,
                        volume_ratio,
                        signal_strength,
                        modal_position,
                        modal_bin,
                        direction,
                        status,
                        pnl,
                        entry_time,
                        exit_time
                    FROM trades 
                    WHERE cluster_timestamp IS NOT NULL
                    ORDER BY cluster_timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                recent_clusters = []
                for row in cursor.fetchall():
                    cluster = dict(row)
                    
                    # Determine cluster outcome
                    if cluster['status'] == 'closed':
                        cluster['outcome'] = 'Win' if cluster['pnl'] > 0 else 'Loss'
                        cluster['outcome_class'] = 'success' if cluster['pnl'] > 0 else 'danger'
                    elif cluster['status'] == 'open':
                        cluster['outcome'] = 'Open'
                        cluster['outcome_class'] = 'info'
                    else:
                        cluster['outcome'] = 'Unknown'
                        cluster['outcome_class'] = 'secondary'
                    
                    # Format timestamp
                    if cluster['cluster_timestamp']:
                        cluster['cluster_time_formatted'] = datetime.fromisoformat(
                            cluster['cluster_timestamp'].replace('Z', '')
                        ).strftime('%m/%d %H:%M')
                    
                    recent_clusters.append(cluster)
                
                # Get volume ratio statistics for threshold analysis
                cursor.execute("""
                    SELECT 
                        AVG(volume_ratio) as avg_volume_ratio,
                        MIN(volume_ratio) as min_volume_ratio,
                        MAX(volume_ratio) as max_volume_ratio,
                        COUNT(*) as total_clusters
                    FROM trades 
                    WHERE volume_ratio IS NOT NULL AND cluster_timestamp >= datetime('now', '-7 days')
                """)
                volume_stats = cursor.fetchone()
                
                # Get signal strength distribution
                cursor.execute("""
                    SELECT 
                        AVG(signal_strength) as avg_signal_strength,
                        MIN(signal_strength) as min_signal_strength,
                        MAX(signal_strength) as max_signal_strength,
                        COUNT(CASE WHEN signal_strength >= 0.25 THEN 1 END) as strong_signals,
                        COUNT(*) as total_signals
                    FROM trades 
                    WHERE signal_strength IS NOT NULL AND cluster_timestamp >= datetime('now', '-7 days')
                """)
                signal_stats = cursor.fetchone()
                
                return {
                    'recent_clusters': recent_clusters,
                    'volume_stats': dict(volume_stats) if volume_stats else {},
                    'signal_stats': dict(signal_stats) if signal_stats else {},
                    'thresholds': {
                        'volume_threshold': 4.0,  # 4x multiplier from 02_signal.py
                        'signal_threshold': 0.25,  # Minimum signal strength
                        'long_threshold': 0.25,  # Modal position threshold for long signals
                        'retest_tolerance': 0.75,  # Points tolerance for retest
                        'retest_timeout': 30  # Minutes timeout for retest
                    }
                }
        except Exception as e:
            logger.error(f"Error getting cluster analysis: {e}")
            return {'recent_clusters': [], 'volume_stats': {}, 'signal_stats': {}, 'thresholds': {}}
    
    def get_pnl_chart_data(self, days: int = MAX_CHART_DAYS) -> Dict[str, Any]:
        """Get P&L data for charting"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get daily P&L data
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                cursor.execute("""
                    SELECT 
                        DATE(exit_time) as trade_date,
                        SUM(pnl) as daily_pnl,
                        COUNT(*) as daily_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as daily_winners
                    FROM trades 
                    WHERE status = 'closed' AND exit_time >= ?
                    GROUP BY DATE(exit_time)
                    ORDER BY trade_date ASC
                """, (cutoff_date,))
                
                daily_data = []
                cumulative_pnl = 0
                
                for row in cursor.fetchall():
                    daily_pnl = row['daily_pnl'] if row['daily_pnl'] else 0
                    cumulative_pnl += daily_pnl
                    
                    daily_data.append({
                        'date': row['trade_date'],
                        'daily_pnl': daily_pnl,
                        'cumulative_pnl': cumulative_pnl,
                        'trades': row['daily_trades'],
                        'winners': row['daily_winners'],
                        'win_rate': (row['daily_winners'] / row['daily_trades'] * 100) if row['daily_trades'] > 0 else 0
                    })
                
                # Get hourly P&L for today
                today = datetime.now().strftime('%Y-%m-%d')
                cursor.execute("""
                    SELECT 
                        strftime('%H', exit_time) as trade_hour,
                        SUM(pnl) as hourly_pnl,
                        COUNT(*) as hourly_trades
                    FROM trades 
                    WHERE status = 'closed' AND DATE(exit_time) = ?
                    GROUP BY strftime('%H', exit_time)
                    ORDER BY trade_hour ASC
                """, (today,))
                
                hourly_data = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'daily_data': daily_data,
                    'hourly_data': hourly_data
                }
        except Exception as e:
            logger.error(f"Error getting P&L chart data: {e}")
            return {'daily_data': [], 'hourly_data': []}


# Initialize database
dashboard_db = DashboardDatabase(DATABASE_FILE)


@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/portfolio')
def api_portfolio():
    """API endpoint for portfolio status"""
    try:
        data = dashboard_db.get_portfolio_status()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in portfolio API: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades')
def api_trades():
    """API endpoint for latest trades"""
    try:
        limit = request.args.get('limit', MAX_TRADES_DISPLAY, type=int)
        data = dashboard_db.get_latest_trades(limit)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in trades API: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/bayesian')
def api_bayesian():
    """API endpoint for Bayesian statistics"""
    try:
        data = dashboard_db.get_bayesian_stats()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in Bayesian API: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clusters')
def api_clusters():
    """API endpoint for cluster analysis"""
    try:
        data = dashboard_db.get_cluster_analysis()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in clusters API: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/pnl')
def api_pnl_chart():
    """API endpoint for P&L chart data"""
    try:
        days = request.args.get('days', MAX_CHART_DAYS, type=int)
        data = dashboard_db.get_pnl_chart_data(days)
        
        # Create Plotly charts
        daily_data = data['daily_data']
        
        if daily_data:
            # Daily P&L chart
            fig_daily = go.Figure()
            
            # Cumulative P&L line
            fig_daily.add_trace(go.Scatter(
                x=[d['date'] for d in daily_data],
                y=[d['cumulative_pnl'] for d in daily_data],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ))
            
            # Daily P&L bars
            colors = ['green' if pnl >= 0 else 'red' for pnl in [d['daily_pnl'] for d in daily_data]]
            fig_daily.add_trace(go.Bar(
                x=[d['date'] for d in daily_data],
                y=[d['daily_pnl'] for d in daily_data],
                name='Daily P&L',
                marker_color=colors,
                opacity=0.7,
                yaxis='y2'
            ))
            
            fig_daily.update_layout(
                title='P&L Performance',
                xaxis_title='Date',
                yaxis=dict(title='Cumulative P&L ($)', side='left'),
                yaxis2=dict(title='Daily P&L ($)', side='right', overlaying='y'),
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            chart_json = json.dumps(fig_daily, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            chart_json = json.dumps({})
        
        return jsonify({'chart': chart_json, 'data': data})
    except Exception as e:
        logger.error(f"Error in P&L chart API: {e}")
        return jsonify({'error': str(e)}), 500


# Create templates directory and HTML template
def create_templates():
    """Create templates directory and files if they don't exist"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    template_path = os.path.join(templates_dir, 'dashboard.html')
    if not os.path.exists(template_path):
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algo Trading Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .metric-card { transition: transform 0.2s; }
        .metric-card:hover { transform: translateY(-2px); }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-open { background-color: #007bff; }
        .status-closed-win { background-color: #28a745; }
        .status-closed-loss { background-color: #dc3545; }
        .table-responsive { max-height: 400px; overflow-y: auto; }
        .chart-container { height: 400px; }
        .refresh-indicator { color: #6c757d; font-size: 0.8em; }
        .navbar-brand { font-weight: bold; }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">
                <i class="fas fa-chart-line"></i> Algorithmic Trading Dashboard
            </span>
            <span class="refresh-indicator" id="lastUpdate">Loading...</span>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- Portfolio Status Row -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card metric-card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title text-muted">Account Balance</h5>
                        <h3 class="text-primary" id="accountBalance">$0.00</h3>
                        <small class="text-muted">Equity: <span id="equity">$0.00</span></small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title text-muted">Total P&L</h5>
                        <h3 id="totalPnl">$0.00</h3>
                        <small class="text-muted">Today: <span id="todayPnl">$0.00</span></small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title text-muted">Open Positions</h5>
                        <h3 class="text-info" id="openPositions">0</h3>
                        <small class="text-muted">Contracts: <span id="openQuantity">0</span></small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title text-muted">Win Rate</h5>
                        <h3 class="text-success" id="winRate">0%</h3>
                        <small class="text-muted">Total Trades: <span id="totalTrades">0</span></small>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-chart-area"></i> P&L Performance</h5>
                    </div>
                    <div class="card-body">
                        <div id="pnlChart" class="chart-container"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Data Tables Row -->
        <div class="row mb-4">
            <!-- Latest Trades -->
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-history"></i> Latest Trades</h5>
                    </div>
                    <div class="card-body p-0">
                        <div class="table-responsive">
                            <table class="table table-sm table-hover mb-0">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Status</th>
                                        <th>Symbol</th>
                                        <th>Dir</th>
                                        <th>Qty</th>
                                        <th>Entry</th>
                                        <th>Exit</th>
                                        <th>P&L</th>
                                        <th>Signal</th>
                                        <th>Modal</th>
                                        <th>Time</th>
                                    </tr>
                                </thead>
                                <tbody id="tradesTableBody">
                                    <tr><td colspan="10" class="text-center">Loading...</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Bayesian Statistics -->
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-brain"></i> Bayesian Stats (Top 10)</h5>
                    </div>
                    <div class="card-body p-0">
                        <div class="table-responsive">
                            <table class="table table-sm table-hover mb-0">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Bin</th>
                                        <th>Trades</th>
                                        <th>Win%</th>
                                        <th>Mult</th>
                                    </tr>
                                </thead>
                                <tbody id="bayesianTableBody">
                                    <tr><td colspan="4" class="text-center">Loading...</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Cluster Analysis Row -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-analytics"></i> Volume Cluster Analysis</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <!-- Thresholds -->
                            <div class="col-md-6">
                                <h6>System Thresholds</h6>
                                <table class="table table-sm">
                                    <tr><td>Volume Threshold:</td><td><span id="volumeThreshold">4.0x</span></td></tr>
                                    <tr><td>Signal Threshold:</td><td><span id="signalThreshold">0.25</span></td></tr>
                                    <tr><td>Long Threshold:</td><td><span id="longThreshold">≤ 0.25</span></td></tr>
                                    <tr><td>Retest Tolerance:</td><td><span id="retestTolerance">0.75 pts</span></td></tr>
                                    <tr><td>Retest Timeout:</td><td><span id="retestTimeout">30 min</span></td></tr>
                                </table>
                            </div>
                            <!-- Recent Stats -->
                            <div class="col-md-6">
                                <h6>Recent Performance (7 days)</h6>
                                <table class="table table-sm">
                                    <tr><td>Total Clusters:</td><td><span id="totalClusters">0</span></td></tr>
                                    <tr><td>Avg Volume Ratio:</td><td><span id="avgVolumeRatio">0.0x</span></td></tr>
                                    <tr><td>Avg Signal Strength:</td><td><span id="avgSignalStrength">0.0</span></td></tr>
                                    <tr><td>Strong Signals:</td><td><span id="strongSignals">0</span></td></tr>
                                </table>
                            </div>
                        </div>
                        
                        <h6 class="mt-3">Recent Clusters</h6>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead class="table-light">
                                    <tr>
                                        <th>Time</th>
                                        <th>Symbol</th>
                                        <th>Volume Ratio</th>
                                        <th>Signal Strength</th>
                                        <th>Modal Position</th>
                                        <th>Direction</th>
                                        <th>Outcome</th>
                                    </tr>
                                </thead>
                                <tbody id="clustersTableBody">
                                    <tr><td colspan="7" class="text-center">Loading...</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Dashboard JavaScript
        let refreshInterval;
        
        function formatCurrency(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 2
            }).format(value);
        }
        
        function formatPercent(value) {
            return value.toFixed(1) + '%';
        }
        
        function updatePortfolio() {
            fetch('/api/portfolio')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('accountBalance').textContent = formatCurrency(data.account_balance);
                    document.getElementById('equity').textContent = formatCurrency(data.equity);
                    document.getElementById('totalPnl').textContent = formatCurrency(data.total_pnl);
                    document.getElementById('totalPnl').className = data.total_pnl >= 0 ? 'text-success' : 'text-danger';
                    document.getElementById('todayPnl').textContent = formatCurrency(data.today_pnl);
                    document.getElementById('openPositions').textContent = data.open_positions;
                    document.getElementById('openQuantity').textContent = data.open_quantity;
                    document.getElementById('winRate').textContent = formatPercent(data.all_time_win_rate);
                    document.getElementById('totalTrades').textContent = data.total_trades;
                })
                .catch(error => console.error('Error updating portfolio:', error));
        }
        
        function updateTrades() {
            fetch('/api/trades?limit=20')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('tradesTableBody');
                    tbody.innerHTML = '';
                    
                    data.forEach(trade => {
                        const row = document.createElement('tr');
                        
                        let statusIndicator = '';
                        if (trade.status === 'open') {
                            statusIndicator = '<span class="status-indicator status-open"></span>Open';
                        } else if (trade.status === 'closed') {
                            if (trade.pnl > 0) {
                                statusIndicator = '<span class="status-indicator status-closed-win"></span>Win';
                            } else {
                                statusIndicator = '<span class="status-indicator status-closed-loss"></span>Loss';
                            }
                        }
                        
                        const pnlClass = trade.pnl > 0 ? 'text-success' : trade.pnl < 0 ? 'text-danger' : '';
                        const pnlText = trade.pnl ? formatCurrency(trade.pnl) : '-';
                        
                        row.innerHTML = `
                            <td>${statusIndicator}</td>
                            <td>${trade.symbol}</td>
                            <td><span class="badge bg-${trade.direction === 'long' ? 'success' : 'danger'}">${trade.direction.toUpperCase()}</span></td>
                            <td>${trade.quantity}</td>
                            <td>$${trade.entry_price.toFixed(2)}</td>
                            <td>${trade.exit_price ? '$' + trade.exit_price.toFixed(2) : '-'}</td>
                            <td class="${pnlClass}">${pnlText}</td>
                            <td>${trade.signal_strength.toFixed(3)}</td>
                            <td>${trade.modal_position.toFixed(3)}</td>
                            <td>${trade.entry_time_formatted}</td>
                        `;
                        tbody.appendChild(row);
                    });
                })
                .catch(error => console.error('Error updating trades:', error));
        }
        
        function updateBayesian() {
            fetch('/api/bayesian')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('bayesianTableBody');
                    tbody.innerHTML = '';
                    
                    // Show only top 10 bins with trades
                    const activeBins = data.filter(bin => bin.total_trades > 0)
                                          .sort((a, b) => b.total_trades - a.total_trades)
                                          .slice(0, 10);
                    
                    activeBins.forEach(bin => {
                        const row = document.createElement('tr');
                        const multiplierClass = bin.bayesian_multiplier > 1.5 ? 'text-success' : 
                                              bin.bayesian_multiplier < 0.8 ? 'text-danger' : '';
                        
                        row.innerHTML = `
                            <td>${bin.bin_range}</td>
                            <td>${bin.total_trades}</td>
                            <td>${bin.win_rate_pct.toFixed(1)}%</td>
                            <td class="${multiplierClass}">${bin.bayesian_multiplier.toFixed(2)}x</td>
                        `;
                        tbody.appendChild(row);
                    });
                    
                    if (activeBins.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No trading data yet</td></tr>';
                    }
                })
                .catch(error => console.error('Error updating Bayesian:', error));
        }
        
        function updateClusters() {
            fetch('/api/clusters')
                .then(response => response.json())
                .then(data => {
                    // Update thresholds
                    document.getElementById('volumeThreshold').textContent = data.thresholds.volume_threshold + 'x';
                    document.getElementById('signalThreshold').textContent = data.thresholds.signal_threshold;
                    document.getElementById('longThreshold').textContent = '≤ ' + data.thresholds.long_threshold;
                    document.getElementById('retestTolerance').textContent = data.thresholds.retest_tolerance + ' pts';
                    document.getElementById('retestTimeout').textContent = data.thresholds.retest_timeout + ' min';
                    
                    // Update stats
                    document.getElementById('totalClusters').textContent = data.signal_stats.total_signals || 0;
                    document.getElementById('avgVolumeRatio').textContent = (data.volume_stats.avg_volume_ratio || 0).toFixed(1) + 'x';
                    document.getElementById('avgSignalStrength').textContent = (data.signal_stats.avg_signal_strength || 0).toFixed(3);
                    document.getElementById('strongSignals').textContent = data.signal_stats.strong_signals || 0;
                    
                    // Update recent clusters table
                    const tbody = document.getElementById('clustersTableBody');
                    tbody.innerHTML = '';
                    
                    data.recent_clusters.forEach(cluster => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${cluster.cluster_time_formatted}</td>
                            <td>${cluster.symbol}</td>
                            <td>${cluster.volume_ratio.toFixed(1)}x</td>
                            <td>${cluster.signal_strength.toFixed(3)}</td>
                            <td>${cluster.modal_position.toFixed(3)}</td>
                            <td><span class="badge bg-${cluster.direction === 'long' ? 'success' : 'danger'}">${cluster.direction.toUpperCase()}</span></td>
                            <td><span class="badge bg-${cluster.outcome_class}">${cluster.outcome}</span></td>
                        `;
                        tbody.appendChild(row);
                    });
                    
                    if (data.recent_clusters.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No recent clusters</td></tr>';
                    }
                })
                .catch(error => console.error('Error updating clusters:', error));
        }
        
        function updatePnLChart() {
            fetch('/api/charts/pnl')
                .then(response => response.json())
                .then(data => {
                    if (data.chart) {
                        const chart = JSON.parse(data.chart);
                        Plotly.newPlot('pnlChart', chart.data, chart.layout, {responsive: true});
                    }
                })
                .catch(error => console.error('Error updating P&L chart:', error));
        }
        
        function updateAll() {
            updatePortfolio();
            updateTrades();
            updateBayesian();
            updateClusters();
            updatePnLChart();
            
            document.getElementById('lastUpdate').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            updateAll();
            refreshInterval = setInterval(updateAll, ''' + str(REFRESH_INTERVAL) + ''');
        });
        
        // Handle page visibility to pause/resume updates
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                clearInterval(refreshInterval);
            } else {
                updateAll();
                refreshInterval = setInterval(updateAll, ''' + str(REFRESH_INTERVAL) + ''');
            }
        });
    </script>
</body>
</html>'''
        
        with open(template_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📄 Created dashboard template: {template_path}")


if __name__ == '__main__':
    logger.info("🚀 Starting Algorithmic Trading Dashboard")
    logger.info("=" * 60)
    logger.info(f"📊 Database: {DATABASE_FILE}")
    logger.info(f"🔄 Refresh interval: {REFRESH_INTERVAL/1000}s")
    logger.info(f"📈 Max chart days: {MAX_CHART_DAYS}")
    logger.info(f"📋 Max trades display: {MAX_TRADES_DISPLAY}")
    logger.info("=" * 60)
    
    # Create templates directory and HTML file
    create_templates()
    
    # Check if database exists
    if os.path.exists(DATABASE_FILE):
        logger.info("✅ Database found - dashboard ready")
    else:
        logger.warning("⚠️ Database not found - dashboard will show empty data")
        logger.info("💡 Run the trading system (01_connect.py) to generate data")
    
    logger.info("🌐 Dashboard will be available at: http://localhost:8080")
    logger.info("🛑 Press Ctrl+C to stop the dashboard")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections for DigitalOcean deployment
        port=8080,
        debug=False,  # Set to False for production
        threaded=True
    )
