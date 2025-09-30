#!/bin/bash
# Deployment script using SSH key for Algorithmic Trading System
# Usage: ./deploy_with_key.sh

set -e  # Exit on any error

# Configuration - Load from environment variables
source .env
DROPLET_IP="${DROPLET_IP}"
DROPLET_USER="root"
REMOTE_DIR="/root/algo_trader"
LOCAL_DIR="/Users/albertbeccu/Library/CloudStorage/OneDrive-Personal/NordicOwl/Thoendel/New Algo Trader/New_Algo"
SSH_KEY="${DROPLET_SSH_KEY_PATH}"

echo "🚀 Deploying Algorithmic Trading System to DigitalOcean Droplet"
echo "=================================================="
echo "Droplet: ${DROPLET_USER}@${DROPLET_IP}"
echo "Remote Directory: ${REMOTE_DIR}"
echo "Local Directory: ${LOCAL_DIR}"
echo "SSH Key: ${SSH_KEY}"
echo "=================================================="

# Check if .env file exists
if [ ! -f "${LOCAL_DIR}/.env" ]; then
    echo "❌ ERROR: .env file not found!"
    echo "Please create a .env file with your API keys before deploying."
    exit 1
fi

echo "✅ .env file found"

# Test SSH connection
echo "🔑 Testing SSH connection..."
ssh -i ${SSH_KEY} -o ConnectTimeout=10 ${DROPLET_USER}@${DROPLET_IP} "echo 'SSH connection successful'" || {
    echo "❌ SSH connection failed!"
    echo "Please run ./deploy_manual.sh for setup instructions"
    exit 1
}

# Create remote directory structure
echo "📁 Creating remote directory structure..."
ssh -i ${SSH_KEY} ${DROPLET_USER}@${DROPLET_IP} "mkdir -p ${REMOTE_DIR}/{Databases,templates,logs}"

# Transfer core Python files
echo "📦 Transferring core application files..."
scp -i ${SSH_KEY} "${LOCAL_DIR}/01_connect.py" ${DROPLET_USER}@${DROPLET_IP}:${REMOTE_DIR}/
scp -i ${SSH_KEY} "${LOCAL_DIR}/02_signal.py" ${DROPLET_USER}@${DROPLET_IP}:${REMOTE_DIR}/
scp -i ${SSH_KEY} "${LOCAL_DIR}/03_trader.py" ${DROPLET_USER}@${DROPLET_IP}:${REMOTE_DIR}/
scp -i ${SSH_KEY} "${LOCAL_DIR}/04_dashboard.py" ${DROPLET_USER}@${DROPLET_IP}:${REMOTE_DIR}/

# Transfer configuration files
echo "🔧 Transferring configuration files..."
scp -i ${SSH_KEY} "${LOCAL_DIR}/requirements.txt" ${DROPLET_USER}@${DROPLET_IP}:${REMOTE_DIR}/
scp -i ${SSH_KEY} "${LOCAL_DIR}/.env" ${DROPLET_USER}@${DROPLET_IP}:${REMOTE_DIR}/

# Transfer templates directory
echo "🎨 Transferring templates..."
if [ -d "${LOCAL_DIR}/templates" ]; then
    scp -i ${SSH_KEY} -r "${LOCAL_DIR}/templates/"* ${DROPLET_USER}@${DROPLET_IP}:${REMOTE_DIR}/templates/
fi

# Transfer existing database if it exists
echo "🗄️ Transferring database (if exists)..."
if [ -f "${LOCAL_DIR}/Databases/trading_system.db" ]; then
    scp -i ${SSH_KEY} "${LOCAL_DIR}/Databases/trading_system.db" ${DROPLET_USER}@${DROPLET_IP}:${REMOTE_DIR}/Databases/
    echo "✅ Database transferred"
else
    echo "ℹ️ No existing database found - will be created on first run"
fi

# Install system dependencies and setup Python environment
echo "🔧 Setting up remote environment..."
ssh -i ${SSH_KEY} ${DROPLET_USER}@${DROPLET_IP} << 'EOF'
# Update system
apt update && apt upgrade -y

# Install Python 3 and pip if not already installed
apt install -y python3 python3-pip python3-venv

# Install system dependencies for Python packages
apt install -y build-essential libssl-dev libffi-dev python3-dev

# Create virtual environment
cd /root/algo_trader
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Remote environment setup complete"
EOF

# Set proper permissions for database directory
echo "🔒 Setting up permissions..."
ssh -i ${SSH_KEY} ${DROPLET_USER}@${DROPLET_IP} << 'EOF'
cd /root/algo_trader
chmod -R 755 .
chmod 766 Databases/
if [ -f "Databases/trading_system.db" ]; then
    chmod 666 Databases/trading_system.db
fi
chmod +x *.py
echo "✅ Permissions configured"
EOF

# Update the database path in Python files for production
echo "⚙️ Configuring for production environment..."
ssh -i ${SSH_KEY} ${DROPLET_USER}@${DROPLET_IP} << 'EOF'
cd /root/algo_trader

# Update database paths in Python files
sed -i 's|/Users/albertbeccu/Library/CloudStorage/OneDrive-Personal/NordicOwl/Thoendel/New Algo Trader/New_Algo/Databases|/root/algo_trader/Databases|g' 03_trader.py
sed -i 's|/Users/albertbeccu/Library/CloudStorage/OneDrive-Personal/NordicOwl/Thoendel/New Algo Trader/New_Algo/Databases|/root/algo_trader/Databases|g' 04_dashboard.py

# Update the dynamic import paths in 02_signal.py
sed -i 's|/Users/albertbeccu/Library/CloudStorage/OneDrive-Personal/NordicOwl/Thoendel/New Algo Trader/New_Algo/03_trader.py|/root/algo_trader/03_trader.py|g' 02_signal.py

echo "✅ Production paths configured"
EOF

# Create the launch script on the remote server
echo "🚀 Creating launch script on remote server..."
ssh -i ${SSH_KEY} ${DROPLET_USER}@${DROPLET_IP} << 'EOF'
cd /root/algo_trader

cat > launch_system.sh << 'LAUNCH_EOF'
#!/bin/bash
# Launch script for Algorithmic Trading System
# This script starts all components of the trading system

set -e

WORK_DIR="/root/algo_trader"
LOG_DIR="${WORK_DIR}/logs"

echo "🚀 Starting Algorithmic Trading System"
echo "======================================"
echo "Work Directory: ${WORK_DIR}"
echo "Log Directory: ${LOG_DIR}"
echo "======================================"

# Change to work directory
cd ${WORK_DIR}

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p ${LOG_DIR}

# Function to check if a process is running
is_running() {
    pgrep -f "$1" > /dev/null 2>&1
}

# Function to stop existing processes
stop_processes() {
    echo "🛑 Stopping existing processes..."
    pkill -f "01_connect.py" 2>/dev/null || true
    pkill -f "04_dashboard.py" 2>/dev/null || true
    sleep 2
    echo "✅ Existing processes stopped"
}

# Function to start data connector
start_connector() {
    echo "📡 Starting Data Connector (01_connect.py)..."
    nohup python 01_connect.py > ${LOG_DIR}/connector.log 2>&1 &
    CONNECTOR_PID=$!
    echo "✅ Data Connector started (PID: ${CONNECTOR_PID})"
    echo ${CONNECTOR_PID} > ${LOG_DIR}/connector.pid
}

# Function to start dashboard
start_dashboard() {
    echo "📊 Starting Dashboard (04_dashboard.py)..."
    nohup python 04_dashboard.py > ${LOG_DIR}/dashboard.log 2>&1 &
    DASHBOARD_PID=$!
    echo "✅ Dashboard started (PID: ${DASHBOARD_PID})"
    echo ${DASHBOARD_PID} > ${LOG_DIR}/dashboard.pid
    echo "🌐 Dashboard will be available at: http://${DROPLET_IP}:8080"
}

# Function to show status
show_status() {
    echo ""
    echo "📊 System Status:"
    echo "=================="
    
    if is_running "01_connect.py"; then
        echo "✅ Data Connector: RUNNING"
    else
        echo "❌ Data Connector: STOPPED"
    fi
    
    if is_running "04_dashboard.py"; then
        echo "✅ Dashboard: RUNNING"
    else
        echo "❌ Dashboard: STOPPED"
    fi
    
    echo ""
    echo "📋 Process Information:"
    ps aux | grep -E "(01_connect|04_dashboard)" | grep -v grep || echo "No processes found"
    
    echo ""
    echo "📁 Log Files:"
    ls -la ${LOG_DIR}/ 2>/dev/null || echo "No log files yet"
    
    echo ""
    echo "🌐 Access URLs:"
    echo "   Dashboard: http://${DROPLET_IP}:8080"
    echo ""
    echo "📖 Useful Commands:"
    echo "   View connector logs: tail -f ${LOG_DIR}/connector.log"
    echo "   View dashboard logs: tail -f ${LOG_DIR}/dashboard.log"
    echo "   Stop system: ./stop_system.sh"
    echo "   Check status: ./launch_system.sh status"
}

# Main execution
case "${1:-start}" in
    start)
        stop_processes
        sleep 3
        start_connector
        sleep 5  # Give connector time to initialize
        start_dashboard
        sleep 2
        show_status
        ;;
    stop)
        stop_processes
        echo "✅ System stopped"
        ;;
    restart)
        stop_processes
        sleep 3
        start_connector
        sleep 5
        start_dashboard
        sleep 2
        show_status
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all system components"
        echo "  stop    - Stop all system components"
        echo "  restart - Restart all system components"
        echo "  status  - Show current system status"
        exit 1
        ;;
esac
LAUNCH_EOF

chmod +x launch_system.sh
echo "✅ Launch script created"
EOF

# Create stop script
ssh -i ${SSH_KEY} ${DROPLET_USER}@${DROPLET_IP} << 'EOF'
cd /root/algo_trader

cat > stop_system.sh << 'STOP_EOF'
#!/bin/bash
# Stop script for Algorithmic Trading System

echo "🛑 Stopping Algorithmic Trading System..."

# Kill processes
pkill -f "01_connect.py" 2>/dev/null || true
pkill -f "04_dashboard.py" 2>/dev/null || true

# Remove PID files
rm -f logs/connector.pid logs/dashboard.pid 2>/dev/null || true

echo "✅ System stopped"
STOP_EOF

chmod +x stop_system.sh
echo "✅ Stop script created"
EOF

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "======================="
echo ""
echo "📍 Your trading system has been deployed to:"
echo "   Server: ${DROPLET_USER}@${DROPLET_IP}"
echo "   Directory: ${REMOTE_DIR}"
echo ""
echo "🚀 To start the system, SSH to your droplet and run:"
echo "   ssh -i ${SSH_KEY} ${DROPLET_USER}@${DROPLET_IP}"
echo "   cd ${REMOTE_DIR}"
echo "   ./launch_system.sh"
echo ""
echo "🌐 Once started, access the dashboard at:"
echo "   http://${DROPLET_IP}:8080"
echo ""
echo "📖 Available commands on the droplet:"
echo "   ./launch_system.sh start    - Start the system"
echo "   ./launch_system.sh stop     - Stop the system"
echo "   ./launch_system.sh restart  - Restart the system"
echo "   ./launch_system.sh status   - Check system status"
echo "   ./stop_system.sh            - Quick stop"
echo ""
echo "📊 Monitor logs with:"
echo "   tail -f logs/connector.log   - Data connector logs"
echo "   tail -f logs/dashboard.log   - Dashboard logs"
echo ""
echo "🔧 The system is configured for production with:"
echo "   ✅ Proper file permissions for SQLite database"
echo "   ✅ Virtual environment with all dependencies"
echo "   ✅ Background process management"
echo "   ✅ Comprehensive logging"
echo "   ✅ Dashboard accessible from external IP"
echo ""
