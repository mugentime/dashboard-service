from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import asyncio
import json
from datetime import datetime
from typing import Dict, List
import logging

from config.config import config

logger = logging.getLogger(__name__)

class WebDashboard:
    """Web dashboard for real-time trading bot monitoring"""

    def __init__(self, trading_engine=None, hive_mind=None, performance_tracker=None):
        self.app = FastAPI(title="Autonomous Trading Bot Dashboard")
        self.trading_engine = trading_engine
        self.hive_mind = hive_mind
        self.performance_tracker = performance_tracker
        self.active_connections: List[WebSocket] = []

        self._setup_routes()
        self._setup_static_files()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self._get_dashboard_html()

        @self.app.get("/api/status")
        async def get_status():
            """Get current bot status"""
            try:
                status = {
                    "timestamp": datetime.now().isoformat(),
                    "bot_running": self.trading_engine.running if self.trading_engine else False,
                    "hive_mind_active": self.hive_mind.coordination_active if self.hive_mind else False,
                }

                if self.trading_engine:
                    trading_status = await self.trading_engine.get_status()
                    status.update(trading_status)

                if self.hive_mind:
                    hive_status = await self.hive_mind.get_swarm_status()
                    status["hive_mind"] = hive_status

                return status
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                return {"error": str(e)}

        @self.app.get("/api/performance")
        async def get_performance():
            """Get performance metrics"""
            try:
                if not self.performance_tracker:
                    return {"error": "Performance tracker not available"}

                daily_perf = await self.performance_tracker.calculate_daily_performance()
                portfolio_value = 10000  # Default value

                if self.trading_engine and self.trading_engine.portfolio_manager:
                    portfolio_value = await self.trading_engine.portfolio_manager.get_total_value()

                portfolio_metrics = await self.performance_tracker.calculate_portfolio_metrics(portfolio_value)

                return {
                    "daily_performance": daily_perf,
                    "portfolio_metrics": portfolio_metrics,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting performance: {e}")
                return {"error": str(e)}

        @self.app.get("/api/trades")
        async def get_recent_trades():
            """Get recent trades"""
            try:
                if not self.performance_tracker:
                    return {"trades": [], "count": 0}

                recent_trades = self.performance_tracker.trade_history[-50:]  # Last 50 trades
                return {
                    "trades": [
                        {
                            **trade,
                            "timestamp": trade["timestamp"].isoformat() if hasattr(trade["timestamp"], "isoformat") else str(trade["timestamp"])
                        }
                        for trade in recent_trades
                    ],
                    "count": len(recent_trades)
                }
            except Exception as e:
                logger.error(f"Error getting trades: {e}")
                return {"error": str(e)}

        @self.app.get("/api/strategies")
        async def get_strategies():
            """Get active strategies"""
            try:
                if not self.trading_engine:
                    return {"strategies": {}}

                return {
                    "active_strategies": self.trading_engine.active_strategies,
                    "strategy_count": len(self.trading_engine.active_strategies)
                }
            except Exception as e:
                logger.error(f"Error getting strategies: {e}")
                return {"error": str(e)}

        @self.app.get("/api/hive-mind")
        async def get_hive_mind_status():
            """Get detailed hive-mind status"""
            try:
                if not self.hive_mind:
                    return {"error": "Hive-mind not available"}

                status = await self.hive_mind.get_swarm_status()
                return status
            except Exception as e:
                logger.error(f"Error getting hive-mind status: {e}")
                return {"error": str(e)}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            self.active_connections.append(websocket)

            try:
                while True:
                    # Send real-time updates every 5 seconds
                    await asyncio.sleep(5)

                    status_data = {
                        "type": "status_update",
                        "data": await self._get_real_time_data()
                    }

                    await websocket.send_text(json.dumps(status_data))

            except WebSocketDisconnect:
                self.active_connections.remove(websocket)

        @self.app.post("/api/control/start")
        async def start_bot():
            """Start the trading bot"""
            try:
                if self.trading_engine and not self.trading_engine.running:
                    # Start bot in background task
                    asyncio.create_task(self.trading_engine.start())
                    return {"message": "Bot starting...", "status": "success"}
                else:
                    return {"message": "Bot already running", "status": "info"}
            except Exception as e:
                return {"error": str(e), "status": "error"}

        @self.app.post("/api/control/stop")
        async def stop_bot():
            """Stop the trading bot"""
            try:
                if self.trading_engine and self.trading_engine.running:
                    await self.trading_engine.stop()
                    return {"message": "Bot stopped", "status": "success"}
                else:
                    return {"message": "Bot not running", "status": "info"}
            except Exception as e:
                return {"error": str(e), "status": "error"}

    async def _get_real_time_data(self) -> Dict:
        """Get real-time data for WebSocket updates"""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "bot_running": self.trading_engine.running if self.trading_engine else False,
            }

            if self.performance_tracker:
                daily_perf = await self.performance_tracker.calculate_daily_performance()
                data["daily_performance"] = daily_perf

            if self.trading_engine and self.trading_engine.portfolio_manager:
                portfolio_state = self.trading_engine.portfolio_manager.get_portfolio_state()
                data["portfolio"] = {
                    "total_value": portfolio_state["total_value"],
                    "unrealized_pnl": portfolio_state["unrealized_pnl"],
                    "num_positions": portfolio_state["num_positions"]
                }

            return data
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return {"error": str(e)}

    def _setup_static_files(self):
        """Setup static file serving"""
        # We'll serve static files from a static directory
        try:
            self.app.mount("/static", StaticFiles(directory="static"), name="static")
        except:
            # Create static directory if it doesn't exist
            pass

    def _get_dashboard_html(self) -> str:
        """Return the dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Trading Bot Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        .header {
            background: rgba(0,0,0,0.2);
            padding: 1rem 2rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .header h1 {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ff4444;
            animation: pulse 2s infinite;
        }
        .status-dot.active { background: #44ff44; }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .dashboard {
            padding: 2rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
        }

        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }

        .card h3 {
            margin-bottom: 1rem;
            color: #ffd700;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            margin: 0.5rem 0;
            padding: 0.5rem;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
        }

        .metric-value {
            font-weight: bold;
            color: #4ade80;
        }

        .metric-value.negative { color: #f87171; }

        .controls {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }

        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.2s;
        }

        .btn:hover { transform: translateY(-2px); }
        .btn-start { background: #4ade80; color: black; }
        .btn-stop { background: #f87171; color: white; }

        .trades-list {
            max-height: 300px;
            overflow-y: auto;
        }

        .trade-item {
            padding: 0.5rem;
            margin: 0.5rem 0;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
            font-size: 0.9rem;
        }

        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 1rem;
        }

        .hive-agents {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .agent {
            text-align: center;
            padding: 1rem;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }

        .agent-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .loading {
            text-align: center;
            padding: 2rem;
            color: #ffd700;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            🤖 Autonomous Trading Bot
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText">Connecting...</span>
        </h1>
    </div>

    <div class="dashboard">
        <!-- Performance Card -->
        <div class="card">
            <h3>📊 Performance</h3>
            <div class="metric">
                <span>Daily Return:</span>
                <span class="metric-value" id="dailyReturn">0.00%</span>
            </div>
            <div class="metric">
                <span>Portfolio Value:</span>
                <span class="metric-value" id="portfolioValue">$0.00</span>
            </div>
            <div class="metric">
                <span>Unrealized P&L:</span>
                <span class="metric-value" id="unrealizedPnl">$0.00</span>
            </div>
            <div class="metric">
                <span>Active Positions:</span>
                <span class="metric-value" id="activePositions">0</span>
            </div>
            <div class="metric">
                <span>Daily Target Progress:</span>
                <span class="metric-value" id="targetProgress">0%</span>
            </div>
        </div>

        <!-- Controls Card -->
        <div class="card">
            <h3>⚡ Controls</h3>
            <div class="controls">
                <button class="btn btn-start" onclick="startBot()">Start Bot</button>
                <button class="btn btn-stop" onclick="stopBot()">Stop Bot</button>
            </div>
            <div class="metric" style="margin-top: 1rem;">
                <span>Bot Status:</span>
                <span class="metric-value" id="botStatus">Stopped</span>
            </div>
            <div class="metric">
                <span>Hive-Mind:</span>
                <span class="metric-value" id="hiveMindStatus">Inactive</span>
            </div>
        </div>

        <!-- Hive-Mind Agents -->
        <div class="card">
            <h3>🧠 Hive-Mind Collective Intelligence</h3>
            <div class="hive-agents">
                <div class="agent">
                    <div class="agent-icon">🎯</div>
                    <div>Strategy Generator</div>
                </div>
                <div class="agent">
                    <div class="agent-icon">📡</div>
                    <div>Market Intelligence</div>
                </div>
                <div class="agent">
                    <div class="agent-icon">⚡</div>
                    <div>Execution Engine</div>
                </div>
                <div class="agent">
                    <div class="agent-icon">🛡️</div>
                    <div>Risk Guardian</div>
                </div>
                <div class="agent">
                    <div class="agent-icon">📈</div>
                    <div>Performance Analyzer</div>
                </div>
                <div class="agent">
                    <div class="agent-icon">🧬</div>
                    <div>Self-Evolution</div>
                </div>
            </div>
        </div>

        <!-- Recent Trades -->
        <div class="card">
            <h3>📋 Recent Trades</h3>
            <div class="trades-list" id="tradesList">
                <div class="loading">Loading trades...</div>
            </div>
        </div>

        <!-- Performance Chart -->
        <div class="card" style="grid-column: 1 / -1;">
            <h3>📈 Performance Chart</h3>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let performanceChart;

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
            initChart();
            loadInitialData();
        });

        function initWebSocket() {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);

            ws.onopen = function() {
                document.getElementById('statusDot').classList.add('active');
                document.getElementById('statusText').textContent = 'Connected';
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'status_update') {
                    updateDashboard(data.data);
                }
            };

            ws.onclose = function() {
                document.getElementById('statusDot').classList.remove('active');
                document.getElementById('statusText').textContent = 'Disconnected';
                // Reconnect after 5 seconds
                setTimeout(initWebSocket, 5000);
            };
        }

        function initChart() {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [],
                        borderColor: '#4ade80',
                        backgroundColor: 'rgba(74, 222, 128, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: 'white' } }
                    },
                    scales: {
                        x: { ticks: { color: 'white' } },
                        y: { ticks: { color: 'white' } }
                    }
                }
            });
        }

        async function loadInitialData() {
            try {
                // Load performance data
                const perfResponse = await fetch('/api/performance');
                const perfData = await perfResponse.json();
                updatePerformanceDisplay(perfData);

                // Load trades
                const tradesResponse = await fetch('/api/trades');
                const tradesData = await tradesResponse.json();
                updateTradesDisplay(tradesData);

                // Load status
                const statusResponse = await fetch('/api/status');
                const statusData = await statusResponse.json();
                updateStatusDisplay(statusData);

            } catch (error) {
                console.error('Error loading initial data:', error);
            }
        }

        function updateDashboard(data) {
            if (data.daily_performance) {
                updatePerformanceDisplay({ daily_performance: data.daily_performance });
            }
            if (data.portfolio) {
                updatePortfolioDisplay(data.portfolio);
            }
            updateStatusDisplay({ bot_running: data.bot_running });
        }

        function updatePerformanceDisplay(data) {
            if (data.daily_performance) {
                const perf = data.daily_performance;
                document.getElementById('dailyReturn').textContent = `${(perf.return * 100).toFixed(2)}%`;
                document.getElementById('targetProgress').textContent = `${(perf.target_progress * 100).toFixed(1)}%`;

                // Update colors based on performance
                const returnEl = document.getElementById('dailyReturn');
                returnEl.className = `metric-value ${perf.return >= 0 ? '' : 'negative'}`;
            }
        }

        function updatePortfolioDisplay(portfolio) {
            document.getElementById('portfolioValue').textContent = `$${portfolio.total_value.toFixed(2)}`;
            document.getElementById('unrealizedPnl').textContent = `$${portfolio.unrealized_pnl.toFixed(2)}`;
            document.getElementById('activePositions').textContent = portfolio.num_positions;

            // Update unrealized P&L color
            const pnlEl = document.getElementById('unrealizedPnl');
            pnlEl.className = `metric-value ${portfolio.unrealized_pnl >= 0 ? '' : 'negative'}`;
        }

        function updateStatusDisplay(status) {
            document.getElementById('botStatus').textContent = status.bot_running ? 'Running' : 'Stopped';
            const statusEl = document.getElementById('botStatus');
            statusEl.className = `metric-value ${status.bot_running ? '' : 'negative'}`;
        }

        function updateTradesDisplay(data) {
            const tradesContainer = document.getElementById('tradesList');

            if (data.error || !data.trades || data.trades.length === 0) {
                tradesContainer.innerHTML = '<div class="loading">No trades yet</div>';
                return;
            }

            tradesContainer.innerHTML = data.trades.slice(-10).reverse().map(trade => `
                <div class="trade-item">
                    <strong>${trade.symbol}</strong> ${trade.side}
                    ${trade.quantity?.toFixed(4)} @ $${trade.price?.toFixed(2)}
                    <br><small>${new Date(trade.timestamp).toLocaleTimeString()}</small>
                </div>
            `).join('');
        }

        async function startBot() {
            try {
                const response = await fetch('/api/control/start', { method: 'POST' });
                const result = await response.json();
                alert(result.message);
            } catch (error) {
                alert('Error starting bot: ' + error.message);
            }
        }

        async function stopBot() {
            try {
                const response = await fetch('/api/control/stop', { method: 'POST' });
                const result = await response.json();
                alert(result.message);
            } catch (error) {
                alert('Error stopping bot: ' + error.message);
            }
        }

        // Refresh data every 30 seconds
        setInterval(loadInitialData, 30000);
    </script>
</body>
</html>
        """

    async def broadcast_update(self, data: Dict):
        """Broadcast update to all connected WebSocket clients"""
        if self.active_connections:
            message = json.dumps({"type": "update", "data": data})
            disconnected = []

            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except:
                    disconnected.append(connection)

            # Remove disconnected clients
            for connection in disconnected:
                self.active_connections.remove(connection)

# Global dashboard instance
dashboard = None

def create_dashboard(trading_engine=None, hive_mind=None, performance_tracker=None):
    """Create dashboard instance"""
    global dashboard
    dashboard = WebDashboard(trading_engine, hive_mind, performance_tracker)
    return dashboard