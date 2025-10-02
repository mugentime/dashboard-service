#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Optimizer Bot Monitoring Dashboard
READ-ONLY dashboard that displays bot activity from Redis
"""
import os
import sys
import io
import json
from datetime import datetime, timedelta

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import redis
from binance.client import Client

# Load environment variables
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Initialize Redis connection
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_connected = redis_client.ping()
    print(f"✅ Redis connected: {REDIS_URL}")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    redis_client = None
    redis_connected = False

# Initialize Binance client (READ-ONLY for balance)
print(f"🔑 Binance API Key present: {bool(BINANCE_API_KEY)}")
print(f"🔑 Binance API Secret present: {bool(BINANCE_API_SECRET)}")
if BINANCE_API_KEY:
    print(f"🔑 API Key (first 8 chars): {BINANCE_API_KEY[:8]}...")
if BINANCE_API_SECRET:
    print(f"🔑 API Secret (first 8 chars): {BINANCE_API_SECRET[:8]}...")

# Check if using testnet
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
print(f"🌐 Binance Testnet Mode: {BINANCE_TESTNET}")

def initialize_binance_client(max_retries=3):
    """Initialize Binance client with retry logic and detailed error logging"""
    for attempt in range(max_retries):
        try:
            if not BINANCE_API_KEY or not BINANCE_API_SECRET:
                error_msg = "Binance API credentials not found in environment variables"
                print(f"❌ {error_msg}")
                print("⚠️  Please set BINANCE_API_KEY and BINANCE_API_SECRET in Railway environment variables")
                return None

            print(f"🔄 Initializing Binance client (attempt {attempt + 1}/{max_retries})...")

            # Initialize client with testnet support
            if BINANCE_TESTNET:
                print("🧪 Using Binance TESTNET endpoints")
                client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)
            else:
                print("🌍 Using Binance MAINNET endpoints")
                client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

            # Sync time with retry
            import time
            print("⏰ Syncing server time...")
            server_time = client.get_server_time()
            time_offset = server_time['serverTime'] - int(time.time() * 1000)
            client.timestamp_offset = time_offset
            print(f"✅ Time offset synchronized: {time_offset}ms")

            # Test connection by fetching account info
            print("🔍 Testing API connection with account info request...")
            try:
                account_info = client.futures_account()
                balance = float(account_info.get('totalWalletBalance', 0))
                print(f"✅ Binance API connected successfully!")
                print(f"💰 Account Balance: ${balance:.2f} USDT")
                print(f"📊 Can Place Orders: {account_info.get('canTrade', False)}")
                print(f"🔐 API Permissions: Trade={account_info.get('canTrade')}, Withdraw={account_info.get('canWithdraw')}")
                return client
            except Exception as test_error:
                print(f"⚠️  Connection test failed: {type(test_error).__name__}: {test_error}")

                # Check for common error patterns
                error_str = str(test_error).lower()
                if 'timestamp' in error_str:
                    print("❌ ERROR: Timestamp synchronization issue detected")
                    print("🔧 Possible fix: Server time might be out of sync")
                elif 'signature' in error_str or 'invalid' in error_str:
                    print("❌ ERROR: API signature validation failed")
                    print("🔧 Possible fix: Check if API Key and Secret are correct")
                    print("🔧 Verify API keys are for the correct environment (testnet vs mainnet)")
                elif 'permission' in error_str or 'forbidden' in error_str:
                    print("❌ ERROR: API permission denied")
                    print("🔧 Possible fix: Enable Futures trading permission in Binance API settings")
                elif 'ip' in error_str:
                    print("❌ ERROR: IP address not whitelisted")
                    print("🔧 Possible fix: Add Railway's IP to API whitelist or remove IP restrictions")
                elif 'rate' in error_str or 'limit' in error_str:
                    print("❌ ERROR: Rate limit exceeded")
                    print("🔧 Waiting before retry...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"❌ ERROR: Unknown error - {test_error}")
                    print(f"🔍 Full error details: {repr(test_error)}")

                # If not last attempt, retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ All {max_retries} connection attempts failed")
                    return None

        except Exception as e:
            print(f"❌ Binance client initialization failed (attempt {attempt + 1}/{max_retries})")
            print(f"❌ Error Type: {type(e).__name__}")
            print(f"❌ Error Message: {e}")
            import traceback
            print(f"❌ Stack Trace:")
            traceback.print_exc()

            if attempt < max_retries - 1:
                import time
                wait_time = 2 ** attempt
                print(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ All {max_retries} initialization attempts failed")
                return None

    return None

# Initialize Binance client with retry logic
binance_client = initialize_binance_client()

# Initialize Dash app
app = dash.Dash(__name__, title="Self-Optimizer Bot Monitor")

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("🤖 Self-Optimizer Bot Monitor",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Read-only monitoring dashboard [v2.0 - Live Binance Data]",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px'}),

        # Status Row
        html.Div([
            html.Div([
                html.H3("Bot Status", style={'color': '#27ae60', 'marginBottom': '10px'}),
                html.Div(id="bot-status", style={'fontSize': '16px', 'fontWeight': 'bold'})
            ], className="status-card", style={'width': '19%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),

            html.Div([
                html.H3("Balance", style={'color': '#3498db', 'marginBottom': '10px'}),
                html.Div(id="balance-display", style={'fontSize': '16px', 'fontWeight': 'bold'})
            ], className="status-card", style={'width': '19%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),

            html.Div([
                html.H3("Active Trades", style={'color': '#e74c3c', 'marginBottom': '10px'}),
                html.Div(id="active-trades", style={'fontSize': '16px', 'fontWeight': 'bold'})
            ], className="status-card", style={'width': '19%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),

            html.Div([
                html.H3("Optimization Cycle", style={'color': '#9b59b6', 'marginBottom': '10px'}),
                html.Div(id="optimization-cycle", style={'fontSize': '16px', 'fontWeight': 'bold'})
            ], className="status-card", style={'width': '19%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),

            html.Div([
                html.H3("Redis Status", style={'color': '#f39c12', 'marginBottom': '10px'}),
                html.Div(id="supervisor-status", style={'fontSize': '14px', 'fontWeight': 'bold'})
            ], className="status-card", style={'width': '19%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
        ], style={'marginBottom': '20px'}),

        # Charts Row 1
        html.Div([
            html.Div([
                dcc.Graph(id="balance-chart")
            ], style={'width': '50%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(id="optimization-params-chart")
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),

        # Charts Row 2
        html.Div([
            html.Div([
                dcc.Graph(id="baseline-comparison")
            ], style={'width': '50%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(id="supervisor-actions")
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),

        # Recent Trades Section
        html.Div([
            html.Div([
                html.H3("Recent Open Trades", style={'color': '#2c3e50', 'marginTop': '20px'}),
                html.Div(id="recent-trades-table")
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '1%'}),

            html.Div([
                html.H3("Recently Closed Trades", style={'color': '#2c3e50', 'marginTop': '20px'}),
                html.Div(id="closed-trades-table")
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'})
        ]),

        # Supervisor Detailed Status
        html.Div([
            html.H3("🤖 Supervisor Detailed Status", style={'color': '#2c3e50', 'marginTop': '20px'}),
            html.Div(id="supervisor-detailed-status", style={
                'padding': '15px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '10px',
                'marginBottom': '20px'
            })
        ]),

        # Supervisor Alerts and Actions
        html.Div([
            html.Div([
                html.H3("⚠️ Supervisor Alerts", style={'color': '#2c3e50', 'marginTop': '20px'}),
                html.Div(id="supervisor-alerts-table")
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '1%'}),

            html.Div([
                html.H3("📋 Supervisor Actions", style={'color': '#2c3e50', 'marginTop': '20px'}),
                html.Div(id="supervisor-actions-table")
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'})
        ]),

        # Recent Activity
        html.Div([
            html.H3("Recent Bot Activity", style={'color': '#2c3e50', 'marginTop': '20px'}),
            html.Div(id="bot-activity-log")
        ]),

        # Auto-refresh every 5 seconds
        dcc.Interval(
            id='interval-component',
            interval=5*1000,
            n_intervals=0
        )
    ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})
])

def get_bot_data_from_redis():
    """Get bot data from Redis"""
    if not redis_client or not redis_connected:
        return None

    try:
        # Get bot status
        bot_data = {
            'status': redis_client.get('bot:status') or 'UNKNOWN',
            'balance': float(redis_client.get('bot:balance') or 0),
            'active_trades': int(redis_client.get('bot:active_trades') or 0),
            'optimization_cycle': int(redis_client.get('bot:optimization_cycle') or 0),
            'last_update': redis_client.get('bot:last_update'),
            'adaptive_params': json.loads(redis_client.get('bot:adaptive_params') or '{}'),
            'supervisor_advice': redis_client.get('bot:supervisor_advice'),
            'supervisor_last_update': redis_client.get('bot:supervisor_last_update'),
            'baseline_performance': json.loads(redis_client.get('bot:baseline_performance') or '{}'),
            'current_performance': json.loads(redis_client.get('bot:current_performance') or '{}'),
        }

        # Get activity log (last 20 entries)
        activity = redis_client.lrange('bot:activity_log', 0, 19)
        bot_data['activity_log'] = [json.loads(a) for a in activity]

        # Get performance history
        performance_history = redis_client.lrange('bot:performance_history', 0, 49)
        bot_data['performance_history'] = [json.loads(p) for p in performance_history]

        # Get supervisor alerts (last 20)
        supervisor_alerts = redis_client.lrange('bot:supervisor_alerts', 0, 19)
        bot_data['supervisor_alerts'] = [json.loads(a) for a in supervisor_alerts]

        # Get supervisor actions (last 20)
        supervisor_actions = redis_client.lrange('bot:supervisor_actions', 0, 19)
        bot_data['supervisor_actions'] = [json.loads(a) for a in supervisor_actions]

        return bot_data
    except Exception as e:
        print(f"Error getting bot data from Redis: {e}")
        return None

def get_balance_from_binance():
    """Get real-time balance from Binance (fallback)"""
    if not binance_client:
        return 0.0

    try:
        account = binance_client.futures_account()
        return float(account['totalWalletBalance'])
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return 0.0

@app.callback(
    [Output('bot-status', 'children'),
     Output('balance-display', 'children'),
     Output('active-trades', 'children'),
     Output('optimization-cycle', 'children'),
     Output('supervisor-status', 'children'),
     Output('balance-chart', 'figure'),
     Output('optimization-params-chart', 'figure'),
     Output('baseline-comparison', 'figure'),
     Output('supervisor-actions', 'figure'),
     Output('recent-trades-table', 'children'),
     Output('closed-trades-table', 'children'),
     Output('bot-activity-log', 'children'),
     Output('supervisor-detailed-status', 'children'),
     Output('supervisor-alerts-table', 'children'),
     Output('supervisor-actions-table', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update dashboard with bot data"""

    try:
        # Get bot data from Redis
        bot_data = get_bot_data_from_redis()
    except Exception as e:
        print(f"❌ Error in update_dashboard callback: {e}")
        import traceback
        traceback.print_exc()
        bot_data = None

    if not bot_data:
        # Redis not available - fallback to Binance data
        try:
            balance = get_balance_from_binance()
        except Exception as e:
            print(f"❌ Error fetching balance: {e}")
            balance = 0.0

        # Check if bot is active by looking at recent trades - GET REAL COUNT FROM BINANCE
        try:
            if not binance_client:
                print("❌ No Binance client - API credentials missing")
                bot_status = "🔴 NO API CONNECTION"
                open_positions_count = 0
            else:
                positions = binance_client.futures_position_information()
                # Count only non-zero positions
                open_positions_count = len([p for p in positions if abs(float(p['positionAmt'])) > 0])
                bot_status = f"🟢 ACTIVE - {open_positions_count} positions" if open_positions_count > 0 else "🟡 MONITORING"
                print(f"✅ Found {open_positions_count} open positions (real-time from Binance)")
        except Exception as e:
            print(f"❌ Error checking positions: {e}")
            import traceback
            traceback.print_exc()
            bot_status = "🟡 STATUS UNKNOWN"
            open_positions_count = 0

        try:
            recent_trades_table = create_recent_trades_table()
        except Exception as e:
            print(f"❌ Error creating recent trades table: {e}")
            recent_trades_table = html.Div("Error loading trades", style={'textAlign': 'center', 'color': '#e74c3c'})

        try:
            closed_trades_table = create_closed_trades_table()
        except Exception as e:
            print(f"❌ Error creating closed trades table: {e}")
            closed_trades_table = html.Div("Error loading trades", style={'textAlign': 'center', 'color': '#e74c3c'})

        return (
            bot_status,
            f"${balance:.2f} USDT",
            f"{open_positions_count} positions",
            "N/A - Redis offline",
            "🔴 Redis offline - showing live data only",
            create_empty_chart("Balance Over Time", "Bot data not in Redis yet"),
            create_empty_chart("Optimization Parameters", "Bot data not in Redis yet"),
            create_empty_chart("Baseline vs Current", "Bot data not in Redis yet"),
            create_empty_chart("Supervisor Actions", "Bot data not in Redis yet"),
            recent_trades_table,
            closed_trades_table,
            html.Div([
                html.P("✅ Dashboard showing LIVE trading data from Binance API",
                       style={'color': '#27ae60', 'padding': '10px', 'textAlign': 'center', 'fontWeight': 'bold'}),
                html.P("⚠️ Bot optimization metrics not available - Bot not writing to Redis yet",
                       style={'color': '#f39c12', 'padding': '5px', 'textAlign': 'center', 'fontSize': '14px'})
            ]),
            html.Div("⏳ Waiting for supervisor to connect...", style={'textAlign': 'center', 'color': '#95a5a6'}),
            html.Div("No alerts yet", style={'textAlign': 'center', 'color': '#95a5a6'}),
            html.Div("No actions yet", style={'textAlign': 'center', 'color': '#95a5a6'})
        )

    # Status indicators
    try:
        bot_status = f"🟢 {bot_data['status']}" if bot_data['status'] == 'ACTIVE' else f"🟡 {bot_data['status']}"

        # Get balance (from Redis or Binance fallback)
        balance = bot_data.get('balance', 0) or get_balance_from_binance()
        balance_display = f"${balance:.2f} USDT"

        active_trades = f"{bot_data.get('active_trades', 0)} positions"
        optimization_cycle = f"Cycle #{bot_data.get('optimization_cycle', 0)}"

        supervisor_advice = bot_data.get('supervisor_advice') or "No advice"
        # Make status more informative - show if Redis is connected
        if supervisor_advice and "STALE" in supervisor_advice:
            supervisor_status = "⚠️ Bot not updating Redis"
        elif supervisor_advice and supervisor_advice != "No advice":
            supervisor_status = supervisor_advice[:40] + "..." if len(supervisor_advice) > 40 else supervisor_advice
        else:
            supervisor_status = "✅ Redis connected"

        # Balance chart (from performance history)
        balance_fig = create_balance_chart(bot_data.get('performance_history', []))

        # Optimization parameters chart
        params_fig = create_optimization_params_chart(bot_data.get('adaptive_params', {}))

        # Baseline vs Current comparison
        comparison_fig = create_baseline_comparison(
            bot_data.get('baseline_performance', {}),
            bot_data.get('current_performance', {})
        )

        # Supervisor actions chart
        supervisor_fig = create_supervisor_actions_chart(bot_data.get('activity_log', []))

        # Recent trades tables
        recent_trades_table = create_recent_trades_table()
        closed_trades_table = create_closed_trades_table()

        # Activity log
        activity_log = create_activity_log(bot_data.get('activity_log', []))

        # Supervisor detailed status
        supervisor_detailed = create_supervisor_detailed_status(bot_data)

        # Supervisor alerts table
        supervisor_alerts_table = create_supervisor_alerts_table(bot_data.get('supervisor_alerts', []))

        # Supervisor actions table
        supervisor_actions_table_view = create_supervisor_actions_table(bot_data.get('supervisor_actions', []))

        return (bot_status, balance_display, active_trades, optimization_cycle, supervisor_status,
                balance_fig, params_fig, comparison_fig, supervisor_fig,
                recent_trades_table, closed_trades_table, activity_log,
                supervisor_detailed, supervisor_alerts_table, supervisor_actions_table_view)
    except Exception as e:
        print(f"❌ Error processing bot data: {e}")
        import traceback
        traceback.print_exc()
        # Return fallback data
        return (
            "🔴 ERROR",
            "$0.00 USDT",
            "0 positions",
            "N/A",
            "Error loading data",
            create_empty_chart("Balance Over Time", "Error loading data"),
            create_empty_chart("Optimization Parameters", "Error loading data"),
            create_empty_chart("Baseline vs Current", "Error loading data"),
            create_empty_chart("Supervisor Actions", "Error loading data"),
            html.Div("Error loading trades", style={'textAlign': 'center', 'color': '#e74c3c'}),
            html.Div("Error loading trades", style={'textAlign': 'center', 'color': '#e74c3c'}),
            html.Div("Error loading activity", style={'textAlign': 'center', 'color': '#e74c3c'}),
            html.Div(f"❌ Error: {str(e)}", style={'textAlign': 'center', 'color': '#e74c3c'}),
            html.Div("Error loading alerts", style={'textAlign': 'center', 'color': '#e74c3c'}),
            html.Div("Error loading actions", style={'textAlign': 'center', 'color': '#e74c3c'})
        )

def create_empty_chart(title, message):
    """Create an empty chart with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#95a5a6")
    )
    fig.update_layout(title=title, template="plotly_white", height=250)
    return fig

def create_balance_chart(performance_history):
    """Create balance over time chart"""
    if not performance_history:
        return create_empty_chart("Balance Over Time", "No performance data yet")

    df = pd.DataFrame(performance_history)
    if 'timestamp' in df.columns and 'balance' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['balance'],
            mode='lines+markers',
            name='Balance',
            line=dict(color='#3498db', width=2)
        ))
        fig.update_layout(
            title="Balance Over Time",
            xaxis_title="Time",
            yaxis_title="Balance (USDT)",
            template="plotly_white",
            height=250
        )
        return fig

    return create_empty_chart("Balance Over Time", "Invalid data format")

def create_optimization_params_chart(adaptive_params):
    """Create chart showing current optimization parameters"""
    if not adaptive_params:
        return create_empty_chart("Optimization Parameters", "No parameters yet")

    params_df = pd.DataFrame([
        {'param': k, 'value': v}
        for k, v in adaptive_params.items()
        if isinstance(v, (int, float))
    ])

    if params_df.empty:
        return create_empty_chart("Optimization Parameters", "No numeric parameters")

    fig = px.bar(
        params_df,
        x='param',
        y='value',
        title="Current Optimization Parameters",
        color='value',
        color_continuous_scale='viridis'
    )
    fig.update_layout(template="plotly_white", height=250)
    return fig

def create_baseline_comparison(baseline, current):
    """Create baseline vs current performance comparison"""
    if not baseline or not current:
        return create_empty_chart("Baseline vs Current Performance", "No comparison data yet")

    metrics = list(set(list(baseline.keys()) + list(current.keys())))
    comparison_data = []

    for metric in metrics:
        if metric in baseline and metric in current:
            comparison_data.append({
                'metric': metric,
                'baseline': baseline[metric],
                'current': current[metric],
                'improvement': ((current[metric] - baseline[metric]) / baseline[metric] * 100) if baseline[metric] != 0 else 0
            })

    if not comparison_data:
        return create_empty_chart("Baseline vs Current Performance", "No metrics available")

    df = pd.DataFrame(comparison_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Baseline', x=df['metric'], y=df['baseline'], marker_color='#95a5a6'))
    fig.add_trace(go.Bar(name='Current', x=df['metric'], y=df['current'], marker_color='#3498db'))

    fig.update_layout(
        title="Baseline vs Current Performance",
        barmode='group',
        template="plotly_white",
        height=250
    )
    return fig

def create_supervisor_actions_chart(activity_log):
    """Create chart showing supervisor actions"""
    if not activity_log:
        return create_empty_chart("Supervisor Actions", "No supervisor activity yet")

    # Try both 'type' and 'action_type' fields for compatibility
    supervisor_activities = [
        a for a in activity_log
        if a.get('type') == 'supervisor_action' or a.get('action_type') is not None
    ]

    if not supervisor_activities:
        return create_empty_chart("Supervisor Actions", "No supervisor actions recorded")

    df = pd.DataFrame(supervisor_activities)
    # Use 'action_type' field if available, fallback to 'action'
    action_field = 'action_type' if 'action_type' in df.columns else 'action'

    if action_field not in df.columns:
        return create_empty_chart("Supervisor Actions", "No action data available")

    action_counts = df[action_field].value_counts().reset_index()
    action_counts.columns = ['action', 'count']

    fig = px.pie(
        action_counts,
        values='count',
        names='action',
        title="Supervisor Actions Distribution"
    )
    fig.update_layout(template="plotly_white", height=250)
    return fig

def create_activity_log(activity_log):
    """Create activity log table"""
    if not activity_log:
        return html.Div([
            html.P("No activity yet - Bot starting up...",
                   style={'textAlign': 'center', 'color': '#95a5a6', 'padding': '20px'})
        ])

    rows = []
    for activity in activity_log[:15]:  # Show last 15 activities
        timestamp = activity.get('timestamp', 'N/A')
        activity_type = activity.get('type', 'unknown')
        message = activity.get('message', 'No message')

        type_color = {
            'trade': '#3498db',
            'optimization': '#9b59b6',
            'supervisor_action': '#f39c12',
            'error': '#e74c3c',
            'info': '#27ae60'
        }.get(activity_type, '#7f8c8d')

        rows.append(html.Tr([
            html.Td(timestamp, style={'fontSize': '12px'}),
            html.Td(activity_type.upper(), style={'color': type_color, 'fontWeight': 'bold', 'fontSize': '12px'}),
            html.Td(message, style={'fontSize': '12px'})
        ]))

    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('Time'),
                html.Th('Type'),
                html.Th('Message')
            ])
        ]),
        html.Tbody(rows)
    ], style={'width': '100%', 'border': '1px solid #ddd'})

    return table

def create_recent_trades_table():
    """Create table showing recent open positions"""
    if not binance_client:
        return html.Div([
            html.P("No API connection - Cannot fetch open positions",
                   style={'textAlign': 'center', 'color': '#e74c3c', 'padding': '20px'})
        ])

    try:
        # Get open positions from Binance
        positions = binance_client.futures_position_information()
        # Filter only non-zero positions
        open_positions = [pos for pos in positions if abs(float(pos['positionAmt'])) > 0]

        if not open_positions:
            return html.Div([
                html.P("No open positions currently",
                       style={'textAlign': 'center', 'color': '#95a5a6', 'padding': '20px'})
            ])

        # Sort by unrealized PnL (descending - profits first, then losses)
        open_positions.sort(key=lambda x: float(x['unRealizedProfit']), reverse=True)

        rows = []
        # Show ALL positions, not just top 15
        for pos in open_positions:
            symbol = pos['symbol']
            amt = float(pos['positionAmt'])
            entry_price = float(pos['entryPrice'])
            mark_price = float(pos['markPrice'])
            unrealized_pnl = float(pos['unRealizedProfit'])
            leverage = int(pos['leverage'])

            side = "LONG" if amt > 0 else "SHORT"
            side_color = '#3498db' if amt > 0 else '#e74c3c'
            pnl_color = '#27ae60' if unrealized_pnl >= 0 else '#e74c3c'

            pnl_percent = ((mark_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            if amt < 0:  # Short position
                pnl_percent = -pnl_percent

            rows.append(html.Tr([
                html.Td(symbol, style={'fontWeight': 'bold', 'fontSize': '12px'}),
                html.Td(side, style={'color': side_color, 'fontWeight': 'bold', 'fontSize': '12px'}),
                html.Td(f"{abs(amt):.4f}", style={'fontSize': '12px'}),
                html.Td(f"${entry_price:.2f}", style={'fontSize': '12px'}),
                html.Td(f"${mark_price:.2f}", style={'fontSize': '12px'}),
                html.Td(f"{leverage}x", style={'fontSize': '12px'}),
                html.Td(f"${unrealized_pnl:+.2f}", style={'color': pnl_color, 'fontWeight': 'bold', 'fontSize': '12px'}),
                html.Td(f"{pnl_percent:+.2f}%", style={'color': pnl_color, 'fontSize': '12px'})
            ]))

        table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th('Symbol', style={'fontSize': '12px'}),
                    html.Th('Side', style={'fontSize': '12px'}),
                    html.Th('Amount', style={'fontSize': '12px'}),
                    html.Th('Entry', style={'fontSize': '12px'}),
                    html.Th('Mark', style={'fontSize': '12px'}),
                    html.Th('Leverage', style={'fontSize': '12px'}),
                    html.Th('PnL', style={'fontSize': '12px'}),
                    html.Th('PnL %', style={'fontSize': '12px'})
                ])
            ]),
            html.Tbody(rows)
        ], style={'width': '100%', 'border': '1px solid #ddd', 'fontSize': '12px'})

        return table

    except Exception as e:
        print(f"Error fetching open positions: {e}")
        return html.Div([
            html.P(f"Error loading positions: {str(e)}",
                   style={'textAlign': 'center', 'color': '#e74c3c', 'padding': '20px', 'fontSize': '12px'})
        ])

def create_closed_trades_table():
    """Create table showing recently closed positions"""
    if not binance_client:
        return html.Div([
            html.P("No API connection - Cannot fetch trade history",
                   style={'textAlign': 'center', 'color': '#e74c3c', 'padding': '20px'})
        ])

    try:
        # Get recent trades from major pairs
        all_trades = []
        major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']

        for symbol in major_pairs:
            try:
                trades = binance_client.futures_account_trades(symbol=symbol, limit=20)
                all_trades.extend(trades)
            except:
                continue

        if not all_trades:
            return html.Div([
                html.P("No recent trades found",
                       style={'textAlign': 'center', 'color': '#95a5a6', 'padding': '20px'})
            ])

        # Sort by time (most recent first)
        all_trades.sort(key=lambda x: x['time'], reverse=True)

        rows = []
        for trade in all_trades[:15]:  # Show last 15 trades
            timestamp = datetime.fromtimestamp(trade['time'] / 1000)
            symbol = trade['symbol']
            side = trade['side']
            qty = float(trade['qty'])
            price = float(trade['price'])
            realized_pnl = float(trade.get('realizedPnl', 0))
            commission = float(trade.get('commission', 0))

            side_color = '#3498db' if side == 'BUY' else '#e74c3c'
            pnl_color = '#27ae60' if realized_pnl >= 0 else '#e74c3c'
            status = "✓ PROFIT" if realized_pnl > 0 else ("✗ LOSS" if realized_pnl < 0 else "○ NEUTRAL")
            status_color = '#27ae60' if realized_pnl > 0 else ('#e74c3c' if realized_pnl < 0 else '#95a5a6')

            rows.append(html.Tr([
                html.Td(timestamp.strftime('%m-%d %H:%M'), style={'fontSize': '12px'}),
                html.Td(symbol, style={'fontWeight': 'bold', 'fontSize': '12px'}),
                html.Td(side, style={'color': side_color, 'fontWeight': 'bold', 'fontSize': '12px'}),
                html.Td(f"{qty:.4f}", style={'fontSize': '12px'}),
                html.Td(f"${price:.2f}", style={'fontSize': '12px'}),
                html.Td(f"${realized_pnl:+.2f}", style={'color': pnl_color, 'fontWeight': 'bold', 'fontSize': '12px'}),
                html.Td(f"${commission:.4f}", style={'fontSize': '12px', 'color': '#95a5a6'}),
                html.Td(status, style={'color': status_color, 'fontWeight': 'bold', 'fontSize': '11px'})
            ]))

        table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th('Time', style={'fontSize': '12px'}),
                    html.Th('Symbol', style={'fontSize': '12px'}),
                    html.Th('Side', style={'fontSize': '12px'}),
                    html.Th('Qty', style={'fontSize': '12px'}),
                    html.Th('Price', style={'fontSize': '12px'}),
                    html.Th('PnL', style={'fontSize': '12px'}),
                    html.Th('Fee', style={'fontSize': '12px'}),
                    html.Th('Status', style={'fontSize': '12px'})
                ])
            ]),
            html.Tbody(rows)
        ], style={'width': '100%', 'border': '1px solid #ddd', 'fontSize': '12px'})

        return table

    except Exception as e:
        print(f"Error fetching trade history: {e}")
        return html.Div([
            html.P(f"Error loading trades: {str(e)}",
                   style={'textAlign': 'center', 'color': '#e74c3c', 'padding': '20px', 'fontSize': '12px'})
        ])

def create_supervisor_detailed_status(bot_data):
    """Create detailed supervisor status display"""
    if not bot_data:
        return html.Div("No supervisor data available", style={'textAlign': 'center', 'color': '#95a5a6'})

    supervisor_advice = bot_data.get('supervisor_advice') or "No advice from supervisor yet"
    supervisor_last_update = bot_data.get('supervisor_last_update') or "Never"

    # Format last update time
    try:
        if supervisor_last_update != "Never":
            update_time = datetime.fromisoformat(supervisor_last_update)
            supervisor_last_update = update_time.strftime('%Y-%m-%d %H:%M:%S')
    except:
        pass

    return html.Div([
        html.Div([
            html.Strong("Current Status:", style={'color': '#2c3e50', 'marginRight': '10px'}),
            html.Span(supervisor_advice, style={'color': '#34495e', 'fontSize': '14px'})
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Strong("Last Update:", style={'color': '#7f8c8d', 'marginRight': '10px', 'fontSize': '12px'}),
            html.Span(supervisor_last_update, style={'color': '#95a5a6', 'fontSize': '12px'})
        ])
    ])

def create_supervisor_alerts_table(supervisor_alerts):
    """Create table showing supervisor alerts"""
    if not supervisor_alerts:
        return html.Div([
            html.P("✅ No alerts - System healthy",
                   style={'textAlign': 'center', 'color': '#27ae60', 'padding': '20px'})
        ])

    rows = []
    for alert in supervisor_alerts[:15]:  # Show last 15 alerts
        timestamp = alert.get('timestamp', 'N/A')
        try:
            alert_time = datetime.fromisoformat(timestamp)
            timestamp = alert_time.strftime('%m-%d %H:%M:%S')
        except:
            pass

        alert_type = alert.get('type', 'unknown')
        severity = alert.get('severity', 'INFO')
        message = alert.get('message', 'No message')

        # Color coding by severity
        severity_colors = {
            'CRITICAL': '#e74c3c',
            'WARNING': '#f39c12',
            'INFO': '#3498db',
            'LOW': '#95a5a6'
        }
        severity_color = severity_colors.get(severity, '#95a5a6')

        rows.append(html.Tr([
            html.Td(timestamp, style={'fontSize': '12px'}),
            html.Td(severity, style={'color': severity_color, 'fontWeight': 'bold', 'fontSize': '12px'}),
            html.Td(alert_type, style={'fontSize': '12px', 'fontWeight': 'bold'}),
            html.Td(message, style={'fontSize': '12px'})
        ]))

    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('Time', style={'fontSize': '12px'}),
                html.Th('Severity', style={'fontSize': '12px'}),
                html.Th('Type', style={'fontSize': '12px'}),
                html.Th('Message', style={'fontSize': '12px'})
            ])
        ]),
        html.Tbody(rows)
    ], style={'width': '100%', 'border': '1px solid #ddd'})

    return table

def create_supervisor_actions_table(supervisor_actions):
    """Create table showing supervisor actions"""
    if not supervisor_actions:
        return html.Div([
            html.P("⏳ No actions recorded yet",
                   style={'textAlign': 'center', 'color': '#95a5a6', 'padding': '20px'})
        ])

    rows = []
    for action in supervisor_actions[:15]:  # Show last 15 actions
        timestamp = action.get('timestamp', 'N/A')
        try:
            action_time = datetime.fromisoformat(timestamp)
            timestamp = action_time.strftime('%m-%d %H:%M:%S')
        except:
            pass

        action_type = action.get('action_type', 'unknown')
        action_taken = action.get('action_taken', 'No action')

        # Get metadata if available
        metadata = action.get('metadata', {})
        balance = metadata.get('balance', 'N/A')
        if balance != 'N/A':
            balance = f"${balance:.2f}"

        overfitting_risk = metadata.get('overfitting_risk', 'N/A')

        # Color code by action type
        action_colors = {
            'STATUS_REPORT': '#3498db',
            'ALERT_TRIGGERED': '#f39c12',
            'INTERVENTION': '#e74c3c',
            'OPTIMIZATION_ADVICE': '#9b59b6'
        }
        action_color = action_colors.get(action_type, '#7f8c8d')

        rows.append(html.Tr([
            html.Td(timestamp, style={'fontSize': '12px'}),
            html.Td(action_type, style={'color': action_color, 'fontWeight': 'bold', 'fontSize': '12px'}),
            html.Td(action_taken, style={'fontSize': '12px'}),
            html.Td(balance, style={'fontSize': '12px'}),
            html.Td(overfitting_risk, style={'fontSize': '12px', 'fontWeight': 'bold'})
        ]))

    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('Time', style={'fontSize': '12px'}),
                html.Th('Action Type', style={'fontSize': '12px'}),
                html.Th('Details', style={'fontSize': '12px'}),
                html.Th('Balance', style={'fontSize': '12px'}),
                html.Th('Risk', style={'fontSize': '12px'})
            ])
        ]),
        html.Tbody(rows)
    ], style={'width': '100%', 'border': '1px solid #ddd'})

    return table

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8050))
    print(f"\n{'='*60}")
    print(f"🚀 Starting Self-Optimizer Bot Monitor")
    print(f"{'='*60}")
    print(f"📊 Port: {port}")
    print(f"📊 Dashboard URL: http://0.0.0.0:{port}")
    print(f"🔗 Redis: {'✅ Connected' if redis_connected else '❌ Disconnected'}")
    print(f"🔗 Binance: {'✅ Connected' if binance_client else '❌ Disconnected'}")

    if not binance_client:
        print(f"\n⚠️  WARNING: Binance API not connected!")
        print(f"⚠️  Dashboard will show 'NO API CONNECTION' status")
        print(f"⚠️  Please check Railway environment variables:")
        print(f"   - BINANCE_API_KEY")
        print(f"   - BINANCE_API_SECRET")
        print(f"   - BINANCE_TESTNET (optional, set to 'true' for testnet)")
        print(f"\n📝 Common issues:")
        print(f"   1. API keys not set in Railway environment")
        print(f"   2. API keys are for wrong environment (testnet vs mainnet)")
        print(f"   3. API permissions not enabled (enable Futures trading)")
        print(f"   4. IP whitelist restrictions (disable or add Railway IP)")

    print(f"{'='*60}\n")

    app.run_server(debug=False, host='0.0.0.0', port=port)
