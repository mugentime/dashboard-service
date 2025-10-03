# Live Trading Configuration
# WARNING: This configuration is for LIVE trading with real money
# Ensure all parameters are carefully reviewed before deployment

LIVE_TRADING_CONFIG = {
    # Risk Management Parameters
    "max_position_size_percent": 2.0,  # Max 2% of account per trade
    "max_daily_loss_percent": 5.0,     # Stop trading if daily loss exceeds 5%
    "max_drawdown_percent": 10.0,      # Maximum allowed drawdown
    "stop_loss_percent": 2.0,          # Stop loss at 2%
    "take_profit_percent": 4.0,        # Take profit at 4%

    # Position Sizing
    "base_position_size": 8.0,         # Base position size in USDT
    "min_position_size": 5.0,          # Minimum position size in USDT
    "max_position_size": 40.0,         # Maximum position size in USDT (safe for 55 USDT balance)

    # Trading Pairs (Expanded for maximum opportunities)
    "trading_pairs": [
        # Major pairs
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT",
        # DeFi & Layer 1
        "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT", "UNIUSDT", "AAVEUSDT", "ATOMUSDT",
        # Layer 2 & Scaling
        "ARBUSDT", "OPUSDT", "SUIUSDT", "APTUSDT", "NEARUSDT", "FILUSDT",
        # Meme & Trending
        "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "WIFUSDT", "BONKUSDT",
        # Gaming & Metaverse
        "SANDUSDT", "MANAUSDT", "AXSUSDT", "ENJUSDT", "GALAUSDT",
        # AI & Tech
        "FETUSDT", "AGIXUSDT", "OCEANUSDT", "RENDERUSDT", "THETAUSDT",
        # Privacy & Security
        "XMRUSDT", "ZECUSDT", "DASHUSDT",
        # High Volume Alts
        "LTCUSDT", "BCHUSDT", "ETCUSDT", "TRXUSDT", "XLMUSDT", "VETUSDT",
        # Emerging & High Potential
        "INJUSDT", "SEIUSDT", "TIAUSDT", "PYUSDT", "JUPUSDT", "STRKUSDT"
    ],

    # Trading Parameters
    "leverage": 3,                     # Conservative leverage for live trading
    "order_type": "MARKET",            # Market orders for fast execution
    "time_in_force": "GTC",           # Good Till Cancelled

    # Hive-Mind Decision Thresholds
    "min_consensus_score": 0.75,      # Minimum consensus score to execute trade
    "min_confidence_level": 0.8,      # Minimum confidence level
    "strategy_weight_threshold": 0.6, # Strategy agreement threshold

    # Performance Targets
    "daily_target_return": 0.02,      # 2% daily target (reduced from 5% for safety)
    "weekly_target_return": 0.10,     # 10% weekly target
    "monthly_target_return": 0.30,    # 30% monthly target

    # Monitoring
    "performance_check_interval": 300, # Check performance every 5 minutes
    "risk_check_interval": 60,        # Check risk every minute
    "heartbeat_interval": 30,         # System health check every 30 seconds

    # Emergency Controls
    "emergency_stop_loss": 0.15,      # Emergency stop at 15% loss
    "circuit_breaker_threshold": 0.10, # Stop trading if 10% daily loss
    "max_consecutive_losses": 3,       # Stop after 3 consecutive losses

    # Logging and Alerts
    "log_level": "INFO",
    "alert_on_trades": True,
    "alert_on_profits": True,
    "alert_on_losses": True,
    "save_trade_history": True,

    # Advanced Optimization Parameters (Phase 1 Enhancements)
    "multi_timeframe_analysis": {
        "enabled": True,
        "timeframes": ["1m", "5m", "15m", "1h"],
        "confluence_threshold": 0.7,
        "minimum_timeframe_agreement": 3
    },

    "volatility_based_sizing": {
        "enabled": True,
        "atr_period": 14,
        "volatility_multiplier_range": {"low": 1.2, "high": 0.3},
        "dynamic_stop_loss_multiplier": 2.0,
        "risk_reward_ratio": 2.0
    },

    "ml_signal_enhancement": {
        "enabled": True,
        "ml_weight": 0.4,
        "technical_weight": 0.6,
        "min_ml_confidence": 0.3,
        "ensemble_threshold": 0.7,
        "lstm_enabled": True,
        "feature_analysis_enabled": True
    },

    "inter_bot_communication": {
        "enabled": True,
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 0,
        "heartbeat_interval": 30,
        "position_conflict_prevention": True,
        "coordination_timeout": 60
    },

    "portfolio_heat_management": {
        "enabled": True,
        "max_portfolio_heat": 0.15,
        "max_sector_heat": 0.08,
        "heat_calculation_weights": {
            "volatility": 0.3,
            "correlation": 0.3,
            "position_size": 0.2,
            "time_decay": 0.2
        }
    },

    "correlation_analysis": {
        "enabled": True,
        "lookback_period_days": 30,
        "high_correlation_threshold": 0.8,
        "extreme_correlation_threshold": 0.9,
        "diversification_target": 0.5,
        "max_correlated_pairs": 3
    }
}

# Market Hours (24/7 for crypto, but avoid low volume periods)
MARKET_CONDITIONS = {
    "avoid_low_volume_hours": True,
    "min_volume_threshold": 1000000,  # Minimum 24h volume in USDT
    "volatility_threshold": 0.02,     # Minimum volatility for trading opportunities
}

# Safety Checks
SAFETY_CHECKS = {
    "verify_balance_before_trade": True,
    "check_margin_requirements": True,
    "validate_order_size": True,
    "confirm_network_status": True,
    "backup_positions_data": True
}