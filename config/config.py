import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from decimal import Decimal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class TradingConfig:
    """Core trading configuration"""
    target_daily_return: float = 0.05  # 5% daily target
    max_daily_loss: float = 0.02  # 2% max daily loss
    max_drawdown: float = 0.10  # 10% max drawdown
    min_win_rate: float = 0.60  # 60% minimum win rate
    target_sharpe_ratio: float = 2.0
    position_sizing_method: str = "kelly"  # kelly, fixed, volatility
    max_position_size: float = 0.20  # 20% max position size

@dataclass
class BinanceConfig:
    """Binance API configuration"""
    api_key: str = os.getenv("BINANCE_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    testnet: bool = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    base_url: str = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
    max_retries: int = 3
    timeout: int = 30
    rate_limit_buffer: float = 0.1  # 10% buffer for rate limits

@dataclass
class MLConfig:
    """Machine Learning configuration"""
    models: List[str] = None
    lookback_period: int = 168  # 7 days in hours
    prediction_horizon: int = 24  # 24 hours ahead
    feature_engineering: Dict = None
    model_retrain_interval: int = 24  # hours
    min_backtest_period: int = 4320  # 6 months in hours
    validation_split: float = 0.2

    def __post_init__(self):
        if self.models is None:
            self.models = ["lstm", "transformer", "ensemble"]
        if self.feature_engineering is None:
            self.feature_engineering = {
                "technical_indicators": True,
                "market_microstructure": True,
                "sentiment_analysis": False,
                "volatility_features": True,
                "time_features": True
            }

@dataclass
class RiskConfig:
    """Risk management configuration"""
    stop_loss_methods: List[str] = None
    take_profit_methods: List[str] = None
    position_limits: Dict = None
    correlation_threshold: float = 0.7
    var_confidence: float = 0.95
    stress_test_scenarios: int = 1000

    def __post_init__(self):
        if self.stop_loss_methods is None:
            self.stop_loss_methods = ["atr", "volatility", "ml_based", "time_based"]
        if self.take_profit_methods is None:
            self.take_profit_methods = ["risk_reward", "trailing", "ml_based"]
        if self.position_limits is None:
            self.position_limits = {
                "max_open_positions": 10,
                "max_exposure_per_symbol": 0.15,
                "max_sector_exposure": 0.30
            }

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    alert_channels: List[str] = None
    performance_check_interval: int = 300  # 5 minutes
    health_check_interval: int = 60  # 1 minute
    log_level: str = "INFO"
    metrics_retention_days: int = 90
    alert_thresholds: Dict = None

    def __post_init__(self):
        if self.alert_channels is None:
            self.alert_channels = ["email", "webhook"]
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "daily_loss_threshold": 0.015,  # 1.5%
                "drawdown_threshold": 0.08,     # 8%
                "latency_threshold": 100,       # ms
                "error_rate_threshold": 0.05    # 5%
            }

@dataclass
class HiveMindConfig:
    """Hive-mind collective intelligence configuration"""
    swarm_size: int = 6
    coordination_topology: str = "mesh"  # mesh, hierarchical, adaptive
    memory_sharing: bool = True
    consensus_threshold: float = 0.6
    agent_types: List[str] = None
    neural_training: bool = True
    performance_optimization: bool = True

    def __post_init__(self):
        if self.agent_types is None:
            self.agent_types = [
                "strategy_generator",
                "market_intelligence",
                "execution_engine",
                "risk_guardian",
                "performance_analyzer",
                "self_evolution"
            ]

class Config:
    """Main configuration class"""

    def __init__(self):
        self.trading = TradingConfig()
        self.binance = BinanceConfig()
        self.ml = MLConfig()
        self.risk = RiskConfig()
        self.monitoring = MonitoringConfig()
        self.hive_mind = HiveMindConfig()

        # Database configuration
        self.database = {
            "url": os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/trading_bot"),
            "pool_size": 20,
            "max_overflow": 30,
            "echo": False
        }

        # Redis configuration for caching
        self.redis = {
            "url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            "max_connections": 50
        }

        # Validate critical settings
        self._validate_config()

    def _validate_config(self):
        """Validate configuration settings"""
        if not self.binance.api_key or not self.binance.api_secret:
            raise ValueError("Binance API credentials not provided")

        if self.trading.target_daily_return <= 0:
            raise ValueError("Target daily return must be positive")

        if self.trading.max_daily_loss >= self.trading.target_daily_return:
            raise ValueError("Max daily loss should be less than target return")

# Global config instance
config = Config()