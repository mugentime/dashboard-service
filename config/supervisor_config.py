"""
Supervisor Configuration
Settings for automated 6-hour monitoring and analysis
"""

# 6-Hour Monitoring Configuration
MONITORING_CONFIG = {
    'cycle_duration_hours': 6,
    'hourly_updates': True,
    'detailed_logging': True,
    'save_reports': True,
    'report_directory': 'supervisor_reports'
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'excellent_performance': 2.0,      # % gain
    'good_performance': 1.0,           # % gain
    'acceptable_performance': 0.0,     # break-even
    'warning_performance': -1.0,       # % loss
    'critical_performance': -2.0       # % loss
}

# Risk Management Thresholds
RISK_THRESHOLDS = {
    'very_low_exposure': 100,          # %
    'low_exposure': 200,               # %
    'medium_exposure': 300,            # %
    'high_exposure': 400,              # %
    'critical_exposure': 500           # %
}

# Portfolio Diversification Targets
DIVERSIFICATION_TARGETS = {
    'min_positions': 5,
    'optimal_positions': 15,
    'max_positions': 30,
    'target_exposure_ratio': 250      # %
}

# Optimization Scoring Weights
OPTIMIZATION_SCORING = {
    'performance_weight': 40,          # /100 points
    'risk_management_weight': 30,      # /100 points
    'diversification_weight': 20,      # /100 points
    'balance_weight': 10               # /100 points
}

# Alert Configuration
ALERT_CONFIG = {
    'enable_email_alerts': False,
    'enable_discord_alerts': False,
    'critical_performance_alert': True,
    'high_risk_alert': True,
    'optimization_score_threshold': 40  # Below this triggers alert
}

# Monitoring Features
MONITORING_FEATURES = {
    'track_multi_timeframe_signals': True,
    'track_volatility_adjustments': True,
    'track_ml_enhancements': True,
    'track_portfolio_heat': True,
    'track_correlation_analysis': True,
    'track_trade_quality': True,
    'track_signal_confidence': True
}

# Historical Analysis
HISTORICAL_CONFIG = {
    'keep_reports_days': 30,
    'generate_weekly_summary': True,
    'generate_monthly_summary': True,
    'performance_comparison': True
}

# Recommendation Engine
RECOMMENDATION_CONFIG = {
    'auto_generate_recommendations': True,
    'recommendation_severity_levels': {
        'info': 'Informational guidance',
        'warning': 'Attention required',
        'critical': 'Immediate action needed',
        'urgent': 'Stop trading and investigate'
    }
}

# Export Configuration
EXPORT_CONFIG = {
    'json_reports': True,
    'csv_summaries': True,
    'html_dashboard': False,
    'backup_reports': True
}