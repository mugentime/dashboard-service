import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

from config.config import config

# Optional imports for demo mode
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit
    import joblib
    from src.ml.feature_engineering import FeatureEngineer
    from src.ml.backtesting import BacktestEngine
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    # Create mock classes for demo mode
    class MinMaxScaler:
        def fit_transform(self, data): return data
        def transform(self, data): return data
    class FeatureEngineer:
        def __init__(self): pass
        async def engineer_features(self, data): return {"features": data, "targets": data}
    class BacktestEngine:
        def __init__(self): pass
        async def run_backtest(self, *args): return {"returns": [0.01], "sharpe_ratio": 1.5}

logger = logging.getLogger(__name__)

class StrategyGenerator:
    """ML-based trading strategy generator targeting 5% daily returns"""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.backtest_engine = BacktestEngine()
        self.models = {}
        self.scalers = {}
        self.current_strategies = {}
        self.performance_tracker = {}

    async def generate_strategies(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate multiple trading strategies using different ML models"""
        if not DEPENDENCIES_AVAILABLE:
            logger.warning("🎭 ML dependencies not available - running in DEMO MODE")
            return self._generate_demo_strategies()

        strategies = {}

        # Prepare features
        features_df = await self.feature_engineer.create_features(market_data)

        # Generate strategies with different models
        strategies['lstm'] = await self._create_lstm_strategy(features_df)
        strategies['transformer'] = await self._create_transformer_strategy(features_df)
        strategies['ensemble'] = await self._create_ensemble_strategy(features_df)

        # Backtest all strategies
        for name, strategy in strategies.items():
            backtest_results = await self.backtest_engine.run_backtest(
                strategy, market_data, lookback_days=180
            )
            strategy['backtest_results'] = backtest_results
            strategy['performance_score'] = self._calculate_performance_score(backtest_results)

        # Select best performing strategies
        best_strategies = self._select_best_strategies(strategies)

        return best_strategies

    def _generate_demo_strategies(self) -> Dict[str, Any]:
        """Generate demo strategies for testing without ML dependencies"""
        return {
            'demo_strategy': {
                'name': 'Demo Strategy',
                'model_type': 'demo',
                'predictions': [0.02, 0.015, 0.025, 0.01, 0.03],
                'confidence': 0.85,
                'expected_return': 0.05,
                'risk_score': 0.3,
                'backtest_results': {
                    'total_return': 0.15,
                    'sharpe_ratio': 1.8,
                    'max_drawdown': 0.08,
                    'win_rate': 0.62
                },
                'performance_score': 0.87,
                'signals': {
                    'buy_signals': [1, 0, 1, 0, 1],
                    'sell_signals': [0, 1, 0, 1, 0],
                    'hold_signals': [0, 0, 0, 0, 0]
                }
            }
        }

    async def _create_lstm_strategy(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Create LSTM-based trading strategy"""
        X, y = self._prepare_sequences(features_df)

        # Split data for training
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Build LSTM model
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='tanh')  # Output between -1 and 1 for position sizing
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )

        # Generate trading signals
        predictions = model.predict(X_test)
        signals = self._generate_signals_from_predictions(predictions)

        strategy = {
            'type': 'lstm',
            'model': model,
            'scaler': self.scalers.get('lstm'),
            'signals': signals,
            'features': list(features_df.columns),
            'lookback_period': config.ml.lookback_period,
            'prediction_horizon': config.ml.prediction_horizon,
            'training_history': history.history,
            'last_trained': datetime.now()
        }

        return strategy

    async def _create_transformer_strategy(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Create Transformer-based trading strategy"""
        X, y = self._prepare_sequences(features_df)

        # Build Transformer model
        inputs = Input(shape=(X.shape[1], X.shape[2]))

        # Multi-head attention layer
        attention = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        attention = LayerNormalization()(attention + inputs)

        # Feed forward network
        ffn = Dense(128, activation='relu')(attention)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(X.shape[2])(ffn)
        ffn = LayerNormalization()(ffn + attention)

        # Global average pooling and output
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn)
        outputs = Dense(1, activation='tanh')(pooled)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train model
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=30,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
            ]
        )

        predictions = model.predict(X_test)
        signals = self._generate_signals_from_predictions(predictions)

        strategy = {
            'type': 'transformer',
            'model': model,
            'scaler': self.scalers.get('transformer'),
            'signals': signals,
            'features': list(features_df.columns),
            'training_history': history.history,
            'last_trained': datetime.now()
        }

        return strategy

    async def _create_ensemble_strategy(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Create ensemble strategy combining multiple models"""
        X_flat = features_df.values

        # Prepare data
        split_idx = int(len(X_flat) * 0.8)
        X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]

        # Create target (future returns)
        returns = features_df['close'].pct_change().shift(-1)
        y_train, y_test = returns.values[:split_idx], returns.values[split_idx:]

        # Remove NaN values
        mask_train = ~np.isnan(y_train)
        mask_test = ~np.isnan(y_test)
        X_train, y_train = X_train[mask_train], y_train[mask_train]
        X_test, y_test = X_test[mask_test], y_test[mask_test]

        # Train ensemble models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        predictions = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions[name] = pred

        # Combine predictions (simple average)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        signals = self._generate_signals_from_predictions(ensemble_pred.reshape(-1, 1))

        strategy = {
            'type': 'ensemble',
            'models': models,
            'signals': signals,
            'features': list(features_df.columns),
            'individual_predictions': predictions,
            'last_trained': datetime.now()
        }

        return strategy

    def _prepare_sequences(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM/Transformer training"""
        lookback = config.ml.lookback_period
        data = features_df.values

        # Normalize data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        X, y = [], []
        for i in range(lookback, len(data_scaled)):
            X.append(data_scaled[i-lookback:i])
            # Target: future return (normalized)
            future_return = (data[i, 0] - data[i-1, 0]) / data[i-1, 0]  # Close price return
            y.append(np.tanh(future_return * 100))  # Scale and bound between -1, 1

        return np.array(X), np.array(y)

    def _generate_signals_from_predictions(self, predictions: np.ndarray) -> List[Dict]:
        """Convert model predictions to trading signals"""
        signals = []

        for i, pred in enumerate(predictions.flatten()):
            # Convert prediction to position size (-1 to 1)
            position_size = np.clip(pred, -1, 1)

            # Generate signal based on position size
            if position_size > 0.3:
                action = 'BUY'
                confidence = min(position_size, 1.0)
            elif position_size < -0.3:
                action = 'SELL'
                confidence = min(abs(position_size), 1.0)
            else:
                action = 'HOLD'
                confidence = 0.0

            signals.append({
                'timestamp': datetime.now() + timedelta(hours=i),
                'action': action,
                'confidence': confidence,
                'position_size': position_size * config.trading.max_position_size,
                'prediction': float(pred)
            })

        return signals

    def _calculate_performance_score(self, backtest_results: Dict) -> float:
        """Calculate comprehensive performance score for strategy selection"""
        if not backtest_results:
            return 0.0

        # Key metrics for scoring
        total_return = backtest_results.get('total_return', 0)
        sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
        max_drawdown = backtest_results.get('max_drawdown', 1)
        win_rate = backtest_results.get('win_rate', 0)
        profit_factor = backtest_results.get('profit_factor', 0)

        # Weighted scoring
        score = (
            total_return * 0.3 +
            sharpe_ratio * 0.25 +
            (1 - max_drawdown) * 0.2 +
            win_rate * 0.15 +
            min(profit_factor / 2, 1) * 0.1
        )

        # Penalty for not meeting targets
        if total_return < config.trading.target_daily_return * 180:  # 6 months
            score *= 0.5

        if sharpe_ratio < config.trading.target_sharpe_ratio:
            score *= 0.8

        if win_rate < config.trading.min_win_rate:
            score *= 0.7

        return max(score, 0.0)

    def _select_best_strategies(self, strategies: Dict[str, Any]) -> Dict[str, Any]:
        """Select best performing strategies for live trading"""
        # Sort by performance score
        sorted_strategies = sorted(
            strategies.items(),
            key=lambda x: x[1]['performance_score'],
            reverse=True
        )

        # Select top strategies that meet minimum requirements
        selected = {}
        for name, strategy in sorted_strategies:
            results = strategy['backtest_results']

            # Minimum requirements
            if (results.get('total_return', 0) >= config.trading.target_daily_return * 90 and  # 3 months
                results.get('sharpe_ratio', 0) >= 1.5 and
                results.get('max_drawdown', 1) <= config.trading.max_drawdown and
                results.get('win_rate', 0) >= 0.55):

                selected[name] = strategy

                # Limit to top 3 strategies
                if len(selected) >= 3:
                    break

        return selected

    async def update_strategy_performance(self, strategy_name: str, trade_result: Dict):
        """Update strategy performance tracking"""
        if strategy_name not in self.performance_tracker:
            self.performance_tracker[strategy_name] = {
                'trades': [],
                'total_return': 0.0,
                'win_rate': 0.0,
                'last_updated': datetime.now()
            }

        tracker = self.performance_tracker[strategy_name]
        tracker['trades'].append(trade_result)

        # Update metrics
        returns = [trade['return'] for trade in tracker['trades']]
        tracker['total_return'] = sum(returns)
        tracker['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
        tracker['last_updated'] = datetime.now()

        # Check if strategy needs retraining
        if len(tracker['trades']) >= 100:  # Retrain after 100 trades
            await self._trigger_strategy_retrain(strategy_name)

    async def _trigger_strategy_retrain(self, strategy_name: str):
        """Trigger strategy retraining based on performance"""
        logger.info(f"Triggering retrain for strategy: {strategy_name}")
        # This would trigger the self-evolution agent to retrain the strategy

    def save_strategies(self, strategies: Dict[str, Any], filepath: str):
        """Save trained strategies to disk"""
        joblib.dump(strategies, filepath)
        logger.info(f"Strategies saved to {filepath}")

    def load_strategies(self, filepath: str) -> Dict[str, Any]:
        """Load trained strategies from disk"""
        strategies = joblib.load(filepath)
        logger.info(f"Strategies loaded from {filepath}")
        return strategies