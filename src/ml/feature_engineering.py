import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for trading strategies"""

    def __init__(self):
        self.feature_cache = {}

    async def create_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for ML models"""
        df = market_data.copy()

        # Basic price features
        df = self._add_price_features(df)

        # Technical indicators
        df = self._add_technical_indicators(df)

        # Volatility features
        df = self._add_volatility_features(df)

        # Market microstructure features
        df = self._add_microstructure_features(df)

        # Time-based features
        df = self._add_time_features(df)

        # Statistical features
        df = self._add_statistical_features(df)

        # Advanced features
        df = self._add_advanced_features(df)

        # Clean data
        df = self._clean_features(df)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # Price position within range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # Gap features
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)

        # Momentum indicators
        df['rsi_14'] = talib.RSI(close, timeperiod=14)
        df['rsi_21'] = talib.RSI(close, timeperiod=21)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)

        # Williams %R
        df['williams_r'] = talib.WILLR(high, low, close)

        # Commodity Channel Index
        df['cci'] = talib.CCI(high, low, close)

        # Average True Range
        df['atr'] = talib.ATR(high, low, close)

        # Parabolic SAR
        df['sar'] = talib.SAR(high, low)

        # Volume indicators
        df['obv'] = talib.OBV(close, volume)
        df['ad'] = talib.AD(high, low, close, volume)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Historical volatility
        for period in [10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std() * np.sqrt(252)

        # GARCH-like features
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_30']

        # True Range features
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # Volatility breakout
        df['vol_breakout'] = df['true_range'] / df['atr']

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Bid-ask spread proxy (using high-low)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']

        # Price impact proxy
        df['price_impact'] = abs(df['returns']) / np.log(df['volume'] + 1)

        # Volume-weighted average price
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']

        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Money flow
        df['money_flow'] = df['close'] * df['volume']
        df['money_flow_ratio'] = df['money_flow'] / df['money_flow'].rolling(20).mean()

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' not in df.columns:
            df['timestamp'] = df.index

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Hour of day
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Day of month
        df['day_of_month'] = df['timestamp'].dt.day
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)

        # Market session features
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Rolling statistics
        for period in [10, 20, 50]:
            df[f'mean_{period}'] = df['close'].rolling(period).mean()
            df[f'std_{period}'] = df['close'].rolling(period).std()
            df[f'skew_{period}'] = df['returns'].rolling(period).skew()
            df[f'kurt_{period}'] = df['returns'].rolling(period).kurt()

        # Z-scores
        df['zscore_10'] = (df['close'] - df['mean_10']) / df['std_10']
        df['zscore_20'] = (df['close'] - df['mean_20']) / df['std_20']

        # Percentile features
        for period in [20, 50]:
            df[f'percentile_{period}'] = df['close'].rolling(period).rank(pct=True)

        return df

    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced engineered features"""
        # Regime detection features
        df['trend_strength'] = abs(df['close'] - df['sma_50']) / df['atr']

        # Momentum persistence
        df['momentum_persistence'] = (
            (df['returns'] > 0).astype(int).rolling(10).sum() / 10
        )

        # Volatility regime
        df['vol_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).quantile(0.75)).astype(int)

        # Support/Resistance levels
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_level'] = df['low'].rolling(20).min()
        df['distance_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support_level']) / df['close']

        # Cross-timeframe features
        if len(df) > 240:  # 10 days of hourly data
            df['daily_return'] = df['close'].pct_change(24)
            df['weekly_return'] = df['close'].pct_change(168)

        # Fractal features
        df['local_max'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['high'] > df['high'].shift(-1))
        ).astype(int)

        df['local_min'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['low'] < df['low'].shift(-1))
        ).astype(int)

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill missing values for technical indicators
        technical_cols = [col for col in df.columns if any(indicator in col.lower()
                         for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr'])]

        for col in technical_cols:
            df[col] = df[col].fillna(method='ffill')

        # Drop rows with too many missing values
        missing_threshold = 0.5
        df = df.dropna(thresh=int(len(df.columns) * missing_threshold))

        # Fill remaining NaN values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        return df

    def select_features(self, df: pd.DataFrame, target_col: str = 'returns',
                       method: str = 'correlation') -> List[str]:
        """Select most relevant features for prediction"""
        if method == 'correlation':
            # Select features based on correlation with target
            correlations = df.corr()[target_col].abs().sort_values(ascending=False)
            selected_features = correlations[correlations > 0.1].index.tolist()
            selected_features = [f for f in selected_features if f != target_col]

        elif method == 'mutual_info':
            # Use mutual information for feature selection
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.preprocessing import StandardScaler

            # Prepare data
            feature_cols = [col for col in df.columns if col not in ['timestamp', target_col]]
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(0)

            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y)
            feature_importance = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

            # Select top features
            selected_features = feature_importance[feature_importance > feature_importance.mean()].index.tolist()

        else:
            # Return all numeric features
            selected_features = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_features = [f for f in selected_features if f != target_col]

        return selected_features[:50]  # Limit to top 50 features