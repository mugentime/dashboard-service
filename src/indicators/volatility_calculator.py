"""
Advanced Volatility Calculator with ATR
Provides volatility-based position sizing and risk adjustments
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class VolatilityCalculator:
    """
    Advanced volatility calculator using ATR and other volatility measures
    for dynamic position sizing and risk management
    """

    def __init__(self, default_period: int = 14):
        self.default_period = default_period
        self.volatility_cache = {}  # Cache for volatility calculations

    async def calculate_atr_volatility(self, client, symbol: str,
                                     timeframe: str = '5m',
                                     period: int = None) -> Dict:
        """
        Calculate Average True Range (ATR) volatility
        """
        try:
            period = period or self.default_period

            # Get more data than needed to ensure accurate ATR
            limit = max(period * 2, 50)
            klines = await client.get_klines(symbol, timeframe, limit=limit)

            if not klines or len(klines) < period + 1:
                logger.warning(f"Insufficient data for ATR calculation: {symbol}")
                return self._get_default_volatility_data()

            # Extract OHLC data
            highs = [float(kline[2]) for kline in klines]
            lows = [float(kline[3]) for kline in klines]
            closes = [float(kline[4]) for kline in klines]

            # Calculate True Range
            true_ranges = []
            for i in range(1, len(klines)):
                high = highs[i]
                low = lows[i]
                prev_close = closes[i-1]

                tr1 = high - low  # Current high - current low
                tr2 = abs(high - prev_close)  # Current high - previous close
                tr3 = abs(low - prev_close)   # Current low - previous close

                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)

            if len(true_ranges) < period:
                return self._get_default_volatility_data()

            # Calculate ATR (Simple Moving Average of True Range)
            atr = sum(true_ranges[-period:]) / period
            current_price = closes[-1]

            # Calculate relative ATR as percentage of price
            atr_percent = (atr / current_price) * 100

            # Calculate additional volatility metrics
            volatility_data = self._calculate_extended_volatility(
                highs, lows, closes, atr, atr_percent, period
            )

            # Cache the result
            cache_key = f"{symbol}_{timeframe}_{period}"
            self.volatility_cache[cache_key] = {
                'data': volatility_data,
                'timestamp': datetime.now()
            }

            return volatility_data

        except Exception as e:
            logger.error(f"ATR volatility calculation error for {symbol}: {e}")
            return self._get_default_volatility_data()

    def _calculate_extended_volatility(self, highs: List[float], lows: List[float],
                                     closes: List[float], atr: float,
                                     atr_percent: float, period: int) -> Dict:
        """
        Calculate extended volatility metrics
        """
        try:
            current_price = closes[-1]

            # 1. Standard Deviation Volatility
            price_changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            std_volatility = np.std(price_changes[-period:]) if len(price_changes) >= period else 0
            std_volatility_percent = (std_volatility / current_price) * 100 if current_price > 0 else 0

            # 2. High-Low Range Volatility
            recent_highs = highs[-period:]
            recent_lows = lows[-period:]
            hl_ranges = [h - l for h, l in zip(recent_highs, recent_lows)]
            avg_hl_range = sum(hl_ranges) / len(hl_ranges) if hl_ranges else 0
            hl_volatility_percent = (avg_hl_range / current_price) * 100 if current_price > 0 else 0

            # 3. Price Movement Velocity (rate of change)
            short_period = max(5, period // 3)
            recent_change = (closes[-1] - closes[-short_period]) / closes[-short_period] if len(closes) >= short_period else 0
            movement_velocity = abs(recent_change) * 100

            # 4. Volatility Trend (is volatility increasing or decreasing?)
            if len(closes) >= period * 2:
                recent_atr = self._calculate_simple_atr(highs[-period:], lows[-period:], closes[-period:])
                older_atr = self._calculate_simple_atr(
                    highs[-period*2:-period],
                    lows[-period*2:-period],
                    closes[-period*2:-period]
                )
                volatility_trend = (recent_atr - older_atr) / older_atr if older_atr > 0 else 0
            else:
                volatility_trend = 0

            # 5. Volatility Classification
            volatility_class = self._classify_volatility(atr_percent)

            # 6. Position Sizing Multiplier based on volatility
            position_multiplier = self._calculate_position_multiplier(atr_percent, volatility_class)

            # 7. Risk Adjustment Factor
            risk_adjustment = self._calculate_risk_adjustment(atr_percent, std_volatility_percent)

            return {
                'atr': atr,
                'atr_percent': atr_percent,
                'std_volatility': std_volatility,
                'std_volatility_percent': std_volatility_percent,
                'hl_volatility_percent': hl_volatility_percent,
                'movement_velocity': movement_velocity,
                'volatility_trend': volatility_trend,
                'volatility_class': volatility_class,
                'position_multiplier': position_multiplier,
                'risk_adjustment': risk_adjustment,
                'current_price': current_price,
                'calculation_timestamp': datetime.now().isoformat(),
                'period': period
            }

        except Exception as e:
            logger.error(f"Extended volatility calculation error: {e}")
            return self._get_default_volatility_data()

    def _calculate_simple_atr(self, highs: List[float], lows: List[float], closes: List[float]) -> float:
        """Calculate simple ATR for a given price series"""
        if len(highs) < 2:
            return 0

        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))

        return sum(true_ranges) / len(true_ranges) if true_ranges else 0

    def _classify_volatility(self, atr_percent: float) -> str:
        """
        Classify volatility into categories
        """
        if atr_percent >= 4.0:
            return 'EXTREME'
        elif atr_percent >= 2.5:
            return 'HIGH'
        elif atr_percent >= 1.5:
            return 'MEDIUM'
        elif atr_percent >= 0.8:
            return 'LOW'
        else:
            return 'VERY_LOW'

    def _calculate_position_multiplier(self, atr_percent: float, volatility_class: str) -> float:
        """
        Calculate position size multiplier based on volatility
        Higher volatility = smaller positions
        """
        base_multiplier = 1.0

        # Inverse relationship: higher volatility = lower multiplier
        if volatility_class == 'EXTREME':
            return base_multiplier * 0.3  # 30% of normal size
        elif volatility_class == 'HIGH':
            return base_multiplier * 0.5  # 50% of normal size
        elif volatility_class == 'MEDIUM':
            return base_multiplier * 0.75  # 75% of normal size
        elif volatility_class == 'LOW':
            return base_multiplier * 1.0  # Normal size
        else:  # VERY_LOW
            return base_multiplier * 1.2  # 120% of normal size (but cap it)

    def _calculate_risk_adjustment(self, atr_percent: float, std_volatility_percent: float) -> float:
        """
        Calculate risk adjustment factor for stop-loss and take-profit levels
        """
        # Use the higher of ATR or standard deviation volatility
        volatility_measure = max(atr_percent, std_volatility_percent)

        # Base risk adjustment (minimum 1.0x, maximum 3.0x)
        risk_factor = 1.0 + (volatility_measure / 2.0)  # Scale factor

        # Cap the risk adjustment
        return min(max(risk_factor, 1.0), 3.0)

    def calculate_dynamic_stop_loss(self, entry_price: float, volatility_data: Dict,
                                  side: str = 'BUY', multiplier: float = 2.0) -> Dict:
        """
        Calculate dynamic stop-loss based on volatility
        """
        try:
            atr = volatility_data.get('atr', 0)
            risk_adjustment = volatility_data.get('risk_adjustment', 1.5)

            # Adjust ATR with risk factor and multiplier
            stop_distance = atr * risk_adjustment * multiplier

            if side.upper() == 'BUY':
                stop_loss_price = entry_price - stop_distance
                stop_loss_percent = ((entry_price - stop_loss_price) / entry_price) * 100
            else:  # SELL
                stop_loss_price = entry_price + stop_distance
                stop_loss_percent = ((stop_loss_price - entry_price) / entry_price) * 100

            return {
                'stop_loss_price': stop_loss_price,
                'stop_loss_percent': stop_loss_percent,
                'stop_distance': stop_distance,
                'risk_adjustment_used': risk_adjustment,
                'side': side
            }

        except Exception as e:
            logger.error(f"Dynamic stop-loss calculation error: {e}")
            # Fallback to fixed percentage
            fallback_percent = 2.0
            if side.upper() == 'BUY':
                stop_loss_price = entry_price * (1 - fallback_percent / 100)
            else:
                stop_loss_price = entry_price * (1 + fallback_percent / 100)

            return {
                'stop_loss_price': stop_loss_price,
                'stop_loss_percent': fallback_percent,
                'stop_distance': abs(entry_price - stop_loss_price),
                'risk_adjustment_used': 1.0,
                'side': side,
                'fallback': True
            }

    def calculate_dynamic_take_profit(self, entry_price: float, volatility_data: Dict,
                                    side: str = 'BUY', risk_reward_ratio: float = 2.0) -> Dict:
        """
        Calculate dynamic take-profit based on volatility
        """
        try:
            # First calculate stop-loss to determine risk
            stop_loss_data = self.calculate_dynamic_stop_loss(entry_price, volatility_data, side)
            stop_distance = stop_loss_data['stop_distance']

            # Take-profit distance based on risk-reward ratio
            tp_distance = stop_distance * risk_reward_ratio

            if side.upper() == 'BUY':
                take_profit_price = entry_price + tp_distance
                take_profit_percent = ((take_profit_price - entry_price) / entry_price) * 100
            else:  # SELL
                take_profit_price = entry_price - tp_distance
                take_profit_percent = ((entry_price - take_profit_price) / entry_price) * 100

            return {
                'take_profit_price': take_profit_price,
                'take_profit_percent': take_profit_percent,
                'tp_distance': tp_distance,
                'risk_reward_ratio': risk_reward_ratio,
                'side': side,
                'stop_loss_data': stop_loss_data
            }

        except Exception as e:
            logger.error(f"Dynamic take-profit calculation error: {e}")
            # Fallback calculation
            fallback_percent = 4.0  # 2:1 ratio with 2% stop
            if side.upper() == 'BUY':
                take_profit_price = entry_price * (1 + fallback_percent / 100)
            else:
                take_profit_price = entry_price * (1 - fallback_percent / 100)

            return {
                'take_profit_price': take_profit_price,
                'take_profit_percent': fallback_percent,
                'tp_distance': abs(take_profit_price - entry_price),
                'risk_reward_ratio': risk_reward_ratio,
                'side': side,
                'fallback': True
            }

    def get_volatility_adjusted_position_size(self, base_position_size: float,
                                            volatility_data: Dict) -> Dict:
        """
        Adjust position size based on volatility
        """
        try:
            multiplier = volatility_data.get('position_multiplier', 1.0)
            volatility_class = volatility_data.get('volatility_class', 'MEDIUM')
            atr_percent = volatility_data.get('atr_percent', 1.0)

            # Apply volatility adjustment
            adjusted_size = base_position_size * multiplier

            # Additional safety check for extreme volatility
            if atr_percent > 5.0:  # Extreme volatility
                adjusted_size *= 0.5  # Further reduce by 50%

            return {
                'original_size': base_position_size,
                'adjusted_size': adjusted_size,
                'volatility_multiplier': multiplier,
                'volatility_class': volatility_class,
                'atr_percent': atr_percent,
                'extreme_volatility_applied': atr_percent > 5.0
            }

        except Exception as e:
            logger.error(f"Volatility position sizing error: {e}")
            return {
                'original_size': base_position_size,
                'adjusted_size': base_position_size,
                'volatility_multiplier': 1.0,
                'error': str(e)
            }

    def _get_default_volatility_data(self) -> Dict:
        """
        Return default volatility data when calculation fails
        """
        return {
            'atr': 0,
            'atr_percent': 1.0,
            'std_volatility': 0,
            'std_volatility_percent': 1.0,
            'hl_volatility_percent': 1.0,
            'movement_velocity': 0,
            'volatility_trend': 0,
            'volatility_class': 'MEDIUM',
            'position_multiplier': 1.0,
            'risk_adjustment': 1.5,
            'current_price': 0,
            'calculation_timestamp': datetime.now().isoformat(),
            'period': self.default_period,
            'default_data': True
        }

    def is_high_volatility_period(self, volatility_data: Dict) -> bool:
        """
        Determine if current market conditions show high volatility
        """
        volatility_class = volatility_data.get('volatility_class', 'MEDIUM')
        return volatility_class in ['HIGH', 'EXTREME']

    def get_cached_volatility(self, symbol: str, timeframe: str = '5m',
                            period: int = None, max_age_minutes: int = 5) -> Optional[Dict]:
        """
        Get cached volatility data if available and not too old
        """
        period = period or self.default_period
        cache_key = f"{symbol}_{timeframe}_{period}"

        if cache_key in self.volatility_cache:
            cached_data = self.volatility_cache[cache_key]
            age_minutes = (datetime.now() - cached_data['timestamp']).total_seconds() / 60

            if age_minutes <= max_age_minutes:
                return cached_data['data']

        return None

    def clear_volatility_cache(self):
        """Clear the volatility cache"""
        self.volatility_cache.clear()
        logger.info("Volatility cache cleared")

    def get_volatility_summary(self, volatility_data: Dict) -> str:
        """
        Get a human-readable summary of volatility conditions
        """
        try:
            volatility_class = volatility_data.get('volatility_class', 'UNKNOWN')
            atr_percent = volatility_data.get('atr_percent', 0)
            trend = volatility_data.get('volatility_trend', 0)

            trend_desc = "increasing" if trend > 0.1 else "decreasing" if trend < -0.1 else "stable"

            return f"{volatility_class} volatility ({atr_percent:.2f}% ATR), {trend_desc}"

        except Exception as e:
            return f"Volatility summary error: {e}"