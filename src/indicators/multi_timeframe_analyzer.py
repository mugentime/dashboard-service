"""
Multi-Timeframe Signal Analysis Module
Provides confluence-based signal analysis across multiple timeframes
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """
    Analyzes market signals across multiple timeframes for confluence-based trading
    """

    def __init__(self, timeframes: List[str] = None):
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h']

        # Timeframe weights (higher for longer timeframes)
        self.timeframe_weights = {
            '1m': 0.15,   # Short-term noise, lower weight
            '5m': 0.25,   # Current primary timeframe
            '15m': 0.35,  # Medium-term trend
            '1h': 0.25    # Long-term context
        }

        # Minimum confluence threshold
        self.confluence_threshold = 0.6  # 60% weighted agreement needed

    async def get_multi_timeframe_signal(self, client, symbol: str) -> Dict:
        """
        Get signals across multiple timeframes and calculate confluence
        """
        try:
            timeframe_signals = {}
            timeframe_data = {}

            # Gather data from all timeframes
            for tf in self.timeframes:
                try:
                    # Get klines for this timeframe
                    limit = self._get_limit_for_timeframe(tf)
                    klines = await client.get_klines(symbol, tf, limit=limit)

                    if not klines or len(klines) < 20:
                        logger.warning(f"Insufficient data for {symbol} on {tf}")
                        continue

                    # Calculate signal for this timeframe
                    signal_data = await self._calculate_timeframe_signal(klines, tf)
                    timeframe_signals[tf] = signal_data
                    timeframe_data[tf] = klines

                except Exception as e:
                    logger.error(f"Error processing {tf} for {symbol}: {e}")
                    continue

            # Calculate confluence
            confluence_result = self._calculate_confluence(timeframe_signals)

            # Add additional context
            confluence_result.update({
                'timeframes': timeframe_signals,  # Add expected key
                'timeframe_signals': timeframe_signals,  # Keep for backward compatibility
                'analysis_timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframes_analyzed': list(timeframe_signals.keys())
            })

            return confluence_result

        except Exception as e:
            logger.error(f"Multi-timeframe analysis error for {symbol}: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'confluence_score': 0,
                'error': str(e)
            }

    def _get_limit_for_timeframe(self, timeframe: str) -> int:
        """Get appropriate data limit based on timeframe"""
        limits = {
            '1m': 60,   # 1 hour of data
            '5m': 60,   # 5 hours of data
            '15m': 40,  # 10 hours of data
            '1h': 30    # 30 hours of data
        }
        return limits.get(timeframe, 50)

    async def _calculate_timeframe_signal(self, klines: List, timeframe: str) -> Dict:
        """
        Calculate trading signal for a specific timeframe
        Enhanced version of the original signal calculation
        """
        try:
            prices = [float(kline[4]) for kline in klines]  # Close prices
            volumes = [float(kline[5]) for kline in klines]
            highs = [float(kline[2]) for kline in klines]
            lows = [float(kline[3]) for kline in klines]

            if len(prices) < 20:
                return {'signal': 'HOLD', 'confidence': 0, 'strength': 0}

            current_price = prices[-1]

            # Adaptive periods based on timeframe
            periods = self._get_adaptive_periods(timeframe)

            # Calculate multiple momentum indicators
            momentum_signals = []

            # 1. Moving Average Confluence
            ma_signal = self._calculate_ma_confluence(prices, periods)
            momentum_signals.append(ma_signal)

            # 2. RSI with timeframe adjustment
            rsi_signal = self._calculate_rsi_signal(prices, periods['rsi'])
            momentum_signals.append(rsi_signal)

            # 3. Volume-weighted momentum
            volume_signal = self._calculate_volume_momentum(prices, volumes, periods['volume'])
            momentum_signals.append(volume_signal)

            # 4. Price action patterns
            pattern_signal = self._calculate_pattern_signal(highs, lows, prices, periods['pattern'])
            momentum_signals.append(pattern_signal)

            # Combine signals with timeframe-specific weights
            combined_signal = self._combine_timeframe_signals(momentum_signals, timeframe)

            return combined_signal

        except Exception as e:
            logger.error(f"Timeframe signal calculation error: {e}")
            return {'signal': 'HOLD', 'confidence': 0, 'strength': 0}

    def _get_adaptive_periods(self, timeframe: str) -> Dict[str, int]:
        """Get adaptive periods based on timeframe"""
        base_periods = {
            '1m': {'short': 5, 'medium': 10, 'long': 20, 'rsi': 14, 'volume': 10, 'pattern': 10},
            '5m': {'short': 5, 'medium': 10, 'long': 20, 'rsi': 14, 'volume': 10, 'pattern': 10},
            '15m': {'short': 8, 'medium': 15, 'long': 25, 'rsi': 14, 'volume': 12, 'pattern': 12},
            '1h': {'short': 10, 'medium': 20, 'long': 30, 'rsi': 14, 'volume': 15, 'pattern': 15}
        }
        return base_periods.get(timeframe, base_periods['5m'])

    def _calculate_ma_confluence(self, prices: List[float], periods: Dict) -> Dict:
        """Calculate moving average confluence signal"""
        try:
            if len(prices) < max(periods['short'], periods['medium'], periods['long']):
                return {'signal': 'HOLD', 'strength': 0, 'type': 'ma_confluence'}

            short_ma = sum(prices[-periods['short']:]) / periods['short']
            medium_ma = sum(prices[-periods['medium']:]) / periods['medium']
            long_ma = sum(prices[-periods['long']:]) / periods['long']
            current_price = prices[-1]

            # Calculate alignment strength
            alignment_score = 0

            # Price above all MAs = bullish
            if current_price > short_ma > medium_ma > long_ma:
                alignment_score = 1.0
                signal = 'BUY'
            elif current_price < short_ma < medium_ma < long_ma:
                alignment_score = 1.0
                signal = 'SELL'
            elif current_price > short_ma > medium_ma:
                alignment_score = 0.7
                signal = 'BUY'
            elif current_price < short_ma < medium_ma:
                alignment_score = 0.7
                signal = 'SELL'
            elif current_price > short_ma:
                alignment_score = 0.4
                signal = 'BUY'
            elif current_price < short_ma:
                alignment_score = 0.4
                signal = 'SELL'
            else:
                alignment_score = 0
                signal = 'HOLD'

            return {
                'signal': signal,
                'strength': alignment_score,
                'type': 'ma_confluence',
                'values': {
                    'short_ma': short_ma,
                    'medium_ma': medium_ma,
                    'long_ma': long_ma,
                    'current_price': current_price
                }
            }

        except Exception as e:
            logger.error(f"MA confluence calculation error: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'type': 'ma_confluence'}

    def _calculate_rsi_signal(self, prices: List[float], period: int) -> Dict:
        """Calculate RSI-based signal with dynamic thresholds"""
        try:
            if len(prices) < period + 1:
                return {'signal': 'HOLD', 'strength': 0, 'type': 'rsi'}

            # Calculate price changes
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            # Calculate average gains and losses
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            # Dynamic thresholds based on recent volatility
            volatility = np.std(prices[-period:]) / np.mean(prices[-period:])

            # Adjust RSI thresholds based on volatility
            if volatility > 0.02:  # High volatility
                oversold_threshold = 25
                overbought_threshold = 75
            else:  # Normal volatility
                oversold_threshold = 30
                overbought_threshold = 70

            # Generate signal
            if rsi < oversold_threshold:
                signal = 'BUY'
                strength = min((oversold_threshold - rsi) / oversold_threshold, 1.0)
            elif rsi > overbought_threshold:
                signal = 'SELL'
                strength = min((rsi - overbought_threshold) / (100 - overbought_threshold), 1.0)
            else:
                signal = 'HOLD'
                strength = 0

            return {
                'signal': signal,
                'strength': strength,
                'type': 'rsi',
                'values': {
                    'rsi': rsi,
                    'oversold_threshold': oversold_threshold,
                    'overbought_threshold': overbought_threshold
                }
            }

        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'type': 'rsi'}

    def _calculate_volume_momentum(self, prices: List[float], volumes: List[float], period: int) -> Dict:
        """Calculate volume-weighted momentum signal"""
        try:
            if len(prices) < period or len(volumes) < period:
                return {'signal': 'HOLD', 'strength': 0, 'type': 'volume_momentum'}

            # Calculate volume-weighted price change
            recent_prices = prices[-period:]
            recent_volumes = volumes[-period:]

            # Volume-weighted average price (VWAP)
            vwap = sum(p * v for p, v in zip(recent_prices, recent_volumes)) / sum(recent_volumes)
            current_price = prices[-1]

            # Price momentum relative to VWAP
            price_momentum = (current_price - vwap) / vwap

            # Volume trend
            recent_vol_avg = sum(recent_volumes) / len(recent_volumes)
            older_vol_avg = sum(volumes[-period*2:-period]) / period if len(volumes) >= period*2 else recent_vol_avg

            volume_trend = (recent_vol_avg - older_vol_avg) / older_vol_avg if older_vol_avg > 0 else 0

            # Combine price momentum with volume trend
            momentum_strength = abs(price_momentum)
            volume_confirmation = 1 + volume_trend  # Volume should support the move

            combined_strength = min(momentum_strength * volume_confirmation, 1.0)

            # Generate signal
            threshold = 0.003  # 0.3% threshold
            if price_momentum > threshold and volume_trend > 0:
                signal = 'BUY'
            elif price_momentum < -threshold and volume_trend > 0:
                signal = 'SELL'
            else:
                signal = 'HOLD'
                combined_strength = 0

            return {
                'signal': signal,
                'strength': combined_strength,
                'type': 'volume_momentum',
                'values': {
                    'price_momentum': price_momentum,
                    'volume_trend': volume_trend,
                    'vwap': vwap
                }
            }

        except Exception as e:
            logger.error(f"Volume momentum calculation error: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'type': 'volume_momentum'}

    def _calculate_pattern_signal(self, highs: List[float], lows: List[float],
                                  closes: List[float], period: int) -> Dict:
        """Calculate price action pattern signal"""
        try:
            if len(closes) < period:
                return {'signal': 'HOLD', 'strength': 0, 'type': 'pattern'}

            recent_highs = highs[-period:]
            recent_lows = lows[-period:]
            recent_closes = closes[-period:]

            # Higher highs and higher lows pattern (bullish)
            hh_hl_strength = 0
            if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                # Check for uptrend
                if (recent_highs[-1] > recent_highs[-2] > recent_highs[-3] and
                    recent_lows[-1] > recent_lows[-2]):
                    hh_hl_strength = 0.8
                    pattern_signal = 'BUY'
                # Check for downtrend
                elif (recent_highs[-1] < recent_highs[-2] < recent_highs[-3] and
                      recent_lows[-1] < recent_lows[-2]):
                    hh_hl_strength = 0.8
                    pattern_signal = 'SELL'
                else:
                    pattern_signal = 'HOLD'
            else:
                pattern_signal = 'HOLD'

            # Support/Resistance breaks
            current_price = closes[-1]
            resistance_level = max(recent_highs[:-1]) if len(recent_highs) > 1 else current_price
            support_level = min(recent_lows[:-1]) if len(recent_lows) > 1 else current_price

            breakout_strength = 0
            if current_price > resistance_level * 1.002:  # 0.2% breakout
                breakout_strength = 0.6
                if pattern_signal != 'SELL':
                    pattern_signal = 'BUY'
            elif current_price < support_level * 0.998:  # 0.2% breakdown
                breakout_strength = 0.6
                if pattern_signal != 'BUY':
                    pattern_signal = 'SELL'

            combined_strength = max(hh_hl_strength, breakout_strength)

            return {
                'signal': pattern_signal,
                'strength': combined_strength,
                'type': 'pattern',
                'values': {
                    'support_level': support_level,
                    'resistance_level': resistance_level,
                    'hh_hl_strength': hh_hl_strength,
                    'breakout_strength': breakout_strength
                }
            }

        except Exception as e:
            logger.error(f"Pattern calculation error: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'type': 'pattern'}

    def _combine_timeframe_signals(self, signals: List[Dict], timeframe: str) -> Dict:
        """Combine multiple signal types for a single timeframe"""
        if not signals:
            return {'signal': 'HOLD', 'confidence': 0, 'strength': 0}

        # Weight different signal types
        signal_weights = {
            'ma_confluence': 0.35,
            'rsi': 0.25,
            'volume_momentum': 0.25,
            'pattern': 0.15
        }

        buy_strength = 0
        sell_strength = 0
        total_weight = 0

        for signal_data in signals:
            signal_type = signal_data.get('type', 'unknown')
            weight = signal_weights.get(signal_type, 0.1)
            strength = signal_data.get('strength', 0)
            signal = signal_data.get('signal', 'HOLD')

            if signal == 'BUY':
                buy_strength += strength * weight
            elif signal == 'SELL':
                sell_strength += strength * weight

            total_weight += weight

        # Normalize
        if total_weight > 0:
            buy_strength /= total_weight
            sell_strength /= total_weight

        # Determine final signal
        threshold = 0.3  # Minimum strength threshold
        if buy_strength > sell_strength and buy_strength > threshold:
            final_signal = 'BUY'
            confidence = buy_strength
        elif sell_strength > buy_strength and sell_strength > threshold:
            final_signal = 'SELL'
            confidence = sell_strength
        else:
            final_signal = 'HOLD'
            confidence = 0

        return {
            'signal': final_signal,
            'confidence': confidence,
            'strength': max(buy_strength, sell_strength),
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'timeframe': timeframe,
            'component_signals': signals
        }

    def _calculate_confluence(self, timeframe_signals: Dict) -> Dict:
        """Calculate confluence across all timeframes"""
        if not timeframe_signals:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'confluence_score': 0
            }

        buy_score = 0
        sell_score = 0
        total_weight = 0

        # Weight signals by timeframe importance
        for tf, signal_data in timeframe_signals.items():
            if tf not in self.timeframe_weights:
                continue

            weight = self.timeframe_weights[tf]
            confidence = signal_data.get('confidence', 0)
            signal = signal_data.get('signal', 'HOLD')

            if signal == 'BUY':
                buy_score += confidence * weight
            elif signal == 'SELL':
                sell_score += confidence * weight

            total_weight += weight

        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight

        # Calculate confluence
        confluence_score = max(buy_score, sell_score)

        # Determine final signal
        if confluence_score >= self.confluence_threshold:
            if buy_score > sell_score:
                final_signal = 'BUY'
                final_confidence = buy_score
            else:
                final_signal = 'SELL'
                final_confidence = sell_score
        else:
            final_signal = 'HOLD'
            final_confidence = 0

        # Calculate timeframe alignment (percentage of timeframes agreeing)
        agreeing_timeframes = 0
        for tf_signal in timeframe_signals.values():
            if tf_signal.get('signal') == final_signal:
                agreeing_timeframes += 1

        timeframe_alignment = agreeing_timeframes / len(timeframe_signals) if timeframe_signals else 0

        return {
            'overall_signal': final_signal,  # Add expected key
            'overall_confidence': final_confidence,  # Add expected key
            'confluence_score': confluence_score,
            'timeframe_alignment': timeframe_alignment,  # Add expected key
            'buy_score': buy_score,
            'sell_score': sell_score,
            'timeframes_count': len(timeframe_signals),
            'confluence_threshold': self.confluence_threshold,
            # Keep original keys for backward compatibility
            'signal': final_signal,
            'confidence': final_confidence
        }