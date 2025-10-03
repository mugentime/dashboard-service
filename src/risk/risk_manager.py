import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from decimal import Decimal

from config.config import config

logger = logging.getLogger(__name__)

class RiskManager:
    """Comprehensive risk management system for capital protection"""

    def __init__(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 100
        self.position_limits = config.risk.position_limits
        self.stop_loss_methods = config.risk.stop_loss_methods
        self.current_drawdown = 0.0
        self.peak_portfolio_value = 0.0

    async def evaluate_trade_risk(self, symbol: str, signal: Dict, portfolio_state: Dict) -> bool:
        """Evaluate if a trade passes risk checks"""
        try:
            # Check daily loss limit
            if not self._check_daily_loss_limit(portfolio_state):
                logger.warning("Trade rejected: Daily loss limit exceeded")
                return False

            # Check position limits
            if not self._check_position_limits(symbol, signal, portfolio_state):
                logger.warning(f"Trade rejected: Position limits exceeded for {symbol}")
                return False

            # Check correlation limits
            if not await self._check_correlation_limits(symbol, portfolio_state):
                logger.warning(f"Trade rejected: Correlation limits exceeded for {symbol}")
                return False

            # Check leverage limits
            if not self._check_leverage_limits(signal, portfolio_state):
                logger.warning("Trade rejected: Leverage limits exceeded")
                return False

            # Check volatility risk
            if not await self._check_volatility_risk(symbol, signal):
                logger.warning(f"Trade rejected: Volatility risk too high for {symbol}")
                return False

            # Check signal quality
            if not self._check_signal_quality(signal):
                logger.warning("Trade rejected: Signal quality insufficient")
                return False

            return True

        except Exception as e:
            logger.error(f"Error in risk evaluation: {e}")
            return False

    def _check_daily_loss_limit(self, portfolio_state: Dict) -> bool:
        """Check if daily loss limit would be exceeded"""
        current_loss_ratio = abs(self.daily_pnl) / portfolio_state['total_value']
        return current_loss_ratio < config.trading.max_daily_loss

    def _check_position_limits(self, symbol: str, signal: Dict, portfolio_state: Dict) -> bool:
        """Check position size and count limits"""
        # Check maximum number of positions
        current_positions = len(portfolio_state['positions'])
        if current_positions >= self.position_limits['max_open_positions']:
            return False

        # Check maximum exposure per symbol
        current_exposure = 0.0
        if symbol in portfolio_state['positions']:
            position = portfolio_state['positions'][symbol]
            current_exposure = position['notional'] / portfolio_state['total_value']

        proposed_size = abs(signal.get('position_size', 0.1))
        new_exposure = current_exposure + proposed_size

        if new_exposure > self.position_limits['max_exposure_per_symbol']:
            return False

        return True

    async def _check_correlation_limits(self, symbol: str, portfolio_state: Dict) -> bool:
        """Check correlation with existing positions"""
        # This would require historical correlation analysis
        # For now, implement a simple sector exposure check

        # Get symbol sector (simplified mapping)
        symbol_sector = self._get_symbol_sector(symbol)

        # Calculate current sector exposure
        sector_exposure = 0.0
        for pos_symbol, position in portfolio_state['positions'].items():
            if self._get_symbol_sector(pos_symbol) == symbol_sector:
                sector_exposure += position['notional'] / portfolio_state['total_value']

        return sector_exposure < self.position_limits['max_sector_exposure']

    def _check_leverage_limits(self, signal: Dict, portfolio_state: Dict) -> bool:
        """Check leverage limits"""
        current_leverage = portfolio_state.get('leverage_used', 0.0)
        max_leverage = 10.0  # Maximum 10x leverage

        proposed_size = abs(signal.get('position_size', 0.1))
        new_leverage = current_leverage + proposed_size

        return new_leverage <= max_leverage

    async def _check_volatility_risk(self, symbol: str, signal: Dict) -> bool:
        """Check if volatility risk is acceptable"""
        # This would use real volatility data
        # For now, implement a simple confidence-based check
        confidence = signal.get('confidence', 0.0)
        min_confidence = 0.6

        return confidence >= min_confidence

    def _check_signal_quality(self, signal: Dict) -> bool:
        """Check signal quality metrics"""
        confidence = signal.get('confidence', 0.0)

        # Minimum confidence threshold
        if confidence < 0.5:
            return False

        # Check if signal is too old
        signal_time = signal.get('timestamp', datetime.now())
        age_limit = timedelta(hours=1)

        if datetime.now() - signal_time > age_limit:
            return False

        return True

    async def calculate_position_size(self, symbol: str, signal: Dict, portfolio_state: Dict) -> float:
        """Calculate optimal position size using Kelly Criterion and risk limits"""
        try:
            # Base position size from signal
            base_size = abs(signal.get('position_size', 0.1))
            confidence = signal.get('confidence', 0.5)

            # Kelly Criterion calculation (simplified)
            win_rate = 0.6  # This would come from strategy performance
            avg_win = 0.03
            avg_loss = 0.015

            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

            # Risk-adjusted size
            volatility_adjustment = self._get_volatility_adjustment(symbol)
            confidence_adjustment = confidence

            # Calculate final position size
            portfolio_value = portfolio_state['total_value']
            risk_adjusted_size = (
                base_size *
                kelly_fraction *
                volatility_adjustment *
                confidence_adjustment
            )

            # Apply position limits
            max_position_value = portfolio_value * self.position_limits['max_exposure_per_symbol']
            position_value = min(risk_adjusted_size * portfolio_value, max_position_value)

            return position_value

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _get_volatility_adjustment(self, symbol: str) -> float:
        """Get volatility-based position size adjustment"""
        # This would use real volatility data
        # For now, return a standard adjustment
        return 0.8

    async def calculate_stop_loss(self, symbol: str, entry_price: float,
                                position_size: float, method: str = 'atr') -> float:
        """Calculate dynamic stop loss"""
        try:
            if method == 'atr':
                # ATR-based stop loss
                atr_multiplier = 2.0
                estimated_atr = entry_price * 0.02  # 2% of price as ATR estimate
                stop_distance = atr_multiplier * estimated_atr

            elif method == 'volatility':
                # Volatility-based stop loss
                volatility = 0.03  # 3% daily volatility estimate
                stop_distance = entry_price * volatility * 2

            elif method == 'ml_based':
                # ML model predicted stop loss
                stop_distance = await self._ml_stop_loss_prediction(symbol, entry_price)

            elif method == 'time_based':
                # Time-based stop loss (tighter for longer holding)
                base_stop = entry_price * 0.02
                time_factor = 1.0  # Would adjust based on holding period
                stop_distance = base_stop * time_factor

            else:
                # Default percentage stop loss
                stop_distance = entry_price * 0.02  # 2%

            return stop_distance

        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return entry_price * 0.02  # Default 2% stop

    async def calculate_take_profit(self, symbol: str, entry_price: float,
                                   stop_loss: float, method: str = 'risk_reward') -> float:
        """Calculate dynamic take profit"""
        try:
            if method == 'risk_reward':
                # Risk-reward ratio based take profit
                risk_reward_ratio = 2.0  # 2:1 reward to risk
                risk = abs(entry_price - stop_loss)
                take_profit_distance = risk * risk_reward_ratio

            elif method == 'trailing':
                # Trailing take profit
                take_profit_distance = entry_price * 0.05  # 5% initial target

            elif method == 'ml_based':
                # ML predicted take profit
                take_profit_distance = await self._ml_take_profit_prediction(symbol, entry_price)

            else:
                # Default percentage take profit
                take_profit_distance = entry_price * 0.04  # 4%

            return take_profit_distance

        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return entry_price * 0.04  # Default 4%

    async def _ml_stop_loss_prediction(self, symbol: str, entry_price: float) -> float:
        """ML-based stop loss prediction"""
        # This would use a trained model to predict optimal stop loss
        # For now, return a dynamic stop based on recent volatility
        return entry_price * 0.025  # 2.5%

    async def _ml_take_profit_prediction(self, symbol: str, entry_price: float) -> float:
        """ML-based take profit prediction"""
        # This would use a trained model to predict optimal take profit
        # For now, return a dynamic target based on recent price action
        return entry_price * 0.06  # 6%

    async def check_risk_limits(self, portfolio_state: Dict) -> List[Dict]:
        """Check all risk limits and return alerts"""
        alerts = []

        # Daily loss check
        daily_loss_ratio = abs(self.daily_pnl) / portfolio_state['total_value']
        if daily_loss_ratio >= config.trading.max_daily_loss * 0.8:  # 80% of limit
            alerts.append({
                'type': 'daily_loss_limit',
                'severity': 'high',
                'message': f"Daily loss at {daily_loss_ratio*100:.1f}% of limit",
                'value': daily_loss_ratio
            })

        # Drawdown check
        current_drawdown = self._calculate_current_drawdown(portfolio_state)
        if current_drawdown >= config.trading.max_drawdown * 0.8:  # 80% of limit
            alerts.append({
                'type': 'drawdown_limit',
                'severity': 'high',
                'message': f"Drawdown at {current_drawdown*100:.1f}%",
                'value': current_drawdown
            })

        # Position concentration check
        risk_metrics = self._calculate_portfolio_risk_metrics(portfolio_state)
        if risk_metrics['concentration_risk'] > 0.6:  # 60% in top 3 positions
            alerts.append({
                'type': 'concentration_risk',
                'severity': 'medium',
                'message': f"High concentration: {risk_metrics['concentration_risk']*100:.1f}%",
                'value': risk_metrics['concentration_risk']
            })

        # Leverage check
        if portfolio_state.get('leverage_used', 0) > 8.0:  # 80% of max leverage
            alerts.append({
                'type': 'leverage_limit',
                'severity': 'high',
                'message': f"High leverage: {portfolio_state['leverage_used']:.1f}x",
                'value': portfolio_state['leverage_used']
            })

        return alerts

    def _calculate_current_drawdown(self, portfolio_state: Dict) -> float:
        """Calculate current drawdown from peak"""
        current_value = portfolio_state['total_value']

        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
            return 0.0

        if self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
            return drawdown

        return 0.0

    def _calculate_portfolio_risk_metrics(self, portfolio_state: Dict) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        positions = portfolio_state.get('positions', {})
        total_value = portfolio_state['total_value']

        if not positions or total_value == 0:
            return {
                'concentration_risk': 0.0,
                'sector_exposure': {},
                'correlation_risk': 0.0,
                'tail_risk': 0.0
            }

        # Position weights
        weights = []
        sectors = {}

        for symbol, position in positions.items():
            weight = position['notional'] / total_value
            weights.append(weight)

            sector = self._get_symbol_sector(symbol)
            sectors[sector] = sectors.get(sector, 0) + weight

        # Concentration risk (HHI)
        hhi = sum(w**2 for w in weights)

        # Top 3 concentration
        top3_concentration = sum(sorted(weights, reverse=True)[:3])

        return {
            'concentration_risk': top3_concentration,
            'hhi': hhi,
            'sector_exposure': sectors,
            'largest_position': max(weights) if weights else 0,
            'num_positions': len(positions)
        }

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector classification for symbol"""
        # Simplified sector mapping
        crypto_sectors = {
            'BTC': 'Layer1',
            'ETH': 'Layer1',
            'ADA': 'Layer1',
            'SOL': 'Layer1',
            'DOT': 'Layer1',
            'AVAX': 'Layer1',
            'MATIC': 'Layer2',
            'LINK': 'Oracle',
            'UNI': 'DeFi',
            'AAVE': 'DeFi'
        }

        base_symbol = symbol.replace('USDT', '').replace('BUSD', '')
        return crypto_sectors.get(base_symbol, 'Other')

    async def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl_change

    def reset_daily_metrics(self):
        """Reset daily tracking metrics"""
        self.daily_pnl = 0.0
        self.daily_trades = 0

    async def stress_test_portfolio(self, portfolio_state: Dict, scenarios: List[Dict]) -> Dict:
        """Run stress tests on current portfolio"""
        results = {}

        for scenario in scenarios:
            scenario_name = scenario['name']
            price_shocks = scenario['price_shocks']  # Dict of symbol: shock_percentage

            portfolio_value = portfolio_state['total_value']
            stressed_value = portfolio_value

            for symbol, position in portfolio_state['positions'].items():
                if symbol in price_shocks:
                    shock = price_shocks[symbol]
                    position_pnl = position['notional'] * shock * (1 if position['side'] == 'LONG' else -1)
                    stressed_value += position_pnl

            stress_loss = (portfolio_value - stressed_value) / portfolio_value

            results[scenario_name] = {
                'portfolio_loss': stress_loss,
                'new_value': stressed_value,
                'passes_limit': stress_loss <= config.trading.max_drawdown
            }

        return results