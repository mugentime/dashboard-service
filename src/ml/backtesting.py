import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from config.config import config

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Comprehensive backtesting engine for strategy validation"""

    def __init__(self):
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.0002   # 0.02% slippage
        self.initial_capital = 10000

    async def run_backtest(self, strategy: Dict, market_data: pd.DataFrame,
                          lookback_days: int = 180) -> Dict:
        """Run comprehensive backtest on strategy"""
        # Prepare data
        end_date = market_data.index[-1]
        start_date = end_date - timedelta(days=lookback_days)
        test_data = market_data[market_data.index >= start_date].copy()

        if len(test_data) < 100:
            logger.warning("Insufficient data for backtesting")
            return {}

        # Get strategy signals
        signals = strategy.get('signals', [])
        if not signals:
            logger.warning("No signals provided for backtesting")
            return {}

        # Run backtest simulation
        trades = await self._simulate_trades(test_data, signals)

        # Calculate performance metrics
        results = self._calculate_performance_metrics(trades, test_data)

        # Add risk metrics
        results.update(self._calculate_risk_metrics(trades, test_data))

        # Add strategy-specific metrics
        results.update(self._calculate_strategy_metrics(strategy, trades))

        return results

    async def _simulate_trades(self, market_data: pd.DataFrame,
                              signals: List[Dict]) -> List[Dict]:
        """Simulate trades based on strategy signals"""
        trades = []
        portfolio = {
            'cash': self.initial_capital,
            'position': 0.0,
            'entry_price': 0.0,
            'unrealized_pnl': 0.0
        }

        for i, row in market_data.iterrows():
            current_price = row['close']
            timestamp = i if isinstance(i, datetime) else datetime.now()

            # Find matching signal
            signal = self._find_signal_for_timestamp(signals, timestamp)

            if signal and signal['action'] != 'HOLD':
                # Execute trade
                trade = await self._execute_trade(
                    signal, current_price, portfolio, timestamp
                )
                if trade:
                    trades.append(trade)

            # Update unrealized PnL for open positions
            if portfolio['position'] != 0:
                portfolio['unrealized_pnl'] = (
                    (current_price - portfolio['entry_price']) *
                    portfolio['position']
                )

        # Close any remaining positions
        if portfolio['position'] != 0:
            final_trade = await self._close_position(
                portfolio, market_data.iloc[-1]['close'],
                market_data.index[-1]
            )
            if final_trade:
                trades.append(final_trade)

        return trades

    def _find_signal_for_timestamp(self, signals: List[Dict],
                                  timestamp: datetime) -> Optional[Dict]:
        """Find the most recent signal for given timestamp"""
        relevant_signals = [
            s for s in signals
            if s['timestamp'] <= timestamp
        ]

        if relevant_signals:
            return max(relevant_signals, key=lambda x: x['timestamp'])

        return None

    async def _execute_trade(self, signal: Dict, price: float,
                           portfolio: Dict, timestamp: datetime) -> Optional[Dict]:
        """Execute a trade based on signal"""
        action = signal['action']
        confidence = signal['confidence']
        position_size = abs(signal.get('position_size', 0.1))

        # Adjust for slippage
        execution_price = price * (1 + self.slippage if action == 'BUY' else 1 - self.slippage)

        # Close existing position if direction change
        if portfolio['position'] != 0:
            if (portfolio['position'] > 0 and action == 'SELL') or \
               (portfolio['position'] < 0 and action == 'BUY'):
                close_trade = await self._close_position(portfolio, price, timestamp)

        # Open new position
        if action in ['BUY', 'SELL']:
            # Calculate position size based on available capital and confidence
            max_position_value = portfolio['cash'] * position_size * confidence
            shares = max_position_value / execution_price

            if action == 'SELL':
                shares = -shares

            # Account for commission
            commission_cost = abs(shares * execution_price * self.commission)

            if portfolio['cash'] >= commission_cost:
                portfolio['position'] = shares
                portfolio['entry_price'] = execution_price
                portfolio['cash'] -= commission_cost

                return {
                    'timestamp': timestamp,
                    'action': action,
                    'price': execution_price,
                    'shares': shares,
                    'commission': commission_cost,
                    'signal_confidence': confidence,
                    'type': 'OPEN'
                }

        return None

    async def _close_position(self, portfolio: Dict, price: float,
                            timestamp: datetime) -> Optional[Dict]:
        """Close current position"""
        if portfolio['position'] == 0:
            return None

        # Adjust for slippage
        execution_price = price * (1 - self.slippage if portfolio['position'] > 0 else 1 + self.slippage)

        # Calculate PnL
        pnl = (execution_price - portfolio['entry_price']) * portfolio['position']
        commission = abs(portfolio['position'] * execution_price * self.commission)
        net_pnl = pnl - commission

        # Update portfolio
        portfolio['cash'] += portfolio['position'] * execution_price - commission

        trade = {
            'timestamp': timestamp,
            'action': 'SELL' if portfolio['position'] > 0 else 'BUY',
            'price': execution_price,
            'shares': -portfolio['position'],
            'commission': commission,
            'pnl': net_pnl,
            'return': net_pnl / (abs(portfolio['position']) * portfolio['entry_price']),
            'type': 'CLOSE'
        }

        # Reset position
        portfolio['position'] = 0
        portfolio['entry_price'] = 0
        portfolio['unrealized_pnl'] = 0

        return trade

    def _calculate_performance_metrics(self, trades: List[Dict],
                                     market_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {'total_return': 0, 'num_trades': 0}

        # Basic metrics
        closed_trades = [t for t in trades if t.get('type') == 'CLOSE']
        returns = [t['return'] for t in closed_trades if 'return' in t]

        if not returns:
            return {'total_return': 0, 'num_trades': 0}

        total_return = sum(returns)
        num_trades = len(closed_trades)
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0

        # Risk metrics
        returns_series = pd.Series(returns)
        volatility = returns_series.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (returns_series.mean() * 252) / volatility if volatility > 0 else 0

        # Drawdown calculation
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0

        # Profit factor
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]

        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Average metrics
        avg_return = np.mean(returns) if returns else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0

        return {
            'total_return': total_return,
            'annualized_return': total_return * 252 / len(market_data) if len(market_data) > 0 else 0,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': max(returns) if returns else 0,
            'worst_trade': min(returns) if returns else 0
        }

    def _calculate_risk_metrics(self, trades: List[Dict],
                               market_data: pd.DataFrame) -> Dict:
        """Calculate advanced risk metrics"""
        closed_trades = [t for t in trades if t.get('type') == 'CLOSE']
        returns = [t['return'] for t in closed_trades if 'return' in t]

        if not returns or len(returns) < 10:
            return {}

        returns_series = pd.Series(returns)

        # Value at Risk (VaR)
        var_95 = returns_series.quantile(0.05)
        var_99 = returns_series.quantile(0.01)

        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns_series[returns_series <= var_95].mean()
        cvar_99 = returns_series[returns_series <= var_99].mean()

        # Skewness and Kurtosis
        skewness = returns_series.skew()
        kurtosis = returns_series.kurtosis()

        # Calmar ratio (annual return / max drawdown)
        annual_return = returns_series.mean() * 252
        max_dd = self._calculate_max_drawdown(returns_series)
        calmar_ratio = annual_return / max_dd if max_dd > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0

        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio
        }

    def _calculate_strategy_metrics(self, strategy: Dict, trades: List[Dict]) -> Dict:
        """Calculate strategy-specific metrics"""
        strategy_type = strategy.get('type', 'unknown')

        # Signal accuracy
        signals = strategy.get('signals', [])
        signal_accuracy = self._calculate_signal_accuracy(signals, trades)

        # Model confidence vs performance correlation
        confidence_performance = self._analyze_confidence_performance(trades)

        return {
            'strategy_type': strategy_type,
            'signal_accuracy': signal_accuracy,
            'confidence_performance_corr': confidence_performance,
            'last_trained': strategy.get('last_trained', datetime.now()).isoformat()
        }

    def _calculate_max_drawdown(self, returns_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min()) if not drawdown.empty else 0

    def _calculate_signal_accuracy(self, signals: List[Dict], trades: List[Dict]) -> float:
        """Calculate how accurate signals were"""
        if not signals or not trades:
            return 0.0

        # This would require more sophisticated matching logic
        # For now, return a simple accuracy based on win rate
        closed_trades = [t for t in trades if t.get('type') == 'CLOSE']
        if not closed_trades:
            return 0.0

        returns = [t['return'] for t in closed_trades if 'return' in t]
        return sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0

    def _analyze_confidence_performance(self, trades: List[Dict]) -> float:
        """Analyze correlation between signal confidence and trade performance"""
        if not trades:
            return 0.0

        # Extract confidence and returns for closed trades
        confidence_returns = []
        for trade in trades:
            if trade.get('type') == 'CLOSE' and 'signal_confidence' in trade and 'return' in trade:
                confidence_returns.append((trade['signal_confidence'], trade['return']))

        if len(confidence_returns) < 5:
            return 0.0

        # Calculate correlation
        confidences = [cr[0] for cr in confidence_returns]
        returns = [cr[1] for cr in confidence_returns]

        return np.corrcoef(confidences, returns)[0, 1] if len(confidences) > 1 else 0.0