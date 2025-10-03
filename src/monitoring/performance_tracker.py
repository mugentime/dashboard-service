import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json

from config.config import config

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Comprehensive performance tracking and analytics"""

    def __init__(self):
        self.trade_history = []
        self.daily_performance = {}
        self.strategy_performance = {}
        self.portfolio_snapshots = []
        self.benchmark_data = {}

    async def record_trade(self, strategy_name: str, trade_result: Dict):
        """Record a completed trade"""
        trade_record = {
            'timestamp': trade_result.get('timestamp', datetime.now()),
            'strategy': strategy_name,
            'symbol': trade_result['symbol'],
            'side': trade_result['side'],
            'quantity': trade_result['quantity'],
            'price': trade_result['price'],
            'order_id': trade_result.get('order_id'),
            'pnl': trade_result.get('pnl', 0),
            'return': trade_result.get('return', 0),
            'confidence': trade_result.get('confidence', 0),
            'commission': trade_result.get('commission', 0)
        }

        self.trade_history.append(trade_record)

        # Update strategy performance
        await self._update_strategy_performance(strategy_name, trade_record)

        # Update daily performance
        await self._update_daily_performance(trade_record)

        logger.info(f"Trade recorded: {strategy_name} - {trade_record['symbol']} - P&L: {trade_record['pnl']:.4f}")

    async def _update_strategy_performance(self, strategy_name: str, trade_record: Dict):
        """Update performance metrics for a specific strategy"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'trades': [],
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'last_updated': datetime.now()
            }

        strategy_perf = self.strategy_performance[strategy_name]
        strategy_perf['trades'].append(trade_record)

        # Calculate updated metrics
        returns = [trade['return'] for trade in strategy_perf['trades']]
        pnls = [trade['pnl'] for trade in strategy_perf['trades']]

        strategy_perf['total_pnl'] = sum(pnls)
        strategy_perf['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
        strategy_perf['avg_return'] = np.mean(returns)

        if len(returns) > 1:
            strategy_perf['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            strategy_perf['max_drawdown'] = self._calculate_max_drawdown(returns)
            strategy_perf['profit_factor'] = self._calculate_profit_factor(returns)

        strategy_perf['last_updated'] = datetime.now()

    async def _update_daily_performance(self, trade_record: Dict):
        """Update daily performance metrics"""
        trade_date = trade_record['timestamp'].date()

        if trade_date not in self.daily_performance:
            self.daily_performance[trade_date] = {
                'trades': [],
                'total_pnl': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'num_trades': 0
            }

        daily_perf = self.daily_performance[trade_date]
        daily_perf['trades'].append(trade_record)
        daily_perf['total_pnl'] += trade_record['pnl']
        daily_perf['num_trades'] += 1

        # Calculate metrics
        returns = [trade['return'] for trade in daily_perf['trades']]
        daily_perf['total_return'] = sum(returns)
        daily_perf['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)

    async def calculate_daily_performance(self) -> Dict:
        """Calculate current day's performance"""
        today = datetime.now().date()

        if today not in self.daily_performance:
            return {
                'date': today.isoformat(),
                'return': 0.0,
                'pnl': 0.0,
                'trades': 0,
                'win_rate': 0.0,
                'target_progress': 0.0
            }

        daily_perf = self.daily_performance[today]

        # Calculate progress toward daily target
        target_return = config.trading.target_daily_return
        target_progress = daily_perf['total_return'] / target_return if target_return > 0 else 0

        return {
            'date': today.isoformat(),
            'return': daily_perf['total_return'],
            'pnl': daily_perf['total_pnl'],
            'trades': daily_perf['num_trades'],
            'win_rate': daily_perf['win_rate'],
            'target_progress': target_progress,
            'meets_target': daily_perf['total_return'] >= target_return
        }

    async def get_daily_pnl(self) -> float:
        """Get today's P&L"""
        today = datetime.now().date()
        if today in self.daily_performance:
            return self.daily_performance[today]['total_pnl']
        return 0.0

    async def calculate_portfolio_metrics(self, portfolio_value: float) -> Dict:
        """Calculate comprehensive portfolio performance metrics"""
        if not self.trade_history:
            return {}

        # Convert trade history to DataFrame for analysis
        df = pd.DataFrame(self.trade_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Calculate cumulative returns
        df['cumulative_return'] = (1 + df['return']).cumprod() - 1

        # Time-based metrics
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        trading_days = (end_date - start_date).days

        # Basic metrics
        total_trades = len(df)
        total_return = df['cumulative_return'].iloc[-1] if not df.empty else 0
        total_pnl = df['pnl'].sum()

        # Win/Loss metrics
        winning_trades = df[df['return'] > 0]
        losing_trades = df[df['return'] < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['return'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['return'].mean() if not losing_trades.empty else 0

        # Risk metrics
        returns = df['return'].values
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)

        # Profit factor
        profit_factor = self._calculate_profit_factor(returns)

        # Calmar ratio
        annualized_return = total_return * (252 / trading_days) if trading_days > 0 else 0
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0

        return {
            'total_trades': total_trades,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'annualized_return': annualized_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'trading_days': trading_days,
            'avg_trades_per_day': total_trades / trading_days if trading_days > 0 else 0,
            'current_portfolio_value': portfolio_value
        }

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        excess_returns = np.array(returns) - (risk_free_rate / 252)
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0

        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return abs(np.min(drawdown))

    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor"""
        if not returns:
            return 0.0

        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))

        return gross_profit / gross_loss if gross_loss > 0 else 0.0

    async def get_strategy_rankings(self) -> List[Dict]:
        """Get strategies ranked by performance"""
        rankings = []

        for strategy_name, performance in self.strategy_performance.items():
            if len(performance['trades']) < 5:  # Need minimum trades for ranking
                continue

            # Calculate composite score
            score = self._calculate_strategy_score(performance)

            rankings.append({
                'strategy': strategy_name,
                'score': score,
                'total_pnl': performance['total_pnl'],
                'win_rate': performance['win_rate'],
                'sharpe_ratio': performance['sharpe_ratio'],
                'num_trades': len(performance['trades']),
                'last_updated': performance['last_updated']
            })

        # Sort by score descending
        rankings.sort(key=lambda x: x['score'], reverse=True)
        return rankings

    def _calculate_strategy_score(self, performance: Dict) -> float:
        """Calculate composite performance score for strategy ranking"""
        # Weighted scoring of multiple metrics
        score = (
            performance['total_pnl'] * 0.3 +
            performance['win_rate'] * 0.25 +
            performance['sharpe_ratio'] * 0.2 +
            (1 - performance['max_drawdown']) * 0.15 +
            performance['profit_factor'] * 0.1
        )

        return max(score, 0.0)

    async def generate_performance_report(self, portfolio_value: float) -> Dict:
        """Generate comprehensive performance report"""
        portfolio_metrics = await self.calculate_portfolio_metrics(portfolio_value)
        daily_performance = await self.calculate_daily_performance()
        strategy_rankings = await self.get_strategy_rankings()

        # Performance vs targets
        target_daily_return = config.trading.target_daily_return
        target_annual_return = target_daily_return * 252

        daily_target_progress = (
            daily_performance['return'] / target_daily_return
            if target_daily_return > 0 else 0
        )

        annual_target_progress = (
            portfolio_metrics.get('annualized_return', 0) / target_annual_return
            if target_annual_return > 0 else 0
        )

        # Risk assessment
        risk_score = self._calculate_risk_score(portfolio_metrics)

        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_metrics': portfolio_metrics,
            'daily_performance': daily_performance,
            'strategy_rankings': strategy_rankings,
            'target_progress': {
                'daily': daily_target_progress,
                'annual': annual_target_progress
            },
            'risk_assessment': {
                'score': risk_score,
                'max_drawdown': portfolio_metrics.get('max_drawdown', 0),
                'volatility': portfolio_metrics.get('volatility', 0),
                'sharpe_ratio': portfolio_metrics.get('sharpe_ratio', 0)
            },
            'recommendations': self._generate_recommendations(portfolio_metrics, daily_performance)
        }

    def _calculate_risk_score(self, portfolio_metrics: Dict) -> float:
        """Calculate risk score (0-100, lower is better)"""
        if not portfolio_metrics:
            return 50.0

        # Components of risk score
        drawdown_score = min(portfolio_metrics.get('max_drawdown', 0) * 100, 100)
        volatility_score = min(portfolio_metrics.get('volatility', 0) * 50, 100)
        sharpe_penalty = max(0, (2.0 - portfolio_metrics.get('sharpe_ratio', 0)) * 20)

        risk_score = (drawdown_score * 0.4 + volatility_score * 0.4 + sharpe_penalty * 0.2)
        return min(risk_score, 100.0)

    def _generate_recommendations(self, portfolio_metrics: Dict, daily_performance: Dict) -> List[str]:
        """Generate performance-based recommendations"""
        recommendations = []

        if not portfolio_metrics:
            return ["Insufficient data for recommendations"]

        # Performance recommendations
        if daily_performance['return'] < config.trading.target_daily_return * 0.5:
            recommendations.append("Daily performance below 50% of target - consider strategy adjustment")

        if portfolio_metrics.get('win_rate', 0) < 0.5:
            recommendations.append("Win rate below 50% - review entry criteria")

        if portfolio_metrics.get('max_drawdown', 0) > config.trading.max_drawdown * 0.8:
            recommendations.append("Drawdown approaching limit - implement tighter risk controls")

        if portfolio_metrics.get('sharpe_ratio', 0) < 1.0:
            recommendations.append("Low Sharpe ratio - optimize risk-adjusted returns")

        if portfolio_metrics.get('profit_factor', 0) < 1.5:
            recommendations.append("Low profit factor - improve trade selection or sizing")

        # Strategy recommendations
        if len(self.strategy_performance) > 3:
            recommendations.append("Consider reducing number of strategies for better focus")

        return recommendations if recommendations else ["Performance within acceptable parameters"]

    async def record_portfolio_snapshot(self, portfolio_state: Dict):
        """Record portfolio snapshot for time-series analysis"""
        snapshot = {
            'timestamp': datetime.now(),
            'total_value': portfolio_state['total_value'],
            'unrealized_pnl': portfolio_state['unrealized_pnl'],
            'num_positions': len(portfolio_state['positions']),
            'leverage': portfolio_state.get('leverage_used', 0),
            'daily_pnl': await self.get_daily_pnl()
        }

        self.portfolio_snapshots.append(snapshot)

        # Keep only last 30 days of snapshots
        cutoff_date = datetime.now() - timedelta(days=30)
        self.portfolio_snapshots = [
            snap for snap in self.portfolio_snapshots
            if snap['timestamp'] > cutoff_date
        ]

    def export_performance_data(self, filepath: str):
        """Export performance data to file"""
        export_data = {
            'trade_history': [
                {**trade, 'timestamp': trade['timestamp'].isoformat()}
                for trade in self.trade_history
            ],
            'daily_performance': {
                str(date): perf for date, perf in self.daily_performance.items()
            },
            'strategy_performance': self.strategy_performance,
            'portfolio_snapshots': [
                {**snap, 'timestamp': snap['timestamp'].isoformat()}
                for snap in self.portfolio_snapshots
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Performance data exported to {filepath}")