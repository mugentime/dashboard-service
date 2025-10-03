import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from config.config import config

logger = logging.getLogger(__name__)

class PortfolioManager:
    """Manages trading portfolio, positions, and account state"""

    def __init__(self):
        self.binance_client = None
        self.positions = {}
        self.account_balance = {}
        self.trade_history = []
        self.portfolio_value = 0.0
        self.unrealized_pnl = 0.0

    async def initialize(self, binance_client):
        """Initialize portfolio manager"""
        self.binance_client = binance_client

        # Get account info
        await self._update_account_info()

        # Get current positions
        await self._update_positions()

        logger.info("Portfolio Manager initialized")

    async def _update_account_info(self):
        """Update account balance and information"""
        try:
            account_info = await self.binance_client.get_account_info()

            # Update balance
            for balance in account_info.get('assets', []):
                asset = balance['asset']
                self.account_balance[asset] = {
                    'free': float(balance['walletBalance']),
                    'locked': float(balance['marginBalance']) - float(balance['walletBalance']),
                    'total': float(balance['marginBalance'])
                }

            # Update portfolio value
            self.portfolio_value = sum(
                balance['total'] for balance in self.account_balance.values()
            )

            logger.debug(f"Portfolio value updated: ${self.portfolio_value:.2f}")

        except Exception as e:
            logger.error(f"Error updating account info: {e}")

    async def _update_positions(self):
        """Update current positions"""
        try:
            positions = await self.binance_client.get_position_info()

            self.positions = {}
            total_unrealized = 0.0

            for position in positions:
                symbol = position['symbol']
                size = float(position['positionAmt'])

                if size != 0:  # Only track non-zero positions
                    self.positions[symbol] = {
                        'size': size,
                        'entry_price': float(position['entryPrice']),
                        'mark_price': float(position['markPrice']),
                        'unrealized_pnl': float(position['unRealizedProfit']),
                        'percentage': float(position['percentage']),
                        'side': 'LONG' if size > 0 else 'SHORT',
                        'notional': abs(size) * float(position['markPrice'])
                    }

                    total_unrealized += float(position['unRealizedProfit'])

            self.unrealized_pnl = total_unrealized
            logger.debug(f"Positions updated: {len(self.positions)} active positions")

        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    async def update_position(self, trade_result: Dict):
        """Update position after trade execution"""
        symbol = trade_result['symbol']
        side = trade_result['side']
        quantity = trade_result['quantity']
        price = trade_result['price']

        # Add to trade history
        self.trade_history.append(trade_result)

        # Update positions
        await self._update_positions()

        # Update account info
        await self._update_account_info()

        logger.info(f"Position updated for {symbol}: {side} {quantity} @ {price}")

    async def close_position(self, symbol: str) -> bool:
        """Close a specific position"""
        try:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return False

            position = self.positions[symbol]
            side = 'SELL' if position['side'] == 'LONG' else 'BUY'
            quantity = abs(position['size'])

            # Place closing order
            order_result = await self.binance_client.place_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity
            )

            logger.info(f"Position closed for {symbol}: {order_result}")

            # Update positions
            await self._update_positions()
            return True

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False

    async def close_all_positions(self):
        """Close all open positions"""
        logger.info("Closing all positions...")

        tasks = []
        for symbol in list(self.positions.keys()):
            tasks.append(self.close_position(symbol))

        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("All positions closed")

    async def reduce_positions(self, reduction_factor: float = 0.5):
        """Reduce all positions by a factor"""
        logger.info(f"Reducing positions by {reduction_factor*100}%")

        for symbol, position in self.positions.items():
            try:
                current_size = abs(position['size'])
                reduction_size = current_size * reduction_factor

                side = 'SELL' if position['side'] == 'LONG' else 'BUY'

                # Place reduction order
                await self.binance_client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type='MARKET',
                    quantity=reduction_size
                )

                logger.info(f"Reduced position for {symbol} by {reduction_size}")

            except Exception as e:
                logger.error(f"Error reducing position for {symbol}: {e}")

        # Update positions
        await self._update_positions()

    def get_portfolio_state(self) -> Dict:
        """Get current portfolio state"""
        return {
            'total_value': self.portfolio_value,
            'unrealized_pnl': self.unrealized_pnl,
            'positions': self.positions.copy(),
            'balance': self.account_balance.copy(),
            'num_positions': len(self.positions),
            'leverage_used': self._calculate_total_leverage(),
            'margin_ratio': self._calculate_margin_ratio()
        }

    def _calculate_total_leverage(self) -> float:
        """Calculate total portfolio leverage"""
        if self.portfolio_value == 0:
            return 0.0

        total_notional = sum(
            position['notional'] for position in self.positions.values()
        )

        return total_notional / self.portfolio_value if self.portfolio_value > 0 else 0.0

    def _calculate_margin_ratio(self) -> float:
        """Calculate margin ratio"""
        if 'USDT' not in self.account_balance:
            return 0.0

        free_balance = self.account_balance['USDT']['free']
        total_balance = self.account_balance['USDT']['total']

        return free_balance / total_balance if total_balance > 0 else 0.0

    async def get_total_value(self) -> float:
        """Get total portfolio value"""
        await self._update_account_info()
        return self.portfolio_value

    def get_position_size(self, symbol: str) -> float:
        """Get current position size for symbol"""
        if symbol in self.positions:
            return self.positions[symbol]['size']
        return 0.0

    def get_available_balance(self, asset: str = 'USDT') -> float:
        """Get available balance for trading"""
        if asset in self.account_balance:
            return self.account_balance[asset]['free']
        return 0.0

    def is_position_open(self, symbol: str) -> bool:
        """Check if position is open for symbol"""
        return symbol in self.positions and self.positions[symbol]['size'] != 0

    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed position information"""
        return self.positions.get(symbol)

    async def calculate_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        # This would require tracking positions from start of day
        # For now, return current unrealized PnL
        return self.unrealized_pnl

    def get_risk_metrics(self) -> Dict:
        """Get portfolio risk metrics"""
        portfolio_value = self.portfolio_value

        if portfolio_value == 0:
            return {
                'leverage': 0.0,
                'margin_ratio': 0.0,
                'largest_position': 0.0,
                'concentration_risk': 0.0
            }

        # Calculate largest position as % of portfolio
        largest_position = 0.0
        if self.positions:
            largest_position = max(
                position['notional'] / portfolio_value
                for position in self.positions.values()
            )

        # Calculate concentration risk (sum of top 3 positions)
        position_weights = sorted([
            position['notional'] / portfolio_value
            for position in self.positions.values()
        ], reverse=True)

        concentration_risk = sum(position_weights[:3])

        return {
            'leverage': self._calculate_total_leverage(),
            'margin_ratio': self._calculate_margin_ratio(),
            'largest_position': largest_position,
            'concentration_risk': concentration_risk,
            'num_positions': len(self.positions),
            'unrealized_pnl_ratio': self.unrealized_pnl / portfolio_value if portfolio_value > 0 else 0
        }