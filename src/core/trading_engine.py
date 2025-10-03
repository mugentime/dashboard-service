import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import json

from config.config import config
from src.api.binance_client import BinanceFuturesClient, WebSocketManager
from src.ml.strategy_generator import StrategyGenerator
from src.risk.risk_manager import RiskManager
from src.core.portfolio_manager import PortfolioManager
from src.monitoring.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

class TradingEngine:
    """Core trading engine orchestrating all trading operations"""

    def __init__(self):
        self.binance_client = None
        self.ws_manager = None
        self.strategy_generator = StrategyGenerator()
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager()
        self.performance_tracker = PerformanceTracker()

        self.active_strategies = {}
        self.market_data = {}
        self.running = False
        self.last_strategy_update = None

    async def initialize(self):
        """Initialize the trading engine"""
        logger.info("Initializing Trading Engine...")

        # Initialize Binance client
        self.binance_client = BinanceFuturesClient()
        await self.binance_client.__aenter__()

        # Initialize WebSocket manager
        self.ws_manager = WebSocketManager(self.binance_client)

        # Initialize portfolio
        await self.portfolio_manager.initialize(self.binance_client)

        # Load existing strategies
        await self._load_strategies()

        logger.info("Trading Engine initialized successfully")

    async def start(self):
        """Start the trading engine"""
        if self.running:
            logger.warning("Trading engine is already running")
            return

        self.running = True
        logger.info("Starting Trading Engine...")

        # Start all tasks concurrently
        tasks = [
            asyncio.create_task(self._market_data_loop()),
            asyncio.create_task(self._strategy_execution_loop()),
            asyncio.create_task(self._risk_monitoring_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._strategy_evolution_loop())
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Trading engine error: {e}")
            await self.stop()

    async def stop(self):
        """Stop the trading engine"""
        logger.info("Stopping Trading Engine...")
        self.running = False

        # Close all positions
        await self.portfolio_manager.close_all_positions()

        # Close Binance client
        if self.binance_client:
            await self.binance_client.__aexit__(None, None, None)

    async def _market_data_loop(self):
        """Continuously collect and process market data"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']

        while self.running:
            try:
                for symbol in symbols:
                    # Get latest klines
                    klines = await self.binance_client.get_klines(
                        symbol=symbol,
                        interval='1h',
                        limit=200
                    )

                    # Process and store market data
                    market_df = self._process_klines(klines)
                    self.market_data[symbol] = market_df

                    # Update real-time data via WebSocket
                    await self._setup_websocket_streams(symbol)

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Market data loop error: {e}")
                await asyncio.sleep(30)

    async def _strategy_execution_loop(self):
        """Execute trading strategies"""
        while self.running:
            try:
                for symbol, market_data in self.market_data.items():
                    if len(market_data) < 100:  # Need sufficient data
                        continue

                    # Get active strategies for this symbol
                    strategies = self.active_strategies.get(symbol, {})

                    for strategy_name, strategy in strategies.items():
                        await self._execute_strategy(strategy_name, strategy, symbol, market_data)

                await asyncio.sleep(30)  # Execute every 30 seconds

            except Exception as e:
                logger.error(f"Strategy execution error: {e}")
                await asyncio.sleep(60)

    async def _execute_strategy(self, strategy_name: str, strategy: Dict,
                               symbol: str, market_data: Any):
        """Execute a specific strategy"""
        try:
            # Get latest signal
            signal = await self._get_latest_signal(strategy, market_data)

            if not signal or signal['action'] == 'HOLD':
                return

            # Risk check
            risk_approved = await self.risk_manager.evaluate_trade_risk(
                symbol, signal, self.portfolio_manager.get_portfolio_state()
            )

            if not risk_approved:
                logger.warning(f"Trade rejected by risk manager: {symbol} - {signal}")
                return

            # Calculate position size
            position_size = await self.risk_manager.calculate_position_size(
                symbol, signal, self.portfolio_manager.get_portfolio_state()
            )

            # Execute trade
            if position_size > 0:
                trade_result = await self._place_trade(symbol, signal, position_size)

                if trade_result:
                    # Update portfolio
                    await self.portfolio_manager.update_position(trade_result)

                    # Track performance
                    await self.performance_tracker.record_trade(
                        strategy_name, trade_result
                    )

                    # Update strategy performance
                    await self.strategy_generator.update_strategy_performance(
                        strategy_name, trade_result
                    )

        except Exception as e:
            logger.error(f"Strategy execution error for {strategy_name}: {e}")

    async def _get_latest_signal(self, strategy: Dict, market_data: Any) -> Optional[Dict]:
        """Get the latest trading signal from strategy"""
        try:
            # This would generate real-time signals using the trained model
            # For now, return a sample signal
            signals = strategy.get('signals', [])
            if signals:
                return signals[-1]  # Return latest signal

            return None
        except Exception as e:
            logger.error(f"Error getting signal: {e}")
            return None

    async def _place_trade(self, symbol: str, signal: Dict, position_size: float) -> Optional[Dict]:
        """Place a trade on Binance"""
        try:
            action = signal['action']
            confidence = signal['confidence']

            # Get current price
            ticker = await self.binance_client.get_ticker_price(symbol)
            current_price = float(ticker['price'])

            # Calculate quantity
            quantity = position_size / current_price

            # Place order
            order_result = await self.binance_client.place_order(
                symbol=symbol,
                side=action,
                order_type='MARKET',
                quantity=quantity
            )

            # Create trade record
            trade_result = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': action,
                'quantity': quantity,
                'price': current_price,
                'order_id': order_result['orderId'],
                'confidence': confidence,
                'signal': signal
            }

            logger.info(f"Trade executed: {trade_result}")
            return trade_result

        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            return None

    async def _risk_monitoring_loop(self):
        """Continuously monitor risk metrics"""
        while self.running:
            try:
                # Get portfolio state
                portfolio_state = self.portfolio_manager.get_portfolio_state()

                # Check risk limits
                risk_alerts = await self.risk_manager.check_risk_limits(portfolio_state)

                if risk_alerts:
                    logger.warning(f"Risk alerts: {risk_alerts}")

                    # Take risk mitigation actions
                    for alert in risk_alerts:
                        await self._handle_risk_alert(alert)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(60)

    async def _performance_monitoring_loop(self):
        """Monitor overall system performance"""
        while self.running:
            try:
                # Calculate daily performance
                daily_performance = await self.performance_tracker.calculate_daily_performance()

                # Check performance targets
                if daily_performance['return'] < -config.trading.max_daily_loss:
                    logger.critical("Daily loss limit exceeded - stopping trading")
                    await self._emergency_stop()

                # Log performance
                logger.info(f"Daily performance: {daily_performance}")

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(300)

    async def _strategy_evolution_loop(self):
        """Continuously evolve and improve strategies"""
        while self.running:
            try:
                # Check if strategies need updating (every 24 hours)
                if (not self.last_strategy_update or
                    datetime.now() - self.last_strategy_update > timedelta(hours=24)):

                    await self._update_strategies()
                    self.last_strategy_update = datetime.now()

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Strategy evolution error: {e}")
                await asyncio.sleep(3600)

    async def _update_strategies(self):
        """Update and evolve trading strategies"""
        logger.info("Updating trading strategies...")

        for symbol, market_data in self.market_data.items():
            if len(market_data) >= config.ml.min_backtest_period:
                # Generate new strategies
                new_strategies = await self.strategy_generator.generate_strategies(market_data)

                # Update active strategies
                if new_strategies:
                    self.active_strategies[symbol] = new_strategies
                    logger.info(f"Updated strategies for {symbol}: {list(new_strategies.keys())}")

    async def _setup_websocket_streams(self, symbol: str):
        """Set up WebSocket streams for real-time data"""
        try:
            # Subscribe to ticker updates
            await self.ws_manager.subscribe_ticker(
                symbol, self._handle_ticker_update
            )

            # Subscribe to kline updates
            await self.ws_manager.subscribe_kline(
                symbol, '1m', self._handle_kline_update
            )

        except Exception as e:
            logger.error(f"WebSocket setup error: {e}")

    async def _handle_ticker_update(self, data: Dict):
        """Handle real-time ticker updates"""
        symbol = data['s']
        price = float(data['c'])

        # Update current price in market data
        if symbol in self.market_data:
            # Update the latest price
            pass  # Implementation depends on data structure

    async def _handle_kline_update(self, data: Dict):
        """Handle real-time kline updates"""
        symbol = data['s']
        kline = data['k']

        if kline['x']:  # Kline is closed
            # Update market data with new kline
            pass  # Implementation depends on data structure

    async def _handle_risk_alert(self, alert: Dict):
        """Handle risk management alerts"""
        alert_type = alert['type']

        if alert_type == 'daily_loss_limit':
            await self._emergency_stop()
        elif alert_type == 'drawdown_limit':
            await self.portfolio_manager.reduce_positions(0.5)
        elif alert_type == 'position_limit':
            await self.portfolio_manager.close_position(alert['symbol'])

    async def _emergency_stop(self):
        """Emergency stop all trading"""
        logger.critical("EMERGENCY STOP ACTIVATED")

        # Close all positions
        await self.portfolio_manager.close_all_positions()

        # Stop strategy execution
        self.running = False

    def _process_klines(self, klines: List) -> Any:
        """Process raw klines into DataFrame"""
        import pandas as pd

        # Convert klines to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                  'taker_buy_quote_volume', 'ignore']

        df = pd.DataFrame(klines, columns=columns)

        # Convert to appropriate types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    async def _load_strategies(self):
        """Load existing strategies from storage"""
        try:
            # Load strategies from file/database
            # For now, generate initial strategies
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']

            for symbol in symbols:
                # Initialize with empty strategies
                self.active_strategies[symbol] = {}

            logger.info("Strategies loaded successfully")

        except Exception as e:
            logger.error(f"Error loading strategies: {e}")

    async def get_status(self) -> Dict:
        """Get current engine status"""
        return {
            'running': self.running,
            'active_strategies': len(self.active_strategies),
            'portfolio_value': await self.portfolio_manager.get_total_value(),
            'daily_pnl': await self.performance_tracker.get_daily_pnl(),
            'last_update': datetime.now().isoformat()
        }