#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SELF-OPTIMIZING MULTI-PAIR TRADING BOT
Adaptive learning system that optimizes parameters based on performance
"""

import asyncio
import logging
import sys
import os

# Fix Unicode encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import json
from datetime import datetime, timedelta
from pathlib import Path
import statistics
import redis

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

from config.live_trading_config import LIVE_TRADING_CONFIG
from src.api.binance_client import BinanceFuturesClient
from src.indicators.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from src.indicators.volatility_calculator import VolatilityCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/self_optimizing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SelfOptimizingTradingBot:
    def __init__(self):
        self.client = None
        self.is_running = False
        self.account_balance = 0
        self.initial_balance = 0
        self.trade_count = 0

        # Redis client for dashboard integration
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = None
        try:
            logger.info(f"🔄 Attempting Redis connection to: {redis_url}")
            self.redis_client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=5)
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Redis client connected successfully: {redis_url}")
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}. Dashboard integration disabled.")
            logger.error(f"   Redis URL was: {redis_url}")
            self.redis_client = None

        # Self-optimization data
        self.trade_history = []
        self.pair_performance = {}
        self.optimization_cycles = 0

        # Adaptive parameters (will self-optimize)
        self.adaptive_params = {
            'momentum_threshold': 0.008,    # Starting at 0.8%
            'confidence_multiplier': 50,    # Confidence scaling factor
            'volume_weight': 0.2,          # Volume trend importance
            'short_ma_weight': 0.5,        # Short-term momentum weight
            'med_ma_weight': 0.3,          # Medium-term momentum weight
            'min_confidence_threshold': 0.6 # Minimum confidence to trade
        }

        # Performance tracking for optimization
        self.performance_metrics = {
            'win_rate': 0.5,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

        # Multi-timeframe and volatility analysis
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()
        self.volatility_calculator = VolatilityCalculator()

    async def initialize(self):
        """Initialize the self-optimizing trading bot"""
        try:
            logger.info("Initializing Self-Optimizing Multi-Pair Trading Bot...")

            # Initialize Binance client
            self.client = BinanceFuturesClient()
            await self.client.__aenter__()

            # Verify account connection
            account_info = await self.client.get_account_info()
            self.account_balance = float(account_info.get('totalWalletBalance', 0))
            self.initial_balance = self.account_balance

            logger.info(f"Account verified - Balance: {self.account_balance} USDT")

            # Load previous optimization data if exists
            await self.load_optimization_data()

            # Publish initial parameters to Redis
            self.publish_adaptive_params_to_redis()

            logger.info(f"Self-optimizing bot initialized with {len(LIVE_TRADING_CONFIG['trading_pairs'])} pairs")
            logger.info(f"Current adaptive parameters: {self.adaptive_params}")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def load_optimization_data(self):
        """Load previous optimization data for continuous learning"""
        try:
            if os.path.exists('optimization_data.json'):
                with open('optimization_data.json', 'r') as f:
                    data = json.load(f)
                    self.adaptive_params = data.get('adaptive_params', self.adaptive_params)
                    self.pair_performance = data.get('pair_performance', {})
                    self.optimization_cycles = data.get('optimization_cycles', 0)
                    logger.info(f"Loaded optimization data from {self.optimization_cycles} previous cycles")
        except Exception as e:
            logger.warning(f"Could not load optimization data: {e}")

    async def save_optimization_data(self):
        """Save optimization data for persistence"""
        try:
            data = {
                'adaptive_params': self.adaptive_params,
                'pair_performance': self.pair_performance,
                'optimization_cycles': self.optimization_cycles,
                'last_updated': datetime.now().isoformat()
            }
            with open('optimization_data.json', 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save optimization data: {e}")

    async def get_adaptive_signal(self, symbol):
        """Get signal using adaptive parameters with multi-timeframe analysis"""
        try:
            logger.info(f"Getting enhanced adaptive signal for {symbol}")

            # Get multi-timeframe signal analysis
            multi_tf_signal = await self.multi_timeframe_analyzer.get_multi_timeframe_signal(
                self.client, symbol
            )

            # Get volatility data for risk assessment
            volatility_data = await self.volatility_calculator.calculate_atr_volatility(
                self.client, symbol, '5m'
            )

            # Extract multi-timeframe components
            mt_signal = multi_tf_signal.get('overall_signal', 'HOLD')
            mt_confidence = multi_tf_signal.get('overall_confidence', 0)
            confluence_score = multi_tf_signal.get('confluence_score', 0)
            timeframe_alignment = multi_tf_signal.get('timeframe_alignment', 0)

            # Get traditional adaptive signal components for blending
            klines = await self.client.get_klines(symbol, '5m', limit=20)
            if not klines or len(klines) < 15:
                return {'signal': 'HOLD', 'confidence': 0, 'momentum': 0}

            prices = [float(kline[4]) for kline in klines]
            volumes = [float(kline[5]) for kline in klines]
            current_price = prices[-1]

            # Calculate momentum with adaptive weights
            short_avg = sum(prices[-5:]) / 5
            short_momentum = (current_price - short_avg) / short_avg

            med_avg = sum(prices[-10:]) / 10
            med_momentum = (current_price - med_avg) / med_avg

            # Volume trend
            recent_volume = sum(volumes[-5:]) / 5
            prev_volume = sum(volumes[-10:-5]) / 5
            volume_trend = (recent_volume - prev_volume) / prev_volume if prev_volume > 0 else 0

            # Use adaptive weights for momentum calculation
            momentum_score = (
                short_momentum * self.adaptive_params['short_ma_weight'] +
                med_momentum * self.adaptive_params['med_ma_weight'] +
                volume_trend * self.adaptive_params['volume_weight']
            )

            # Enhanced signal generation combining multiple timeframes and adaptive logic
            threshold = self.adaptive_params['momentum_threshold']

            # Multi-timeframe signal takes priority with high confluence
            if confluence_score >= 0.7 and mt_confidence >= 0.6:
                signal = mt_signal
                # Boost confidence when all timeframes align
                base_confidence = mt_confidence * (1 + confluence_score * 0.3)
                confidence = min(base_confidence, 0.95)
                logger.info(f"Using multi-timeframe signal: {signal} (confluence: {confluence_score:.2f})")
            else:
                # Use traditional momentum with multi-timeframe bias
                if momentum_score > threshold:
                    signal = 'BUY'
                    base_confidence = min(abs(momentum_score) * self.adaptive_params['confidence_multiplier'], 0.95)
                elif momentum_score < -threshold:
                    signal = 'SELL'
                    base_confidence = min(abs(momentum_score) * self.adaptive_params['confidence_multiplier'], 0.95)
                else:
                    signal = 'HOLD'
                    base_confidence = 0

                # Apply multi-timeframe bias
                if mt_signal == signal and mt_confidence > 0.4:
                    # Same direction signals - boost confidence
                    confidence = min(base_confidence * 1.2, 0.95)
                    logger.info(f"Multi-timeframe alignment boost applied")
                elif mt_signal != signal and mt_signal != 'HOLD' and mt_confidence > 0.5:
                    # Conflicting signals - reduce confidence
                    confidence = base_confidence * 0.7
                    logger.info(f"Multi-timeframe conflict detected, reducing confidence")
                else:
                    confidence = base_confidence

            # Apply volatility adjustment to confidence
            volatility_class = volatility_data.get('volatility_class', 'MEDIUM')
            if volatility_class in ['HIGH', 'EXTREME']:
                confidence *= 0.8  # Reduce confidence in high volatility
                logger.info(f"High volatility detected ({volatility_class}), reducing confidence")

            # Apply pair-specific adjustments
            if symbol in self.pair_performance:
                pair_data = self.pair_performance[symbol]
                win_rate = pair_data.get('win_rate', 0.5)

                # Adjust confidence based on pair's historical performance
                if win_rate > 0.6:  # Good performing pair
                    confidence *= 1.2
                elif win_rate < 0.4:  # Poor performing pair
                    confidence *= 0.8

            return {
                'signal': signal,
                'confidence': confidence,
                'momentum': momentum_score,
                'threshold_used': threshold,
                'pair_adjustment': symbol in self.pair_performance,
                # Enhanced multi-timeframe data
                'multi_timeframe': {
                    'signal': mt_signal,
                    'confidence': mt_confidence,
                    'confluence_score': confluence_score,
                    'timeframe_alignment': timeframe_alignment,
                    'timeframes': multi_tf_signal.get('timeframes', {})
                },
                'volatility': {
                    'class': volatility_class,
                    'atr_percent': volatility_data.get('atr_percent', 0),
                    'risk_adjustment': volatility_data.get('risk_adjustment', 1.0)
                },
                'enhanced_analysis': True
            }

        except Exception as e:
            logger.error(f"Adaptive signal generation error for {symbol}: {e}")
            return {'signal': 'HOLD', 'confidence': 0, 'momentum': 0}

    async def execute_trade_with_tracking(self, symbol, signal, position_size, analysis):
        """Execute trade and track for optimization"""
        try:
            # Get current price for tracking
            ticker = await self.client.get_ticker_price(symbol)
            entry_price = float(ticker['price'])

            logger.info(f"EXECUTING ADAPTIVE TRADE: {signal} {symbol} - Size: {position_size:.2f} USDT")

            # Calculate quantity with leverage
            leverage = LIVE_TRADING_CONFIG['leverage']
            effective_size = position_size * leverage
            quantity = effective_size / entry_price

            # Get symbol info for proper precision
            try:
                exchange_info = await self.client._make_request('GET', '/fapi/v1/exchangeInfo')
                symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)

                if symbol_info:
                    # Get quantity precision from lot size filter
                    quantity_precision = 3  # default
                    for filter_info in symbol_info['filters']:
                        if filter_info['filterType'] == 'LOT_SIZE':
                            step_size = float(filter_info['stepSize'])
                            quantity_precision = len(str(step_size).split('.')[-1]) if '.' in str(step_size) else 0
                            break

                    # Get minimum quantity
                    min_qty = 0.001  # default
                    for filter_info in symbol_info['filters']:
                        if filter_info['filterType'] == 'LOT_SIZE':
                            min_qty = float(filter_info['minQty'])
                            break

                    # Apply proper precision and minimum
                    quantity = max(quantity, min_qty)
                    quantity = round(quantity, quantity_precision)
                else:
                    # Fallback
                    quantity = max(quantity, 0.001)
                    quantity = round(quantity, 3)

            except Exception as e:
                logger.warning(f"Could not get symbol info for {symbol}: {e}")
                # Fallback
                quantity = max(quantity, 0.001)
                quantity = round(quantity, 3)

            # Record trade for optimization
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'signal': signal,
                'entry_price': entry_price,
                'position_size': position_size,
                'quantity': quantity,
                'momentum': analysis['momentum'],
                'confidence': analysis['confidence'],
                'threshold_used': analysis['threshold_used'],
                'parameters_used': self.adaptive_params.copy()
            }

            # Execute order
            result = await self.client.place_order(
                symbol=symbol,
                side=signal,
                order_type='MARKET',
                quantity=quantity,
                time_in_force=None
            )

            if result:
                self.trade_count += 1
                trade_record['order_id'] = result.get('orderId')
                trade_record['status'] = 'EXECUTED'

                # Add to trade history for optimization
                self.trade_history.append(trade_record)

                logger.info(f"ADAPTIVE TRADE EXECUTED:")
                logger.info(f"  Order ID: {result.get('orderId')}")
                logger.info(f"  Adaptive Threshold: {analysis['threshold_used']:.4f}")
                logger.info(f"  Total Trades: {self.trade_count}")

                # Update pair performance tracking
                if symbol not in self.pair_performance:
                    self.pair_performance[symbol] = {
                        'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.5,
                        'total_pnl': 0.0, 'last_trades': []
                    }

                self.pair_performance[symbol]['trades'] += 1

                # Publish to Redis after every successful trade
                self.publish_adaptive_params_to_redis()

                return True
            else:
                trade_record['status'] = 'FAILED'
                self.trade_history.append(trade_record)
                return False

        except Exception as e:
            logger.error(f"ADAPTIVE TRADE EXECUTION ERROR: {e}")
            return False

    async def optimize_parameters(self):
        """Self-optimize parameters based on recent performance"""
        try:
            logger.info("STARTING PARAMETER OPTIMIZATION...")

            if len(self.trade_history) < 10:
                logger.info("Not enough trade history for optimization")
                return

            # Analyze recent trades (last 50 trades)
            recent_trades = self.trade_history[-50:]

            # Calculate performance metrics
            profitable_trades = []
            losing_trades = []

            for trade in recent_trades:
                # Simulate P&L calculation (in real implementation, you'd track actual P&L)
                simulated_pnl = self.simulate_trade_outcome(trade)

                if simulated_pnl > 0:
                    profitable_trades.append(trade)
                else:
                    losing_trades.append(trade)

            win_rate = len(profitable_trades) / len(recent_trades) if recent_trades else 0.5

            logger.info(f"Recent performance: {win_rate:.2%} win rate from {len(recent_trades)} trades")

            # Optimize momentum threshold
            if win_rate < 0.4:  # Poor performance - be more selective
                self.adaptive_params['momentum_threshold'] *= 1.1  # Increase threshold
                self.adaptive_params['min_confidence_threshold'] *= 1.05
                logger.info("Performance poor - increasing selectivity")
            elif win_rate > 0.65:  # Good performance - be more aggressive
                self.adaptive_params['momentum_threshold'] *= 0.95  # Decrease threshold
                self.adaptive_params['min_confidence_threshold'] *= 0.98
                logger.info("Performance good - increasing aggressiveness")

            # Optimize confidence multiplier
            avg_confidence_of_winners = statistics.mean([t['confidence'] for t in profitable_trades]) if profitable_trades else 0.5
            avg_confidence_of_losers = statistics.mean([t['confidence'] for t in losing_trades]) if losing_trades else 0.5

            if avg_confidence_of_winners > avg_confidence_of_losers:
                self.adaptive_params['confidence_multiplier'] *= 1.02
            else:
                self.adaptive_params['confidence_multiplier'] *= 0.98

            # Optimize weight distribution
            self.optimize_weight_distribution(profitable_trades, losing_trades)

            # Ensure parameters stay within reasonable bounds
            self.enforce_parameter_bounds()

            self.optimization_cycles += 1
            await self.save_optimization_data()

            # Publish adaptive parameters to Redis for dashboard
            self.publish_adaptive_params_to_redis()

            logger.info(f"OPTIMIZATION COMPLETE (Cycle {self.optimization_cycles}):")
            logger.info(f"   Momentum Threshold: {self.adaptive_params['momentum_threshold']:.4f}")
            logger.info(f"   Confidence Multiplier: {self.adaptive_params['confidence_multiplier']:.1f}")
            logger.info(f"   Min Confidence: {self.adaptive_params['min_confidence_threshold']:.2f}")

        except Exception as e:
            logger.error(f"Parameter optimization error: {e}")

    def simulate_trade_outcome(self, trade):
        """Simulate trade outcome based on momentum and market behavior"""
        # This is a simplified simulation - in real implementation you'd track actual P&L
        momentum = trade['momentum']
        confidence = trade['confidence']
        signal = trade['signal']

        # Simple profit simulation based on momentum strength
        if signal == 'BUY':
            simulated_return = momentum * confidence * 100  # Simplified calculation
        else:  # SELL
            simulated_return = -momentum * confidence * 100

        return simulated_return

    def optimize_weight_distribution(self, winners, losers):
        """Optimize the weight distribution between momentum components"""
        if not winners or not losers:
            return

        # Analyze which momentum components worked best for winners
        winner_short_momentum = statistics.mean([abs(self.get_short_momentum(t)) for t in winners])
        winner_med_momentum = statistics.mean([abs(self.get_med_momentum(t)) for t in winners])

        loser_short_momentum = statistics.mean([abs(self.get_short_momentum(t)) for t in losers])
        loser_med_momentum = statistics.mean([abs(self.get_med_momentum(t)) for t in losers])

        # Adjust weights based on which component was more predictive
        if winner_short_momentum > loser_short_momentum:
            self.adaptive_params['short_ma_weight'] = min(0.7, self.adaptive_params['short_ma_weight'] * 1.05)
        else:
            self.adaptive_params['short_ma_weight'] = max(0.3, self.adaptive_params['short_ma_weight'] * 0.95)

        if winner_med_momentum > loser_med_momentum:
            self.adaptive_params['med_ma_weight'] = min(0.5, self.adaptive_params['med_ma_weight'] * 1.05)
        else:
            self.adaptive_params['med_ma_weight'] = max(0.2, self.adaptive_params['med_ma_weight'] * 0.95)

        # Normalize weights
        total_weight = self.adaptive_params['short_ma_weight'] + self.adaptive_params['med_ma_weight']
        remaining_weight = 1.0 - self.adaptive_params['volume_weight']

        self.adaptive_params['short_ma_weight'] = (self.adaptive_params['short_ma_weight'] / total_weight) * remaining_weight
        self.adaptive_params['med_ma_weight'] = (self.adaptive_params['med_ma_weight'] / total_weight) * remaining_weight

    def get_short_momentum(self, trade):
        """Extract short momentum from trade data"""
        return trade.get('momentum', 0) * 0.5  # Approximation

    def get_med_momentum(self, trade):
        """Extract medium momentum from trade data"""
        return trade.get('momentum', 0) * 0.3  # Approximation

    def enforce_parameter_bounds(self):
        """Keep parameters within reasonable bounds"""
        self.adaptive_params['momentum_threshold'] = max(0.003, min(0.020, self.adaptive_params['momentum_threshold']))
        self.adaptive_params['confidence_multiplier'] = max(20, min(100, self.adaptive_params['confidence_multiplier']))
        self.adaptive_params['min_confidence_threshold'] = max(0.3, min(0.9, self.adaptive_params['min_confidence_threshold']))
        self.adaptive_params['volume_weight'] = max(0.1, min(0.4, self.adaptive_params['volume_weight']))

    def publish_adaptive_params_to_redis(self):
        """Publish adaptive parameters to Redis for dashboard display"""
        if not self.redis_client:
            logger.warning("⚠️ Redis client not available, skipping parameter publish")
            return

        try:
            # Publish all 6 adaptive parameters to Redis
            params_json = json.dumps(self.adaptive_params)
            self.redis_client.set('bot:adaptive_params', params_json)

            # Update timestamp for stale data detection
            timestamp = datetime.now().isoformat()
            self.redis_client.set('bot:last_update', timestamp)

            # Publish bot status
            self.redis_client.set('bot:status', 'RUNNING')
            self.redis_client.set('bot:trade_count', str(self.trade_count))
            self.redis_client.set('bot:heartbeat', timestamp)

            logger.info(f"📤 Published adaptive parameters to Redis:")
            logger.info(f"   momentum_threshold: {self.adaptive_params['momentum_threshold']}")
            logger.info(f"   confidence_multiplier: {self.adaptive_params['confidence_multiplier']}")
            logger.info(f"   volume_weight: {self.adaptive_params['volume_weight']}")
            logger.info(f"   short_ma_weight: {self.adaptive_params['short_ma_weight']}")
            logger.info(f"   med_ma_weight: {self.adaptive_params['med_ma_weight']}")
            logger.info(f"   min_confidence_threshold: {self.adaptive_params['min_confidence_threshold']}")
            logger.info(f"   Timestamp: {timestamp}")
        except Exception as e:
            logger.error(f"❌ Failed to publish adaptive parameters to Redis: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def start_trading(self):
        """Start the self-optimizing trading loop"""
        if not await self.initialize():
            logger.error("Failed to initialize. Aborting trading.")
            return

        logger.info("=" * 60)
        logger.info("STARTING SELF-OPTIMIZING LIVE TRADING")
        logger.info("=" * 60)
        logger.info("Features:")
        logger.info("  - Adaptive parameter optimization")
        logger.info("  - Performance-based learning")
        logger.info("  - Pair-specific adjustments")
        logger.info("  - Continuous strategy evolution")
        logger.info(f"Trading Pairs: {len(LIVE_TRADING_CONFIG['trading_pairs'])}")
        logger.info(f"Initial Balance: {self.account_balance} USDT")
        logger.info("=" * 60)

        self.is_running = True
        cycle_count = 0

        try:
            while self.is_running:
                cycle_count += 1
                logger.info(f"ADAPTIVE TRADING CYCLE {cycle_count}")

                # Process all pairs
                for i, symbol in enumerate(LIVE_TRADING_CONFIG['trading_pairs']):
                    try:
                        await self.process_symbol_adaptively(symbol)
                        if i < len(LIVE_TRADING_CONFIG['trading_pairs']) - 1:
                            await asyncio.sleep(2)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")

                # Self-optimize every 5 cycles
                if cycle_count % 5 == 0:
                    await self.optimize_parameters()

                # Performance check every 10 cycles
                if cycle_count % 10 == 0:
                    await self.performance_check()

                # Publish heartbeat to Redis every cycle
                self.publish_adaptive_params_to_redis()
                logger.info(f"📤 Heartbeat update (cycle {cycle_count})")

                logger.info(f"Adaptive cycle {cycle_count} completed")
                await asyncio.sleep(60)

        except KeyboardInterrupt:
            logger.info("Self-optimizing trading stopped by user")
        except Exception as e:
            logger.error(f"Self-optimizing trading error: {e}")
        finally:
            await self.shutdown()

    async def process_symbol_adaptively(self, symbol):
        """Process symbol with adaptive parameters"""
        try:
            analysis = await self.get_adaptive_signal(symbol)
            signal = analysis['signal']
            confidence = analysis['confidence']

            logger.info(f"{symbol}: {signal} (Conf: {confidence:.2%}, Thresh: {analysis['threshold_used']:.4f})")

            if (signal in ['BUY', 'SELL'] and
                confidence >= self.adaptive_params['min_confidence_threshold']):

                position_size = min(
                    LIVE_TRADING_CONFIG['base_position_size'] * confidence,
                    LIVE_TRADING_CONFIG['max_position_size']
                )

                success = await self.execute_trade_with_tracking(symbol, signal, position_size, analysis)

                if success:
                    logger.info(f"ADAPTIVE TRADE COMPLETED: {signal} {symbol}")

        except Exception as e:
            logger.error(f"Adaptive processing error for {symbol}: {e}")

    async def performance_check(self):
        """Check performance metrics"""
        try:
            account_info = await self.client.get_account_info()
            current_balance = float(account_info.get('totalWalletBalance', 0))

            pnl = current_balance - self.initial_balance
            pnl_percent = (pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0

            logger.info("ADAPTIVE PERFORMANCE CHECK:")
            logger.info(f"Current Balance: {current_balance:.2f} USDT")
            logger.info(f"P&L: {pnl:+.2f} USDT ({pnl_percent:+.2f}%)")
            logger.info(f"Total Trades: {self.trade_count}")
            logger.info(f"Optimization Cycles: {self.optimization_cycles}")

            self.account_balance = current_balance

        except Exception as e:
            logger.error(f"Performance check error: {e}")

    async def shutdown(self):
        """Safely shutdown and save optimization data"""
        logger.info("Shutting down self-optimizing trading bot...")
        self.is_running = False

        await self.save_optimization_data()

        if self.client:
            await self.client.__aexit__(None, None, None)

        logger.info("Self-optimizing bot shutdown complete")

async def main():
    """Main entry point"""
    print("=" * 60)
    print("SELF-OPTIMIZING MULTI-PAIR TRADING BOT")
    print("=" * 60)
    print("Features:")
    print("  - Adaptive parameter optimization")
    print("  - Continuous learning from performance")
    print("  - Pair-specific strategy adjustments")
    print("  - Real-time strategy evolution")
    print("  - Persistent optimization memory")
    print("=" * 60)

    os.makedirs("logs", exist_ok=True)

    bot = SelfOptimizingTradingBot()
    await bot.start_trading()

if __name__ == "__main__":
    asyncio.run(main())