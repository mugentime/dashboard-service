"""
Bot Coordinator for Multi-Bot Trading System
Manages coordination between multiple trading bots to prevent conflicts
and optimize portfolio-wide risk management
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict

from .redis_manager import RedisManager

logger = logging.getLogger(__name__)


@dataclass
class BotStatus:
    """Bot status information"""
    bot_id: str
    bot_type: str
    is_active: bool
    total_positions: int
    total_exposure: float
    current_pnl: float
    last_heartbeat: datetime
    trading_pairs: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME


@dataclass
class TradingConflict:
    """Trading conflict information"""
    symbol: str
    conflicting_bots: List[str]
    conflict_type: str  # SAME_SYMBOL, CORRELATION, EXPOSURE_LIMIT
    severity: str  # LOW, MEDIUM, HIGH
    recommendation: str


@dataclass
class CoordinationDecision:
    """Coordination decision result"""
    bot_id: str
    symbol: str
    action: str  # PROCEED, WAIT, REDUCE_SIZE, SKIP
    reason: str
    modified_position_size: Optional[float] = None
    wait_duration: Optional[int] = None


class BotCoordinator:
    """
    Coordinates trading activities between multiple bots to prevent conflicts
    and optimize overall portfolio performance
    """

    def __init__(self, redis_manager: RedisManager, bot_id: str, bot_type: str = "unknown"):
        self.redis_manager = redis_manager
        self.bot_id = bot_id
        self.bot_type = bot_type

        # Coordination settings
        self.max_portfolio_exposure = 0.8  # 80% max total exposure
        self.max_symbol_exposure = 0.3     # 30% max exposure per symbol
        self.correlation_threshold = 0.7    # High correlation threshold
        self.heartbeat_interval = 30        # seconds

        # Tracking
        self.active_bots: Dict[str, BotStatus] = {}
        self.symbol_conflicts: Dict[str, List[str]] = {}
        self.last_coordination_check = datetime.now()

        # Event handlers
        self.conflict_handlers = []
        self.coordination_handlers = []

    async def initialize(self) -> bool:
        """Initialize coordinator and start monitoring"""
        try:
            logger.info(f"Initializing bot coordinator for {self.bot_id} ({self.bot_type})")

            # Subscribe to coordination channels
            await self._setup_subscriptions()

            # Start heartbeat
            asyncio.create_task(self._heartbeat_loop())

            # Start monitoring
            asyncio.create_task(self._monitoring_loop())

            logger.info("Bot coordinator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Bot coordinator initialization failed: {e}")
            return False

    async def _setup_subscriptions(self):
        """Setup Redis subscriptions for coordination"""
        try:
            # Subscribe to heartbeats to track active bots
            self.redis_manager.subscribe_to_heartbeats(self._handle_heartbeat)

            # Subscribe to position updates to track conflicts
            self.redis_manager.subscribe_to_positions(self._handle_position_update)

            # Subscribe to trade executions
            self.redis_manager.subscribe_to_trades(self._handle_trade_execution)

            # Subscribe to risk alerts
            self.redis_manager.subscribe_to_risk_alerts(self._handle_risk_alert)

            logger.info("Coordination subscriptions setup complete")

        except Exception as e:
            logger.error(f"Subscription setup error: {e}")

    async def request_trading_permission(self, symbol: str, side: str,
                                       position_size: float, confidence: float) -> CoordinationDecision:
        """
        Request permission to execute a trade with coordination checks
        """
        try:
            logger.info(f"Requesting trading permission: {side} {symbol} size={position_size}")

            # Update our status first
            await self._update_bot_status()

            # Check for conflicts
            conflicts = await self._check_trading_conflicts(symbol, side, position_size)

            if not conflicts:
                return CoordinationDecision(
                    bot_id=self.bot_id,
                    symbol=symbol,
                    action="PROCEED",
                    reason="No conflicts detected"
                )

            # Analyze conflicts and make decision
            decision = await self._resolve_conflicts(symbol, side, position_size,
                                                   confidence, conflicts)

            logger.info(f"Trading permission decision: {decision.action} - {decision.reason}")
            return decision

        except Exception as e:
            logger.error(f"Trading permission error: {e}")
            return CoordinationDecision(
                bot_id=self.bot_id,
                symbol=symbol,
                action="SKIP",
                reason=f"Error in coordination: {e}"
            )

    async def _check_trading_conflicts(self, symbol: str, side: str,
                                     position_size: float) -> List[TradingConflict]:
        """Check for trading conflicts across all bots"""
        conflicts = []

        try:
            # 1. Check direct symbol conflicts
            symbol_conflicts = self.redis_manager.check_symbol_conflicts(symbol)
            if symbol_conflicts:
                conflicting_bots = [c['bot_id'] for c in symbol_conflicts]
                conflicts.append(TradingConflict(
                    symbol=symbol,
                    conflicting_bots=conflicting_bots,
                    conflict_type="SAME_SYMBOL",
                    severity="HIGH",
                    recommendation="Wait or reduce position size"
                ))

            # 2. Check portfolio exposure limits
            all_positions = self.redis_manager.get_all_bot_positions()
            total_exposure = sum(pos['positions'].get(symbol, {}).get('notional', 0)
                               for pos in all_positions.values())

            # Estimate total portfolio value (simplified)
            portfolio_value = self._estimate_portfolio_value(all_positions)
            current_exposure_pct = (total_exposure / portfolio_value) if portfolio_value > 0 else 0

            if current_exposure_pct + (position_size / portfolio_value) > self.max_symbol_exposure:
                conflicts.append(TradingConflict(
                    symbol=symbol,
                    conflicting_bots=list(all_positions.keys()),
                    conflict_type="EXPOSURE_LIMIT",
                    severity="MEDIUM",
                    recommendation="Reduce position size"
                ))

            # 3. Check correlated symbols (simplified correlation check)
            correlated_symbols = await self._get_correlated_symbols(symbol)
            for corr_symbol in correlated_symbols:
                corr_conflicts = self.redis_manager.check_symbol_conflicts(corr_symbol)
                if corr_conflicts:
                    conflicts.append(TradingConflict(
                        symbol=corr_symbol,
                        conflicting_bots=[c['bot_id'] for c in corr_conflicts],
                        conflict_type="CORRELATION",
                        severity="LOW",
                        recommendation="Consider reducing exposure"
                    ))

            return conflicts

        except Exception as e:
            logger.error(f"Conflict checking error: {e}")
            return []

    async def _resolve_conflicts(self, symbol: str, side: str, position_size: float,
                               confidence: float, conflicts: List[TradingConflict]) -> CoordinationDecision:
        """Resolve trading conflicts using coordination logic"""

        try:
            # Priority scoring based on bot characteristics
            our_priority = self._calculate_bot_priority(confidence)

            high_severity_conflicts = [c for c in conflicts if c.severity == "HIGH"]
            medium_severity_conflicts = [c for c in conflicts if c.severity == "MEDIUM"]

            # High severity conflicts - need careful handling
            if high_severity_conflicts:
                conflict = high_severity_conflicts[0]

                if conflict.conflict_type == "SAME_SYMBOL":
                    # Check if we have higher priority
                    competing_bot_priorities = await self._get_competing_bot_priorities(
                        conflict.conflicting_bots
                    )

                    if our_priority > max(competing_bot_priorities.values()):
                        return CoordinationDecision(
                            bot_id=self.bot_id,
                            symbol=symbol,
                            action="PROCEED",
                            reason="Higher priority than competing bots"
                        )
                    else:
                        return CoordinationDecision(
                            bot_id=self.bot_id,
                            symbol=symbol,
                            action="WAIT",
                            reason="Lower priority, waiting for other bots",
                            wait_duration=60  # Wait 60 seconds
                        )

            # Medium severity conflicts - modify position size
            if medium_severity_conflicts:
                reduction_factor = 0.5  # Reduce by 50%
                modified_size = position_size * reduction_factor

                return CoordinationDecision(
                    bot_id=self.bot_id,
                    symbol=symbol,
                    action="REDUCE_SIZE",
                    reason="Exposure limit conflict, reducing position size",
                    modified_position_size=modified_size
                )

            # Low severity conflicts - proceed with caution
            return CoordinationDecision(
                bot_id=self.bot_id,
                symbol=symbol,
                action="PROCEED",
                reason="Low severity conflicts, proceeding with caution"
            )

        except Exception as e:
            logger.error(f"Conflict resolution error: {e}")
            return CoordinationDecision(
                bot_id=self.bot_id,
                symbol=symbol,
                action="SKIP",
                reason=f"Error in conflict resolution: {e}"
            )

    def _calculate_bot_priority(self, confidence: float) -> float:
        """Calculate bot priority for conflict resolution"""
        # Base priority factors
        base_priority = 1.0

        # Confidence factor (higher confidence = higher priority)
        confidence_factor = confidence * 2.0

        # Bot type factor (some bot types might have priority)
        type_factor = {
            "arbitrage": 1.5,
            "trend_following": 1.2,
            "mean_reversion": 1.0,
            "scalping": 0.8
        }.get(self.bot_type, 1.0)

        # Historical performance factor (simplified)
        performance_factor = self._get_historical_performance_factor()

        return base_priority * confidence_factor * type_factor * performance_factor

    def _get_historical_performance_factor(self) -> float:
        """Get performance-based priority factor"""
        # This would be based on historical performance metrics
        # For now, return neutral factor
        return 1.0

    async def _get_competing_bot_priorities(self, bot_ids: List[str]) -> Dict[str, float]:
        """Get priority scores for competing bots"""
        priorities = {}

        for bot_id in bot_ids:
            if bot_id in self.active_bots:
                # Simplified priority calculation for other bots
                # In reality, this would consider their performance metrics
                priorities[bot_id] = 1.0

        return priorities

    async def _get_correlated_symbols(self, symbol: str) -> List[str]:
        """Get symbols correlated with the given symbol"""
        # Simplified correlation mapping
        correlation_map = {
            "BTCUSDT": ["ETHUSDT", "ADAUSDT", "DOTUSDT"],
            "ETHUSDT": ["BTCUSDT", "LINKUSDT", "UNIUSDT"],
            # Add more correlations as needed
        }

        return correlation_map.get(symbol, [])

    def _estimate_portfolio_value(self, all_positions: Dict) -> float:
        """Estimate total portfolio value across all bots"""
        total_value = 0.0

        try:
            for bot_id, position_data in all_positions.items():
                positions = position_data.get('positions', {})
                for symbol, pos in positions.items():
                    total_value += abs(pos.get('notional', 0))

            return max(total_value, 1000)  # Minimum 1000 USDT for calculation

        except Exception:
            return 1000  # Fallback value

    async def _update_bot_status(self):
        """Update this bot's status information"""
        try:
            # Get current positions (this would come from the trading bot)
            # For now, create a simplified status
            status = BotStatus(
                bot_id=self.bot_id,
                bot_type=self.bot_type,
                is_active=True,
                total_positions=0,  # Would be updated with real data
                total_exposure=0.0,  # Would be updated with real data
                current_pnl=0.0,  # Would be updated with real data
                last_heartbeat=datetime.now(),
                trading_pairs=[],  # Would be updated with real pairs
                risk_level="MEDIUM"
            )

            # Store in active bots
            self.active_bots[self.bot_id] = status

            # Publish heartbeat
            bot_status = asdict(status)
            bot_status['last_heartbeat'] = status.last_heartbeat.isoformat()

            self.redis_manager.publish_heartbeat(bot_status)

        except Exception as e:
            logger.error(f"Bot status update error: {e}")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                await self._update_bot_status()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(self.heartbeat_interval)

    async def _monitoring_loop(self):
        """Monitor coordination and clean up expired data"""
        while True:
            try:
                # Clean up inactive bots
                await self._cleanup_inactive_bots()

                # Update coordination status
                await self._update_coordination_status()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_inactive_bots(self):
        """Remove inactive bots from tracking"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=5)
            inactive_bots = []

            for bot_id, status in self.active_bots.items():
                if status.last_heartbeat < cutoff_time:
                    inactive_bots.append(bot_id)

            for bot_id in inactive_bots:
                del self.active_bots[bot_id]
                logger.info(f"Removed inactive bot: {bot_id}")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    async def _update_coordination_status(self):
        """Update coordination status and detect new conflicts"""
        try:
            # Get fresh data from Redis
            active_bots_data = self.redis_manager.get_active_bots()

            # Update active bots tracking
            for bot_data in active_bots_data:
                bot_id = bot_data.get('bot_id', '')
                if bot_id:
                    status = BotStatus(
                        bot_id=bot_id,
                        bot_type=bot_data.get('bot_status', {}).get('bot_type', 'unknown'),
                        is_active=True,
                        total_positions=bot_data.get('bot_status', {}).get('total_positions', 0),
                        total_exposure=bot_data.get('bot_status', {}).get('total_exposure', 0.0),
                        current_pnl=bot_data.get('bot_status', {}).get('current_pnl', 0.0),
                        last_heartbeat=datetime.fromisoformat(bot_data.get('timestamp', datetime.now().isoformat())),
                        trading_pairs=bot_data.get('bot_status', {}).get('trading_pairs', []),
                        risk_level=bot_data.get('bot_status', {}).get('risk_level', 'MEDIUM')
                    )
                    self.active_bots[bot_id] = status

            self.last_coordination_check = datetime.now()

        except Exception as e:
            logger.error(f"Coordination status update error: {e}")

    # Event handlers
    def _handle_heartbeat(self, data: Dict):
        """Handle heartbeat messages from other bots"""
        try:
            bot_id = data.get('bot_id', '')
            if bot_id and bot_id != self.bot_id:
                bot_status_data = data.get('bot_status', {})

                status = BotStatus(
                    bot_id=bot_id,
                    bot_type=bot_status_data.get('bot_type', 'unknown'),
                    is_active=True,
                    total_positions=bot_status_data.get('total_positions', 0),
                    total_exposure=bot_status_data.get('total_exposure', 0.0),
                    current_pnl=bot_status_data.get('current_pnl', 0.0),
                    last_heartbeat=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                    trading_pairs=bot_status_data.get('trading_pairs', []),
                    risk_level=bot_status_data.get('risk_level', 'MEDIUM')
                )

                self.active_bots[bot_id] = status

        except Exception as e:
            logger.error(f"Heartbeat handling error: {e}")

    def _handle_position_update(self, data: Dict):
        """Handle position update messages"""
        try:
            bot_id = data.get('bot_id', '')
            symbol = data.get('symbol', '')

            if bot_id != self.bot_id:  # Only handle updates from other bots
                logger.info(f"Position update from {bot_id}: {symbol}")

                # Trigger conflict handlers if registered
                for handler in self.conflict_handlers:
                    try:
                        handler(data)
                    except Exception as e:
                        logger.error(f"Conflict handler error: {e}")

        except Exception as e:
            logger.error(f"Position update handling error: {e}")

    def _handle_trade_execution(self, data: Dict):
        """Handle trade execution messages"""
        try:
            bot_id = data.get('bot_id', '')
            if bot_id != self.bot_id:
                logger.info(f"Trade execution from {bot_id}: {data.get('symbol', 'Unknown')}")

        except Exception as e:
            logger.error(f"Trade execution handling error: {e}")

    def _handle_risk_alert(self, data: Dict):
        """Handle risk alert messages"""
        try:
            bot_id = data.get('bot_id', '')
            alert_data = data.get('alert_data', {})

            logger.warning(f"Risk alert from {bot_id}: {alert_data.get('message', 'Unknown alert')}")

            # Handle critical alerts that might require coordination
            if alert_data.get('severity') == 'CRITICAL':
                # Implement emergency coordination logic
                pass

        except Exception as e:
            logger.error(f"Risk alert handling error: {e}")

    # Public methods for bot integration
    def register_conflict_handler(self, handler):
        """Register a conflict event handler"""
        self.conflict_handlers.append(handler)

    def register_coordination_handler(self, handler):
        """Register a coordination event handler"""
        self.coordination_handlers.append(handler)

    def get_active_bots_summary(self) -> Dict:
        """Get summary of all active bots"""
        return {
            'total_bots': len(self.active_bots),
            'bot_types': list(set(bot.bot_type for bot in self.active_bots.values())),
            'total_positions': sum(bot.total_positions for bot in self.active_bots.values()),
            'total_exposure': sum(bot.total_exposure for bot in self.active_bots.values()),
            'last_check': self.last_coordination_check.isoformat()
        }

    def get_coordination_metrics(self) -> Dict:
        """Get coordination performance metrics"""
        return {
            'conflicts_resolved': getattr(self, '_conflicts_resolved_count', 0),
            'permissions_granted': getattr(self, '_permissions_granted_count', 0),
            'permissions_denied': getattr(self, '_permissions_denied_count', 0),
            'average_resolution_time': getattr(self, '_avg_resolution_time', 0),
            'active_monitoring': True
        }