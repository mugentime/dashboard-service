"""
Redis Manager for Inter-Bot Communication
Handles real-time communication between trading bots
"""

import redis
import json
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import threading

logger = logging.getLogger(__name__)


class RedisManager:
    """
    Manages Redis connections and provides pub/sub functionality for bot communication
    """

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0,
                 password: Optional[str] = None, decode_responses: bool = True):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.decode_responses = decode_responses

        self.redis_client = None
        self.pubsub = None
        self.subscribers = {}
        self.is_connected = False
        self.bot_id = str(uuid.uuid4())[:8]  # Unique bot identifier

        # Message channels
        self.channels = {
            'positions': 'trading_bot_positions',
            'trades': 'trading_bot_trades',
            'signals': 'trading_bot_signals',
            'heartbeat': 'trading_bot_heartbeat',
            'coordination': 'trading_bot_coordination',
            'risk_alerts': 'trading_bot_risk_alerts'
        }

    def connect(self) -> bool:
        """
        Establish connection to Redis server
        """
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=self.decode_responses,
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # Test connection
            self.redis_client.ping()
            self.is_connected = True
            logger.info(f"Redis connection established - Bot ID: {self.bot_id}")

            # Initialize pubsub
            self.pubsub = self.redis_client.pubsub(ignore_subscribe_messages=True)

            return True

        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """
        Close Redis connections
        """
        try:
            if self.pubsub:
                self.pubsub.close()
            if self.redis_client:
                self.redis_client.close()
            self.is_connected = False
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Redis disconnect error: {e}")

    def is_redis_available(self) -> bool:
        """
        Check if Redis is available (for graceful degradation)
        """
        if not self.is_connected:
            return False

        try:
            self.redis_client.ping()
            return True
        except Exception:
            self.is_connected = False
            return False

    # Position Management
    def publish_position_update(self, symbol: str, position_data: Dict) -> bool:
        """
        Publish position update to other bots
        """
        if not self.is_redis_available():
            return False

        try:
            message = {
                'bot_id': self.bot_id,
                'symbol': symbol,
                'position_data': position_data,
                'timestamp': datetime.now().isoformat(),
                'message_type': 'position_update'
            }

            self.redis_client.publish(self.channels['positions'], json.dumps(message))
            return True

        except Exception as e:
            logger.error(f"Failed to publish position update: {e}")
            return False

    def publish_trade_execution(self, symbol: str, trade_data: Dict) -> bool:
        """
        Publish trade execution to other bots
        """
        if not self.is_redis_available():
            return False

        try:
            message = {
                'bot_id': self.bot_id,
                'symbol': symbol,
                'trade_data': trade_data,
                'timestamp': datetime.now().isoformat(),
                'message_type': 'trade_execution'
            }

            self.redis_client.publish(self.channels['trades'], json.dumps(message))
            return True

        except Exception as e:
            logger.error(f"Failed to publish trade execution: {e}")
            return False

    def publish_signal(self, symbol: str, signal_data: Dict) -> bool:
        """
        Publish trading signal to other bots
        """
        if not self.is_redis_available():
            return False

        try:
            message = {
                'bot_id': self.bot_id,
                'symbol': symbol,
                'signal_data': signal_data,
                'timestamp': datetime.now().isoformat(),
                'message_type': 'trading_signal'
            }

            self.redis_client.publish(self.channels['signals'], json.dumps(message))
            return True

        except Exception as e:
            logger.error(f"Failed to publish signal: {e}")
            return False

    def publish_heartbeat(self, bot_status: Dict) -> bool:
        """
        Publish bot heartbeat for coordination
        """
        if not self.is_redis_available():
            return False

        try:
            message = {
                'bot_id': self.bot_id,
                'bot_status': bot_status,
                'timestamp': datetime.now().isoformat(),
                'message_type': 'heartbeat'
            }

            self.redis_client.publish(self.channels['heartbeat'], json.dumps(message))

            # Also store in Redis with TTL for status tracking
            key = f"bot_status:{self.bot_id}"
            self.redis_client.setex(key, 60, json.dumps(message))  # 60 second TTL

            return True

        except Exception as e:
            logger.error(f"Failed to publish heartbeat: {e}")
            return False

    def publish_risk_alert(self, alert_data: Dict) -> bool:
        """
        Publish risk alert to all bots
        """
        if not self.is_redis_available():
            return False

        try:
            message = {
                'bot_id': self.bot_id,
                'alert_data': alert_data,
                'timestamp': datetime.now().isoformat(),
                'message_type': 'risk_alert'
            }

            self.redis_client.publish(self.channels['risk_alerts'], json.dumps(message))
            return True

        except Exception as e:
            logger.error(f"Failed to publish risk alert: {e}")
            return False

    # Subscription Management
    def subscribe_to_positions(self, callback: Callable[[Dict], None]) -> bool:
        """
        Subscribe to position updates from other bots
        """
        return self._subscribe_to_channel('positions', callback)

    def subscribe_to_trades(self, callback: Callable[[Dict], None]) -> bool:
        """
        Subscribe to trade executions from other bots
        """
        return self._subscribe_to_channel('trades', callback)

    def subscribe_to_signals(self, callback: Callable[[Dict], None]) -> bool:
        """
        Subscribe to trading signals from other bots
        """
        return self._subscribe_to_channel('signals', callback)

    def subscribe_to_heartbeats(self, callback: Callable[[Dict], None]) -> bool:
        """
        Subscribe to bot heartbeats for coordination
        """
        return self._subscribe_to_channel('heartbeat', callback)

    def subscribe_to_risk_alerts(self, callback: Callable[[Dict], None]) -> bool:
        """
        Subscribe to risk alerts from other bots
        """
        return self._subscribe_to_channel('risk_alerts', callback)

    def _subscribe_to_channel(self, channel_name: str, callback: Callable[[Dict], None]) -> bool:
        """
        Internal method to subscribe to a specific channel
        """
        if not self.is_redis_available():
            return False

        try:
            channel = self.channels.get(channel_name)
            if not channel:
                logger.error(f"Unknown channel: {channel_name}")
                return False

            self.pubsub.subscribe(channel)
            self.subscribers[channel_name] = callback

            # Start listening in a separate thread
            if not hasattr(self, '_listener_thread') or not self._listener_thread.is_alive():
                self._listener_thread = threading.Thread(target=self._listen_for_messages, daemon=True)
                self._listener_thread.start()

            logger.info(f"Subscribed to channel: {channel}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to {channel_name}: {e}")
            return False

    def _listen_for_messages(self):
        """
        Listen for incoming messages on subscribed channels
        """
        try:
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])

                        # Don't process messages from this bot
                        if data.get('bot_id') == self.bot_id:
                            continue

                        # Find the appropriate callback
                        channel = message['channel']
                        for channel_name, channel_key in self.channels.items():
                            if channel_key == channel and channel_name in self.subscribers:
                                callback = self.subscribers[channel_name]
                                callback(data)
                                break

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"Message listener error: {e}")

    # Data Storage and Retrieval
    def store_portfolio_positions(self, positions: Dict) -> bool:
        """
        Store current portfolio positions
        """
        if not self.is_redis_available():
            return False

        try:
            key = f"portfolio_positions:{self.bot_id}"
            data = {
                'positions': positions,
                'timestamp': datetime.now().isoformat(),
                'bot_id': self.bot_id
            }

            self.redis_client.setex(key, 300, json.dumps(data))  # 5 minute TTL
            return True

        except Exception as e:
            logger.error(f"Failed to store portfolio positions: {e}")
            return False

    def get_all_bot_positions(self) -> Dict:
        """
        Get positions from all active bots
        """
        if not self.is_redis_available():
            return {}

        try:
            pattern = "portfolio_positions:*"
            keys = self.redis_client.keys(pattern)

            all_positions = {}
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    position_data = json.loads(data)
                    bot_id = position_data.get('bot_id')
                    if bot_id and bot_id != self.bot_id:  # Exclude own positions
                        all_positions[bot_id] = position_data

            return all_positions

        except Exception as e:
            logger.error(f"Failed to get all bot positions: {e}")
            return {}

    def get_active_bots(self) -> List[Dict]:
        """
        Get list of currently active bots
        """
        if not self.is_redis_available():
            return []

        try:
            pattern = "bot_status:*"
            keys = self.redis_client.keys(pattern)

            active_bots = []
            cutoff_time = datetime.now() - timedelta(minutes=2)

            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    bot_data = json.loads(data)
                    timestamp = datetime.fromisoformat(bot_data.get('timestamp', ''))

                    if timestamp > cutoff_time:  # Active within last 2 minutes
                        active_bots.append(bot_data)

            return active_bots

        except Exception as e:
            logger.error(f"Failed to get active bots: {e}")
            return []

    def check_symbol_conflicts(self, symbol: str) -> List[Dict]:
        """
        Check if other bots are trading the same symbol
        """
        all_positions = self.get_all_bot_positions()
        conflicts = []

        for bot_id, position_data in all_positions.items():
            positions = position_data.get('positions', {})
            if symbol in positions:
                conflicts.append({
                    'bot_id': bot_id,
                    'position': positions[symbol],
                    'timestamp': position_data.get('timestamp')
                })

        return conflicts

    def store_trade_history(self, trade_data: Dict) -> bool:
        """
        Store trade in shared history for analysis
        """
        if not self.is_redis_available():
            return False

        try:
            key = f"trade_history"
            trade_record = {
                'bot_id': self.bot_id,
                'trade_data': trade_data,
                'timestamp': datetime.now().isoformat()
            }

            # Use Redis list to store trade history
            self.redis_client.lpush(key, json.dumps(trade_record))
            self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 trades

            return True

        except Exception as e:
            logger.error(f"Failed to store trade history: {e}")
            return False

    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """
        Get recent trades from all bots
        """
        if not self.is_redis_available():
            return []

        try:
            key = "trade_history"
            trades = self.redis_client.lrange(key, 0, limit - 1)

            trade_list = []
            for trade in trades:
                try:
                    trade_data = json.loads(trade)
                    trade_list.append(trade_data)
                except json.JSONDecodeError:
                    continue

            return trade_list

        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []

    def cleanup_expired_data(self):
        """
        Clean up expired data from Redis
        """
        if not self.is_redis_available():
            return

        try:
            # Clean up old bot status entries
            pattern = "bot_status:*"
            keys = self.redis_client.keys(pattern)
            cutoff_time = datetime.now() - timedelta(minutes=5)

            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    try:
                        bot_data = json.loads(data)
                        timestamp = datetime.fromisoformat(bot_data.get('timestamp', ''))

                        if timestamp < cutoff_time:
                            self.redis_client.delete(key)
                    except (json.JSONDecodeError, ValueError):
                        # Delete corrupted entries
                        self.redis_client.delete(key)

            logger.info("Redis cleanup completed")

        except Exception as e:
            logger.error(f"Redis cleanup error: {e}")

    def get_bot_id(self) -> str:
        """Get this bot's unique ID"""
        return self.bot_id