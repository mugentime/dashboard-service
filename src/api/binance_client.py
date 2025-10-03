import asyncio
import time
import hmac
import hashlib
import json
from typing import Dict, List, Optional, Any
from decimal import Decimal
import aiohttp
import websockets
from datetime import datetime, timezone
import logging

from config.config import config

logger = logging.getLogger(__name__)

class BinanceFuturesClient:
    """Async Binance Futures API client with WebSocket support"""

    def __init__(self):
        self.api_key = config.binance.api_key
        self.api_secret = config.binance.api_secret
        self.base_url = config.binance.base_url
        self.ws_url = "wss://stream.binancefuture.com" if not config.binance.testnet else "wss://stream.binancefuture.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connections: Dict[str, Any] = {}
        self.rate_limiter = RateLimiter()
        self.time_offset = 0  # Offset to sync with Binance server time

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.binance.timeout)
        )

        # In demo mode, skip actual API validation
        if config.binance.api_key == "demo_api_key_for_testing":
            logger.info("🎭 Running in DEMO MODE - API calls will be mocked")
            self.demo_mode = True
        else:
            self.demo_mode = False

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        for ws in self.ws_connections.values():
            if not ws.closed:
                await ws.close()

    def _generate_signature(self, query_string: str) -> str:
        """Generate API signature"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_headers(self) -> Dict[str, str]:
        """Get API headers"""
        return {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }

    async def _get_server_time(self) -> int:
        """Get synchronized server time to avoid timestamp errors"""
        try:
            # Get Binance server time
            if not self.session:
                return int(time.time() * 1000) + self.time_offset

            async with self.session.get(f"{self.base_url}/fapi/v1/time") as response:
                if response.status == 200:
                    data = await response.json()
                    server_time = data['serverTime']
                    local_time = int(time.time() * 1000)
                    self.time_offset = server_time - local_time
                    return server_time
                else:
                    # Fallback to local time with existing offset
                    return int(time.time() * 1000) + self.time_offset
        except Exception as e:
            logger.warning(f"Failed to sync server time: {e}")
            # Fallback to local time with existing offset
            return int(time.time() * 1000) + self.time_offset

    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make authenticated API request with rate limiting"""
        await self.rate_limiter.wait()

        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        if params is None:
            params = {}

        if signed:
            timestamp = await self._get_server_time()
            params['timestamp'] = timestamp
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = self._generate_signature(query_string)
            params['signature'] = signature

        for attempt in range(config.binance.max_retries):
            try:
                async with self.session.request(method, url, headers=headers, params=params) as response:
                    data = await response.json()

                    if response.status == 200:
                        return data
                    elif response.status == 429:  # Rate limit
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"API error: {response.status} - {data}")
                        raise Exception(f"API error: {data}")

            except Exception as e:
                if attempt == config.binance.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise Exception("Max retries exceeded")

    async def get_account_info(self) -> Dict:
        """Get futures account information"""
        if hasattr(self, 'demo_mode') and self.demo_mode:
            return {
                'feeTier': 0,
                'canTrade': True,
                'canDeposit': True,
                'canWithdraw': True,
                'updateTime': int(datetime.now().timestamp() * 1000),
                'totalWalletBalance': '10000.00000000',
                'totalUnrealizedProfit': '0.00000000',
                'totalMarginBalance': '10000.00000000',
                'totalPositionInitialMargin': '0.00000000',
                'totalOpenOrderInitialMargin': '0.00000000',
                'maxWithdrawAmount': '10000.00000000',
                'assets': [
                    {
                        'asset': 'USDT',
                        'walletBalance': '10000.00000000',
                        'unrealizedProfit': '0.00000000',
                        'marginBalance': '10000.00000000',
                        'maintMargin': '0.00000000',
                        'initialMargin': '0.00000000',
                        'positionInitialMargin': '0.00000000',
                        'openOrderInitialMargin': '0.00000000',
                        'maxWithdrawAmount': '10000.00000000',
                        'crossWalletBalance': '10000.00000000',
                        'crossUnrealizedProfit': '0.00000000',
                        'availableBalance': '10000.00000000'
                    }
                ],
                'positions': []
            }
        return await self._make_request('GET', '/fapi/v2/account', signed=True)

    async def get_position_info(self, symbol: str = None) -> List[Dict]:
        """Get position information"""
        params = {'symbol': symbol} if symbol else {}
        return await self._make_request('GET', '/fapi/v2/positionRisk', params, signed=True)

    async def get_balance(self) -> List[Dict]:
        """Get account balance"""
        return await self._make_request('GET', '/fapi/v2/balance', signed=True)

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                         price: float = None, time_in_force: str = 'GTC', **kwargs) -> Dict:
        """Place a futures order"""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(quantity)
        }

        # Don't include timeInForce for MARKET orders
        if order_type.upper() != 'MARKET' and time_in_force:
            params['timeInForce'] = time_in_force

        if price:
            params['price'] = str(price)

        params.update(kwargs)

        return await self._make_request('POST', '/fapi/v1/order', params, signed=True)

    async def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an order"""
        params = {'symbol': symbol, 'orderId': order_id}
        return await self._make_request('DELETE', '/fapi/v1/order', params, signed=True)

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        params = {'symbol': symbol} if symbol else {}
        return await self._make_request('GET', '/fapi/v1/openOrders', params, signed=True)

    async def get_klines(self, symbol: str, interval: str, limit: int = 500,
                        start_time: int = None, end_time: int = None) -> List[List]:
        """Get kline/candlestick data"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return await self._make_request('GET', '/fapi/v1/klines', params)

    async def get_ticker_price(self, symbol: str = None) -> Dict:
        """Get ticker price"""
        params = {'symbol': symbol} if symbol else {}
        return await self._make_request('GET', '/fapi/v1/ticker/price', params)

    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book depth"""
        params = {'symbol': symbol, 'limit': limit}
        return await self._make_request('GET', '/fapi/v1/depth', params)

    async def start_websocket_stream(self, streams: List[str], callback) -> None:
        """Start WebSocket stream for real-time data"""
        stream_url = f"{self.ws_url}/stream?streams={'/'.join(streams)}"

        try:
            async with websockets.connect(stream_url) as websocket:
                self.ws_connections['/'.join(streams)] = websocket

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await callback(data)
                    except Exception as e:
                        logger.error(f"WebSocket callback error: {e}")

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise

class RateLimiter:
    """Rate limiter for API requests"""

    def __init__(self):
        self.request_times = []
        self.max_requests_per_minute = 1200  # Binance limit
        self.buffer = config.binance.rate_limit_buffer

    async def wait(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()

        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        # Check if we need to wait
        if len(self.request_times) >= self.max_requests_per_minute * (1 - self.buffer):
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.request_times.append(current_time)

class WebSocketManager:
    """Manages WebSocket connections for real-time data"""

    def __init__(self, client: BinanceFuturesClient):
        self.client = client
        self.callbacks = {}
        self.streams = {}

    async def subscribe_ticker(self, symbol: str, callback):
        """Subscribe to ticker updates"""
        stream = f"{symbol.lower()}@ticker"
        await self._subscribe(stream, callback)

    async def subscribe_kline(self, symbol: str, interval: str, callback):
        """Subscribe to kline updates"""
        stream = f"{symbol.lower()}@kline_{interval}"
        await self._subscribe(stream, callback)

    async def subscribe_depth(self, symbol: str, callback, levels: int = 20):
        """Subscribe to order book depth updates"""
        stream = f"{symbol.lower()}@depth{levels}@100ms"
        await self._subscribe(stream, callback)

    async def subscribe_trade(self, symbol: str, callback):
        """Subscribe to trade updates"""
        stream = f"{symbol.lower()}@aggTrade"
        await self._subscribe(stream, callback)

    async def _subscribe(self, stream: str, callback):
        """Subscribe to a WebSocket stream"""
        self.callbacks[stream] = callback

        if stream not in self.streams:
            asyncio.create_task(
                self.client.start_websocket_stream([stream], self._handle_message)
            )
            self.streams[stream] = True

    async def _handle_message(self, data: Dict):
        """Handle incoming WebSocket messages"""
        if 'stream' in data:
            stream = data['stream']
            if stream in self.callbacks:
                await self.callbacks[stream](data['data'])