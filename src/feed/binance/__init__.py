import unittest
import requests

from .. import Feed 

class TestFeed(unittest.TestCase):
    def setUp(self):
        self.feed = binanceFeed()

    def test_init(self):
        pass

    def test_get(self):
        pass

    def test_get_orderbook(self):
        res = self.feed.get_orderbook("BTCUSDT")
        self.assertIn("bids", res)
        self.assertIn("asks", res)

    def test_get_ticker(self):
        res = self.feed.get_ticker("BTCUSDT")
        self.assertIn("price", res)

    def test_get_trades(self):
        res = self.feed.get_trades("BTCUSDT")
        self.assertIsInstance(res, list)
        self.assertNotEqual(res, [])

    def test_get_candles(self):
        res = self.feed.get_candles("BTCUSDT", "1h")
        self.assertIsInstance(res, list)
        self.assertNotEqual(res, [])

class binanceFeed(Feed):
    def __init__(self):
        # self.base_url = "https://testnet.binance.vision/api"
        self.base_url = "https://api.binance.com/api"
        self.ws_base_url = "wss://testnet.binance.vision/ws"
        # self.ws_base_url = "wss://stream.binance.com:9443/ws"

    def get(self, endpoint, params):
        return requests.get(self.base_url + endpoint, params)

    def get_orderbook(self, symbol, limit = 5):
        params = {
            "symbol": symbol,
            "limit": limit
        }
        orderbook = self.get("/v3/depth", params).json()

        return orderbook

    def get_ticker(self, symbol):
        params = {
            "symbol": symbol
        }
        ticker = self.get("/v3/ticker/price", params).json()

        return ticker

    def get_trades(self, symbol, limit = 5):
        params = {
            "symbol": symbol,
            "limit": limit
        }
        trades = self.get("/v3/trades", params).json()

        return trades

    def get_candles(self, symbol, interval, start_time = None, end_time = None, limit = 500):
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        candles = self.get("/v3/klines", params).json()

        return candles
