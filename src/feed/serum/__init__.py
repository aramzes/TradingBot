import unittest
import requests

from .. import Feed 

class TestFeed(unittest.TestCase):
    def setUp(self):
        self.feed = SerumFeed()

    def test_init(self):
        pass

    def test_get(self):
        pass

    def test_get_orderbook(self):
        res = self.feed.get_orderbook("BTCUSDT")
        self.assertEqual(res["success"], True)
        data = res["data"]
        self.assertIn("bids", data)
        self.assertIsInstance(data["bids"], list)
        self.assertTrue(len(data["bids"]) > 0)
        self.assertEqual(["price", "size"], list(data["bids"][0].keys()))
        self.assertIn("asks", data)
        self.assertIsInstance(data["asks"], list)
        self.assertTrue(len(data["asks"]) > 0)
        self.assertEqual(["price", "size"], list(data["asks"][0].keys()))

    # def test_get_ticker(self):
    #     res = self.feed.get_ticker("BTCUSDT")
    #     self.assertIn("price", res)

    def test_get_trades(self):
        res = self.feed.get_trades("BTCUSDT")
        self.assertEqual(res["success"], True)
        data = res["data"]
        self.assertIsInstance(data, list)
        self.assertNotEqual(len(data), 0)
        fields = [
            "market",
            "price",
            "size",
            "side",
            "time",
            "orderId",
            "feeCost",
            "marketAddress"
        ]
        self.assertEqual(fields, list(data[0].keys()))


    # def test_get_candles(self):
    #     res = self.feed.get_candles("BTCUSDT", 60)
    #     print(res)
    #     self.assertIsInstance(res, list)
    #     self.assertNotEqual(res, [])

class SerumFeed(Feed):
    def __init__(self):
        self.base_url = "https://serum-api.bonfida.com/"
        self.ws_base_url = "wss://serum-ws.bonfida.com/"

    def get(self, endpoint, params={}):
        return requests.get(self.base_url + endpoint, params)

    def get_orderbook(self, symbol):
        orderbook = self.get(f"orderbooks/{symbol}").json()
        return orderbook

    # def get_ticker(self, symbol):
    #     params = {
    #         "symbol": symbol
    #     }
    #     ticker = self.get("/v3/ticker/price", params).json()

        return ticker

    def get_trades(self, symbol):
        trades = self.get(f"trades/{symbol}").json()
        return trades

    # def get_candles(self, symbol, interval, start_time = None, end_time = None, limit = 500):
    #     params = {
    #         "marketName": symbol,
    #         "limit": limit,
    #         "resolution": interval
    #     }
    #     if start_time:
    #         params["startTime"] = start_time
    #     if end_time:
    #         params["endTime"] = end_time

    #     candles = self.get("candles", params).json()

    #     return candles
