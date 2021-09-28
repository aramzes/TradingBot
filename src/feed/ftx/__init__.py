import requests
import unittest

from .. import Ifeed

class TestFeed(unittest.TestCase):
    def setUp(self):
        self.feed = ftxFeed()

    def test_init(self):
        self.assertEqual(self.feed.name, "FTX")
    
    def test_get_orderbook(self):
        res =  self.feed.get_orderbook("BTC-PERP")
        self.assertIn("bids", res)
        self.assertIn("asks", res)
        self.assertIsInstance(res["bids"], list)
        self.assertIsInstance(res["asks"], list)

    def test_get_public_trades(self):
        res = self.feed.get_trades("BTC-PERP")
        self.assertEqual(len(res), 100)
        for attribute in ["price", "side", "size"]:
            self.assertIn(attribute, res[0])

    def test_get_ticker(self):
        res = self.feed.get_ticker("BTC-PERP")
        self.assertIn("price", res)
        self.assertIn("ask", res)
        self.assertIn("bid", res)
        

    def test_get_candles(self):
        res = self.feed.get_candles("BTC-PERP")
        self.assertIn("open", res[0])
        self.assertIn("close", res[0])
        self.assertIn("high", res[0])
        self.assertIn("low", res[0])

class ftxFeed(Ifeed):
    def __init__(self):
        self.base_url = "https://ftx.com/api/"
        self.ws_base_url = "wss://ftx.com/ws/"
        self.name = "FTX"

    def get(self, endpoint, params={}):
        return requests.get(self.base_url + endpoint, params).json()

    def get_orderbook(self, symbol, depth=20):
        params = {
            "depth": depth
        }
        orderbook = self.get(f"markets/{symbol}/orderbook", params)["result"]
        return orderbook 

    def get_trades(self, symbol, limit=100):
        params = {
            "limit": limit
        }
        trades = self.get(f"markets/{symbol}/trades", params)["result"]
        return trades

    def get_ticker(self, symbol):
        ticker = self.get(f"markets/{symbol}")["result"]
        return ticker

    def get_candles(self, symbol, resolution=300, start=None, end=None):
        params = {
            "resolution": resolution,
            "start_time": start,
            "end_time": end
        }
        candles = self.get(f"markets/{symbol}/candles", params)["result"]
        return candles