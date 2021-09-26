import unittest
import requests

from .. import Ifeed 

class TestFeed(unittest.TestCase):
    def setUp(self):
        self.feed = BonfidaFeed()

    def test_get(self):
        pass

    def test_get_markets(self):
        res = self.feed.get_markets()
        self.assertTrue(res["success"])
        self.assertIsInstance(res["result"], list)
        markets = { m["name"] : m["address"] for m in res["result"] }
        self.assertIn("BTC-PERP", markets)
        self.assertIn("ETH-PERP", markets)
        self.assertIn("SOL-PERP", markets)

    def test_get_market_address(self):
        res = self.feed.get_market_address("BTC-PERP")
        self.assertEqual(res, "475P8ZX3NrzyEMJSFHt9KCMjPpWBWGa6oNxkWcwww2BR")

    def test_get_market_data(self):
        res1 = self.feed.get_market_data(symbol="BTC-PERP")
        self.assertTrue(res1["success"])
        res1 = res1["result"]
        self.assertIn("markPrice", res1)
        self.assertIn("indexPrice", res1)
        self.assertIn("fundingLong", res1)
        self.assertIn("fundingShort", res1)
        print(res1)
        res2 = self.feed.get_market_data(address="475P8ZX3NrzyEMJSFHt9KCMjPpWBWGa6oNxkWcwww2BR") 
        self.assertTrue(res2["success"])
        res2 = res2["result"]
        self.assertIn("markPrice", res2)
        self.assertIn("indexPrice", res2)
        self.assertIn("fundingLong", res2)
        self.assertIn("fundingShort", res2)


class BonfidaFeed(Ifeed):
    def __init__(self):
        self.base_url = "http://localhost:3000/"

    def get(self, endpoint, params={}):
        return requests.get(self.base_url + endpoint, params).json()

    def get_markets(self):
        return self.get("markets")
        
    def get_market_address(self, symbol):
        markets =  self.get_markets()["result"]
        for market in markets:
            if market["name"] == symbol:
                return market["address"]
        return None

    def get_market_data(self, symbol=None, address=None):
        assert([symbol, address] != [None, None])
        market_address = None
        if symbol:
            market_address = self.get_market_address(symbol)
        elif address:
            market_address = address

        assert(market_address != None)
        return self.get(f"markets/{market_address}")
