import requests
import unittest
import time
import pandas as pd

from datetime import date

from .. import Ifeed
from utils.date import daterange

class TestFeed(unittest.TestCase):
    def setUp(self):
        self.feed = FtxFeed()

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

    def test_download_candles(self):
        res = self.feed.download_candles("BTC-PERP", "2021-09-01", "2021-09-10", "data/test/ftx/")
        new_df = pd.read_csv("data/test/ftx/2021-09-10.csv")
        test_df = pd.read_csv("data/test/ftx/test_df.csv")
        self.assertTrue(new_df.equals(test_df))

class FtxFeed(Ifeed):
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

    def download_candles(self, market, start_date="2019-09-01", end_date=None, dir=None):
        start_year, start_month, start_day = start_date.split("-")
        start_date = date(int(start_year), int(start_month), int(start_day))
        if not end_date:
            end_date = datetime.today().strftime('%Y-%m-%d')
        end_year, end_month, end_day = end_date.split("-")
        end_date = date(int(end_year), int(end_month), int(end_day))

        candles_list = []
        for single_date in daterange(start_date, end_date):
            unixtime = time.mktime(single_date.timetuple())
            res = self.get_candles(market, start=unixtime, end = unixtime + 24 * 12 * 300)
            candles_list += res
        all = pd.DataFrame(candles_list)
        all.drop("time", axis=1, inplace=True)
        all.set_index("startTime", inplace=True)
        all.sort_index(inplace=True)
        all.drop_duplicates(inplace=True)
        
        if not dir:
            dir = "data/candles/ftx/"
            
        all.to_csv(f"{dir}{end_date}.csv")