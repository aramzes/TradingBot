import requests
import unittest
import time
import pandas as pd
import datetime

from .. import Feed
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
        res = self.feed.download_candles("BTC-PERP",  300, "2021-09-01", "2021-09-10", "data/test/ftx/candles/", filename="2021-09-10")
        new_df = pd.read_csv("data/test/ftx/candles/2021-09-10.csv")
        test_df = pd.read_csv("data/test/ftx/candles/test_df.csv")
        self.assertTrue(new_df.equals(test_df))

    def test_load_candles(self):
        res = self.feed.load_candles("BTC-PERP", 300, 10, "2021-09-10", "data/test/ftx/candles/")

class FtxFeed(Feed):
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

    def download_candles(self, market, resolution, start_date="2019-09-01", end_date=None, dir="data/ftx/candles/", filename=None, save=True):
        start_year, start_month, start_day = start_date.split("-")
        start_date = datetime.date(int(start_year), int(start_month), int(start_day))
        if not end_date:
            end_date = datetime.today().strftime('%Y-%m-%d')
        end_year, end_month, end_day = end_date.split("-")
        end_date = datetime.date(int(end_year), int(end_month), int(end_day))

        candles_list = []
        for single_date in daterange(start_date, end_date):
            unixtime = time.mktime(single_date.timetuple())
            res = self.get_candles(market, resolution, start=unixtime, end = unixtime + 24 * 12 * resolution)
            candles_list += res
        all = pd.DataFrame(candles_list)
        all.drop("time", axis=1, inplace=True)
        all.set_index("startTime", inplace=True)
        all.sort_index(inplace=True)
        all.drop_duplicates(inplace=True)
        if save: 
            if not filename:
                filename = f"data-{market.lower()}-{resolution}"
            all.to_csv(f"{dir}{filename}.csv")
        return all
    
    def load_candles(self, market, resolution, days, end_date=None, dir="data/ftx/candles/"):
        if not end_date:
            end_date = datetime.today().strftime("%Y-%m-%d")

        start_date = datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=days) 
        start_date = start_date.strftime("%Y-%m-%d")
        try:
            latest_data = pd.read_csv(f"{dir}/data-{market.lower()}-{resolution}.csv")
            if latest_data.startTime[-1] < start_date:
               new_data = self.download_candles(market, resolution, start_date=latest_data.startTime[-1][:9], end_date=end_date, dir=dir, save=False)
            else:
               new_data = self.download_candles(market, resolution, start_date=start_date, end_date=end_date, dir=dir, save=False)
            new_data.set_index("startTime", inplace=True)
            all_data = pd.concat(latest_data, new_data)
            all_data = all_data.drop_duplicates(inplace=True)
            all_data.sort_index(inplace=True)
            all_data.to_csv(f"{dir}/data-{market.lower()}-{resolution}.csv")
        except:
            all_data = self.download_candles(market, resolution, start_date=start_date, end_date=end_date, dir=dir)

        return all_data

        