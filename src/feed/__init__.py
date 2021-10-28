from json import load
import unittest

from importlib import import_module

class Test(unittest.TestCase):
    def setUp(self):
        pass

    def test_load(self):
        loaded_feed = Feed.load("ftx")
        self.assertEqual(loaded_feed.name, "FTX")

class Feed:
    def __init__(self):
        self.base_url = None
    
    @staticmethod
    def load(exchange):
        feed_module = import_module(f"feed.{exchange}")
        feed_class = getattr(feed_module, f"{exchange.capitalize()}Feed")
        instance = feed_class()

        return instance        

    def get_orderbook(self, symbol):
        raise NotImplementedError

    def get_public_trades(self, symbol):
        raise NotImplementedError

    def get_ticker(self, symbol):
        raise NotImplementedError
    
    def get_candles(self, symbol):
        raise NotImplementedError

    