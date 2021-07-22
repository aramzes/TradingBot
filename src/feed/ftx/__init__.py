import unittest

from .. import Ifeed

class TestFeed(unittest.TestCase):
    def setUp(self):
        self.feed = ftxFeed()

    def test_init(self):
        pass

class ftxFeed(Ifeed):
    def __init__(self):
        self.base_url = "https://ftx.com/api"
        self.ws_base_url = "wss://ftx.com/ws/"

    