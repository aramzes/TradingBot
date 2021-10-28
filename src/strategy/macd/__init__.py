import unittest
import json
import pandas as pd
import matplotlib.pyplot as plt

from importlib import import_module
from ta.trend import MACD

from ...feed import Feed
from .. import Strategy

class Test(unittest.TestCase):
    def setUp(self):
        with open("config/test/strategy/macd.json") as test_file:
            strategyArgs = json.load(test_file)
        self.strategy = MACDStrategy(strategyArgs)

    def test_load_data(self):
        self.strategy.load_data_1year() 

    def test_analyze_data(self):
        self.strategy.load_data_1year()
        self.strategy.analyze_data()
        
    def tearDown(self):
        return 

    
class MACDStrategy(Strategy):
    def __init__(self, strategyArgs):
        self.feed = Feed.load(strategyArgs["exchange"]) 
        self.market = strategyArgs["market"] 
        self.name = "MACD Strategy"

    def load_data_1year(self):
        #self.data = self.feed.load_data(self.market, "5m", 365)
        self.data = pd.read_csv("data/test/ftx/candles/test_df.csv")

    def analyze_data(self):
        self.data["macd"] = MACD(self.data["close"]).macd()       
        print(self.data)
        fig, ax = plt.subplots()
        ax.plot(self.data["startTime"], self.data["close"])
        ax2 = ax.twinx()
        ax2.plot(self.data["startTime"], self.data["macd"], color="red")
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()