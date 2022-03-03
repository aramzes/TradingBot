from calendar import prcal
from re import A
import pandas as pd
import unittest
import seaborn as sns
import matplotlib.pyplot as plt
from strategy import Strategy


from utils import load


class Test(unittest.TestCase):
    def setUp(self):
        self.strategy = load.load_strategy("LSTM")

    def test_load_data(self):
        #self.strategy.load_data_1year() 
        pass

    def test_analyze_data(self):
        self.strategy.analyze_data()
        
    def tearDown(self):
        return 

class LSTMStrategy(Strategy):
    def __init__(self, strategyArgs: dict):
        self.feed = load.load("feed", strategyArgs["exchange"]) 
        self.market = strategyArgs["market"] 
        self.name = "LSTM Strategy"

    def _analyze_blockchain_data(self):
        number_of_transactions = pd.read_csv("data/glassnode/transaction-count-btc-1h.csv")
        number_of_transactions.set_index("timestamp", inplace=True)
        number_of_transactions.index = pd.to_datetime(number_of_transactions.index)
        number_of_transactions.rename(columns={"value": "txCount"}, inplace=True)
        supply_profit = pd.read_csv("data/glassnode/supply-in-profit-btc-1h.csv")
        supply_profit.set_index("timestamp", inplace=True)
        supply_profit.index = pd.to_datetime(supply_profit.index)
        supply_profit.rename(columns={"value": "supplyProfit"}, inplace=True)
        utxos_profit = pd.read_csv("data/glassnode/utxos-in-profit-btc-1h.csv")
        utxos_profit.set_index("timestamp", inplace=True)
        utxos_profit.index = pd.to_datetime(utxos_profit.index)
        utxos_profit.rename(columns={"value": "utxosProfit"}, inplace=True)
        lth_nupl = pd.read_csv("data/glassnode/lth-nupl-btc-1h.csv")
        lth_nupl.set_index("timestamp", inplace=True)
        lth_nupl.index = pd.to_datetime(lth_nupl.index)
        lth_nupl.rename(columns={"value": "LTH-NUPL"}, inplace=True)
 

        price_data = self.data.resample("1H", on="startTime") .agg(
            {"close": "last"}
        )
        
        # choose last one year of data from March 1st
        price_data = price_data.loc[price_data.index >= "2021-03-01"]
        all_data = pd.concat([price_data, 
            number_of_transactions,
            supply_profit,
            utxos_profit,
            lth_nupl 
        ], axis=1)
        all_data.dropna(inplace=True)

        heatmap = sns.heatmap(all_data.corr()[["close"]].drop(["close"], axis=0), annot=True)
        # plt.show()
        self.blockchain_data = all_data.drop(["txCount"], axis=1)
    
    def _analyze_twitter_data(self):
        tweets = pd.read_csv("data/twitter/Bitcoin_tweets.csv")
        tweets.drop([
            "user_name",
            "user_location",
            "user_created",
            "user_friends",
            "user_favourites",
            "source",

        ], axis=1, inplace=True)
        tweets.set_index("date", inplace=True)
        tweets.index = pd.to_datetime(tweets.index, errors="coerce")
        tweets.dropna(inplace=True)
        tweets.sort_index(inplace=True)
        print(tweets)

    def analyze_data(self):
        # fetch price data from exchange
        self.load_data(period=700)
        # analyze blockchain data to determine relevant ones
        self._analyze_blockchain_data()
        # analyze twitter to transform them into sentiment values
        self._analyze_twitter_data()
