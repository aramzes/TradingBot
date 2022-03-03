import pandas as pd
import numpy as np
import unittest
import re
import time
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential

from utils import load
from utils.date import floor_dt
from strategy import Strategy


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
        number_of_transactions.index = pd.to_datetime(number_of_transactions.index).tz_localize(None)
        number_of_transactions.rename(columns={"value": "txCount"}, inplace=True)
        supply_profit = pd.read_csv("data/glassnode/supply-in-profit-btc-1h.csv")
        supply_profit.set_index("timestamp", inplace=True)
        supply_profit.index = pd.to_datetime(supply_profit.index).tz_localize(None)
        supply_profit.rename(columns={"value": "supplyProfit"}, inplace=True)
        utxos_profit = pd.read_csv("data/glassnode/utxos-in-profit-btc-1h.csv")
        utxos_profit.set_index("timestamp", inplace=True)
        utxos_profit.index = pd.to_datetime(utxos_profit.index).tz_localize(None)
        utxos_profit.rename(columns={"value": "utxosProfit"}, inplace=True)
        lth_nupl = pd.read_csv("data/glassnode/lth-nupl-btc-1h.csv")
        lth_nupl.set_index("timestamp", inplace=True)
        lth_nupl.index = pd.to_datetime(lth_nupl.index).tz_localize(None)
        lth_nupl.rename(columns={"value": "LTH-NUPL"}, inplace=True)
 

        price_data = self.data.resample("1H", on="startTime") .agg(
            {"close": "last"}
        )
        price_data.index = price_data.index.tz_localize(None)
        
        # choose last one year of data from March 1st
        price_data = price_data.loc[price_data.index >= "2021-02-05"]
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
        def contains_keywords(row):
            _KEYWORDS = ["bitcoin", "btc", "crypto"]
            if any(w in row for w in _KEYWORDS):
                return 1.2
            return 1

        def _tweets_to_words(text):
            text = text.lower()
            text = re.sub(r"[^a-zA-Z0-9]", " ", text)
            words =  text.split()

            words = [PorterStemmer().stem(w) for w in words if w not in stopwords.words("english") and not w.isnumeric()]
            return words

        def _computer_vader_scores(words):
            scores = SentimentIntensityAnalyzer().polarity_scores(words)
            return scores["pos"], scores["neu"], scores["neg"], scores["compound"]

        tweets = pd.read_csv("data/twitter/Bitcoin_tweets.csv")
        tweets.drop([
            "user_name",
            "user_location",
            "user_created",
            "user_friends",
            "user_favourites",
            "source",
            "is_retweet"

        ], axis=1, inplace=True)
        tweets["date"] = pd.to_datetime(tweets["date"], errors="coerce")
        tweets.dropna(inplace=True)
        tweets.set_index("date", inplace=True)
        tweets.index = tweets.index.to_series().apply(lambda i: floor_dt(i, 5))
        tweets.sort_index(inplace=True)
        # tweets = tweets[:500]
        
        for t in ["user_description", "text", "hashtags"]:
            tweets[t] = tweets[t].str.lower()

        # weights
        tweets["weight"] = 1
        tweets["weight"] *= np.where(tweets["user_verified"] == True, 1.5, 1)
        tweets["weight"] *= tweets["user_description"].apply(lambda row: contains_keywords(row))
        # tweets["user_followers"] = (tweets["user_followers"] - tweets["user_followers"].mean()) / tweets["user_followers"].std()
        tweets["weight"] *= np.where(tweets["user_followers"] > tweets["user_followers"].quantile(0.75), 1.25,
                                            np.where(tweets["user_followers"] < tweets["user_followers"].quantile(0.25), 0.75, 1))


        # text
        tweets["text"] += tweets["hashtags"].str.join(" ")

        tweets.drop([
            "user_verified",
            "user_description",
            "user_followers",
            "hashtags"
        ], axis=1, inplace=True)

        tweets["words"] = tweets["text"].apply(lambda row: _tweets_to_words(row))
        tweets["pos"], tweets["neu"], tweets["neg"], tweets["comp"] = zip(*tweets["text"].apply(lambda row: _computer_vader_scores(row)))


        for feature in ["pos", "neu", "neg", "comp"]:
            tweets[feature] *= tweets["weight"]
        

        tweets.drop([
            "weight",
            "text",
            "words"
        ], axis=1, inplace=True)
        
        tweets = tweets.groupby(level=0).mean()
        tweets = tweets.resample("1h").mean()

        print(tweets)
        print(self.blockchain_data)
        all_data = pd.concat([tweets, self.blockchain_data], axis=1,)
        self.all_data = all_data.dropna()
        self.all_data.to_csv("data/dataset/lstm.csv")
    
    def _build_lstm_model(self):
        data = pd.read_csv("data/dataset/lstm.csv")

        # normalize data
        scaler = MinMaxScaler()
        data_norm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(data_norm.drop(["close"]), data["close"], test_size=0.3,) 

        # build model
        lookback = 48 # hours to look back to predict the next closing price

        model = Sequential()

        model.add(Bidirectional(LSTM(lookback, return_sequences=True), input_shape=(lookback, X_train.shape[-1]),))
        model.add(Dropout(0.2))

        model.add(Bidirectional(LSTM((lookback*2), return_sequences=True)))
        model.add(Dropout(0.2))

        model.add(Bidirectional(LSTM(lookback, return_sequences=False)))

        model.add(Dense(output_dim=1))
        model.add(Activation('relu'))

        model.compile(loss='mse', optimizer='adam')
        print(model.summary())

        # fit the model
        start = time.time()
        model.fit(X_train, y_train, batch_size=1024, epochs=10)
        print('training time : ', time.time() - start) 

        # predict the test data
        y_predict = model.predict(X_test)

        # plot the prediction against actual
        plt.plot(y_test, color='green')
        plt.plot(y_predict, color='blue')
        plt.title("Bitcoin Price")
        plt.ylabel("Price (USD)")
        plt.xlabel("Time (Hours)")
        # plt.savefig("data/result/price.png", dpi = 600)
        plt.show()


    def analyze_data(self):
        # # fetch price data from exchange
        # self.load_data(period=700)
        # # analyze blockchain data to determine relevant ones
        # self._analyze_blockchain_data()
        # # analyze twitter to transform them into sentiment values
        # self._analyze_twitter_data()
        # # train the LSTM RNN using blockchain and twitter data
        self._build_lstm_model()

