from cProfile import label
import pandas as pd
import numpy as np
import unittest
import re
import time
import seaborn as sns
import math
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
        def split_series(series, n_past, n_future):
            #
            # n_past ==> no of past observations
            #
            # n_future ==> no of future observations 
            #
            X, y = list(), list()
            for window_start in range(len(series)):
                past_end = window_start + n_past
                future_end = past_end + n_future
                if future_end > len(series):
                    break
                # slicing the past and future parts of the window
                past, future = series[window_start:past_end, :], series[past_end:future_end, :]
                X.append(past)
                y.append(future)
            return np.array(X), np.array(y)
        
        lookback = 10 # hours to look back to predict the next closing price
        data_org = pd.read_csv("data/dataset/lstm.csv", index_col=0)
        data_org = data_org.drop(["pos", "neu", "neg"], axis=1)
        data = data_org.copy()
    

        # normalize data
        scaler= MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        # split train and test data

        n_past = lookback 
        n_future = 1
        n_features = 5 
        train, test = data[:math.floor(0.7*len(data))], data[math.floor(0.7*len(data)):]
        X_train, y_train = split_series(train.values,n_past, n_future)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
        X_test, y_test = split_series(test.values,n_past, n_future)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
        y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
        
        # build model

        model = Sequential()

        model.add(LSTM(lookback, return_sequences=True, input_shape=(lookback, X_train.shape[-1])))
        model.add(Dropout(0.2))

        model.add(LSTM((lookback*2), return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(lookback, return_sequences=False))

        model.add(Dense(1))
        model.add(Activation('tanh'))

        model.compile(loss='mse', optimizer='adam')
        print(model.summary())

        # fit the model
        epochs = 20 
        start = time.time()
        hist = model.fit(X_train, y_train, batch_size=1024, epochs=epochs, validation_split=0.2)
        print('training time : ', time.time() - start) 

        # predict the test data
        y_predict = model.predict(X_test)
        plt.plot(hist.history["loss"])
        plt.plot(hist.history["val_loss"])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.legend(['train', 'val'], loc='upper left')
        # plt.savefig("data/result/model_accuracy.png", dpi=600)

        df = data[len(data) - len(y_predict):]
        df = df.set_index(pd.date_range(end="2022-02-18", periods=len(y_predict), freq="1h"))
        df["predicted_close"] = y_predict / y_predict[0]
        df["actual_close"] = y_test 
        df["diff"] = df["predicted_close"].shift(freq=(f"-{lookback}h")) - df["predicted_close"]
        df["leverage"] = df["diff"]
        print(data_org["close"])
        exit()
        df["real_close"] = data_org["close"]
        # print(df)

        # plot the prediction against actual
        fig, ax = plt.subplots()
        fig.autofmt_xdate()
        ax.plot(df.index, df["close"] / df["close"].iloc[0], color='green', label="Actual Close")
        # ax1 = ax.twinx()
        ax.plot(df.index, df["predicted_close"], color='blue', label="Predicted Close")
        plt.title("Bitcoin Price")
        plt.ylabel("Price (USD)")
        plt.xlabel("Time (Hours)")
        plt.legend()
        # plt.savefig(f"data/result/price_{lookback}_{epochs}.png", dpi = 600)
        # plt.show()

        print(df.columns)
        return df["leverage"], df["real_close"]

    def analyze_data(self):
        # # fetch price data from exchange
        # self.load_data(period=700)
        # # analyze blockchain data to determine relevant ones
        # self._analyze_blockchain_data()
        # # analyze twitter to transform them into sentiment values
        # self._analyze_twitter_data()
        # # train the LSTM RNN using blockchain and twitter data
        df = self._build_lstm_model()
        print(df)

