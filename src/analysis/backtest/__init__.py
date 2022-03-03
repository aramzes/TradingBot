from datetime import timedelta
from itertools import accumulate
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.load import load_strategy
from analysis.backtest.position import Position, Side 

class Backtest:
    def __init__(self, strategy_name: str) -> None:
        self.initial_capital = 10000
        self.final_capital = None
        self.trading_fees_rate = 0.0008
        self.min_holding_period = timedelta(hours=8)
        self.min_waiting_period = timedelta(hours=0)
        self.sampling_rate =  timedelta(minutes=5)
        self.positions = []
        self.strategy = load_strategy(strategy_name) 
        self.strategy.load_data(resolution=300, period=365)
        self._data = self.strategy.analyze_data()

    def backtest(self):
        self.generate_positions(self._data)
        # self.check_lookahead(self.strategy.data, self._data, interval="1H")
        pnl_pct_list = self.get_metrics()
        # self.sample(self.strategy.data, self.positions, pnl_pct_list)
        self.plot(self.positions)
        self._get_pnl()
    
    def _get_pnl(self):
        self._data["pnl"] = (
            self._data["leverage"].shift(1).fillna(0)
            * self._data["close"].pct_change().fillna(0)
            - abs(self._data["leverage"].diff().fillna(0) * self.trading_fees_rate)
            + 1
        ).cumprod()
        print(self._data)

    def _get_real_leverage(self, positions: list):
        entry_time = list(map(lambda x: x.entry_time.to_pydatetime(), positions))
        lev = list(map(lambda x: x.leverage, positions))
        self._data["real_lev"] = pd.DataFrame({"real_lev": lev}, index=entry_time)
        for p in positions:
            self._data.loc[(self._data.index >= p.entry_time) & (self._data.index < p.exit_time), "real_lev"] = p.leverage
        self._data["real_lev"].fillna(0, inplace=True)
        

    def plot(self, positions: list):
        exit_time = list(map(lambda x: x.exit_time.to_pydatetime(), positions))
        pnl = list(accumulate(list(map(lambda x: x.pnl, positions))))
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax.plot(self.strategy.data.startTime, self.strategy.data.close, label="price")
        ax.plot(self._data.index, self._data.ema_84, color="magenta")
        ax.plot(self._data.index, self._data.ema_300, color="darkorange")
        ax.plot(self._data.index, self._data.ema_1200, color="green")
        ax.plot(self._data.index, self._data.ST, color="red")
        self._get_real_leverage(positions)
        # ax2.plot(self._data.index, self._data["leverage"], color="black", label="leverage")
        # ax2.plot(self._data.index, self._data["real_lev"], color="darkorange", label="leverage")
        # ax3.plot(exit_time, pnl, color="green", label="pnl")
        plt.legend()
        plt.show()
         

    # calculate pnl and other values 
    # input: dataframe with datetimeIndex, close price and leverage columns
    def generate_positions(self, data: pd.DataFrame):
        data = data.reset_index() 

        data = data.drop_duplicates()
        data.iloc[[0, -1], data.columns.get_loc("leverage")] = [0, 0]

        capital = self.initial_capital
        positions = []
        pos = Position()
        lev = 0
        for index, row in data.iterrows():
            if index == 0:
                continue
            cur_lev = row["leverage"] 
            cur_close = row["close"]
            cur_time = row["startTime"]
            last_lev = data["leverage"].iloc[index - 1]
            # last_lev = lev
            if cur_lev != last_lev and cur_lev != lev and pos.ge_min_holding_period(cur_time, last_lev):
                # closing position
                if cur_lev == 0:
                    pos.close(cur_close, cur_time)
                    capital += pos.pnl
                    positions.append(pos)
                    pos = Position()
                else:
                    # if last position is not closed 
                    if last_lev != 0:
                        pos.close(cur_close, cur_time)
                        capital += pos.pnl
                        positions.append(pos)
                    pos = Position()
                    pos.open(capital, cur_lev, cur_time, cur_close, Side(cur_lev), self.trading_fees_rate, self.min_holding_period)
                lev = cur_lev 

        if not pos.closed():
            pos.close(data.iloc[-1].close, data.iloc[-1].startTime)
            positions.append(pos)

        self.final_capital = capital
        self.positions = positions

        return positions 

    def get_metrics(self):
        longs = list(filter(lambda x: x.side == Side.BUY, self.positions))
        shorts = list(filter(lambda x: x.side == Side.SELL, self.positions))
        wins = list(filter(lambda x: x.pnl > 0, self.positions))
        
        holding_periods_hours = list(map(lambda x: x.duration.total_seconds() / 3600, self.positions))
        pnl_list = list(map(lambda x: x.pnl, self.positions))
        pnl_pct_list = list(map(lambda x: x.pnl_pct / 100, self.positions))
        trading_fees = list(map(lambda x: x.trading_fees, self.positions))

        self._print_metrics(
            pnl_list,
            pnl_pct_list,
            sum(trading_fees),
            longs,
            shorts,
            holding_periods_hours,
            wins
        )

        return pnl_pct_list


    def _print_metrics(self,
        pnl_list: list,
        pnl_pct_list: list,
        trading_fees: float,
        longs: list,
        shorts: list,
        holding_periods: list, 
        wins: list
    ):
        print(f"""
            pnl of trades (with trading fees): {round(sum(pnl_list), 0)}
            trading fees: {round(trading_fees)}
            Total: {round(self.final_capital)}
            mean return per trade: {round(np.mean(pnl_pct_list), 3)}
            standard deviation of pnl: {round(np.std(pnl_pct_list), 2)}
            median: {round(np.median(pnl_pct_list), 3)}
            max drawdown: {round(min(pnl_pct_list), 2)}
            max profit: {round(max(pnl_pct_list), 2)}
            trades: {len(self.positions)}
            longs: {len(longs)}
            shorts: {len(shorts)}
            average holding period: {round(np.mean([holding_periods]), 1)}
            winrate: {round(len(wins) / len(self.positions) * 100, 1)}%
        """)
        self.longer_trades_winrate(self.positions)


    def sample(self, data: pd.DataFrame, trades: list, pnl_pct_list: list):
        n = len(trades) // 5 
        samples = random.sample(trades, n) 
        random_dates = data["startTime"].sample(n = n).reset_index(drop=True)
        samples_returns = []

        for i, s in enumerate(samples):
            entry_time = random_dates.iloc[i]
            exit_time = entry_time + s.duration
            if  exit_time < data.startTime.iloc[-1]:
                entry_price = data[data.startTime == entry_time].close.item()
                exit_price = data[data.startTime == exit_time].close.item()
                pnl = (exit_price - entry_price) * s.size
                samples_returns.append(pnl / s.notional_size)
        
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        sns.histplot(np.log1p(pnl_pct_list), label="real", kde=True, ax=ax)
        sns.histplot(np.log1p(samples_returns), label="samples", kde=True, color="orange", ax=ax2)
        sns.set_theme(style="darkgrid")
        plt.legend()
        plt.show()
        
    
    def longer_trades_winrate(self, trades: list, period: int = 3):
        wins = []
        for t in trades:
            if t.duration > timedelta(hours=period) and t.pnl > 0:
                wins.append(t)
        print(f"win rate of {len(wins)} trades longer than {period} hours: {round(len(wins) / len(trades) * 100, 1)}%")


    def check_lookahead(self, raw_data, signal_data, interval):
        # raw_data = raw_data.resample(interval, on="startTime").mean()
        
        for n in range(101):
            cut_data = raw_data.drop(raw_data.tail(n).index)
            cut_data.reset_index(inplace=True)
            processed_data = self.strategy.analyze_data(cut_data, interval)
            cut_leverage = processed_data["leverage"]
            signal_leverage = signal_data["leverage"]
            signal_leverage.drop(signal_leverage.tail(n).index, inplace=True)
            if not cut_leverage.equals(signal_leverage):
                print(signal_leverage - cut_leverage)
            elif n % 10 == 0:
                print(f"same for n: {n}")

if __name__ == "__main__":
    strategy_name = sys.argv[1:][0]
    Backtest(strategy_name).backtest() 