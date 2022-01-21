from datetime import timedelta
import sys
import random
from turtle import position

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
        self.positions = []
        self.strategy = load_strategy(strategy_name) 
        self.strategy.load_data()
        self._data = self.strategy.analyze_data()

    def backtest(self):
        self.generate_positions(self._data)
        pnl_pct_list = self.get_metrics()
        self.sample(self.strategy.data, self.positions, pnl_pct_list)
        self.plot(self.positions)

    def plot(self, positions: list):
        plt.plot(self.strategy.data.index, self.strategy.data.close)
        time = list(map(lambda x: x.entry_time, positions))
        leverage = list(map(lambda x: x.leverage, positions))
        plt.plot(time, leverage)
        plt.show()
         

    # calculate pnl and other values 
    # input: dataframe with datetimeIndex, close price and leverage columns
    def generate_positions(self, data: pd.DataFrame):
        data = data.reset_index() 

        capital = self.initial_capital
        positions = []
        for index, row in data.iterrows():
            if index == 0:
                data.loc[index, "leverage"] = 0
                continue
            cur_lev = row["leverage"] 
            cur_close = row["close_5m"]
            cur_time = row["startTime"]
            last_lev = data["leverage"].iloc[index - 1]
            if cur_lev != last_lev:
                # closing position
                if cur_lev == 0:
                    pos.close(cur_close, cur_time)
                    capital += pos.pnl
                    positions.append(pos)
                else:
                    # if last position is not closed 
                    if last_lev != 0:
                        pos.close(cur_close, cur_time)
                        capital += pos.pnl
                        positions.append(pos)
                    pos = Position()
                    pos.open(capital, cur_lev, cur_time, cur_close, Side(cur_lev), self.trading_fees_rate)

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
            trading fees: {trading_fees}
            Total: {round(self.final_capital, 0)}
            mean return per trade: {round(np.mean(pnl_pct_list))}
            standard deviation of pnl: {round(np.std(pnl_pct_list), 2)}
            median: {round(np.median(pnl_pct_list), 1)}
            max drawdown: {round(min(pnl_pct_list), 2)}
            max profit: {round(max(pnl_pct_list), 2)}
            trades: {len(self.positions)}
            longs: {len(longs)}
            shorts: {len(shorts)}
            average holding period: {round(np.mean([holding_periods]), 1)}
            winrate: {round(len(wins) / len(self.positions) * 100, 1)}%
        """)


    def sample(self, data: pd.DataFrame, trades: list, pnl_pct_list: list):
        n = len(trades) // 7
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
        
        sns.histplot(np.log1p(pnl_pct_list), label="real", kde=True)
        sns.histplot(np.log1p(samples_returns), label="samples", kde=True, color="orange")
        sns.set_theme(style="darkgrid")
        plt.legend()
        plt.show()
        
    
    def longer_trades_winrate(self, trades: list, period: int = 3):
        win = []
        for t in trades:
            if t.duration > timedelta(hours=period) and t.pnl_pct > 0:
                win.append(t)
        print(f"win rate of trades longer than {period}: {len(win) / len(longs + shorts) * 100}%")


    def check_lookahead(self, raw_data, signal_data, interval):
        raw_data = raw_data.resample(interval, on="startTime").mean()
        
        for n in range(101):
            cut_data = raw_data.drop(raw_data.tail(n).index)
            cut_data.reset_index(inplace=True)
            processed_data = self._random_strategy(cut_data, interval)
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