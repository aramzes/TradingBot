from datetime import timedelta
from enum import Enum

from outcome import capture

class Side(Enum):
    BUY = 1
    SELL = -1
    NONE = 0

class Position:
    def __init__(self) -> None:
        self.entry_price = 0
        self.exit_price = 0
        self.side = Side.NONE 
        self.size = 0
        self.leverage = 0
        self.notional_size = 0
        self.capital = 0
        self.pnl = 0
        self.pnl_pct = 0
        self.entry_time = None
        self.duration = None
        self.exit_time = None
        self.trading_fees = 0
        self.trading_fees_rate = 0
        self.min_holding_period = None
    
    def __repr__(self):
        return f"(entry price: {self.entry_price}, exit price: {self.exit_price} size: {self.size}, duration: {self.duration}) pnl: {self.pnl})\n" 

    def reset(self):
        self.entry_price = 0
        self.exit_price = 0
        self.side = Side.NONE
        self.size = 0
        self.leverage = 0
        self.notional_size = 0
        self.capital = 0
        self.pnl = 0
        self.pnl_pct = 0
        self.entry_time = None
        self.duration = None
        self.exit_time = None
        self.trading_fees = 0
        self.trading_fees_rate = 0
        self.min_holding_period = None

    def open(self, capital, leverage, entry_time, entry_price, side, trading_fees_rate, min_holding_period):
        self.capital = capital
        self.notional_size = capital * leverage * side.value
        self.leverage = leverage
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.side = side
        self.size = (capital * leverage) / entry_price
        self.trading_fees_rate = trading_fees_rate
        self.trading_fees = capital * leverage * trading_fees_rate
        self.min_holding_period = min_holding_period

    def close(self, exit_price, exit_time):
        self.exit_price = exit_price
        self.exit_time = exit_time 
        self.duration = exit_time - self.entry_time
        self.trading_fees += self.capital * abs(self.leverage) * self.trading_fees_rate
        self.pnl = ((exit_price - self.entry_price) * self.size) - self.trading_fees 
        self.pnl_pct = round(self.pnl / self.capital * 100, 1)

    def closed(self):
        return self.entry_time is None  

    def ge_min_holding_period(self, cur_time, lev):
        if self.min_holding_period is None:
            return True
        return cur_time - self.entry_time >= self.min_holding_period