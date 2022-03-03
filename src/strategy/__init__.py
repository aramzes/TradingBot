class Strategy:
    def __init__(self):
        pass

    def load_data(self, resolution: int = 300, period: int = 365):
        self.data = self.feed.load_candles(self.market, resolution, period)