


class Ifeed:
    def __init__(self):
        self.base_url = None

    def get_orderbook(self, symbol):
        raise NotImplementedError

    def get_public_trades(self, symbol):
        raise NotImplementedError

    def get_ticker(self, symbol):
        raise NotImplementedError
    
    def get_historical_data(self, symbol):
        raise NotImplementedError

    