


class Ifeed:
    def __init__(self):
        self.base_url = None

    # # private info
    # def get_account(self):
    #     raise NotImplementedError

    # def get_balance(self, symbol):
    #     raise NotImplementedError

    # def get_account_trades(self, symbol):
    #     raise NotImplementedError

    # def get_position(self, symbol):
    #     raise NotImplementedError

    # def place_order(self, symbol):
        raise NotImplementedError

    # public info
    def get_orderbook(self, symbol):
        raise NotImplementedError

    def get_public_trades(self, symbol):
        raise NotImplementedError

    def get_ticker(self, symbol):
        raise NotImplementedError
    
    def get_historical_data(self, symbol):
        raise NotImplementedError

    