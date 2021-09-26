

class IAccount:
    def __init__(self):
        self.base_url = None
    
    def get_account(self):
        raise NotImplementedError

    def get_balance(self, symbol):
        raise NotImplementedError

    def get_account_trades(self, symbol):
        raise NotImplementedError

    def get_position(self, symbol):
        raise NotImplementedError

    def place_order(self, symbol):
        raise NotImplementedError
