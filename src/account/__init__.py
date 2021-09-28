

class IAccount:
    def __init__(self):
        self.base_url = None
    
    def get_account(self):
        raise NotImplementedError

    def get_balance(self, symbol):
        raise NotImplementedError

    def get_position(self, symbol):
        raise NotImplementedError

    def get_last_trades(self, symbol):
        raise NotImplementedError
    
    def get_open_orders(self, symbol):
        raise NotImplementedError

    def get_order_history(self, symbol):
        raise NotImplementedError

    def place_order(self, symbol):
        raise NotImplementedError

    def get_order_status(self, order_id):
        raise NotImplementedError

    def modify_order(self, order_id):
        raise NotImplementedError

    def cancel_order(self, order_id):
        raise NotImplementedError 