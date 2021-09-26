import unittest
import requests

from account import IAccount
from feed.bonfida import BonfidaFeed

class TestAccount(unittest.TestCase):
    def setUp(self):
       self.account = BonfidaAccount() 
    
    def test_get(self):
        pass

    def test_post(self):
        pass

    def test_get_account(self):
        res = self.account.get_account()

    # def test_create_subaccount(self):
    #     res = self.account.create_subaccount("BTC-PERP")
    #     print(res)


    # def test_close_subaccount(self):
    #     res = self.account.close_subaccount("BTC-PERP")
    #     print(res)


    # def test_create_subaccount(self):
    #     res = self.account.disposit_to_subaccount("BTC-PERP")
    #     print(res)


    # def test_create_subaccount(self):
    #     res = self.account.withdraw_from_subaccount("BTC-PERP")
    #     print(res)

class BonfidaAccount(IAccount):
    def __init__(self):
        self.feed = BonfidaFeed()
        self.base_url = self.feed.base_url
        self._get_init_status()

    def _get_init_status(self):
        all_markets = self.feed.get_markets()["result"]
        self.subaccounts = { m["name"]: [] for m in all_markets }
        existing_subccounts = self.get_account()["result"]
        for account in existing_subccounts:
            self.subaccounts[account["market"]].append(account["address"])

    def get(self, endpoint, params={}):
        return requests.get(self.base_url + endpoint, params).json()
    
    def post(self, endpoint, params={}):
        return requests.post(self.base_url + endpoint, params)

    def get_account(self):
        return self.get("account")    

    def create_subaccount(self, symbol):
        market_address = self.feed.get_market_address(symbol)
        return self.post(f"account/create/{market_address}")

    def close_subaccount(subaccount_address):
        return self.post(f"account/close/{subaccount_address}")
    
    def deposit_subaccount(subaccount_address, amount):
        params = {
            "amount": amount
        }
        return self.post(f"account/deposit/{subaccount_address}")

    def withdraw_subaccount(subaccount_address, amount):
        params = {
            "amount": amount
        }
        return self.post(f"account/withdraw/{subaccount_address}")