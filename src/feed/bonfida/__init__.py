import unittest
import requests
import math

from solana.rpc.api import Client
from solana.publickey import PublicKey

from .. import Ifeed 

class TestFeed(unittest.TestCase):
    def setUp(self):
        self.feed = BonfidaFeed()

    def test_get(self):
        pass

    def test_get_markets(self):
        res = self.feed.get_markets()
        self.assertTrue(res["success"])
        self.assertIsInstance(res["result"], list)
        markets = { m["name"] : m["address"] for m in res["result"] }
        self.assertIn("BTC-PERP", markets)
        self.assertIn("ETH-PERP", markets)
        self.assertIn("SOL-PERP", markets)

    def test_get_market_address(self):
        res = self.feed.get_market_address("BTC-PERP")
        self.assertEqual(res, "475P8ZX3NrzyEMJSFHt9KCMjPpWBWGa6oNxkWcwww2BR")

    def test_get_market_data(self):
        res1 = self.feed.get_market_data(symbol="BTC-PERP")
        print(res1)
        for r in res1["result"]:
            print(r["funding"] / 1000000)
        self.assertTrue(res1["success"])
        res1 = res1["result"]["marketInfo"]
        self.assertIn("markPrice", res1)
        self.assertIn("indexPrice", res1)
        self.assertIn("fundingLong", res1)
        self.assertIn("fundingShort", res1)

        res2 = self.feed.get_market_data(address="475P8ZX3NrzyEMJSFHt9KCMjPpWBWGa6oNxkWcwww2BR") 
        self.assertTrue(res2["success"])
        res2 = res2["result"]["marketInfo"]
        self.assertIn("markPrice", res2)
        self.assertIn("indexPrice", res2)
        self.assertIn("fundingLong", res2)
        self.assertIn("fundingShort", res2)
    '''    
    def test_get_insurance_fund(self):
        res1 = self.feed.get_insurance_fund("BTC-PERP")
        print(res1)
        res2 = self.feed.get_insurance_fund("ETH-PERP")
        print(res2)
        res3 = self.feed.get_insurance_fund("SOL-PERP")
        print(res3)

    def test_get_k(self):
        res1 = self.feed.get_k("BTC-PERP")
        print(res1)
        res2 = self.feed.get_k("ETH-PERP")
        print(res2)
        res3 = self.feed.get_k("SOL-PERP")
        print(res3)
    '''
                
class BonfidaFeed(Ifeed):
    def __init__(self):
        self.base_url = "http://localhost:3000/"
        self.solana_client = Client("https://api.mainnet-beta.solana.com/")
        self.vault_address = {
            "BTC-PERP": "Fc6keb9ZANEmdEhUgEiB9T9nBU6oSEiTmNbfd9h7g4HA",
            "ETH-PERP": "7A8qLjuagUJ97Q9Un5AJiavhEg55apXsgQ51s3Aq5sLe",
            "SOL-PERP": "LXMAV4hRP44N9VoL3XsSo3HqHP2kL1GxqwvJ8qGitv1"
        }

    def get(self, endpoint, params={}):
        return requests.get(self.base_url + endpoint, params).json()

    def get_markets(self):
        return self.get("markets")
        
    def get_market_address(self, symbol):
        markets =  self.get_markets()["result"]
        for market in markets:
            if market["name"] == symbol:
                return market["address"]
        return None

    def get_market_data(self, symbol=None, address=None):
        assert([symbol, address] != [None, None])
        market_address = None
        if symbol:
            market_address = self.get_market_address(symbol)
        elif address:
            market_address = address

        assert(market_address != None)
        return self.get(f"markets/{market_address}")

    def get_account_balance(self, public_key):
        res = self.solana_client.get_token_account_balance(PublicKey(public_key))
        balance = int(res["result"]["value"]["amount"])
        return balance 

    def get_insurance_fund(self, market):
        def _compute_add_v_pc(original_v_coin_amount, added_v_coin_amount, v_pc_amount):
            final_v_coin_amount = original_v_coin_amount + added_v_coin_amount
            if final_v_coin_amount < 0:
                print("Error in get_insurance_fund: Vcoin amount is too large!")
                return -99999999
            add_pc_amount = abs(added_v_coin_amount) * v_pc_amount / final_v_coin_amount
            return -math.copysign(1, added_v_coin_amount) * add_pc_amount

                
        res = self.get_market_data(market)
        market_state = res["result"]["marketState"]
        original_v_coin_amount = market_state["vCoinAmount"]
        added_v_coin_amount = (market_state["openLongsVCoin"]) - (market_state["openShortsVCoin"]) 
        v_pc_amount = market_state["vQuoteAmount"]
        delta = -1 * _compute_add_v_pc(original_v_coin_amount, added_v_coin_amount, v_pc_amount)

        total_payout = delta + (market_state["totalCollateral"]) + (market_state["openShortsVPc"]) - (market_state["openLongsVPc"]) 

        total_payout = max(0, total_payout)

        market_vault_balance = self.get_account_balance(self.vault_address[market])

        print(market_vault_balance, market_state["totalUserBudgets"], market_state["totalFeeBudget"], market_state["rebalancingFunds"])
        insurance_fund = ( market_vault_balance
             - total_payout
             - market_state["totalUserBudgets"]
             - market_state["totalFeeBudget"]
             - market_state["rebalancingFunds"]
        )
        return insurance_fund / 1000000


    def get_k(self, market):
        res = self.get_market_data(market)
        market_state = res["result"]["marketState"]
        return market_state["vCoinAmount"] * market_state["vQuoteAmount"]

