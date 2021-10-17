import unittest
import time
import urllib.parse
import hmac
import json

from typing import Optional, Dict, Any, List
from requests import Request, Session, Response

from .. import IAccount

class Test(unittest.TestCase):
    def setUp(self):
        with open("config/test/ftx_account.json") as test_file:
            ftx_test = json.load(test_file)
        self.ftx_test = ftx_test
        # read-only API keys can be used without testing place_order
        self.account = FtxAccount(ftx_test["account"])

    def test_init(self):
        pass

    def test_get_account(self):
        res = self.account.get_account()
        self.assertEqual(res["totalAccountValue"], 0)
        total_size = 0
        for pos in res["positions"]:
            total_size += pos["size"]    
        self.assertEqual(total_size, 0)

    def test_get_balances(self):
        res = self.account.get_balances()
        self.assertEqual(res, [])

    def test_get_balance(self):
        res = self.account.get_balance("BTC")
        self.assertEqual(res, None)

    def test_get_positions(self):
        res = self.account.get_positions()
        if res != []:
            total_size = 0
            for pos in res:
                total_size += pos["size"]    
            self.assertEqual(total_size, 0)
    
    def test_get_position(self):
        res = self.account.get_position("BTC-PERP")
        self.assertEqual(res, None) 

    def test_get_trades(self):
        res = self.account.get_trades("BTC-PERP")
        self.assertEqual(res, [])

    def test_get_open_orders(self):
        res = self.account.get_open_orders("BTC-PERP")
        self.assertEqual(res, [])
    
    def test_get_order_history(self):
        res = self.account.get_order_history("BTC-PERP")
        self.assertEqual(res, [])

    def test_place_order(self):
        ftx_order = FtxOrder(self.ftx_test["order"])
        res = self.account.place_order(ftx_order)
        self.assertIn("Error", res)
    
    def test_get_order_status(self):
        res = self.account.get_order_status(self.ftx_test["order_id"])
        self.assertEqual(res["status"], "closed")        

    def tearDown(self):
        self.account.rest_client._session.close()

class FtxRestClient:

    _ENDPOINT = 'https://ftx.com/api/'

    def __init__(self, api_key=None, api_secret=None, subaccount_name=None) -> None:
        self._session = Session()
        self._api_key = api_key
        self._api_secret = api_secret
        self._subaccount_name = subaccount_name

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('GET', path, params=params)

    def post(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('POST', path, json=params)

    def delete(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('DELETE', path, json=params)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        request = Request(method, self._ENDPOINT + path, **kwargs)
        self._sign_request(request)
        response = self._session.send(request.prepare())
        return self._process_response(response)

    def _sign_request(self, request: Request) -> None:
        ts = int(time.time() * 1000)
        prepared = request.prepare()
        signature_payload = f'{ts}{prepared.method}{prepared.path_url}'.encode()
        if prepared.body:
            signature_payload += prepared.body
        signature = hmac.new(self._api_secret.encode(), signature_payload, 'sha256').hexdigest()
        request.headers['FTX-KEY'] = self._api_key
        request.headers['FTX-SIGN'] = signature
        request.headers['FTX-TS'] = str(ts)
        if self._subaccount_name:
            request.headers['FTX-SUBACCOUNT'] = urllib.parse.quote(self._subaccount_name)

    def _process_response(self, response: Response) -> Any:
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        else:
            if not data['success']:
                raise Exception(data['error'])
            return data['result']


class FtxOrder:
    market: str = None             # e.g. "BTC/USDT" and "BTC-PERP"
    side: str = None               # "buy" or "sell"
    type: str = None               # "limit" or "market"
    price: float = None            # null for market orders
    size: float = None                    
    reduce_only: bool = False       # Optional (default: False)
    ioc: bool = False               # Optional (default: False)
    post_only: bool = False         # Optional (default: False)
    client_id: str = False          # Optional (default: null)

    def __init__(self, orderDict: Dict):
        self.market = orderDict["market"]
        self.side = orderDict["side"]
        self.type = orderDict["type"]
        self.size = orderDict["size"]

        if self.type == "limit":
            self.price = orderDict["price"]
        
        self.ioc = orderDict["ioc"] if "ioc" in orderDict else False
        self.reduce_only = orderDict["reduce_only"] if "reduce_only" in orderDict else False
        self.post_only = orderDict["post_only"] if "post_only" in orderDict else False
        self.client_id = orderDict["client_id"] if "client_id" in orderDict else None

    def __str__(self):
        return str(self.__dict__)


class FtxAccount(IAccount):

    def __init__(self, api={}):
        self.rest_client = FtxRestClient(
            api["key"],
            api["secret"],
            api["subaccount"]
        )

    def get_account(self):
        account = self.rest_client.get("account")
        return account

    def get_balance(self, symbol):
        balances = self.get_balances() 
        if symbol in balances:
            return balances[symbol]
        else:
            return None

    def get_balances(self):
        balances = self.rest_client.get("wallet/balances")
        return balances

    def get_position(self, symbol):
        positions = self.get_positions()
        if symbol in positions:
            return positions[symbol]
        else:
            return None

    def get_positions(self):
        positions = self.rest_client.get("positions")
        return positions

    def get_trades(self, market):
        params = {
            "market": market
        }
        trades =  self.rest_client.get(f"fills", params)
        return trades
    
    def get_open_orders(self, market):
        params = {
            "market": market
        }
        open_orders = self.rest_client.get("orders", params)
        return open_orders

    def get_order_history(self, market):
        params = {
            "market": market
        }
        order_history = self.rest_client.get("orders/history", params)
        return order_history 

    def place_order(self, order: FtxOrder):
        try:
            new_order = self.rest_client.post("orders", order)
        except Exception as e:
            return (f"Error: {e}")
        return new_order

    def get_order_status(self, order_id):
        order = self.rest_client.get(f"orders/{order_id}")
        return order

    def modify_order(self, order_id):
        raise NotImplementedError

    def cancel_order(self, order_id):
        raise NotImplementedError 