#!/usr/bin/python
from bitget.client import Client
from bitget.consts import GET, POST


class BitgetApi(Client):
    # Updated __init__ to accept and pass base_url
    def __init__(self, api_key, api_secret_key, passphrase, use_server_time=False, first=False, base_url=None):
        # If base_url is not provided, it will default to c.API_URL in the Client class
        Client.__init__(self, api_key, api_secret_key, passphrase, use_server_time, first, base_url=base_url)

    def post(self, request_path, params):
        return self._request_with_params(POST, request_path, params)

    def get(self, request_path, params):
        return self._request_with_params(GET, request_path, params)
