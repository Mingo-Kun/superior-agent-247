#!/usr/bin/python
# Initialize logger at the VERY TOP
import logging
logger = logging.getLogger(__name__)

import json
import asyncio
import math
import threading
import time
import traceback
from threading import Timer
from zlib import crc32

import websocket

from bitget.consts import GET, REQUEST_PATH, SIGN_TYPE, RSA
from bitget import consts as c
from bitget import utils

WS_OP_LOGIN = 'login'
WS_OP_SUBSCRIBE = "subscribe"
WS_OP_UNSUBSCRIBE = "unsubscribe"

def handle(message):
    print("default:" + message)

def handel_error(message):
    print("default_error:" + message)

class BitgetWsClient:
    def __init__(self, url, need_login=False, loop=None):
        utils.check_none(url, "url")
        self.__need_login = need_login
        self.__connection = False
        self.__should_reconnect = True
        self.__login_status = False
        self.__reconnect_status = False
        self.__api_key = None
        self.__api_secret_key = None
        self.__passphrase = None
        self.__all_suribe = set()
        self.__listener = handle
        self.__error_listener = handel_error
        self.__url = url
        self.__scribe_map = {}
        self.__allbooks_map = {}
        self.__lock = threading.Lock()
        self.__loop = loop  # Store event loop reference

    def build(self):
        self.__ws_client = self.__init_client()
        __thread = threading.Thread(target=self.connect)
        __thread.start()
        
        # Wait for connection to establish
        for _ in range(30):  # 30 second timeout
            if self.has_connect():
                break
            print("start connecting... url: ", self.__url)
            time.sleep(1)

        if self.__need_login:
            self.__login()

        return self

    def api_key(self, api_key):
        self.__api_key = api_key
        return self

    def api_secret_key(self, api_secret_key):
        self.__api_secret_key = api_secret_key
        return self

    def passphrase(self, passphrase):
        self.__passphrase = passphrase
        return self

    def listener(self, listener):
        self.__listener = listener
        return self

    def error_listener(self, error_listener):
        self.__error_listener = error_listener
        return self

    def has_connect(self):
        return self.__connection

    def __init_client(self):
        try:
            return websocket.WebSocketApp(self.__url,
                                      on_open=self.__on_open,
                                      on_message=self.__on_message,
                                      on_error=self.__on_error,
                                      on_close=self.__on_close)
        except Exception as ex:
            logger.error(f"Error initializing WebSocket: {ex}")

    def __login(self):
        utils.check_none(self.__api_key, "api key")
        utils.check_none(self.__api_secret_key, "api secret key")
        utils.check_none(self.__passphrase, "passphrase")
        timestamp = int(round(time.time()))
        sign = utils.sign(utils.pre_hash(timestamp, GET, c.REQUEST_PATH), self.__api_secret_key)
        if c.SIGN_TYPE == c.RSA:
            sign = utils.signByRSA(utils.pre_hash(timestamp, GET, c.REQUEST_PATH), self.__api_secret_key)
        ws_login_req = WsLoginReq(self.__api_key, self.__passphrase, str(timestamp), sign)
        self.send_message(WS_OP_LOGIN, [ws_login_req])
        logger.info("Attempting WebSocket authentication")
        while not self.__login_status:
            time.sleep(1)

    def connect(self):
        try:
            # Disable hostname verification for SSL due to mismatch errors.
            # WARNING: This reduces security. Investigate the root cause if possible.
            self.__ws_client.run_forever(ping_timeout=10, sslopt={"check_hostname": False})
        except Exception as ex:
            logger.error(f"WebSocket connection error: {ex}")

    def send_message(self, op, args):
        message = json.dumps(BaseWsReq(op, args), default=lambda o: o.__dict__)
        if op == 'login':
            logger.debug("Sending login request (credentials redacted)")
        else:
            logger.debug(f"Sending message: {message[:100]}...")
        self.__ws_client.send(message)

    def subscribe(self, channels, listener=None):
        if listener:
            for chanel in channels:
                chanel.inst_type = str(chanel.inst_type)
                with self.__lock:
                    self.__scribe_map[chanel] = listener

        for channel in channels:
            with self.__lock:
                self.__all_suribe.add(channel)

        self.send_message(WS_OP_SUBSCRIBE, channels)

    def unsubscribe(self, channels):
        try:
            with self.__lock:
                for channel in channels:
                    if channel in self.__scribe_map:
                        del self.__scribe_map[channel]

                for channel in channels:
                    if channel in self.__all_suribe:
                        self.__all_suribe.remove(channel)

                self.send_message(WS_OP_UNSUBSCRIBE, channels)
        except Exception as e:
            logger.error(f"Error unsubscribing: {e}")

    def __on_open(self, ws):
        logger.info('connection is success....')
        self.__connection = True
        self.__reconnect_status = False

    def __on_message(self, ws, message):
        if message == 'pong':
            logger.debug("Received pong response")
            return
        json_obj = json.loads(message)
        if "code" in json_obj and json_obj.get("code") != 0:
            if self.__error_listener:
                if asyncio.iscoroutinefunction(self.__error_listener):
                    if self.__loop:
                        asyncio.run_coroutine_threadsafe(self.__error_listener(message), self.__loop)
                    else:
                        logger.warning("No event loop available for async error callback")
                else:
                    self.__error_listener(message)
                return

        if "event" in json_obj and json_obj.get("event") == "login":
            self.__login_status = True
            return

        listenner = None
        if "data" in json_obj and not self.__check_sum(json_obj):
            return

        if listenner:
            if asyncio.iscoroutinefunction(listenner):
                if self.__loop:
                    asyncio.run_coroutine_threadsafe(listenner(message), self.__loop)
                else:
                    logger.warning("No event loop available for async callback")
            else:
                listenner(message)
            return

        if asyncio.iscoroutinefunction(self.__listener):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.__listener(message))
        else:
            self.__listener(message)

    def __on_error(self, ws, msg):
        logger.error(f"WebSocket error: {msg}")
        self.__close()
        if not self.__reconnect_status:
            self.__re_connect()

    def __on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.__close()
        if not self.__reconnect_status:
            self.__re_connect()

    def __re_connect(self):
        if self.__reconnect_status:
            return
            
        if not self.__should_reconnect:
            return

        self.__reconnect_status = True
        logger.info("Starting WebSocket reconnection...")
        self.__reconnect_attempts = getattr(self, '_BitgetWsClient__reconnect_attempts', 0)

        try:
            self.__close()
            self.build()
            if self.__all_suribe:
                logger.info(f"Restoring subscriptions: {self.__all_suribe}")
                self.subscribe(list(self.__all_suribe))

            self.__reconnect_status = False
            self.__reconnect_attempts = 0
            logger.info("WebSocket reconnected successfully")
            return

        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
            self.__reconnect_attempts += 1
            delay = min(30, 2 ** self.__reconnect_attempts)
            logger.info(f"Retrying connection in {delay} seconds...")
            Timer(delay, self.__re_connect).start()

    def __close(self):
        self.__login_status = False
        self.__connection = False
        self.__reconnect_status = False
        if hasattr(self, '_BitgetWsClient__ws_client') and self.__ws_client:
            try:
                self.__ws_client.keep_running = False
                self.__ws_client.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            time.sleep(0.1)
        self.__ws_client = None

    def __check_sum(self, json_obj):
        try:
            if "arg" not in json_obj or "action" not in json_obj:
                return True
            arg = str(json_obj.get('arg')).replace("\'", "\"")
            action = str(json_obj.get('action')).replace("\'", "\"")
            data = str(json_obj.get('data')).replace("\'", "\"")

            subscribe_req = json.loads(arg, object_hook=self.__dict_to_subscribe_req)

            if subscribe_req.channel != "books":
                return True

            books_info = json.loads(data, object_hook=self.__dict_books_info)[0]

            if action == "snapshot":
                self.__allbooks_map[subscribe_req] = books_info
                return True
            if action == "update":
                all_books = self.__allbooks_map.get(subscribe_req)
                if all_books is None:
                    logger.warning(f"Received update for {subscribe_req} before snapshot. Requesting snapshot.")
                    self.unsubscribe([subscribe_req])
                    time.sleep(0.1)
                    self.subscribe([subscribe_req])
                    return False

                all_books = all_books.merge(books_info)
                check_sum = all_books.check_sum(books_info.checksum)
                if not check_sum:
                    logger.warning(f"Checksum failed for {subscribe_req}. Resubscribing.")
                    self.unsubscribe([subscribe_req])
                    time.sleep(0.1)
                    self.subscribe([subscribe_req])
                    if subscribe_req in self.__allbooks_map:
                        del self.__allbooks_map[subscribe_req]
                    return False
                self.__allbooks_map[subscribe_req] = all_books
        except Exception as e:
            msg = traceback.format_exc()
            logger.error(f"Error during checksum: {msg}")

        return True

    def __dict_books_info(self, dict):
        return BooksInfo(dict['asks'], dict['bids'], dict['checksum'])

    def __dict_to_subscribe_req(self, dict):
        if "instId" in dict:
            instId = dict['instId']
        else:
            instId = dict['coin']
        return SubscribeReq(dict['instType'], dict['channel'], instId)

class BooksInfo:
    def __init__(self, asks, bids, checksum):
        self.asks = asks if asks else []
        self.bids = bids if bids else []
        self.checksum = checksum

    def merge(self, book_info):
        self.asks = self.innerMerge(self.asks, book_info.asks, False)
        self.bids = self.innerMerge(self.bids, book_info.bids, True)
        return self

    def innerMerge(self, all_list, update_list, is_reverse):
        price_and_value = {v[0]: v for v in all_list}

        for v in update_list:
            if len(v) >= 2 and v[1] == "0":
                price_and_value.pop(v[0], None)
                continue
            if len(v) >= 2:
                 price_and_value[v[0]] = v

        keys = sorted(price_and_value.keys(), reverse=is_reverse)
        result = []
        count = 0
        for k in keys:
            if count < 25:
                result.append(price_and_value[k])
                count += 1
            else:
                break
        while len(result) < 25:
             result.append(None)

        return result

    def check_sum(self, new_check_sum):
        crc32str = ''
        bids_padded = self.bids + [None] * (25 - len(self.bids))
        asks_padded = self.asks + [None] * (25 - len(self.asks))

        for x in range(25):
            bid = bids_padded[x]
            ask = asks_padded[x]
            if bid and len(bid) >= 2:
                crc32str += f"{bid[0]}:{bid[1]}:"
            if ask and len(ask) >= 2:
                crc32str += f"{ask[0]}:{ask[1]}:"

        if not crc32str:
             logger.warning("Checksum string is empty, cannot verify.")
             return False

        crc32str = crc32str[:-1]
        merge_num = crc32(bytes(crc32str, encoding="utf8"))
        signed_merge_num = self.__signed_int(merge_num)
        return signed_merge_num == new_check_sum

    def __signed_int(self, checknum):
        if checknum >= 2**31:
            return checknum - 2**32
        return checknum

class SubscribeReq:
    def __init__(self, inst_type, channel, instId):
        self.inst_type = inst_type
        self.channel = channel
        self.inst_id = instId
        self.coin = instId

    def __eq__(self, other) -> bool:
        return (isinstance(other, SubscribeReq) and
                self.inst_type == other.inst_type and
                self.channel == other.channel and
                self.inst_id == other.inst_id)

    def __hash__(self) -> int:
        return hash((self.inst_type, self.channel, self.inst_id))

    def __repr__(self):
        return f"SubscribeReq(inst_type='{self.inst_type}', channel='{self.channel}', inst_id='{self.inst_id}')"

class BaseWsReq:
    def __init__(self, op, args):
        self.op = op
        self.args = args

class WsLoginReq:
    def __init__(self, api_key, passphrase, timestamp, sign):
        self.api_key = api_key
        self.passphrase = passphrase
        self.timestamp = timestamp
        self.sign = sign
