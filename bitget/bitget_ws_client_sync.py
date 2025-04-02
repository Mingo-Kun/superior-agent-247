#!/usr/bin/python
import json
import logging
import time
import websocket
from bitget.consts import GET, REQUEST_PATH, SIGN_TYPE, RSA
from bitget import utils

logger = logging.getLogger(__name__)

WS_OP_LOGIN = 'login'
WS_OP_SUBSCRIBE = "subscribe"
WS_OP_UNSUBSCRIBE = "unsubscribe"

def default_handler(message):
    logger.debug(f"Received message: {message}")

def default_error_handler(message):
    logger.error(f"WebSocket error: {message}")

class BitgetWsClientSync:
    def __init__(self, url, need_login=False):
        self.url = url
        self.need_login = need_login
        self.ws = None
        self.connected = False
        self.login_status = False
        self.api_key = None
        self.api_secret_key = None
        self.passphrase = None
        self.subscriptions = set()
        self.message_handler = default_handler
        self.error_handler = default_error_handler

    def connect(self):
        """Establish WebSocket connection"""
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            # Run in foreground (blocking)
            self.ws.run_forever()
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def _on_open(self, ws):
        logger.info("WebSocket connection established")
        self.connected = True
        if self.need_login:
            self._login()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get('event') == 'pong':
                logger.debug("Received pong response")
                return
            
            self.message_handler(message)
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
        self.error_handler(str(error))
        self.connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False

    def _login(self):
        """Authenticate with the WebSocket server"""
        timestamp = str(int(time.time()))
        sign = utils.sign(utils.pre_hash(timestamp, GET, REQUEST_PATH), self.api_secret_key)
        login_req = {
            "op": WS_OP_LOGIN,
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": timestamp,
                "sign": sign
            }]
        }
        self.send_message(login_req)
        logger.info("Login request sent")

    def send_message(self, message):
        """Send a message through the WebSocket"""
        if isinstance(message, dict):
            message = json.dumps(message)
        if self.ws and self.connected:
            self.ws.send(message)
            logger.debug(f"Message sent: {message[:100]}...")
        else:
            logger.error("Cannot send message - WebSocket not connected")

    def subscribe(self, channels):
        """Subscribe to channels"""
        if not isinstance(channels, list):
            channels = [channels]
        
        subscribe_req = {
            "op": WS_OP_SUBSCRIBE,
            "args": channels
        }
        self.send_message(subscribe_req)
        self.subscriptions.update(channels)
        logger.info(f"Subscribed to: {channels}")

    def unsubscribe(self, channels):
        """Unsubscribe from channels"""
        if not isinstance(channels, list):
            channels = [channels]
        
        unsubscribe_req = {
            "op": WS_OP_UNSUBSCRIBE,
            "args": channels
        }
        self.send_message(unsubscribe_req)
        self.subscriptions.difference_update(channels)
        logger.info(f"Unsubscribed from: {channels}")

    def close(self):
        """Close the WebSocket connection"""
        if self.ws:
            self.ws.close()
            self.connected = False
            logger.info("WebSocket connection closed")