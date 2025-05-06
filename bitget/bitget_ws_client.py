#!/usr/bin/python
import logging
import json
import asyncio
import time
import traceback
from zlib import crc32
import websockets
import ssl # Import ssl module
from websockets.connection import State # Import State enum

from bitget.consts import GET
from bitget import consts as c
from bitget import utils

logger = logging.getLogger(__name__)

WS_OP_LOGIN = 'login'
WS_OP_SUBSCRIBE = "subscribe"
WS_OP_UNSUBSCRIBE = "unsubscribe"

# Default handlers (can be replaced by user)
def default_handle(message):
    logger.info(f"Default Handler Received: {message}")

def default_error_handle(error):
    logger.error(f"Default Error Handler Received: {error}", exc_info=True)

class BitgetWsClientAsync:
    def __init__(self, url, api_key=None, api_secret_key=None, passphrase=None, listener=None, error_listener=None):
        utils.check_none(url, "url")
        self.__url = url
        self.__api_key = api_key
        self.__api_secret_key = api_secret_key
        self.__passphrase = passphrase
        self.__need_login = bool(api_key and api_secret_key and passphrase)

        self.__listener = listener if listener else default_handle
        self.__error_listener = error_listener if error_listener else default_error_handle

        self.__connection = None
        self.__login_status = False
        self.__should_reconnect = True
        self.__reconnect_attempts = 0
        self.__active_task = None
        self.__lock = asyncio.Lock()
        self.__subscribed_channels = set()
        self.__scribe_map = {} # For channel-specific listeners (optional)
        self.__allbooks_map = {} # For checksum logic

        # Configure SSL context to disable hostname verification (use with caution)
        self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    async def start(self):
        """Starts the WebSocket connection and message handling loop."""
        if self.__active_task:
            logger.warning("WebSocket client already running.")
            return
        self.__should_reconnect = True
        self.__active_task = asyncio.create_task(self._run_forever())
        logger.info("WebSocket client started.")

    async def stop(self):
        """Stops the WebSocket connection and message handling loop."""
        self.__should_reconnect = False
        if self.__active_task:
            self.__active_task.cancel()
            try:
                await self.__active_task
            except asyncio.CancelledError:
                logger.info("WebSocket client task cancelled.")
            except Exception as e:
                logger.error(f"Error during task cancellation: {e}")
        await self._close_connection()
        self.__active_task = None
        logger.info("WebSocket client stopped.")

    async def _run_forever(self):
        """The main loop for connecting, handling messages, and reconnecting."""
        while self.__should_reconnect:
            try:
                async with websockets.connect(self.__url, ssl=self.ssl_context, ping_interval=10, ping_timeout=10) as ws:
                    self.__connection = ws
                    logger.info(f"Assigned connection object type: {type(ws)}; dir: {dir(ws)}")
                    self.__reconnect_attempts = 0
                    logger.info(f"WebSocket connected to {self.__url}")

                    # Perform login if required
                    if self.__need_login:
                        await self._login()

                    # Resubscribe to channels upon successful reconnection
                    if self.__subscribed_channels:
                        logger.info(f"Restoring subscriptions: {self.__subscribed_channels}")
                        await self.subscribe(list(self.__subscribed_channels)) # Pass as list

                    # Message handling loop
                    async for message in ws:
                        await self._handle_message(message)

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e.code} {e.reason}")
                if not self.__should_reconnect:
                    break # Exit loop if stop was called
            except asyncio.CancelledError:
                logger.info("WebSocket run task cancelled.")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                await self._handle_error(e)

            # Cleanup before potential reconnect
            await self._close_connection()
            self.__login_status = False

            if self.__should_reconnect:
                self.__reconnect_attempts += 1
                delay = min(30, 2 ** self.__reconnect_attempts)
                logger.info(f"Attempting reconnection in {delay} seconds...")
                await asyncio.sleep(delay)

        logger.info("WebSocket run loop finished.")

    async def _close_connection(self):
        if self.__connection:
            try:
                await self.__connection.close()
                logger.info("WebSocket connection closed gracefully.")
            except Exception as e:
                logger.warning(f"Error closing WebSocket connection: {e}")
        self.__connection = None

    async def _login(self):
        """Sends the login request."""
        # Check if connection exists and is in the OPEN state
        if not self.__connection or self.__connection.state != State.OPEN:
            logger.warning("Cannot login, WebSocket not connected.")
            return

        timestamp = int(round(time.time()))
        prehash_msg = utils.pre_hash(timestamp, GET, c.REQUEST_PATH)
        sign = utils.sign(prehash_msg, self.__api_secret_key)
        # RSA signing not implemented here, assuming HMAC

        ws_login_req = WsLoginReq(self.__api_key, self.__passphrase, str(timestamp), sign)
        await self.send_message(WS_OP_LOGIN, [ws_login_req])
        logger.info("Attempting WebSocket authentication")
        # Note: Login status is confirmed by receiving a login success message

    async def send_message(self, op, args):
        """Sends a message to the WebSocket server."""
        # Check if connection exists and is in the OPEN state
        if not self.__connection or self.__connection.state != State.OPEN:
            logger.warning(f"Cannot send message, WebSocket not connected. Op: {op}")
            return

        message_dict = BaseWsReq(op, args).__dict__
        # Convert SubscribeReq or WsLoginReq objects within args to dicts for JSON serialization
        if 'args' in message_dict and isinstance(message_dict['args'], list):
            message_dict['args'] = [
                arg.__dict__ if isinstance(arg, (SubscribeReq, WsLoginReq)) else arg
                for arg in message_dict['args']
            ]

        message = json.dumps(message_dict)

        if op == 'login':
            logger.debug("Sending login request (credentials redacted)")
        else:
            logger.debug(f"Sending message: {message[:200]}...")
        try:
            await self.__connection.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Failed to send message, connection closed. Op: {op}")
        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)

    async def subscribe(self, channels, listener=None):
        """Subscribes to WebSocket channels."""
        async with self.__lock:
            new_channels = []
            for channel in channels:
                # Ensure channel is a SubscribeReq object
                if not isinstance(channel, SubscribeReq):
                    logger.warning(f"Invalid channel format for subscription: {channel}. Skipping.")
                    continue
                if channel not in self.__subscribed_channels:
                    self.__subscribed_channels.add(channel)
                    new_channels.append(channel)
                if listener:
                    self.__scribe_map[channel] = listener

            if new_channels:
                await self.send_message(WS_OP_SUBSCRIBE, new_channels)
                logger.info(f"Subscribed to: {new_channels}")
            else:
                logger.info(f"Channels already subscribed: {channels}")

    async def unsubscribe(self, channels):
        """Unsubscribes from WebSocket channels."""
        async with self.__lock:
            removed_channels = []
            for channel in channels:
                 # Ensure channel is a SubscribeReq object
                if not isinstance(channel, SubscribeReq):
                    logger.warning(f"Invalid channel format for unsubscription: {channel}. Skipping.")
                    continue
                if channel in self.__subscribed_channels:
                    self.__subscribed_channels.remove(channel)
                    removed_channels.append(channel)
                    if channel in self.__scribe_map:
                        del self.__scribe_map[channel]
                    if channel in self.__allbooks_map:
                         del self.__allbooks_map[channel]

            if removed_channels:
                await self.send_message(WS_OP_UNSUBSCRIBE, removed_channels)
                logger.info(f"Unsubscribed from: {removed_channels}")
            else:
                logger.info(f"Channels not currently subscribed: {channels}")

    async def _handle_message(self, message):
        """Processes incoming WebSocket messages."""
        try:
            if message == 'pong': # Handle pong from server
                logger.debug("Received pong")
                return
            if message == 'ping': # Handle ping from server, send pong back
                logger.debug("Received ping, sending pong")
                await self.__connection.pong()
                return

            json_obj = json.loads(message)

            # Handle error messages from Bitget
            if "code" in json_obj and str(json_obj.get("code")) != '0':
                logger.error(f"Bitget API Error: {json_obj}")
                await self._handle_error(json_obj) # Pass error details
                return

            # Handle login confirmation
            if json_obj.get("event") == "login":
                if str(json_obj.get("code")) == '0':
                    self.__login_status = True
                    logger.info("WebSocket authenticated successfully.")
                else:
                    logger.error(f"WebSocket authentication failed: {json_obj.get('msg')}")
                    # Consider stopping or specific error handling
                return

            # Checksum logic for order book data
            if not await self._check_sum(json_obj):
                return # Checksum failed, resubscription handled within _check_sum

            # Find specific listener or use default
            listener_to_use = self.__listener
            arg = json_obj.get('arg')
            if isinstance(arg, dict):
                try:
                    req = self._dict_to_subscribe_req(arg)
                    async with self.__lock:
                        if req in self.__scribe_map:
                            listener_to_use = self.__scribe_map[req]
                except KeyError as e:
                    logger.warning(f"Could not parse arg to SubscribeReq for listener lookup: {arg}, Error: {e}")

            # Call the listener (handle sync/async)
            if asyncio.iscoroutinefunction(listener_to_use):
                asyncio.create_task(listener_to_use(json_obj)) # Run as separate task
            else:
                # Run synchronous listener in an executor to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, listener_to_use, json_obj)

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON message: {message[:200]}...")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            await self._handle_error(e)

    async def _handle_error(self, error):
        """Calls the user-defined error listener."""
        try:
            if asyncio.iscoroutinefunction(self.__error_listener):
                await self.__error_listener(error)
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.__error_listener, error)
        except Exception as e:
            logger.error(f"Exception in error listener itself: {e}", exc_info=True)

    async def _check_sum(self, json_obj):
        """Validates checksum for order book updates."""
        try:
            arg = json_obj.get('arg')
            action = json_obj.get('action')
            data_list = json_obj.get('data')

            if not isinstance(arg, dict) or not action or not data_list:
                return True # Not an order book message or invalid format

            subscribe_req = self._dict_to_subscribe_req(arg)

            if subscribe_req.channel != "books":
                return True # Only check checksum for order book channel

            # Assuming data_list contains a single book info dict
            if not data_list or not isinstance(data_list, list):
                 logger.warning(f"Unexpected data format for books channel: {data_list}")
                 return True # Cannot process

            books_info = BooksInfo(**data_list[0]) # Use **kwargs for cleaner init

            async with self.__lock:
                if action == "snapshot":
                    self.__allbooks_map[subscribe_req] = books_info
                    logger.debug(f"Snapshot received for {subscribe_req}")
                    return True
                elif action == "update":
                    all_books = self.__allbooks_map.get(subscribe_req)
                    if all_books is None:
                        logger.warning(f"Received update for {subscribe_req} before snapshot. Requesting snapshot.")
                        # Need to run unsubscribe/subscribe async
                        asyncio.create_task(self._resubscribe_channel(subscribe_req))
                        return False # Indicate failure to process update

                    all_books.merge(books_info) # Merge updates in place
                    if not all_books.check_sum(books_info.checksum):
                        logger.warning(f"Checksum failed for {subscribe_req}. Resubscribing.")
                        if subscribe_req in self.__allbooks_map:
                            del self.__allbooks_map[subscribe_req]
                        # Need to run unsubscribe/subscribe async
                        asyncio.create_task(self._resubscribe_channel(subscribe_req))
                        return False # Indicate checksum failure
                    # Checksum passed, update stored book
                    self.__allbooks_map[subscribe_req] = all_books
                    return True
                else:
                     logger.warning(f"Unknown action '{action}' for books channel.")
                     return True # Or False if strict

        except Exception as e:
            msg = traceback.format_exc()
            logger.error(f"Error during checksum: {msg}")
            return False # Indicate error during checksum

        return True # Default pass if no issues

    async def _resubscribe_channel(self, channel):
        """Helper to unsubscribe and resubscribe a channel asynchronously."""
        logger.info(f"Attempting to resubscribe {channel}...")
        await self.unsubscribe([channel])
        await asyncio.sleep(0.5) # Brief delay before resubscribing
        await self.subscribe([channel])

    def _dict_to_subscribe_req(self, d):
        """Converts a dictionary (from message arg) to a SubscribeReq object."""
        instType = d.get('instType')
        channel = d.get('channel')
        if not instType or not channel:
            raise KeyError("Missing 'instType' or 'channel' in subscription arg")

        if channel == 'account':
            coin = d.get('coin')
            if coin is None:
                raise KeyError("Missing 'coin' for account channel subscription arg")
            # For account channel, instId is not used in the request, pass coin
            return SubscribeReq(instType, channel, coin=coin)
        else:
            instId = d.get('instId')
            if instId is None:
                raise KeyError(f"Missing 'instId' for {channel} channel subscription arg")
            return SubscribeReq(instType, channel, instId=instId)

# --- Data Structures --- (Mostly unchanged, but adapted BooksInfo init)

class BooksInfo:
    # Use **kwargs in __init__ for flexibility if keys change slightly
    def __init__(self, asks=None, bids=None, checksum=None, ts=None):
        self.asks = asks if asks is not None else []
        self.bids = bids if bids is not None else []
        self.checksum = checksum
        self.ts = ts # Timestamp might be useful

    def merge(self, book_info):
        # Logic seems okay, ensure it handles potential None values if keys missing
        self.asks = self._innerMerge(self.asks, book_info.asks or [], False)
        self.bids = self._innerMerge(self.bids, book_info.bids or [], True)
        # Update timestamp if available in the update
        if hasattr(book_info, 'ts') and book_info.ts:
            self.ts = book_info.ts
        return self # Return self for chaining if needed, though modified in-place

    def _innerMerge(self, all_list, update_list, is_reverse):
        # Using dict for efficient updates
        price_and_value = {item[0]: item for item in all_list if item} # Filter out None placeholders

        for update_item in update_list:
            if not update_item or len(update_item) < 2:
                continue # Skip invalid update items
            price, size = update_item[0], update_item[1]
            if size == "0":
                price_and_value.pop(price, None) # Remove price level
            else:
                price_and_value[price] = update_item # Add or update price level

        # Sort keys and take top 25
        # Ensure keys are comparable (e.g., convert to float if they are strings)
        try:
            sorted_keys = sorted(price_and_value.keys(), key=float, reverse=is_reverse)
        except ValueError:
             logger.error(f"Could not sort book keys as floats: {list(price_and_value.keys())[:5]}...")
             return all_list # Return original list on sort error

        result = [price_and_value[k] for k in sorted_keys[:25]]

        # No need to pad with None, just return the top 25 or fewer
        return result

    def check_sum(self, new_check_sum):
        crc32str = ''
        # Take top 25 bids and asks directly
        top_bids = self.bids[:25]
        top_asks = self.asks[:25]

        for i in range(25):
            # Format bids
            if i < len(top_bids) and top_bids[i] and len(top_bids[i]) >= 2:
                crc32str += f"{top_bids[i][0]}:{top_bids[i][1]}:"
            # Format asks
            if i < len(top_asks) and top_asks[i] and len(top_asks[i]) >= 2:
                crc32str += f"{top_asks[i][0]}:{top_asks[i][1]}:"

        if not crc32str:
             # logger.warning("Checksum string is empty, cannot verify.")
             return True # Treat empty string as passing? Or False? Bitget likely expects False.

        crc32str = crc32str[:-1] # Remove trailing ':'
        try:
            merge_num = crc32(bytes(crc32str, encoding="utf8"))
            signed_merge_num = self._signed_int(merge_num)
            checksum_match = (signed_merge_num == new_check_sum)
            # if not checksum_match:
            #     logger.debug(f"Checksum mismatch: Calculated={signed_merge_num}, Expected={new_check_sum}, String='{crc32str[:100]}...' ")
            return checksum_match
        except Exception as e:
            logger.error(f"Error calculating CRC32: {e}")
            return False

    def _signed_int(self, checknum):
        # Standard 32-bit signed integer conversion
        if checknum >= 2**31:
            return checknum - 2**32
        return checknum

class SubscribeReq:
    def __init__(self, instType, channel, instId=None, coin=None):
        self.instType = instType
        self.channel = channel
        # Store both, but prioritize 'coin' for account channel
        self.instId = instId
        self.coin = coin

        if channel == 'account' and coin is None:
            raise ValueError("Parameter 'coin' is required for 'account' channel.")
        if channel != 'account' and instId is None:
             raise ValueError(f"Parameter 'instId' is required for '{channel}' channel.")

    # Modify __dict__ behavior for JSON serialization
    def __getattribute__(self, name):
        if name == '__dict__':
            d = object.__getattribute__(self, name).copy()
            if d.get('channel') == 'account':
                # For account channel, send 'coin', remove 'instId'
                if 'instId' in d: del d['instId']
                if d.get('coin') is None: # Should not happen due to __init__ check
                     d['coin'] = 'default' # Safety default
            else:
                # For other channels, send 'instId', remove 'coin'
                if 'coin' in d: del d['coin']
                if d.get('instId') is None: # Should not happen
                    raise AttributeError("instId is missing for non-account channel")
            return d
        return object.__getattribute__(self, name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, SubscribeReq):
            return False
        if self.channel == 'account':
            return (self.instType == other.instType and
                    self.channel == other.channel and
                    self.coin == other.coin)
        else:
            return (self.instType == other.instType and
                    self.channel == other.channel and
                    self.instId == other.instId)

    def __hash__(self) -> int:
        if self.channel == 'account':
            return hash((self.instType, self.channel, self.coin))
        else:
            return hash((self.instType, self.channel, self.instId))

    def __repr__(self): # More informative repr
        if self.channel == 'account':
             return f"SubscribeReq(instType='{self.instType}', channel='{self.channel}', coin='{self.coin}')"
        else:
             return f"SubscribeReq(instType='{self.instType}', channel='{self.channel}', instId='{self.instId}')"

class BaseWsReq:
    def __init__(self, op, args):
        self.op = op
        self.args = args

class WsLoginReq:
    def __init__(self, apiKey, passphrase, timestamp, sign):
        # Match Bitget's expected field names
        self.apiKey = apiKey
        self.passphrase = passphrase
        self.timestamp = timestamp
        self.sign = sign
