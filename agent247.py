import asyncio
import os # Re-import for path printing
import logging # Re-import for early logging

# --- Early Path Logging ---
# Get logger early for path confirmation
early_logger = logging.getLogger('init_path_check')
early_logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
early_logger.addHandler(stream_handler)

try:
    # Attempt to get the absolute path of the script
    # __file__ might not be defined in all execution contexts (e.g., interactive interpreter)
    if '__file__' in globals():
        script_path = os.path.abspath(__file__)
        early_logger.info(f"Executing script: {script_path}")
    else:
        early_logger.info("Executing script (path not determined as __file__ is not defined).")
except Exception as e:
    early_logger.warning(f"Could not determine script path: {e}")
# --- End Early Path Logging ---
import json
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP

# --- Custom JSON Encoder for Decimal --- 
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return str(o)  # Convert Decimal to string
        return super(DecimalEncoder, self).default(o)
# --- End Custom JSON Encoder ---

INSTRUMENT_PRECISIONS = {}
MARK_PRICE_BUFFER_PERCENT = Decimal('0.0005') # 0.05% buffer
MIN_PRICE_ADJUSTMENT_FACTOR = Decimal('0.0001') # Minimum factor to ensure price is different from mark_price

import os
import logging
import time
import datetime
import feedparser
import httpx
import collections # For deque
import re # For regex operations in LLM response parsing
import pandas as pd # Added for data manipulation
import pandas_ta as ta # Added for technical indicators
from dotenv import load_dotenv
# Configure logging before any other imports
logging.basicConfig(
    level=logging.INFO, # Changed from DEBUG to INFO to prevent logging sensitive data like API keys
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from bitget import consts as c # Import consts for API URL

# Import Bitget SDK components
try:
    from bitget.bitget_ws_client import BitgetWsClient, SubscribeReq
    from bitget.exceptions import BitgetAPIException
    from bitget.bitget_api import BitgetApi
except ImportError as e:
    # If running as main script, adjust path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from bitget.bitget_ws_client import BitgetWsClientAsync, SubscribeReq
    from bitget.exceptions import BitgetAPIException
    from bitget.bitget_api import BitgetApi

# --- Configuration ---
load_dotenv()

# Logging Setup (Reconfiguring to ensure it takes effect)
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()

logging.basicConfig(
    level=logging.INFO, # Changed from DEBUG to INFO to prevent logging sensitive data like API keys
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) # Re-get logger after reconfig

# API Keys and Credentials
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY","")
BITGET_API_KEY = os.getenv("BITGET_API_KEY","")
BITGET_SECRET_KEY = os.getenv("BITGET_SECRET_KEY","")
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE","")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1:free")

# Global flag to signal trading cycle restart
RESTART_TRADING_CYCLE_FLAG = asyncio.Event()

# News Feeds
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://cointelegraph.com/rss/tag/bitcoin",
    "https://cointelegraph.com/rss/category/analysis",
    "https://cointelegraph.com/rss/category/markets"
]

# Bitget Endpoints
BITGET_WSS_PRIVATE_URL = "wss://ws.bitget.com/v2/ws/private"
BITGET_WSS_PUBLIC_URL = "wss://ws.bitget.com/v2/ws/public" # Public endpoint
BITGET_REST_PLACE_ORDER_ENDPOINT = '/api/v2/mix/order/place-order'
BITGET_REST_PLACE_POS_TPSL_ENDPOINT = '/api/v2/mix/order/place-pos-tpsl' # Endpoint for position TP/SL
BITGET_REST_POSITIONS_ENDPOINT = "/api/v2/mix/position/all-position" # For fetching all positions
BITGET_REST_ACCOUNT_ENDPOINT = '/api/v2/mix/account/account' # Endpoint to get mix account details
BITGET_REST_FILL_HISTORY_ENDPOINT = '/api/v2/mix/order/fill-history' # Endpoint for fill history

# Trading Parameters
TARGET_INSTRUMENT = "SBTCSUSDT"  # Demo BTC/USDT futures
# Define Product Type for REST API (V2 uses 'productType' for mix endpoints)
# Common values: 'USDT-FUTURES', 'COIN-FUTURES', 'SUSDT-FUTURES' (for demo)
PRODUCT_TYPE_V2 = "SUSDT-FUTURES" # Use 'umcbl' for USDT-margined futures in V2 API
INST_TYPE_V2 = PRODUCT_TYPE_V2 # Keep for WS compatibility if needed elsewhere

CANDLE_CHANNEL = "candle1H"      # Candle interval (e.g., 1 hour)
MAX_CANDLES = 200                 # Max number of candles to store
# TRADE_SIZE = 0.02                # Fixed trade size (Replaced by dynamic sizing)
STOP_LOSS_PERCENT = 0.02         # 2.0% stop loss (new range 0.1-2.0%)
TAKE_PROFIT_PERCENT = 0.02       # 2.0% take profit (new range 0.1-2.0%)
RISK_PER_TRADE_PERCENT = 0.01    # Risk 1% of equity per trade
MAX_POSITION_SIZE_USD = 50000     # Maximum position size in USD
MIN_LEVERAGE = 1                 # Minimum leverage
MAX_LEVERAGE = 125                # Maximum leverage
DEFAULT_LEVERAGE = 5             # Default leverage
MIN_TRADE_SIZE = 0.001           # Minimum allowed trade size (adjust based on exchange)
TRADING_FEE_RATE = 0.0006        # Trading fee rate (e.g., 0.06% = 0.0006). Adjust based on your exchange.
TRADE_HISTORY_FILE = "trade_history.json" # File to store trade history

# Global storage for candle data
# Using deque for efficient fixed-size storage
candle_data_store = collections.deque(maxlen=MAX_CANDLES)

# --- Helper functions for trade history ---
async def fetch_historical_positions(rest_client, symbol=None, start_time=None, end_time=None, limit=20):
    """Fetch historical position data from Bitget API."""
    params = {
        'productType': PRODUCT_TYPE_V2,
        'limit': str(limit)
    }
    if symbol:
        params['symbol'] = symbol
    if start_time:
        params['startTime'] = str(start_time)
    if end_time:
        params['endTime'] = str(end_time)
    
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, rest_client.get, '/api/v2/mix/position/history-position', params)
        if result and result.get('code') == '00000' and 'data' in result:
            return result['data']['list']
        return []
    except Exception as e:
        logger.error(f"Error fetching historical positions: {e}")
        return []

async def fetch_and_process_fill_history(rest_client, order_id_to_fetch, client_order_id_to_update):
    """Fetch fill history for a given orderId and update the trade_history.json."""
    if not order_id_to_fetch:
        logger.error(f"[FillHistory] No exchange orderId provided for client_order_id: {client_order_id_to_update}. Cannot fetch fill history.")
        update_trade_history(client_order_id_to_update, {'status': 'Closed_PnL_Fetch_Failed_No_OrderID'})
        return

    logger.info(f"[FillHistory] Fetching fill history for orderId: {order_id_to_fetch} (client_order_id: {client_order_id_to_update})")
    try:
        params = {
            'productType': PRODUCT_TYPE_V2,
            'orderId': order_id_to_fetch,
            # 'limit': '100' # Default is 100, usually one order has fewer fills
        }
        result = await rest_client.get(BITGET_REST_FILL_HISTORY_ENDPOINT, params)
        logger.debug(f"[FillHistory] API Response for {order_id_to_fetch}: {result}")

        if result and result.get('code') == '00000' and 'data' in result and 'fillList' in result['data']:
            fill_list = result['data']['fillList']
            if not fill_list:
                logger.warning(f"[FillHistory] No fills found for orderId: {order_id_to_fetch}.")
                update_trade_history(client_order_id_to_update, {'status': 'Closed_No_Fills_Found'})
                return

            total_profit = 0.0
            total_fees_paid = 0.0
            last_exit_price = None
            
            for fill in fill_list:
                try:
                    profit_str = fill.get('profit')
                    if profit_str is not None:
                        total_profit += float(profit_str)
                    
                    last_exit_price = float(fill.get('price'))

                    fee_details = fill.get('feeDetail', [])
                    for fee_item in fee_details:
                        fee_paid_str = fee_item.get('totalFee')
                        if fee_paid_str is not None:
                            total_fees_paid += float(fee_paid_str)

                except (ValueError, TypeError) as e:
                    logger.error(f"[FillHistory] Error parsing fill data for order {order_id_to_fetch}: {fill}. Error: {e}")
                    continue

            update_data = {
                'status': 'Closed_PnL_Fetched',
                'pnl': round(total_profit, 8),
                'exit_price': last_exit_price,
                'fees_paid': round(total_fees_paid, 8)
            }
            logger.info(f"[FillHistory] Updating trade {client_order_id_to_update} (orderId: {order_id_to_fetch}) with PnL: {update_data['pnl']}, Exit: {update_data['exit_price']}, Fees: {update_data['fees_paid']}")
            update_trade_history(client_order_id_to_update, update_data)

        elif result and result.get('code') != '00000':
            logger.error(f"[FillHistory] API error fetching fills for {order_id_to_fetch}: {result.get('msg')} (Code: {result.get('code')})")
            update_trade_history(client_order_id_to_update, {'status': f"Closed_PnL_Fetch_API_Error_{result.get('code')}"})
        else:
            logger.error(f"[FillHistory] Unexpected response or no data for {order_id_to_fetch}: {result}")
            update_trade_history(client_order_id_to_update, {'status': 'Closed_PnL_Fetch_Error_No_Data'})

    except BitgetAPIException as e:
        logger.error(f"[FillHistory] BitgetAPIException fetching fills for {order_id_to_fetch}: {e}")
        update_trade_history(client_order_id_to_update, {'status': f"Closed_PnL_Fetch_API_Exc_{e.code}"})
    except Exception as e:
        logger.error(f"[FillHistory] Unexpected error fetching fills for {order_id_to_fetch}: {e}", exc_info=True)
        update_trade_history(client_order_id_to_update, {'status': 'Closed_PnL_Fetch_Unexpected_Error'})

# --- WebSocket Handlers ---

async def handle_private_message(message, ws_client): # Added ws_client
    """Callback for PRIVATE WebSocket messages (account, positions, orders)"""
    try:
        if isinstance(message, str):
            data = json.loads(message)
        elif isinstance(message, dict):
            data = message
        else:
            logger.warning(f"[PrivateWS] Unexpected message type: {type(message)}")
            return

        event = data.get('event')
        action = data.get('action')
        arg = data.get('arg', {})
        channel = arg.get('channel') if isinstance(arg, dict) else None

        if event == 'login':
            if data.get('code') == '0':
                logger.info("[PrivateWS] Authenticated successfully")
            else:
                logger.error(f"[PrivateWS] Authentication failed: {data.get('msg')}")
        elif event == 'subscribe':
             logger.info(f"[PrivateWS] Subscribed to: {arg}")
        elif action in ['snapshot', 'update']:
            data_list = data.get('data', [])
            if channel == 'account':
                for item in data_list:
                    # Use PRODUCT_TYPE_V2 for consistency if marginCoin matches
                    if item.get('marginCoin') == PRODUCT_TYPE_V2.split('-')[0]: # e.g., 'SUSDT'
                        logger.info(f"[PrivateWS] Account Update: Equity={item.get('equity')}")
            elif channel == 'positions':
                for pos in data_list:
                     instId = pos.get('instId')
                     total_size_str = pos.get('total', '0') # Position size
                     avg_entry_price_str = pos.get('openPriceAvg', '0') # Average entry price
                     hold_side = pos.get('holdSide') # 'long' or 'short'
                     upl_str = pos.get('unrealizedPL', '0') # Unrealized PnL
                     liq_price_str = pos.get('liqPx', '') # Liquidation price

                     logger.info(f"[PrivateWS] Position Update: Inst={instId}, Side={hold_side}, Total={total_size_str}, AvgEntry={avg_entry_price_str}, UPL={upl_str}, LiqPx={liq_price_str}")

                     # --- Trade Closure Detection --- 
                     if instId == TARGET_INSTRUMENT:
                         try:
                             total_size = float(total_size_str)
                             if total_size == 0:
                                 logger.info(f"Detected closure for position: {instId}")
                                 RESTART_TRADING_CYCLE_FLAG.set()
                                 logger.info(f"RESTART_TRADING_CYCLE_FLAG set due to position closure for {instId}.")
                                 # Find the corresponding 'Filled' trade in history to update
                                 # This assumes only one open trade per instrument at a time
                                 # More complex logic needed for multiple concurrent trades
                                 if os.path.exists(TRADE_HISTORY_FILE):
                                     with open(TRADE_HISTORY_FILE, 'r') as f:
                                         history = json.load(f)
                                     
                                     found_trade = None
                                     trade_index = -1
                                     # Find the latest 'Filled' trade for this instrument
                                     for i, trade in reversed(list(enumerate(history))):
                                         if trade.get('instrument') == instId and trade.get('status') == 'Filled':
                                             found_trade = trade
                                             trade_index = i
                                             break
                                     
                                     if found_trade:
                                         client_order_id_of_closed_trade = found_trade.get('client_order_id')
                                         exchange_order_id_of_closed_trade = found_trade.get('order_id') # This is the actual exchange ID

                                         if exchange_order_id_of_closed_trade:
                                             logger.info(f"Position closed for {instId}. Attempting to fetch fill history for orderId: {exchange_order_id_of_closed_trade} (client_order_id: {client_order_id_of_closed_trade}).")
                                             # Schedule both fill history and historical position fetches
                                             asyncio.create_task(fetch_and_process_fill_history(ws_client.rest_client, exchange_order_id_of_closed_trade, client_order_id_of_closed_trade))
                                             
                                             # Also fetch historical positions for the instrument
                                             now = int(time.time() * 1000)
                                             three_months_ago = now - (90 * 24 * 60 * 60 * 1000)
                                             historical_positions = await fetch_historical_positions(
                                                 ws_client.rest_client,
                                                 symbol=instId,
                                                 start_time=three_months_ago,
                                                 end_time=now
                                             )
                                             
                                             if historical_positions:
                                                 latest_position = max(historical_positions, key=lambda x: int(x.get('uTime', 0)))
                                                 update_trade_history(client_order_id_of_closed_trade, {
                                                     'historical_pnl': float(latest_position.get('pnl', 0)),
                                                     'historical_net_profit': float(latest_position.get('netProfit', 0)),
                                                     'historical_fees': float(latest_position.get('openFee', 0)) + float(latest_position.get('closeFee', 0))
                                                 })
                                         else:
                                             logger.warning(f"Position closed for {instId} (client_order_id: {client_order_id_of_closed_trade}), but no exchange order_id found in history. Cannot fetch fill details.")
                                             update_trade_history(client_order_id_of_closed_trade, {'status': 'Closed_No_Exchange_OrderID'})
                                     else:
                                         logger.warning(f"Position closed for {instId}, but no corresponding 'Filled' trade found in history.")
                                 else:
                                     logger.warning(f"Position closed for {instId}, but trade history file not found.")

                         except ValueError:
                             logger.error(f"Could not parse position size ('{total_size_str}') or UPL ('{upl_str}') to float for {instId}.")
                         except Exception as e:
                             logger.error(f"Error processing position closure for {instId}: {e}", exc_info=True)
                     # --- End Trade Closure Detection ---

            elif channel == 'orders':
                 for order in data_list:
                    try:
                        order_id = order.get('orderId')
                        client_order_id = order.get('clOrdId') # Use client order ID if available for matching
                        status = order.get('status')
                        filled_size_str = order.get('accBaseVolume') # Filled size
                        avg_fill_price_str = order.get('avgPx') # Average fill price

                        logger.info(f"[PrivateWS] Order Update: ID={order_id}, ClientID={client_order_id}, Status={status}, FilledSz={filled_size_str}, AvgPx={avg_fill_price_str}")

                        # Update trade history when filled
                        if status == 'filled':
                        
                                avg_fill_price = float(avg_fill_price_str) if avg_fill_price_str else None
                                filled_size = float(filled_size_str) if filled_size_str else None
                                update_data = {
                                    'status': 'Filled',
                                    'actual_entry_price': avg_fill_price,
                                    'filled_size': filled_size,
                                    'order_id': order_id # Store the actual exchange order ID
                                }
                                # Use client_order_id for lookup as it's generated by us
                                update_trade_history(client_order_id, update_data)
                        elif status in ['cancelled', 'rejected']:
                             # Optionally update history for failed/cancelled orders
                             update_trade_history(client_order_id, {'status': status.capitalize()})
                    except ValueError:
                         logger.error(f"Could not parse fill price ('{avg_fill_price_str}') or size ('{filled_size_str}') to float for order {order_id}.")
                    except Exception as e:
                        logger.error(f"Error processing order update for {order_id or client_order_id}: {e}", exc_info=True)

        elif event == 'error':
             logger.error(f"[PrivateWS] Error Event: {data}")
        # Ignore pong messages for cleaner logs if needed
        # elif event == 'pong':
        #     pass
        # else:
        #     logger.debug(f"[PrivateWS] Message: {data}")

    except Exception as e:
        logger.error(f"[PrivateWS] Error handling message: {e}", exc_info=True)

async def handle_public_message(message):
    """Callback for PUBLIC WebSocket messages (candles)"""
    global candle_data_store
    try:
        if isinstance(message, str):
            data = json.loads(message)
        elif isinstance(message, dict):
            data = message
        else:
            logger.warning(f"[PublicWS] Unexpected message type: {type(message)}")
            return

        action = data.get('action')
        arg = data.get('arg', {})
        channel = arg.get('channel') if isinstance(arg, dict) else None
        instId = arg.get('instId') if isinstance(arg, dict) else None

        # Ensure instType from message matches our expected type for candles
        msg_inst_type = arg.get('instType') if isinstance(arg, dict) else None

        if channel == CANDLE_CHANNEL and instId == TARGET_INSTRUMENT and msg_inst_type == INST_TYPE_V2:
            if action in ['snapshot', 'update']:
                candle_list = data.get('data', [])
                # Data format: [timestamp, open, high, low, close, base_vol, quote_vol]
                for candle in candle_list:
                    if len(candle) >= 7: # Basic validation
                        # Store relevant parts (e.g., timestamp, O, H, L, C, V)
                        formatted_candle = {
                            "ts": int(candle[0]),
                            "o": float(candle[1]),
                            "h": float(candle[2]),
                            "l": float(candle[3]),
                            "c": float(candle[4]),
                            "v": float(candle[5]) # base volume
                        }
                        # Ensure uniqueness based on timestamp before appending
                        if formatted_candle['ts'] not in {c['ts'] for c in candle_data_store}:
                            candle_data_store.append(formatted_candle)
                        # Optional: Log only the latest candle update
                        # logger.debug(f"[PublicWS] Candle Update: {formatted_candle}")
                # Log after processing a batch
                logger.info(f"[PublicWS] Processed {len(candle_list)} candle updates for {TARGET_INSTRUMENT}. Total stored: {len(candle_data_store)}")

        elif data.get('event') == 'subscribe':
            logger.info(f"[PublicWS] Subscribed to: {arg}")
        elif data.get('event') == 'error':
            logger.error(f"[PublicWS] Error Event: {data}")
        # Ignore pong messages
        # elif data.get('event') == 'pong':
        #      pass
        # else:
        #      logger.debug(f"[PublicWS] Message: {data}")

    except Exception as e:
        logger.error(f"[PublicWS] Error handling message: {e}", exc_info=True)


async def handle_ws_error(client_name, err):
    """Generic callback for WebSocket connection/library errors"""
    logger.error(f"[{client_name}] Connection/Library Error: {err}", exc_info=True)

# --- Technical Indicator Calculation ---

def calculate_indicators(candle_data):
    """Calculate technical indicators (SMA, RSI, MACD, Bollinger Bands) from candle data."""
    # Increased minimum data points for Bollinger Bands (default 20) and MACD (needs ~26+ periods)
    if len(candle_data) < 26:
        logger.warning(f"Not enough candle data ({len(candle_data)}) to calculate all indicators.")
        return None

    try:
        # Convert deque of dicts to DataFrame
        df = pd.DataFrame(list(candle_data))

        # Ensure correct column names and types for pandas_ta
        df.rename(columns={'ts': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Drop duplicate timestamps before setting index to prevent 'cannot reindex on an axis with duplicate labels'
        df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        df.set_index('timestamp', inplace=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # Calculate indicators using pandas_ta
        # SMA (Simple Moving Average) - 20 period
        df.ta.sma(length=20, append=True)
        # RSI (Relative Strength Index) - 14 period
        df.ta.rsi(length=14, append=True)
        # MACD (Moving Average Convergence Divergence) - Default settings (12, 26, 9)
        df.ta.macd(append=True)
        # Bollinger Bands - Default settings (20 periods, 2 std deviations)
        df.ta.bbands(append=True)

        # Get the latest indicator values
        # Select relevant columns (adjust names based on pandas_ta output)
        indicator_cols = [
            'SMA_20', 'RSI_14',
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', # MACD line, histogram, signal line
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0' # Lower, Middle, Upper Bands
        ]
        # Filter out columns that might not exist if data is too short (though check prevents this)
        existing_cols = [col for col in indicator_cols if col in df.columns]
        latest_indicators = df.iloc[-1][existing_cols].to_dict()

        # Round for cleaner output (optional)
        for key in latest_indicators:
            if isinstance(latest_indicators[key], (int, float)):
                 # Handle potential NaN values before rounding
                if pd.notna(latest_indicators[key]):
                    latest_indicators[key] = round(latest_indicators[key], 2)
                else:
                    latest_indicators[key] = None # Represent NaN as None

        logger.info(f"Calculated Indicators: {latest_indicators}")
        return latest_indicators

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True)
        return None

# --- Core Logic ---

async def fetch_news():
    """Fetch news from RSS feeds"""
    news_items = []
    fallback_feeds = [
        "https://cointelegraph.com/rss",
        "https://cointelegraph.com/rss/tag/bitcoin",
        "https://cointelegraph.com/rss/category/analysis",
        "https://cointelegraph.com/rss/category/markets"
    ]
    logger.info("Fetching news from RSS feeds...")
    all_feeds_successful = True
    successful_feeds_count = 0
    async with httpx.AsyncClient(timeout=20.0) as client:
        for url in RSS_FEEDS[:4]:
            feed_successful_this_attempt = False
            for attempt in range(3):  # Retry up to 3 times
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    feed = feedparser.parse(response.text)
                    parsed_items = [{ "title": entry.title, "link": entry.link, "published": entry.get("published", "") } for entry in feed.entries[:5]]
                    news_items.extend(parsed_items)
                    if parsed_items: # Consider successful if items were parsed
                        logger.info(f"Successfully fetched {len(parsed_items)} items from {url}")
                        feed_successful_this_attempt = True
                    else:
                        logger.warning(f"Fetched 0 items from {url} (attempt {attempt + 1})")
                    await asyncio.sleep(0.2 * (attempt + 1))  # Exponential backoff
                    break  # Success for this feed, exit retry loop
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error fetching {url} (attempt {attempt + 1}): Status {e.response.status_code}")
                    if attempt == 2:  # Last attempt failed
                        all_feeds_successful = False
                        logger.warning(f"Failed to fetch {url} after 3 attempts due to HTTP error.")
                        # Try fallback only if it's a critical error type, or always if preferred
                        # RSS_FEEDS.extend(fallback_feeds) # Decided against auto-extending fallback here to avoid loop complexity
                except httpx.RequestError as e:
                    logger.error(f"Request error fetching {url} (attempt {attempt + 1}): {e}")
                    if attempt == 2:
                        all_feeds_successful = False
                        logger.warning(f"Failed to fetch {url} after 3 attempts due to request error.")
                except Exception as e:
                    logger.error(f"Error parsing feed {url} (attempt {attempt + 1}): {e}")
                    if attempt == 2:
                        all_feeds_successful = False
                        logger.warning(f"Failed to parse {url} after 3 attempts.")
            if feed_successful_this_attempt:
                successful_feeds_count += 1
            else:
                all_feeds_successful = False # Mark as not all successful if any feed failed all attempts

    if all_feeds_successful and successful_feeds_count == len(RSS_FEEDS[:4]):
        logger.info(f"All {successful_feeds_count} RSS feeds fetched successfully. Total items: {len(news_items)}.")
    elif successful_feeds_count > 0:
        logger.warning(f"Successfully fetched {successful_feeds_count}/{len(RSS_FEEDS[:4])} RSS feeds. Total items: {len(news_items)}. Some feeds may have failed.")
    else:
        logger.error(f"Failed to fetch any news items from RSS feeds. Total items: {len(news_items)}.")
    return news_items

# This function definition is now part of the previous block due to the change in analyze_trade_history_and_learn
# The following is the original start of get_llm_analysis, ensure it's correctly placed after the modified analyze_trade_history_and_learn
async def get_llm_analysis(news_data, historical_candles, indicator_data, account_equity, learning_insights=None):
    """Get trade signal from LLM analysis with dynamic risk management based on account equity."""
    # Validate learning_insights
    if learning_insights is None:
        learning_insights = {}
    elif not isinstance(learning_insights, dict):
        logger.error(f"Invalid learning_insights type: {type(learning_insights)}")
        return None
        
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not configured")
        return None
        
    # Validate account equity data type
    equity_value = None
    if isinstance(account_equity, (int, float)):
        equity_value = float(account_equity)
    elif isinstance(account_equity, dict):
        if 'usdtEquity' in account_equity:
            try:
                equity_value = float(account_equity['usdtEquity'])
            except (ValueError, TypeError):
                logger.error(f"Could not parse usdtEquity: {account_equity['usdtEquity']}")
        else:
            logger.error(f"account_equity dict missing usdtEquity: {account_equity}")
    else:
        logger.error(f"Invalid account_equity type: {type(account_equity)}")
        return None

    # Define risk management parameters string
    risk_management_params_info = f"Current Risk Management Parameters: Max Position Size: {MAX_POSITION_SIZE_USD} USD, Min Leverage: {MIN_LEVERAGE}, Max Leverage: {MAX_LEVERAGE}, Default Leverage: {DEFAULT_LEVERAGE}, Take Profit: {TAKE_PROFIT_PERCENT*100}%, Stop Loss: {STOP_LOSS_PERCENT*100}%"

    # Format news titles
    news_titles = []
    if isinstance(news_data, list):
        news_titles = [item['title'] for item in news_data if isinstance(item, dict) and 'title' in item]
    elif news_data: # Log only if it's not None/empty
        logger.warning(f"news_data is not a list: {type(news_data)}")

    # Format candles
    formatted_candles = []
    if isinstance(historical_candles, list):
        for candle in historical_candles:
            if isinstance(candle, dict):
                if all(k in candle for k in ['T', 'O', 'H', 'L', 'C', 'V']):
                    formatted_candles.append({
                        'T': candle['T'], 'O': candle['O'], 'H': candle['H'], 
                        'L': candle['L'], 'C': candle['C'], 'V': candle['V']
                    })
                elif all(k in candle for k in ['ts', 'o', 'h', 'l', 'c', 'v']):
                    formatted_candles.append({
                        'T': candle['ts'], 'O': candle['o'], 'H': candle['h'], 
                        'L': candle['l'], 'C': candle['c'], 'V': candle['v']
                    })
                else:
                    logger.warning(f"Skipping malformed candle data (unexpected dict keys): {candle}")
            elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                try:
                    formatted_candles.append({'T': candle[0], 'O': candle[1], 'H': candle[2], 'L': candle[3], 'C': candle[4], 'V': candle[5]})
                except IndexError:
                    logger.warning(f"Skipping malformed candle data (tuple/list too short): {candle}")
            else:
                logger.warning(f"Skipping malformed candle data: {candle}")
    elif historical_candles: # Log only if it's not None/empty
        logger.warning(f"historical_candles is not a list: {type(historical_candles)}")

    # Format indicators
    formatted_indicators = json.dumps(indicator_data, indent=2) if indicator_data else "No indicator data available."

    # Add context about the symbol
    symbol_context = f"{TARGET_INSTRUMENT} is the demo trading symbol for Bitcoin (BTC) USDT-margined perpetual futures."

    # Format learning insights for prompt
    formatted_learning = json.dumps(learning_insights, indent=2) if learning_insights else "No specific learning insights available."

    # Define LLM prompt leverage constraints
    MIN_LEVERAGE_LLM_PROMPT = MIN_LEVERAGE
    MAX_LEVERAGE_LLM_PROMPT = MAX_LEVERAGE

    current_account_equity_info = f"Current Account Equity: ${equity_value:.2f} USD." if equity_value is not None else "Account equity not available."

    prompt = f"""
    You're a genius and professional crypto future trader. detailed thinking on Analyze the following market data for {TARGET_INSTRUMENT} ({symbol_context}):

    {current_account_equity_info}
    {risk_management_params_info}

    IMPORTANT INSTRUCTIONS:
    - Recommended margin should be 10% to 60% of account equity (minimum $5, maximum $50,000)
    - Recommended take-profit and stop-loss percentages should be percentages of the **entry_price**.
    1. Only use these exact signal values: "long", "short", or "hold"
    2. For margin calculation, use 10% to 60% of account equity (min $5, max $50000)
    3. For take-profit and stop-loss, recommend percentages based on your analysis (0.1% to 2.0% range for both SL and TP). **The recommended_tp_percent and recommended_sl_percent should be percentages of the anticipated entry price.**
    4. Provide detailed reasoning for your analysis

    Recent News Headlines (Consider potential sentiment impact):
    {json.dumps(news_titles, indent=2)}

    Recent {CANDLE_CHANNEL} Candles (Latest {len(formatted_candles)} periods. T=Timestamp, O=Open, H=High, L=Low, C=Close, V=Volume):
    {json.dumps(formatted_candles, indent=2)}

    Calculated Technical Indicators (Latest values):
    {formatted_indicators}

    Learning Insights from Past Trades (Explain how these historical insights are relevant to your current analysis and the prevailing market conditions):
    {formatted_learning}

    Respond with a JSON object containing:
    {{
      "signal": "long"|"short"|"hold"| (REQUIRED, exact values only),
      "confidence": 0-100 (REQUIRED),
      "recommended_margin": suggested position size in USDT (10%-60% of equity, min $5 max $50000),
      "recommended_leverage": suggested leverage (e.g., 5, 10, 20, min {MIN_LEVERAGE_LLM_PROMPT}, max {MAX_LEVERAGE_LLM_PROMPT}),
      "recommended_tp_percent": suggested take-profit percentage (e.g., 0.01 for 1%, range 0.001 to 0.02 for 0.1% to 2.0%),
      "recommended_sl_percent": suggested stop-loss percentage (e.g., 0.01 for 1%, range 0.001 to 0.02 for 0.1% to 2.0%),
      "reasoning": detailed explanation of analysis
    }}

    Example of a good response:
    ```json
    {{
        "signal": "long",
        "confidence": 75,
        "recommended_margin": 500.0,
        "recommended_leverage": 10,
        "recommended_tp_percent": 0.15,
        "recommended_sl_percent": 0.10,
        "reasoning": "The price is bouncing off a key support level, and RSI shows bullish divergence..."
    }}
    ```
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AITradingBot"
    }

    logger.info("Requesting LLM analysis with expanded indicators...")
    max_retries = 3
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
                result = response.json()

                # Check for error field in the JSON payload even if status is 200 OK
                if 'error' in result:
                    error_payload = result['error']
                    error_message = error_payload.get('message', 'Unknown error in payload')
                    error_code_in_payload = error_payload.get('code') # Can be string or int
                    logger.error(f"LLM API returned 200 OK but with error in payload (Attempt {attempt + 1}/{max_retries}): {error_message} (Code: {error_code_in_payload}). Full response: {result}")
                    
                    # Try to convert error_code_in_payload to int for comparison, if it's a string digit
                    numeric_error_code = None
                    if isinstance(error_code_in_payload, str) and error_code_in_payload.isdigit():
                        numeric_error_code = int(error_code_in_payload)
                    elif isinstance(error_code_in_payload, int):
                        numeric_error_code = error_code_in_payload

                    if numeric_error_code is not None and numeric_error_code >= 500 and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Retrying due to server-side error (Code: {numeric_error_code}) in JSON payload. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue  # Continue to the next attempt in the for loop
                    else:
                        logger.error(f"Non-retriable error in JSON payload or max retries reached for payload error. Full response: {result}")
                        return None # Not retrying or max retries hit for this type of error

                if not result.get('choices'):
                    logger.error("LLM response missing 'choices' (and no 'error' field at root). Full API response: %s", result)
                    return None 

                content = result['choices'][0]['message']['content']
                logger.debug(f"Raw LLM Response Content: {content}")
                content_cleaned = content.strip()
                # DO NOT return here, continue to parsing logic below

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during LLM request (Attempt {attempt + 1}/{max_retries}): {e.response.status_code} - {e.response.text}")
            if e.response.status_code >= 500 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                # For 4xx errors or last retry on 5xx, log and return None
                logger.error(f"Failed to get LLM analysis after {attempt + 1} attempts due to HTTP error.")
                return None
        except httpx.RequestError as e:
            logger.error(f"Request error during LLM request (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Failed to get LLM analysis after {attempt + 1} attempts due to request error.")
                return None
        except Exception as e:
            logger.error(f"Unexpected error during LLM request (Attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
            # For other unexpected errors, probably not worth retrying without understanding them
            return None

    logger.error(f"Failed to get LLM analysis after {max_retries} retries.")
    # If all retries failed and content_cleaned was never set (e.g. all attempts raised exceptions before getting response)
    # or if content_cleaned is empty, return None early.    
    if not 'content_cleaned' in locals() or not content_cleaned:
        logger.error("LLM content is empty or not defined after all retries, cannot parse.")
        return None

    parsed_analysis_dict = None # This will hold the successfully parsed dictionary
    response_text = content_cleaned # Use the cleaned content from LLM

    # Attempt 1: Direct parse
    try:
        parsed_analysis_dict = json.loads(response_text)
        logger.info("Successfully parsed LLM response directly.")
    except json.JSONDecodeError:
        logger.debug("Direct JSON parsing failed. Attempting extraction methods...")

        # Attempt 2: Extract from markdown code block ```json ... ```
        if not parsed_analysis_dict:
            match_md = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
            if match_md:
                extracted_str = match_md.group(1).strip()
                logger.info(f"Found JSON in markdown block. Extracted: {extracted_str[:200]}...")
                try:
                    parsed_analysis_dict = json.loads(extracted_str)
                    logger.info("Successfully parsed JSON from markdown block.")
                except json.JSONDecodeError as e_md:
                    logger.warning(f"Failed to parse JSON from markdown block: {e_md}. Content: {extracted_str}")
        
        # Attempt 3: Extract content between the first '{' and last '}'
        if not parsed_analysis_dict:
            start_brace = response_text.find('{')
            end_brace = response_text.rfind('}')
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                extracted_str = response_text[start_brace : end_brace+1]
                logger.info(f"Attempting to parse content between first '{{' and last '}}'. Extracted: {extracted_str[:200]}...")
                try:
                    parsed_analysis_dict = json.loads(extracted_str)
                    logger.info("Successfully parsed JSON from content between first '{' and last '}'.")
                except json.JSONDecodeError as e_brace:
                    logger.warning(f"Failed to parse JSON from content between braces: {e_brace}. Content: {extracted_str}")

        # Attempt 4: Handle <think>...</think> tags or <think> prefix
        if not parsed_analysis_dict and "<think>" in response_text:
            logger.info("'<think>' tag found in response. Attempting to extract JSON from its content.")
            content_to_search_json_in = response_text 
            match_think_block = re.search(r"<think>([\s\S]*?)</think>", response_text, re.DOTALL)
            if match_think_block:
                content_to_search_json_in = match_think_block.group(1).strip()
                logger.debug(f"Content within <think></think>: {content_to_search_json_in[:200]}...")
            elif response_text.strip().startswith("<think>"):
                # Handle case where it's just <think> {JSON_HERE}
                temp_content = response_text.strip()[len("<think>"):].strip()
                # Check if this content itself is the JSON or contains it
                if temp_content.startswith("{") and temp_content.endswith("}"):
                     content_to_search_json_in = temp_content
                else: # If not, it might be <think> some text {JSON} some text </think> (without closing tag in original response_text)
                    # This case is harder, rely on the brace finding within the original response_text if it's not a clean block
                    pass # Fall through to brace finding on content_to_search_json_in (which is still response_text)
                logger.debug(f"Content after <think> prefix: {content_to_search_json_in[:200]}...")
            
            # Try to find JSON within this extracted/identified content_to_search_json_in
            start_brace_think = content_to_search_json_in.find('{')
            end_brace_think = content_to_search_json_in.rfind('}')
            if start_brace_think != -1 and end_brace_think != -1 and end_brace_think > start_brace_think:
                extracted_str = content_to_search_json_in[start_brace_think : end_brace_think+1]
                logger.info(f"Attempting to parse JSON from <think> related content. Extracted: {extracted_str[:200]}...")
                try:
                    parsed_analysis_dict = json.loads(extracted_str)
                    logger.info("Successfully parsed JSON from <think> related content.")
                except json.JSONDecodeError as e_think:
                    logger.warning(f"Failed to parse JSON from <think> related content: {e_think}. Content: {extracted_str}")
    
    if not parsed_analysis_dict:
        logger.error(f"All attempts to parse LLM analysis JSON failed. Raw response: {response_text}")
        log_trade_event({
            "event": "llm_json_parse_failure_all_attempts",
            "raw_response": response_text,
            "error": "Could not parse JSON after multiple extraction attempts."
        })
        # Return None if all parsing attempts fail
    
    # After all parsing attempts, return the parsed_analysis_dict (which might be None)
    if parsed_analysis_dict:
        logger.info(f"Successfully parsed LLM analysis: {parsed_analysis_dict}")
    else:
        logger.error("Failed to parse LLM analysis into a dictionary after all attempts. Raw response was: " + response_text[:500] + "...")
    return parsed_analysis_dict

    # The following validation logic will now be in the main trading loop after calling get_llm_analysis
    # analysis = parsed_analysis_dict
    # Validate required fields and structure
    base_required_fields = ["signal", "confidence", "reasoning"]
    trade_related_fields = ["recommended_margin", "recommended_leverage", "recommended_tp_percent", "recommended_sl_percent"]

    for field in base_required_fields:
        if field not in analysis:
            logger.error(f"LLM response missing essential field: {field}. Response: {analysis}")
            return None

    # For trade signals, ensure all trade-related fields are present or have defaults
    if analysis.get("signal") in ["long", "short"]:
        for field in trade_related_fields:
            if field not in analysis:
                logger.warning(f"LLM response missing trade-related field: {field} for signal {analysis['signal']}. Will use defaults if possible. Response: {analysis}")
                # Set defaults for missing numeric fields to avoid downstream errors
                if field == 'recommended_margin': analysis[field] = None # Or a sensible default like min_trade_size_usdt
                if field == 'recommended_leverage': analysis[field] = DEFAULT_LEVERAGE # Define DEFAULT_LEVERAGE, e.g., 10
                if field == 'recommended_tp_percent': analysis[field] = TAKE_PROFIT_PERCENT # Use existing global default
                if field == 'recommended_sl_percent': analysis[field] = STOP_LOSS_PERCENT # Use existing global default
    
    # Type checking and conversion for numeric fields
        numeric_fields_with_defaults = {
            'confidence': (0, 100, 50), # min, max, default
            'recommended_margin': (0, float('inf'), None), # Assuming None means it will be calculated or error handled later
            'recommended_leverage': (MIN_LEVERAGE_LLM_PROMPT, MAX_LEVERAGE_LLM_PROMPT, DEFAULT_LEVERAGE), # Define these constants
            'recommended_tp_percent': (0.001, 2.0, TAKE_PROFIT_PERCENT), # 0.1% to 200%
            'recommended_sl_percent': (0.001, 0.5, STOP_LOSS_PERCENT)  # 0.1% to 50%
        }

        for field, (min_val, max_val, default_val) in numeric_fields_with_defaults.items():
            if field in analysis:
                try:
                    val = float(analysis[field])
                    if not (min_val <= val <= max_val):
                        logger.warning(f"LLM provided {field} {val} out of range [{min_val}-{max_val}]. Clamping or using default.")
                        analysis[field] = max(min_val, min(val, max_val)) if default_val is None else default_val # Clamp or use default
                    else:
                        analysis[field] = val
                except (ValueError, TypeError):
                    logger.warning(f"LLM provided non-numeric {field}: {analysis[field]}. Using default {default_val}.")
                    analysis[field] = default_val
            elif default_val is not None and analysis.get("signal") in ["long", "short"]:
                 analysis[field] = default_val # Ensure default is set if field was missing for trade signals

        # --- Start of moved and corrected TP/SL and signal validation block ---
        # This block is now inside get_llm_analysis, before the final successful return.

        # Validate signal first, as TP/SL logic might depend on it.
        signal_value = analysis.get("signal")
        if "signal" not in analysis: # Check if signal key itself is missing
            logger.error(f"'signal' key is missing from LLM analysis. LLM Response: {content_cleaned}")
            return None # Critical: signal is mandatory

        if not isinstance(signal_value, str) or signal_value.lower() not in ["long", "short", "hold"]:
            logger.error(f"Invalid signal value: '{signal_value}'. Must be 'long', 'short', or 'hold'. LLM Response: {content_cleaned}")
            return None
        
        analysis["signal"] = signal_value.lower() # Standardize to lowercase

        # TP/SL defaulting and validation only if it's a trading signal ("long" or "short")
        if analysis["signal"] in ["long", "short"]:
            # Default TP if missing or not a number
            if not isinstance(analysis.get("recommended_tp_percent"), (float, int)):
                try:
                    if 'TAKE_PROFIT_PERCENT' not in globals():
                        logger.error("TAKE_PROFIT_PERCENT global variable is not defined. Cannot set default TP.")
                        return None # Critical configuration missing
                    analysis["recommended_tp_percent"] = TAKE_PROFIT_PERCENT * 100 
                    logger.warning(f"Using default take profit percentage: {analysis['recommended_tp_percent']}% for signal '{analysis['signal']}' as it was missing or invalid.")
                except Exception as e_tp_default:
                    logger.error(f"Error applying default TP: {e_tp_default}. Check TAKE_PROFIT_PERCENT definition.")
                    return None
            
            # Default SL if missing or not a number
            if not isinstance(analysis.get("recommended_sl_percent"), (float, int)):
                try:
                    if 'STOP_LOSS_PERCENT' not in globals():
                        logger.error("STOP_LOSS_PERCENT global variable is not defined. Cannot set default SL.")
                        return None # Critical configuration missing
                    analysis["recommended_sl_percent"] = STOP_LOSS_PERCENT * 100
                    logger.warning(f"Using default stop loss percentage: {analysis['recommended_sl_percent']}% for signal '{analysis['signal']}' as it was missing or invalid.")
                except Exception as e_sl_default:
                    logger.error(f"Error applying default SL: {e_sl_default}. Check STOP_LOSS_PERCENT definition.")
                    return None
                            
            # Validate TP/SL percentages (now that they are ensured to exist and be numeric for trading signals)
            try:
                # Ensure keys exist before float conversion, even after defaulting attempts
                if "recommended_tp_percent" not in analysis or "recommended_sl_percent" not in analysis:
                    logger.error(f"TP or SL key missing after defaulting attempt. Analysis: {analysis}. LLM Response: {content_cleaned}")
                    return None

                tp_percent = float(analysis["recommended_tp_percent"])
                sl_percent = float(analysis["recommended_sl_percent"])
                
                # Range validation (0.5% to 10%)
                if not (0.5 <= tp_percent <= 10):
                    logger.error(f"Take profit percentage ({tp_percent}%) is out of specified range (0.5%-10%). LLM Response: {content_cleaned}")
                    return None 
                if not (0.5 <= sl_percent <= 10):
                    logger.error(f"Stop loss percentage ({sl_percent}%) is out of specified range (0.5%-10%). LLM Response: {content_cleaned}")
                    return None
                # SL must be less than TP
                if sl_percent >= tp_percent:
                    logger.error(f"Stop loss percentage ({sl_percent}%) must be less than take profit ({tp_percent}%). LLM Response: {content_cleaned}")
                    return None
            except (ValueError, TypeError) as e: 
                logger.error(f"Invalid TP/SL percentage format after defaults/LLM: {e}. Values: TP='{analysis.get('recommended_tp_percent')}', SL='{analysis.get('recommended_sl_percent')}'. LLM Response: {content_cleaned}")
                return None
        # --- End of moved and corrected TP/SL and signal validation block ---

                logger.info(f"Successfully parsed and validated LLM analysis: {analysis}")
                return analysis

            except json.JSONDecodeError as e_json:
                logger.error(f"Failed to decode LLM JSON response. Content: {content_cleaned} Error: {e_json}")

            except Exception as e:
                logger.error(f"Error processing LLM response: {e}. Content: {content_cleaned}", exc_info=True)
                return None

# --- Constants for LLM Prompting and Defaults ---
MIN_LEVERAGE_LLM_PROMPT = 1
MAX_LEVERAGE_LLM_PROMPT = 125 # Align with typical exchange limits
DEFAULT_LEVERAGE = 10 # A common default leverage

# Ensure RISK_PER_TRADE_PERCENT, TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT are defined globally
# For example:
# RISK_PER_TRADE_PERCENT = 0.01  # 1% risk per trade
# TAKE_PROFIT_PERCENT = 0.02  # 2% take profit
# STOP_LOSS_PERCENT = 0.01    # 1% stop loss

async def get_account_equity(rest_client):
    """Fetch account equity using the REST API."""
    logger.info("Fetching account equity via REST...")

    # Use constants defined earlier for product type
    # Removing marginCoin, letting API infer from productType
    params = {
        "productType": PRODUCT_TYPE_V2, # Should be SUSDT-FUTURES for demo
        "marginCoin": "SUSDT", # Re-adding marginCoin as it seems required for this endpoint
        "symbol": TARGET_INSTRUMENT # Added missing symbol parameter
    }
    logger.debug(f"Calling Bitget API: {BITGET_REST_ACCOUNT_ENDPOINT} with params: {params}")
    try:
        result = await asyncio.to_thread(
            rest_client.get, BITGET_REST_ACCOUNT_ENDPOINT, params
        )
        logger.debug(f"Raw Account Equity Response: {result}") # Log raw response

        # Correctly check for '00000' success code
        if isinstance(result, dict) and result.get('code') == '00000':
            logger.debug("API call successful (code 00000).")
            account_data = result.get('data')
            if account_data and 'usdtEquity' in account_data:
                try:
                    equity = float(account_data['usdtEquity'])
                    logger.info(f"Successfully fetched and parsed account equity: {equity}")
                    return equity
                except (ValueError, TypeError) as parse_err:
                    logger.error(f"Error parsing usdtEquity '{account_data.get('usdtEquity')}': {parse_err}")
                    return None
            else:
                logger.error(f"'usdtEquity' key not found or data missing in response data: {account_data}")
                return None
        else:
            # Handle non-dict or error code responses
            error_code = result.get('code', 'N/A') if isinstance(result, dict) else 'N/A'
            error_msg = result.get('msg', 'Unknown error') if isinstance(result, dict) else str(result)
            logger.error(f"API call failed. Code: {error_code}, Msg: {error_msg}. Raw response: {result}")
            return None

    except Exception as e:
        # Log the exception more simply first to avoid potential formatting errors
        logger.error(f"Exception caught in get_account_equity: {e}") 
        # Then log the full traceback separately
        logger.exception("Full traceback for equity fetch error:")
        return None

async def connect_private_websocket(loop, rest_client): # Added rest_client
    """Initialize and connect PRIVATE WebSocket client"""
    if not all([BITGET_API_KEY, BITGET_SECRET_KEY, BITGET_PASSPHRASE]):
        logger.error("[PrivateWS] Connection requires API credentials.")
        return None

    try:
        # --- Wrapper for thread-safe async callback ---
        # The 'client' instance will be available in this scope when the wrapper is defined.
        # So, we can capture it in the lambda or a functools.partial if needed, 
        # or simply pass it if the listener mechanism of BitgetWsClientAsync supports passing context.
        # For now, let's assume the listener in BitgetWsClientAsync is called with the message only.
        # We will modify the wrapper to pass the 'client' (BitgetWsClientAsync instance) to handle_private_message.
        # This requires 'client' to be defined before the wrapper that uses it.

        # Forward declaration for the client, will be assigned after wrapper definition
        # This is a bit tricky. Let's define the client first, then the wrapper.

        # Initialize client with all necessary parameters
        # The listener will be set after client is created, so it can capture 'client'
        # Placeholder for listener initially
        client = BitgetWsClientAsync(BITGET_WSS_PRIVATE_URL, 
                                   api_key=BITGET_API_KEY, 
                                   api_secret_key=BITGET_SECRET_KEY, 
                                   passphrase=BITGET_PASSPHRASE, 
                                   listener=None, # Will set this properly after client is created
                                   error_listener=lambda err: asyncio.run_coroutine_threadsafe(handle_ws_error("PrivateWS", err), loop))
        
        # Store rest_client as an attribute since BitgetWsClientAsync doesn't accept it in constructor
        client.rest_client = rest_client

        def private_message_handler_wrapper(message):
            # 'client' is now in the closure of this wrapper
            asyncio.run_coroutine_threadsafe(handle_private_message(message, client), loop)
        
        # Now set the actual listener on the client instance
        client._BitgetWsClientAsync__listener = private_message_handler_wrapper



        # client._BitgetWsClient__loop = loop # Loop is handled internally by websockets/asyncio
        asyncio.create_task(client.start()) # Start the connection and message handling loop

        await asyncio.sleep(10) # Allow more time for connection/auth attempt

        # Removed client.has_connect() check as it's not available
        # Connection status is handled internally by the start() loop and error listener.

        # Subscribe to account and position updates
        # Reverting account channel subscription to SubscribeReq format
        # The format {"instType": ..., "channel": "account", "coin": "default"} caused warnings.
        # Subscribe using SubscribeReq for all channels
        # SubscribeReq is modified to handle 'coin' for 'account' channel correctly
        account_sub = SubscribeReq(INST_TYPE_V2, 'account', coin='default')
        # Use 'default' for positions channel instId, similar to account channel, keep TARGET_INSTRUMENT for orders for now
        position_sub = SubscribeReq(INST_TYPE_V2, 'positions', instId='default')
        order_sub = SubscribeReq(INST_TYPE_V2, 'orders', instId=TARGET_INSTRUMENT)

        channels_to_subscribe = [account_sub, position_sub, order_sub]
        await client.subscribe(channels_to_subscribe)

        logger.info("[PrivateWS] Connection initiated and subscriptions sent.")
        return client

    except Exception as e:
        logger.error(f"[PrivateWS] Failed to initialize: {e}", exc_info=True)
        return None

async def connect_public_websocket(loop):
    """Initialize and connect PUBLIC WebSocket client for candle data"""
    try:
        # --- Wrapper for thread-safe async callback ---
        def public_message_handler_wrapper(message):
            asyncio.run_coroutine_threadsafe(handle_public_message(message), loop)

        # No login needed for public endpoint
        # Initialize client with all necessary parameters
        client = BitgetWsClientAsync(BITGET_WSS_PUBLIC_URL, 
                                   listener=public_message_handler_wrapper, # Use the sync wrapper
                                   error_listener=lambda err: asyncio.run_coroutine_threadsafe(handle_ws_error("PublicWS", err), loop))

        # client._BitgetWsClient__loop = loop # Loop is handled internally by websockets/asyncio
        asyncio.create_task(client.start()) # Start the connection and message handling loop

        await asyncio.sleep(2) # Allow time for connection

        # Removed client.has_connect() check as it's not available
        # Connection status is handled internally by the start() loop and error listener.

        # Subscribe to candle channel
        channels = [
            SubscribeReq(INST_TYPE_V2, CANDLE_CHANNEL, TARGET_INSTRUMENT)
        ]
        # Assuming subscribe uses the listener set earlier.
        await client.subscribe(channels) # Rely on the default listener set earlier

        logger.info(f"[PublicWS] Connection initiated and subscribed to {CANDLE_CHANNEL} for {TARGET_INSTRUMENT}.")
        return client

    except Exception as e:
        logger.error(f"[PublicWS] Failed to initialize: {e}", exc_info=True)
        return None


async def place_trade_via_rest(rest_client, instrument, signal, size, stop_loss_price=None, take_profit_price=None, order_type="market"):
    """Place trade via REST API, optionally including SL/TP"""
    trade_id = f"bot_{int(time.time()*1000)}"
    api_side = None # Initialize

    # Map signal to API side parameter
    # For unilateral position mode, posSide should be 'net'
    pos_side = "net"
    if signal.lower() == "long":
        api_side = "buy"
    elif signal.lower() == "short":
        api_side = "sell"
    else:
        logger.error(f"Invalid signal '{signal}' for placing trade.")
        return False

    logger.debug(f"place_trade_via_rest received signal: '{signal}', mapped to api_side: '{api_side}'")

    # Assuming entry_price will be passed to this function or derived
    # For now, let's assume it's available. This needs to be handled if place_trade_via_rest is used.
    # Placeholder: entry_price = current_market_price or a passed argument
    # This function's signature might need to change to accept entry_price if it's to place limit orders.
    entry_price = 0 # Placeholder, MUST be set correctly if this function is used for limit orders
    # For market orders, entry_price is determined by the exchange, not set by the client.

    params = {
        "symbol": instrument, # Use the instrument passed to the function
        "productType": PRODUCT_TYPE_V2,
        "marginCoin": "SUSDT",
        "marginMode": "isolated",
        # "posSide": pos_side, # 'net' for unilateral - Removing as it might conflict or be unnecessary for one-way mode via API
        "side": api_side,
        "orderType": "market",
        "size": str(size),
        "clientOrderId": trade_id,
        # "tradeSide": "open", # Not used in unilateral mode with 'side'
        "timeInForce": "GTC",
        "leverage": "20" # Ensure this is applicable or managed correctly
    }

    # Add SL/TP if provided, using correct Bitget V2 parameter names
    if stop_loss_price is not None:
        params['presetStopLossPrice'] = str(stop_loss_price)
    if take_profit_price is not None:
        params['presetTakeProfitPrice'] = str(take_profit_price)

    logger.info(f"Attempting to place REST order with params: {params}") # Log the final params being sent

    try:
        result = await asyncio.to_thread(
            rest_client.post, BITGET_REST_PLACE_ORDER_ENDPOINT, params
        )
        logger.info(f"REST Order Placement Response: {result}")

        logger.debug("Checking result type and code...")

        # Check for success code '0'
        is_dict = isinstance(result, dict)
        code = result.get('code') if is_dict else None
        logger.debug(f"Result is_dict: {is_dict}, Code: {code}")

        if is_dict and isinstance(code, str) and code == '0':
            logger.debug("Processing successful trade placement...")
            order_id = result.get('data', {}).get('orderId', 'N/A')
            logger.info(f"REST Trade submitted successfully: ClOrdId={trade_id}, OrderId={order_id}")
            log_trade_event({
                 "event": "place_order_success", "clOrdId": trade_id, "orderId": order_id,
                 "params": params, "response": result
            })
            return True
        else:
            logger.debug("Processing failed trade placement...")
            error_msg = result.get('msg', 'Unknown error') if isinstance(result, dict) else str(result)
            error_code = result.get('code', 'N/A') if isinstance(result, dict) else 'N/A'
            logger.error(f"REST Trade placement failed: Code={error_code}, Msg={error_msg} (ClOrdId: {trade_id}) Response: {result}")
            log_trade_event({
                 "event": "place_order_fail", "clOrdId": trade_id, "params": params,
                 "error_code": error_code, "error_msg": error_msg, "response": result
            })
            return False

    except BitgetAPIException as e:
         logger.error(f"REST API Exception during trade placement: {e}", exc_info=True)
         log_trade_event({"event": "place_order_exception", "clOrdId": trade_id, "params": params, "exception": str(e)})
         return False
    except Exception as e:
        logger.error(f"Unexpected error during REST trade placement: {e}", exc_info=True)
        log_trade_event({"event": "place_order_exception", "clOrdId": trade_id, "params": params, "exception": str(e)})
        return False

async def place_position_tpsl_via_rest(rest_client, instrument, hold_side, stop_loss_price, take_profit_price, stop_loss_trigger_type="mark_price", take_profit_trigger_type="mark_price"):
    """Place position TP/SL via REST API using /api/v2/mix/order/place-pos-tpsl, with dynamic adjustment based on mark price."""
    global PRODUCT_TYPE_V2 # Assuming PRODUCT_TYPE_V2 is globally defined

    # Ensure precisions are loaded for the instrument
    if instrument not in INSTRUMENT_PRECISIONS:
        logger.info(f"Precisions for {instrument} not yet loaded. Attempting to initialize.")
        await initialize_instrument_precisions(rest_client, instrument, PRODUCT_TYPE_V2)
        if instrument not in INSTRUMENT_PRECISIONS:
            logger.error(f"Failed to initialize precisions for {instrument}. TP/SL placement might fail or use default rounding.")

    current_mark_price = await get_current_mark_price(rest_client, instrument, PRODUCT_TYPE_V2)

    if current_mark_price is None:
        logger.error(f"Could not fetch mark price for {instrument}. Cannot reliably place TP/SL. Aborting TP/SL placement.")
        log_trade_event({"event": "place_pos_tpsl_fail_no_mark_price", "instrument": instrument, "reason": "Mark price unavailable"})
        return False, None

    logger.info(f"Current mark price for {instrument}: {current_mark_price}. Original TP: {take_profit_price}, SL: {stop_loss_price} for {hold_side}")

    adjusted_tp = None
    if take_profit_price is not None and take_profit_price > 0:
        tp_decimal = Decimal(str(take_profit_price))
        if hold_side.lower() in ['long', 'buy']:
            if tp_decimal <= current_mark_price:
                original_tp_for_log = tp_decimal
                tp_decimal = current_mark_price * (Decimal('1') + MARK_PRICE_BUFFER_PERCENT)
                # Ensure it's at least a minimum tick size away if buffer is too small
                if tp_decimal <= current_mark_price:
                    tp_decimal = current_mark_price + (current_mark_price * MIN_PRICE_ADJUSTMENT_FACTOR) 
                logger.warning(f"Long TP price {original_tp_for_log} was <= mark price {current_mark_price}. Adjusted to {tp_decimal}")
            adjusted_tp = round_price(tp_decimal, instrument, direction=ROUND_UP) # Round up for long TP
        elif hold_side.lower() in ['short', 'sell']:
            if tp_decimal >= current_mark_price:
                original_tp_for_log = tp_decimal
                tp_decimal = current_mark_price * (Decimal('1') - MARK_PRICE_BUFFER_PERCENT)
                if tp_decimal >= current_mark_price:
                    tp_decimal = current_mark_price - (current_mark_price * MIN_PRICE_ADJUSTMENT_FACTOR)
                logger.warning(f"Short TP price {original_tp_for_log} was >= mark price {current_mark_price}. Adjusted to {tp_decimal}")
            adjusted_tp = round_price(tp_decimal, instrument, direction=ROUND_DOWN) # Round down for short TP
        else:
            logger.error(f"Unknown hold_side '{hold_side}' for TP adjustment.")
            adjusted_tp = round_price(tp_decimal, instrument)
    
    adjusted_sl = None
    if stop_loss_price is not None and stop_loss_price > 0:
        sl_decimal = Decimal(str(stop_loss_price))
        if hold_side.lower() in ['long', 'buy']:
            if sl_decimal >= current_mark_price:
                original_sl_for_log = sl_decimal
                sl_decimal = current_mark_price * (Decimal('1') - MARK_PRICE_BUFFER_PERCENT)
                if sl_decimal >= current_mark_price:
                    sl_decimal = current_mark_price - (current_mark_price * MIN_PRICE_ADJUSTMENT_FACTOR)
                logger.warning(f"Long SL price {original_sl_for_log} was >= mark price {current_mark_price}. Adjusted to {sl_decimal}")
            adjusted_sl = round_price(sl_decimal, instrument, direction=ROUND_DOWN) # Round down for long SL
        elif hold_side.lower() in ['short', 'sell']:
            if sl_decimal <= current_mark_price:
                original_sl_for_log = sl_decimal
                sl_decimal = current_mark_price * (Decimal('1') + MARK_PRICE_BUFFER_PERCENT)
                if sl_decimal <= current_mark_price:
                    sl_decimal = current_mark_price + (current_mark_price * MIN_PRICE_ADJUSTMENT_FACTOR)
                logger.warning(f"Short SL price {original_sl_for_log} was <= mark price {current_mark_price}. Adjusted to {sl_decimal}")
            adjusted_sl = round_price(sl_decimal, instrument, direction=ROUND_UP) # Round up for short SL
        else:
            logger.error(f"Unknown hold_side '{hold_side}' for SL adjustment.")
            adjusted_sl = round_price(sl_decimal, instrument)

    # Final check for TP/SL validity against mark price after adjustments and rounding
    if adjusted_tp is not None:
        if hold_side.lower() in ['long', 'buy'] and adjusted_tp <= current_mark_price:
            logger.error(f"CRITICAL: Adjusted Long TP {adjusted_tp} is still <= mark price {current_mark_price}. Skipping TP.")
            adjusted_tp = None
        elif hold_side.lower() in ['short', 'sell'] and adjusted_tp >= current_mark_price:
            logger.error(f"CRITICAL: Adjusted Short TP {adjusted_tp} is still >= mark price {current_mark_price}. Skipping TP.")
            adjusted_tp = None

    if adjusted_sl is not None:
        if hold_side.lower() in ['long', 'buy'] and adjusted_sl >= current_mark_price:
            logger.error(f"CRITICAL: Adjusted Long SL {adjusted_sl} is still >= mark price {current_mark_price}. Skipping SL.")
            adjusted_sl = None
        elif hold_side.lower() in ['short', 'sell'] and adjusted_sl <= current_mark_price:
            logger.error(f"CRITICAL: Adjusted Short SL {adjusted_sl} is still <= mark price {current_mark_price}. Skipping SL.")
            adjusted_sl = None

    logger.info(f"Final adjusted prices for {instrument} ({hold_side}): TP={adjusted_tp}, SL={adjusted_sl} (Mark Price: {current_mark_price})")

    params = {
        "symbol": instrument,
        "productType": PRODUCT_TYPE_V2,
        "marginCoin": "SUSDT",
        "holdSide": hold_side,
    }

    if adjusted_tp is not None and adjusted_tp > Decimal('0'):
        params["stopSurplusTriggerPrice"] = str(adjusted_tp)
        params["stopSurplusTriggerType"] = take_profit_trigger_type

    if adjusted_sl is not None and adjusted_sl > Decimal('0'):
        params["stopLossTriggerPrice"] = str(adjusted_sl)
        params["stopLossTriggerType"] = stop_loss_trigger_type

    if not (params.get("stopSurplusTriggerPrice") or params.get("stopLossTriggerPrice")):
        logger.warning("place_position_tpsl_via_rest: Neither adjusted stop-loss nor take-profit price provided or valid. Skipping TP/SL placement.")
        return False, None

    logger.info(f"Attempting to place Position TP/SL with dynamically adjusted params: {params}")

    try:
        result = await asyncio.to_thread(
            rest_client.post, BITGET_REST_PLACE_POS_TPSL_ENDPOINT, params
        )
        logger.info(f"Position TP/SL Placement Response: {result}")

        is_dict = isinstance(result, dict)
        code = result.get('code') if is_dict else None

        if is_dict and isinstance(code, str) and code == '00000':
            data = result.get('data', [])
            order_ids = [item.get('orderId') for item in data if isinstance(item, dict) and item.get('orderId')]
            logger.info(f"Position TP/SL submitted successfully. Order IDs: {order_ids}")
            log_trade_event({
                "event": "place_pos_tpsl_success", 
                "params": params, 
                "response": result,
                "tpsl_order_ids": order_ids
            })
            return True, order_ids
        else:
            error_msg = result.get('msg', 'Unknown error') if is_dict else str(result)
            error_code = result.get('code', 'N/A') if is_dict else 'N/A'
            logger.error(f"Position TP/SL placement failed: Code={error_code}, Msg={error_msg}. Response: {result}")
            log_trade_event({
                "event": "place_pos_tpsl_fail", 
                "params": params, 
                "error_code": error_code, 
                "error_msg": error_msg, 
                "response": result
            })
            # Specific check for the original error to provide more context if it still occurs
            if str(error_code) == '40915':
                 logger.error(f"Error 40915 recurred: Long position TP > mark price. Mark: {current_mark_price}, TP Sent: {params.get('stopSurplusTriggerPrice')}")
            return False, None

    except BitgetAPIException as e:
        logger.error(f"REST API Exception during Position TP/SL placement: {e}", exc_info=True)
        log_trade_event({"event": "place_pos_tpsl_exception", "params": params, "exception": str(e)})
        return False, None
    except Exception as e:
        logger.error(f"Unexpected error during Position TP/SL placement: {e}", exc_info=True)
        log_trade_event({"event": "place_pos_tpsl_exception", "params": params, "exception": str(e)})
        return False, None

    params = {
        "symbol": instrument,
        "productType": PRODUCT_TYPE_V2,
        "marginCoin": "SUSDT", # Assuming SUSDT, adjust if necessary
        "holdSide": hold_side, # 'long' or 'short' for two-way, 'buy' or 'sell' for one-way
    }

    if take_profit_price is not None and take_profit_price > 0:
        params["stopSurplusTriggerPrice"] = str(take_profit_price)
        params["stopSurplusTriggerType"] = take_profit_trigger_type
        # params["stopSurplusExecutePrice"] = "0" # Market price execution for TP is default if not sent or 0

    if stop_loss_price is not None and stop_loss_price > 0:
        params["stopLossTriggerPrice"] = str(stop_loss_price)
        params["stopLossTriggerType"] = stop_loss_trigger_type
        # params["stopLossExecutePrice"] = "0" # Market price execution for SL is default if not sent or 0

    if not (params.get("stopSurplusTriggerPrice") or params.get("stopLossTriggerPrice")):
        logger.warning("place_position_tpsl_via_rest: Neither stop-loss nor take-profit price provided or valid. Skipping TP/SL placement.")
        return False, None # Indicate failure, no order IDs

    logger.info(f"Attempting to place Position TP/SL with params: {params}")

    try:
        result = await asyncio.to_thread(
            rest_client.post, BITGET_REST_PLACE_POS_TPSL_ENDPOINT, params
        )
        logger.info(f"Position TP/SL Placement Response: {result}")

        is_dict = isinstance(result, dict)
        code = result.get('code') if is_dict else None

        if is_dict and isinstance(code, str) and code == '00000': # Bitget success code for TP/SL
            data = result.get('data', [])
            order_ids = [item.get('orderId') for item in data if isinstance(item, dict) and item.get('orderId')]
            logger.info(f"Position TP/SL submitted successfully. Order IDs: {order_ids}")
            log_trade_event({
                "event": "place_pos_tpsl_success", 
                "params": params, 
                "response": result,
                "tpsl_order_ids": order_ids
            })
            return True, order_ids
        else:
            error_msg = result.get('msg', 'Unknown error') if is_dict else str(result)
            error_code = result.get('code', 'N/A') if is_dict else 'N/A'
            logger.error(f"Position TP/SL placement failed: Code={error_code}, Msg={error_msg}. Response: {result}")
            log_trade_event({
                "event": "place_pos_tpsl_fail", 
                "params": params, 
                "error_code": error_code, 
                "error_msg": error_msg, 
                "response": result
            })
            return False, None

    except BitgetAPIException as e:
        logger.error(f"REST API Exception during Position TP/SL placement: {e}", exc_info=True)
        log_trade_event({"event": "place_pos_tpsl_exception", "params": params, "exception": str(e)})
        return False, None
    except Exception as e:
        logger.error(f"Unexpected error during Position TP/SL placement: {e}", exc_info=True)
        log_trade_event({"event": "place_pos_tpsl_exception", "params": params, "exception": str(e)})
        return False, None


TRADE_EVENT_LOG_FILE = "trade_events.log" # Added for detailed event logging

def log_trade_event(event_data):
    """Logs a trade-related event to the main logger and a dedicated event log file."""
    try:
        # Add a timestamp to the event data
        event_data_with_ts = {
            'timestamp': datetime.datetime.now().isoformat(),
            **event_data
        }

        # Log a summary to the main logger
        event_type = event_data.get('event', 'unknown_event')
        cl_ord_id = event_data.get('clOrdId') or event_data.get('client_order_id')
        log_summary = f"Trade Event: {event_type}"
        if cl_ord_id:
            log_summary += f" (ID: {cl_ord_id})"
        
        # Check for exception details to log more informatively
        if 'exception' in event_data:
            logger.error(f"{log_summary} - Exception: {event_data['exception']}", exc_info=False) # exc_info=False as we already have the string
        elif 'error_msg' in event_data:
            logger.error(f"{log_summary} - Error: {event_data['error_msg']} (Code: {event_data.get('error_code', 'N/A')})")
        else:
            logger.info(log_summary)

        # Append the full event data to the trade_events.log file
        # Ensure the directory exists before writing
        event_log_dir = os.path.dirname(TRADE_EVENT_LOG_FILE)
        if event_log_dir: # Check if dirname returned a non-empty string
            os.makedirs(event_log_dir, exist_ok=True)

        with open(TRADE_EVENT_LOG_FILE, 'a') as f:
            json.dump(event_data_with_ts, f)
            f.write('\n') # Newline for each JSON entry

    except Exception as e:
        logger.error(f"Failed to log trade event: {event_data.get('event', 'unknown_event')}. Error: {e}", exc_info=True)


def save_trade_history(client_order_id, side, size, entry_price, stop_loss_price, take_profit_price, status, llm_analysis=None, indicators=None, exchange_order_id=None):
    """Saves trade details and context to a JSON file."""
    try: # Add try block here
        history = []
        if os.path.exists(TRADE_HISTORY_FILE):
            try:
                with open(TRADE_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        logger.warning(f"'{TRADE_HISTORY_FILE}' does not contain a valid JSON list. Initializing new list.")
                        history = []
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from '{TRADE_HISTORY_FILE}'. Initializing new list.")
                history = [] # Ensure history is a list even if file is corrupt
            except Exception as e:
                logger.error(f"Error reading trade history file '{TRADE_HISTORY_FILE}': {e}", exc_info=True)
                history = [] # Ensure history is a list on other read errors

        trade_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'client_order_id': client_order_id,
            'symbol': TARGET_INSTRUMENT,
            'side': side,
            'size': size,
            'exchange_order_id': exchange_order_id, # Add exchange_order_id to the record
            'entry_price_target': entry_price, # The price used for calculation
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'status': status, # e.g., 'Placed', 'PlaceFailed_NoID', 'PlaceFailed_APIExc_...', 'Filled', 'Closed_SL', 'Closed_TP'
            'llm_analysis': llm_analysis, # Full analysis dict
            'indicators_at_trade': indicators, # Full indicators dict
            'actual_entry_price': None, # To be updated on fill
            'exit_price': None, # To be updated on close
            'pnl': None # To be updated on close
        }

        history.append(trade_record)

        # Ensure the directory exists before writing
        # Get the directory path from the full file path
        history_dir = os.path.dirname(TRADE_HISTORY_FILE)
        if history_dir: # Check if dirname returned a non-empty string
            os.makedirs(history_dir, exist_ok=True)

        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4, cls=DecimalEncoder)
        logger.info(f"Saved trade record for {client_order_id} with status '{status}'")

    except Exception as e: # This except block now correctly corresponds to the outer try
        logger.error(f"Error saving trade history for {client_order_id}: {e}", exc_info=True)



def update_trade_history(order_id, update_data):
    """Updates an existing trade record in the history file."""
    try:
        if not os.path.exists(TRADE_HISTORY_FILE):
            logger.warning(f"Trade history file {TRADE_HISTORY_FILE} not found for update.")
            return

        try:
            with open(TRADE_HISTORY_FILE, 'r') as f:
                history = json.load(f)
                if not isinstance(history, list):
                    logger.warning(f"'{TRADE_HISTORY_FILE}' does not contain a valid JSON list. Resetting.")
                    history = []
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {TRADE_HISTORY_FILE}. Cannot update. Resetting.")
            history = [] # Start fresh if file is corrupt
        except Exception as e:
            logger.error(f"Error reading trade history file '{TRADE_HISTORY_FILE}' during update: {e}", exc_info=True)
            return # Cannot proceed if reading fails

        updated = False
        for trade in history:
            # Match based on clientOrderId or potentially orderId if available
            if trade.get('client_order_id') == order_id or trade.get('order_id') == order_id:
                # Merge update_data into the trade record
                # Only update fields present in update_data
                for key, value in update_data.items():
                    if value is not None: # Avoid overwriting with None unless explicitly intended
                        trade[key] = value
                # Ensure status is updated if provided
                if 'status' in update_data:
                     trade['status'] = update_data['status']
                updated = True
                logger.info(f"Updating trade record for {order_id} with: {update_data}")
                break # Assume unique order IDs

        if not updated:
            logger.warning(f"Could not find trade record with ID {order_id} to update.")
            return # Don't rewrite if no changes were made

        # Write the updated history back
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4, cls=DecimalEncoder)
        logger.info(f"Successfully updated trade history for {order_id}.")

    except Exception as e:
        logger.error(f"Error updating trade history for {order_id}: {e}", exc_info=True)


# --- Learning Mechanism --- 

async def analyze_trade_history_and_learn(rest_client, ws_client):
    """Analyzes past trades to identify patterns and generate learning insights.
    Args:
        rest_client: The Bitget REST client instance for fetching historical positions
        ws_client: The WebSocket client instance for accessing REST client
    """
    logger.info("--- Analyzing Trade History for Learning --- ")
    if not os.path.exists(TRADE_HISTORY_FILE):
        logger.warning("Trade history file not found. Cannot perform learning analysis.")
        return None # Or return empty insights

    try:
        history = []

        
        with open(TRADE_HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error reading or parsing trade history file: {e}")
        return None

    closed_trades = [t for t in history if t.get('status') in ['Closed', 'Closed_With_API_PnL'] and t.get('pnl') is not None]
    logger.info(f"Found {len(closed_trades)} trades with status 'Closed' or 'Closed_With_API_PnL' and PnL for analysis.")

    if not closed_trades:
        logger.info("No closed trades with PnL found in history to analyze. Fetching historical positions instead.")
        # Fetch historical positions as fallback
        now = int(time.time() * 1000)
        three_months_ago = now - (90 * 24 * 60 * 60 * 1000)
        historical_positions = await fetch_historical_positions(
            ws_client.rest_client,
            symbol=TARGET_INSTRUMENT,
            start_time=three_months_ago,
            end_time=now
        )
        
        if historical_positions:
            logger.info(f"Found {len(historical_positions)} historical positions for analysis")
            # Process historical positions similar to closed trades
            pnl_by_confidence = {}
            count_by_confidence = {}
            for pos in historical_positions:
                confidence = 'Historical'  # Default confidence for historical positions
                pnl = float(pos.get('pnl', 0.0))
                
                pnl_by_confidence[confidence] = pnl_by_confidence.get(confidence, 0) + pnl
                count_by_confidence[confidence] = count_by_confidence.get(confidence, 0) + 1

            avg_pnl = {conf: round(pnl_by_confidence[conf] / count_by_confidence[conf], 4) 
                       for conf in pnl_by_confidence if count_by_confidence[conf] > 0}
            
            logger.info(f"Average PnL from historical positions: {avg_pnl}")
            return {"avg_pnl_from_history": avg_pnl}
        
        logger.info("No historical positions found either.")
        return None

    logger.info(f"Analyzing {len(closed_trades)} closed trades...")

    # Example Analysis: Average PnL by Confidence Level
    pnl_by_confidence = {}
    count_by_confidence = {}
    for trade in closed_trades:
        confidence = trade.get('llm_analysis', {}).get('confidence', 'Unknown')
        pnl = trade.get('pnl', 0.0)
        
        pnl_by_confidence[confidence] = pnl_by_confidence.get(confidence, 0) + pnl
        count_by_confidence[confidence] = count_by_confidence.get(confidence, 0) + 1

    avg_pnl_by_confidence = {conf: round(pnl_by_confidence[conf] / count_by_confidence[conf], 4) 
                             for conf in pnl_by_confidence if count_by_confidence[conf] > 0}
    
    logger.info(f"Average PnL by LLM Confidence: {avg_pnl_by_confidence}")

    # Example Analysis: Average PnL by Volatility Level
    pnl_by_volatility = {}
    count_by_volatility = {}
    for trade in closed_trades:
        volatility = trade.get('llm_analysis', {}).get('volatility', 'Unknown')
        pnl = trade.get('pnl', 0.0)
        
        pnl_by_volatility[volatility] = pnl_by_volatility.get(volatility, 0) + pnl
        count_by_volatility[volatility] = count_by_volatility.get(volatility, 0) + 1

    avg_pnl_by_volatility = {vol: round(pnl_by_volatility[vol] / count_by_volatility[vol], 4) 
                             for vol in pnl_by_volatility if count_by_volatility[vol] > 0}
    
    logger.info(f"Average PnL by LLM Volatility Assessment: {avg_pnl_by_volatility}")

    # --- Generate Feedback/Insights (Placeholder) ---
    # This is where more sophisticated pattern detection would go.
    # For now, we just log the averages.
    # Future: Generate structured feedback like:
    # feedback = ["High confidence trades during Low volatility have been profitable.", 
    #             "Short signals during High volatility have resulted in losses."]
    learning_insights = {
        "avg_pnl_by_confidence": avg_pnl_by_confidence,
        "avg_pnl_by_volatility": avg_pnl_by_volatility
        # Add more analysis results here
    }
    if not learning_insights.get("avg_pnl_by_confidence") and not learning_insights.get("avg_pnl_by_volatility"):
        logger.info("No specific learning insights generated from trade history.")
        return {}

    logger.info("Learning analysis complete.")
    return learning_insights


async def set_leverage_on_exchange(rest_client, symbol, leverage, margin_mode, product_type, hold_side):
    """Sets the leverage for a given symbol on the exchange."""
    endpoint = '/api/v2/mix/account/set-leverage' # Bitget V2 endpoint for setting leverage
    params = {
        "symbol": symbol,
        "marginCoin": "SUSDT", # Assuming SUSDT for demo, adjust if necessary
        "leverage": str(int(leverage)), # Leverage must be an integer string
        "holdSide": hold_side, # Use the passed hold_side ('long', 'short', or 'net')
        "productType": product_type
    }

    logger.info(f"Attempting to set leverage for {symbol} to {leverage}x with params: {params}")
    try:
        result = await asyncio.to_thread(
            rest_client.post, endpoint, params
        )
        logger.debug(f"Set Leverage Response: {result}")
        if isinstance(result, dict) and result.get('code') == '00000':
            logger.info(f"Successfully set leverage for {symbol} to {leverage}x.")
            return True
        else:
            error_msg = result.get('msg', 'Unknown error') if isinstance(result, dict) else str(result)
            logger.error(f"Failed to set leverage for {symbol} to {leverage}x: {error_msg}")
            log_trade_event({
                "event": "set_leverage_fail", 
                "symbol": symbol, 
                "leverage": leverage, 
                "params": params, 
                "response": result
            })
            return False
    except BitgetAPIException as e:
        logger.error(f"API Exception setting leverage for {symbol} to {leverage}x: {e}")
        log_trade_event({
            "event": "set_leverage_api_exception", 
            "symbol": symbol, 
            "leverage": leverage, 
            "params": params, 
            "exception": str(e)
        })
        return False
    except Exception as e:
        logger.error(f"Unexpected error setting leverage for {symbol} to {leverage}x: {e}", exc_info=True)
        log_trade_event({
            "event": "set_leverage_unexpected_exception", 
            "symbol": symbol, 
            "leverage": leverage, 
            "params": params, 
            "exception": str(e)
        })
        return False

async def initialize_instrument_precisions(rest_client, symbol, product_type):
    """Fetch and cache instrument's price and size precision."""
    global INSTRUMENT_PRECISIONS
    if symbol in INSTRUMENT_PRECISIONS:
        return True
    try:
        params = {"productType": product_type, "symbol": symbol}
        logger.info(f"Fetching contract details for precision: {symbol}")
        result = await asyncio.to_thread(rest_client.get, "/api/v2/mix/market/contracts", params)
        if isinstance(result, dict) and result.get('code') == '00000':
            contracts = result.get('data', [])
            if contracts and len(contracts) > 0:
                contract_info = contracts[0]
                price_place = int(contract_info.get('pricePlace', 2)) # Default to 2 if not found
                size_place = int(contract_info.get('sizePlace', 3))  # Default to 3 if not found
                INSTRUMENT_PRECISIONS[symbol] = {"pricePlace": price_place, "sizePlace": size_place}
                logger.info(f"Initialized precision for {symbol}: pricePlace={price_place}, sizePlace={size_place}")
                return True
            else:
                logger.error(f"No contract data found for {symbol} in precision fetch response: {result}")
        else:
            logger.error(f"Failed to fetch contract details for precision for {symbol}: {result}")
    except Exception as e:
        logger.error(f"Error initializing instrument precisions for {symbol}: {e}", exc_info=True)
    return False

def round_price(price, symbol, direction=ROUND_HALF_UP):
    """Round price according to the instrument's precision using Decimal."""
    global INSTRUMENT_PRECISIONS
    try:
        price_decimal = Decimal(str(price))
        if symbol in INSTRUMENT_PRECISIONS:
            price_place = INSTRUMENT_PRECISIONS[symbol]['pricePlace']
            quantizer = Decimal('1e-' + str(price_place))
            return price_decimal.quantize(quantizer, rounding=direction)
        else:
            logger.warning(f"Price precision for {symbol} not found. Using default rounding to 1 decimal place. Consider calling initialize_instrument_precisions first.")
            # Fallback, ensure this is appropriate for your instruments
            return price_decimal.quantize(Decimal('0.1'), rounding=direction)
    except Exception as e:
        logger.error(f"Error rounding price {price} for {symbol}: {e}. Returning original price.", exc_info=True)
        return Decimal(str(price)) # Fallback to original price as Decimal

async def get_current_mark_price(rest_client, symbol, product_type):
    """Fetch the current mark price for the instrument."""
    try:
        params = {"productType": product_type, "symbol": symbol}
        # logger.debug(f"Fetching ticker for mark price: {symbol}")
        result = await asyncio.to_thread(rest_client.get, "/api/v2/mix/market/ticker", params)
        if isinstance(result, dict) and result.get('code') == '00000':
            tickers = result.get('data', [])
            if tickers and len(tickers) > 0:
                mark_price_str = tickers[0].get('markPrice') # Changed 'markPx' to 'markPrice'
                if mark_price_str:
                    # logger.debug(f"Mark price for {symbol}: {mark_price_str}")
                    return Decimal(mark_price_str)
                else:
                    logger.error(f"'markPrice' not found in ticker response for {symbol}: {tickers[0]}") # Changed 'markPx' to 'markPrice' in log
            else:
                logger.error(f"No ticker data found for {symbol} in mark price fetch response: {result}")
        else:
            logger.error(f"Failed to fetch ticker for mark price for {symbol}: {result}")
    except Exception as e:
        logger.error(f"Error fetching mark price for {symbol}: {e}", exc_info=True)
    return None

async def fetch_contract_details(rest_client):
    """Fetch contract details from Bitget API."""
    try:
        params = {
            "productType": PRODUCT_TYPE_V2,
            "symbol": TARGET_INSTRUMENT
        }
        result = await asyncio.to_thread(
            rest_client.get, "/api/v2/mix/market/contracts", params
        )
        if isinstance(result, dict) and result.get('code') == '00000':
            contracts = result.get('data', [])
            if contracts and len(contracts) > 0:
                return contracts[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching contract details: {e}")
        return None

async def place_order(rest_client, side, equity, entry_price, llm_analysis, indicator_data, leverage=None, tp_percent=None, sl_percent=None):
    """Place a market order with dynamic size, stop-loss, and take-profit."""
    # Ensure precisions are loaded for the target instrument first
    if TARGET_INSTRUMENT not in INSTRUMENT_PRECISIONS:
        logger.info(f"Precisions for {TARGET_INSTRUMENT} not yet loaded in place_order. Attempting to initialize.")
        await initialize_instrument_precisions(rest_client, TARGET_INSTRUMENT, PRODUCT_TYPE_V2)
        if TARGET_INSTRUMENT not in INSTRUMENT_PRECISIONS:
            logger.error(f"Failed to initialize precisions for {TARGET_INSTRUMENT} in place_order. Proceeding with default rounding.")

    if not equity or not entry_price or equity <= 0 or entry_price <= 0:
        logger.error(f"Invalid equity ({equity}) or entry_price ({entry_price}) for placing order.")
        return None

    # Fetch contract details
    contract_details = await fetch_contract_details(rest_client)
    min_trade_usdt = float(contract_details.get('minTradeUSDT', '5')) if contract_details else 5.0
    min_lever = float(contract_details.get('minLever', '1')) if contract_details else 1.0
    max_lever = float(contract_details.get('maxLever', '125')) if contract_details else 125.0

    # --- Initialize TP/SL percentages --- 
    # Use TP percent from llm_analysis if available, else use global default
    current_tp_percent = tp_percent if tp_percent is not None else TAKE_PROFIT_PERCENT
    logger.info(f"Using initial Take Profit Percent: {current_tp_percent * 100}%")

    # Use SL percent from llm_analysis if available, else use global default
    current_sl_percent = sl_percent if sl_percent is not None else STOP_LOSS_PERCENT
    logger.info(f"Using initial Stop Loss Percent: {current_sl_percent * 100}%")

    # --- Dynamic Position Sizing & Leverage --- 
    # Use leverage from llm_analysis if available, else use default from constants or contract details
    # This will be used for size calculation if recommended_margin is provided, and for setting leverage on the exchange.
    current_leverage = leverage if leverage is not None else llm_analysis.get('recommended_leverage')
    if current_leverage is None:
        logger.warning(f"'recommended_leverage' not found in LLM analysis, falling back to DEFAULT_LEVERAGE: {DEFAULT_LEVERAGE}")
        current_leverage = DEFAULT_LEVERAGE
    else:
        current_leverage = float(current_leverage) # Ensure it's a float

    current_leverage = min(max(current_leverage, min_lever), max_lever) # Ensure it's within contract limits
    logger.info(f"Target Leverage based on LLM/Default: {current_leverage}x (Min: {min_lever}, Max: {max_lever})")

    # --- Position Sizing --- 
    recommended_margin_usdt = llm_analysis.get('recommended_margin')

    if recommended_margin_usdt is not None and isinstance(recommended_margin_usdt, (float, int)) and recommended_margin_usdt > 0:
        logger.info(f"Using LLM recommended_margin: {recommended_margin_usdt} USDT with {current_leverage}x leverage.")
        # Ensure recommended_margin_usdt does not exceed available equity
        if recommended_margin_usdt > equity:
            logger.warning(f"LLM recommended_margin {recommended_margin_usdt} USDT exceeds available equity {equity} USDT. Clamping to equity.")
            recommended_margin_usdt = equity 
        
        position_value_usdt = recommended_margin_usdt * current_leverage
        if entry_price <= 0:
            logger.error("Entry price is zero or negative, cannot calculate position size from recommended_margin.")
            return None
        size = position_value_usdt / entry_price
        logger.info(f"Calculated size based on recommended_margin: {size} {TARGET_INSTRUMENT.replace('SUSDT', '')}")
    else:
        logger.info(f"recommended_margin not provided or invalid in LLM analysis. Falling back to risk-based sizing.")
        # Calculate dynamic risk percentage based on LLM analysis or default
        dynamic_risk_percent = llm_analysis.get('risk_percent', RISK_PER_TRADE_PERCENT)
        # Ensure risk is within reasonable bounds (e.g., 0.5% to 5%)
        dynamic_risk_percent = min(max(dynamic_risk_percent, 0.005), 0.05)
        
        # Calculate the amount to risk in USDT
        risk_amount_usdt = equity * dynamic_risk_percent
        # Use SL percent from llm_analysis if available, else use global default
        current_sl_percent = sl_percent if sl_percent is not None else STOP_LOSS_PERCENT
        logger.info(f"Using Stop Loss Percent for risk-based sizing: {current_sl_percent * 100}%")

        # Calculate the distance to the stop loss in price terms
        stop_loss_distance = entry_price * current_sl_percent
        if stop_loss_distance <= 0:
            logger.error("Stop loss distance is zero or negative, cannot calculate risk-based position size.")
            return None
        size = risk_amount_usdt / stop_loss_distance
        logger.info(f"Calculated size based on risk: {size} {TARGET_INSTRUMENT.replace('SUSDT', '')} (Equity: {equity}, Risk %: {dynamic_risk_percent*100}%, SL Dist: {stop_loss_distance})")

    # Ensure minimum trade size
    size = max(size, MIN_TRADE_SIZE)
    # Round size to appropriate precision (e.g., 3 decimal places for BTC)
    size = round(size, 3)

    logger.info(f"Final calculated dynamic position size: {size} {TARGET_INSTRUMENT.replace('SUSDT', '')}")

    # --- Set Leverage on Exchange --- 
    # Map 'buy'/'sell' to 'long'/'short' for holdSide
    hold_side_for_leverage = 'long' if side == 'buy' else 'short' if side == 'sell' else 'net'
    # Ensure leverage is set before placing the order
    leverage_set_successfully = await set_leverage_on_exchange(rest_client, TARGET_INSTRUMENT, current_leverage, "isolated", PRODUCT_TYPE_V2, hold_side_for_leverage)
    if not leverage_set_successfully:
        logger.error(f"Failed to set leverage to {current_leverage}x for {TARGET_INSTRUMENT}. Aborting order placement.")
        # Optionally, save a specific trade history event for leverage set failure
        order_id_fail_leverage = f"agent247_levfail_{int(time.time() * 1000)}"
        save_trade_history(order_id_fail_leverage, side, size, entry_price, None, None, "LeverageSetFailed", llm_analysis, indicator_data, exchange_order_id=None)
        return None
    logger.info(f"Successfully set leverage to {current_leverage}x for {TARGET_INSTRUMENT} on the exchange.")

    # --- Stop Loss and Take Profit Calculation (Factoring in Fees for TP) ---
    # Fees are applied on both entry and exit, so double the fee rate for TP calculation.
    total_fee_consideration_rate = 2 * TRADING_FEE_RATE # Renamed for clarity
    stop_loss_price = None
    take_profit_price = None

    # current_tp_percent and current_sl_percent are already initialized at the beginning of the function.

    if recommended_margin_usdt is not None and isinstance(recommended_margin_usdt, (float, int)) and recommended_margin_usdt > 0 and size > 0:
        logger.info(f"Calculating TP/SL based on recommended_margin_usdt: {recommended_margin_usdt} and size: {size}")
        # Calculate TP/SL amounts based on the margin used for this specific trade
        tp_amount_usdt = recommended_margin_usdt * current_tp_percent
        sl_amount_usdt = recommended_margin_usdt * current_sl_percent

        # Calculate fee amount based on the position value for TP adjustment
        # Using size * entry_price for a more accurate position value for fee calculation
        approx_position_value_for_fees = size * entry_price 
        fee_per_leg_usdt = approx_position_value_for_fees * TRADING_FEE_RATE
        total_fees_for_tp_usdt = 2 * fee_per_leg_usdt # Entry and Exit fees

        if side == 'buy': # Long position
            # SL: Margin loss amount / size = price drop per unit
            stop_loss_price = entry_price - (sl_amount_usdt / size)
            # TP: Margin gain amount / size = price increase per unit. Add fees to target gain.
            take_profit_price = entry_price + ((tp_amount_usdt + total_fees_for_tp_usdt) / size)
            if take_profit_price <= entry_price + (total_fees_for_tp_usdt / size):
                 logger.warning(f"Adjusted take profit price ({take_profit_price}) for long (margin-based) is too close to or below entry + fees. Original TP target amount: {tp_amount_usdt}, Total fees for TP: {total_fees_for_tp_usdt}")
        elif side == 'sell': # Short position
            # SL: Margin loss amount / size = price increase per unit
            stop_loss_price = entry_price + (sl_amount_usdt / size)
            # TP: Margin gain amount / size = price drop per unit. Subtract fees from target gain (as price moves down).
            take_profit_price = entry_price - ((tp_amount_usdt + total_fees_for_tp_usdt) / size)
            if take_profit_price >= entry_price - (total_fees_for_tp_usdt / size):
                logger.warning(f"Adjusted take profit price ({take_profit_price}) for short (margin-based) is too close to or above entry - fees. Original TP target amount: {tp_amount_usdt}, Total fees for TP: {total_fees_for_tp_usdt}")
        else:
            logger.error(f"Invalid side '{side}' for margin-based SL/TP calculation.")
            return None
        logger.info(f"Margin-based SL/TP: SL Amount: {sl_amount_usdt}, TP Amount (pre-fee): {tp_amount_usdt}, Total Fees for TP: {total_fees_for_tp_usdt}")

    else:
        logger.info(f"Falling back to entry price based TP/SL calculation (recommended_margin_usdt not available or size is zero).")
        if current_tp_percent <= total_fee_consideration_rate:
            logger.warning(f"TAKE_PROFIT_PERCENT ({current_tp_percent*100}%) is less than or equal to total fee consideration ({total_fee_consideration_rate*100}%). Profit target may not cover fees.")

        if side == 'buy': # Long position
            stop_loss_price = entry_price * (1 - current_sl_percent)
            take_profit_price = entry_price * (1 + current_tp_percent + total_fee_consideration_rate)
            if take_profit_price <= entry_price * (1 + total_fee_consideration_rate):
                logger.warning(f"Adjusted take profit price ({take_profit_price}) for long (entry-price based) is too close to or below entry price + fees. Original TP: {entry_price * (1 + current_tp_percent)}")
        elif side == 'sell': # Short position
            stop_loss_price = entry_price * (1 + current_sl_percent)
            take_profit_price = entry_price * (1 - current_tp_percent - total_fee_consideration_rate)
            if take_profit_price >= entry_price * (1 - total_fee_consideration_rate):
                logger.warning(f"Adjusted take profit price ({take_profit_price}) for short (entry-price based) is too close to or above entry price - fees. Original TP: {entry_price * (1 - current_tp_percent)}")
        else:
            logger.error(f"Invalid side '{side}' for entry-price based SL/TP calculation.")
            return None

    # Round prices using the new round_price function
    # Ensure precisions are loaded for TARGET_INSTRUMENT before calling place_order or early in main_trading_loop
    # Example: await initialize_instrument_precisions(rest_client, TARGET_INSTRUMENT, PRODUCT_TYPE_V2)
    if stop_loss_price is not None:
        stop_loss_price = round_price(stop_loss_price, TARGET_INSTRUMENT, direction=ROUND_DOWN if side == 'buy' else ROUND_UP)
    if take_profit_price is not None:
        take_profit_price = round_price(take_profit_price, TARGET_INSTRUMENT, direction=ROUND_UP if side == 'buy' else ROUND_DOWN)

    logger.info(f"Calculated SL: {stop_loss_price}, TP: {take_profit_price} for {side} order before main placement.")
    if take_profit_price is not None:
        take_profit_price = round(take_profit_price, 1)

    order_id = f"agent247_{int(time.time() * 1000)}" # Unique client order ID
    pos_side = "net"  # Explicitly set for unilateral position mode
    params = {
        "symbol": TARGET_INSTRUMENT,
        "productType": PRODUCT_TYPE_V2,
        "marginMode": "isolated", # Or 'cross'
        "marginCoin": "SUSDT",
        "side": "buy" if side == "buy" else "sell",
        "orderType": "market", # Reverted to market for one-way mode
        # "price": str(round(entry_price, 1)), # Price is not required for market orders
        "size": str(size), # Size must be a string
        "clientOrderId": order_id,
        "posSide": pos_side,
        # tradeSide is ignored in one-way mode as per documentation
        # Add Stop Loss and Take Profit parameters
        # Temporarily commenting out TP/SL to diagnose error 40774 for market orders in unilateral mode.
        # "presetTakeProfitPrice": str(take_profit_price),
        # "presetStopLossPrice": str(stop_loss_price),
        "reduceOnly": "NO"  # Explicitly set for opening new positions in one-way mode
    }

    logger.info(f"Placing Order: {params}")
    try:
        result = await asyncio.to_thread(
            rest_client.post, BITGET_REST_PLACE_ORDER_ENDPOINT, params
        )
        logger.debug(f"Place Order Response: {result}")

        if isinstance(result, dict) and result.get('code') == '00000': # Main order success
            order_data = result.get('data')
            if order_data and 'orderId' in order_data:
                exchange_order_id = order_data['orderId']
                logger.info(f"Main order placed successfully: ID {exchange_order_id}, Client ID {order_id}")

                # Save main trade details first
                save_trade_history(order_id, side, size, entry_price, stop_loss_price, take_profit_price, "PlacedMainOrder", llm_analysis, indicator_data, exchange_order_id=exchange_order_id)

                # --- Place TP/SL for the position using the new function --- 
                if (take_profit_price and take_profit_price > 0) or (stop_loss_price and stop_loss_price > 0):
                    # Determine holdSide for TP/SL based on main order's side
                    # For one-way mode, API expects 'buy' for long position TP/SL, 'sell' for short position TP/SL.
                    tpsl_hold_side = "buy" if side == "buy" else "sell" 
                    logger.info(f"Attempting to place TP/SL for main order {exchange_order_id}. HoldSide: {tpsl_hold_side}, SL: {stop_loss_price}, TP: {take_profit_price}")
                    
                    tpsl_success, tpsl_order_ids = await place_position_tpsl_via_rest(
                        rest_client,
                        TARGET_INSTRUMENT,
                        hold_side=tpsl_hold_side,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price
                    )

                    if tpsl_success:
                        logger.info(f"Position TP/SL placed successfully for main order {exchange_order_id}. TP/SL Order IDs: {tpsl_order_ids}")
                        update_trade_history(order_id, {'tpsl_order_ids': tpsl_order_ids, 'status': 'PlacedWithTPSL'})
                    else:
                        logger.error(f"Failed to place Position TP/SL for main order {exchange_order_id}.")
                        update_trade_history(order_id, {'status': 'PlacedMainOrder_TPSLFail'})
                else:
                    logger.info(f"No valid TP ({take_profit_price}) or SL ({stop_loss_price}) prices provided. Skipping TP/SL placement for main order {exchange_order_id}.")
                    update_trade_history(order_id, {'status': 'PlacedMainOrder_NoTPSLNeeded'}) # Update status to reflect no TP/SL was attempted
                
                return exchange_order_id # Return the main order's exchange ID
            else:
                logger.error(f"Order placement succeeded but no orderId in response data: {order_data}")
                save_trade_history(order_id, side, size, entry_price, stop_loss_price, take_profit_price, "PlaceFailed_NoID", llm_analysis, indicator_data, exchange_order_id=None)
                return None
        else:
            error_msg = result.get('msg', 'Unknown error') if isinstance(result, dict) else str(result)
            logger.error(f"Failed to place order: {error_msg}")
            save_trade_history(order_id, side, size, entry_price, stop_loss_price, take_profit_price, f"PlaceFailed_{error_msg[:50]}", llm_analysis, indicator_data, exchange_order_id=None) # Save truncated error
            return None

    except BitgetAPIException as e:
        logger.error(f"API Exception placing order: {e}")
        save_trade_history(order_id, side, size, entry_price, stop_loss_price, take_profit_price, f"PlaceFailed_APIExc_{e.message[:50]}", llm_analysis, indicator_data, exchange_order_id=None)
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing order: {e}", exc_info=True)
        save_trade_history(order_id, side, size, entry_price, stop_loss_price, take_profit_price, f"PlaceFailed_Exc_{str(e)[:50]}", llm_analysis, indicator_data, exchange_order_id=None)
        return None


async def run_trading_cycle(rest_client, ws_client):
    """Executes one cycle of fetching data, analysis, and potential trading."""
    logger.info("--- Starting Trading Cycle ---")
    try:
        # 0. Check for existing open position for the TARGET_INSTRUMENT
        logger.info(f"Checking for existing position for {TARGET_INSTRUMENT} before proceeding...")
        current_position = await get_current_position(rest_client, TARGET_INSTRUMENT)
        # Ensure 'total' exists and can be converted to float before checking if it's non-zero
        if current_position:
            position_size_str = current_position.get('total')
            if position_size_str is not None:
                try:
                    position_size = float(position_size_str)
                    if position_size != 0:
                        avg_entry_price = current_position.get('avgPx', 'N/A')
                        hold_side = current_position.get('holdSide', 'N/A') # Added to log side
                        unrealized_pnl = current_position.get('upl', 'N/A') # Added to log UPL
                        logger.info(f"Position already open for {TARGET_INSTRUMENT}: Side={hold_side}, Size={position_size}, EntryPx={avg_entry_price}, UPL={unrealized_pnl}. Skipping LLM analysis and new trade placement.")
                        logger.debug(f"Full details of open position: {current_position}") # Optional: more details
                        return # Exit the trading cycle early
                    else:
                        logger.info(f"Position found for {TARGET_INSTRUMENT}, but size is 0. Proceeding with trading cycle.")
                except ValueError:
                    logger.warning(f"Could not parse position size '{position_size_str}' for {TARGET_INSTRUMENT}. Assuming no open position and proceeding.")
            else:
                logger.info(f"Position data for {TARGET_INSTRUMENT} does not contain 'total' field. Assuming no open position and proceeding.")
        else:
            logger.info(f"No existing position found for {TARGET_INSTRUMENT}. Proceeding with trading cycle.")

        # 1. Fetch News
        news_items = await fetch_news()



        # 2. Get Recent Candles
        # Make a copy to avoid modification during iteration if WS updates
        current_candles = list(candle_data_store)
        if len(current_candles) < 2:
            logger.warning("Not enough candle data available yet.")
            return
        latest_candle = current_candles[-1]
        latest_close_price = latest_candle.get('c')
        if not latest_close_price:
            logger.error("Latest candle data is missing closing price.")
            return

        # 3. Calculate Indicators
        indicator_data = calculate_indicators(current_candles)
        # If indicators can't be calculated, we might still proceed with news/price action
        # or wait. For now, we proceed but log the absence.
        if not indicator_data:
            logger.warning("Proceeding without technical indicators for this cycle.")

        # 4. Analyze Trade History for Learning Insights
        learning_insights = await analyze_trade_history_and_learn(rest_client, ws_client) # Returns a dict or {} if no insights
        if learning_insights is None: # Ensure it's always a dict for the LLM prompt
            learning_insights = {}
        logger.info(f"Learning Insights: {learning_insights}")

        # Fetch account equity before LLM analysis as it might be needed by the LLM
        account_equity = await get_account_equity(rest_client)
        if account_equity is None:
            logger.warning("Proceeding with LLM analysis without account equity information.")
            # Decide if you want to return or proceed with a default/None equity
            # For now, we'll pass None, and get_llm_analysis should handle it

        # 5. Get LLM Analysis
        analysis_str = await get_llm_analysis(news_items, current_candles[-10:], indicator_data, account_equity, learning_insights)
        if not analysis_str:
            logger.error("Failed to get LLM analysis string.")
            return

        try:
            if isinstance(analysis_str, dict):
                analysis = analysis_str  # Already a dict, use directly
            elif isinstance(analysis_str, (str, bytes, bytearray)):
                analysis = json.loads(analysis_str)
            else:
                logger.error(f"LLM analysis data is not a string or dict. Type: {type(analysis_str)}, Content: {analysis_str}")
                return

            if not isinstance(analysis, dict):
                logger.error(f"LLM analysis did not result in a dictionary. Parsed type: {type(analysis)}, Content: {analysis_str}")
                return
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM analysis JSON: {e}. Raw response: {analysis_str}")
            return

        logger.info(f"LLM Analysis Result: Signal={analysis.get('signal')}, Confidence={analysis.get('confidence')}, Justification={analysis.get('justification')}")

        # 5. Decision Making & Order Placement
        signal = analysis.get('signal')
        confidence = analysis.get('confidence', 'Low') # Default to Low if missing
        volatility = analysis.get('volatility', 'Medium') # Default to Medium if missing

        # Basic confidence threshold (adjust as needed)
        if confidence in ['Low']:
            logger.info(f"Skipping trade due to low confidence ('{confidence}'). Signal was: {signal}")
            # Still calculate sleep interval based on analysis before returning
        else:
            # Account equity is already fetched before LLM analysis. Use that value.
            # If account_equity was None and we proceeded, current_equity will also be None here.
            current_equity = account_equity # Use the already fetched equity
            if current_equity is None:
                logger.error("Cannot place order: Account equity was not available.")
                return

            # Use latest close price as entry price proxy for calculations
            entry_price_proxy = latest_close_price

            if signal and signal.lower() == "long":
                logger.info("Executing LONG trade based on LLM analysis.")
                await place_order(
                    rest_client, 
                    'buy', 
                    current_equity, 
                    entry_price_proxy, 
                    analysis, 
                    indicator_data,
                    leverage=analysis.get('recommended_leverage'),
                    tp_percent=analysis.get('recommended_tp_percent'),
                    sl_percent=analysis.get('recommended_sl_percent')
                )
            elif signal and signal.lower() == "short":
                logger.info("Executing SHORT trade based on LLM analysis.")
                await place_order(
                    rest_client, 
                    'sell', 
                    current_equity, 
                    entry_price_proxy, 
                    analysis, 
                    indicator_data,
                    leverage=analysis.get('recommended_leverage'),
                    tp_percent=analysis.get('recommended_tp_percent'),
                    sl_percent=analysis.get('recommended_sl_percent')
                )
            elif signal and signal.lower() == "hold":
                logger.info("LLM signal is HOLD. No trade executed.")
            else:
                logger.warning(f"Unrecognized signal from LLM: {signal}")

        logger.info("--- Finished Trading Cycle ---")

    except asyncio.CancelledError:
            logger.info("Trading cycle cancelled, propagating cancellation...")
            raise # Re-raise the CancelledError to be caught by the main loop
    except BitgetAPIException as e:
        logger.error(f"Bitget API error in trading cycle: {e}", exc_info=True)
        await asyncio.sleep(60)  # Wait longer after API errors
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}", exc_info=True)
        # Wait longer after an unexpected error
        await asyncio.sleep(60)


async def get_current_position(rest_client, instrument_id):
    """Fetches current position for a specific instrument_id using Bitget REST API v2."""
    try:
        # Corrected endpoint for fetching all positions, then filtering
        endpoint = BITGET_REST_POSITIONS_ENDPOINT 
        # Params for v2 all-position: productType is required. instId is optional for filtering at client side.
        params = {'productType': PRODUCT_TYPE_V2} # Use the full PRODUCT_TYPE_V2 directly
        
        logger.debug(f"Attempting to fetch all positions for productType: {params['productType']} via {endpoint}")
        response = rest_client.get(endpoint, params) # Assuming rest_client.get(url, params)
        
        # Check for success code (string '00000' or integer 0)
        code = response.get('code')
        if response and (code == '00000' or code == 0):
            logger.info(f"Full raw API response for get_current_position (productType: {params['productType']}): {response}") # Log raw response at INFO level
            positions = response.get('data', [])
            # logger.debug(f"Full positions response data for productType {params['productType']}: {positions}") # Original debug log
            if not positions:
                logger.info(f"No positions found in 'data' field for productType: {params['productType']}.")
                return None # No positions at all

            for pos in positions:
                logger.info(f"Processing position from API: {pos}") # Log each position at INFO level
                api_symbol = pos.get('symbol')
                # Detailed logging for comparison
                logger.info(f"Comparing: api_symbol='{api_symbol}' (type: {type(api_symbol)}) with instrument_id='{instrument_id}' (type: {type(instrument_id)}) - Are they equal? {api_symbol == instrument_id}")
                
                if api_symbol == instrument_id: 
                    logger.info(f"Match SUCCESSFUL for {instrument_id}")
                    pos_size_str = pos.get('total', '0') # 'total' is typically the field for position size
                    try:
                        pos_size = float(pos_size_str)
                        logger.info(f"Found position for {instrument_id}: Size = {pos_size}. Details: {pos}")
                        return pos # Return the full position details
                    except ValueError:
                        logger.error(f"Could not parse position size ('{pos_size_str}') to float for {instrument_id}.")
                        return None # Error parsing size
                else:
                    logger.info(f"Match FAILED for api_symbol='{api_symbol}' vs instrument_id='{instrument_id}'")
            logger.info(f"No specific position matching {instrument_id} found after iterating all positions in 'data' list (checked 'symbol' field).")
            return None # No position for the target instrument among those returned
        elif response: # If there's a response but the code is not success
            logger.warning(f"API call to fetch positions returned code: {response.get('code')}, msg: {response.get('msg')}")
            return None # Error fetching position
        else: # No response or unexpected format
            logger.error("Failed to fetch current positions: No response or unexpected format from API.")
            return None
    except Exception as e:
        logger.error(f"Exception in get_current_position for {instrument_id}: {e}", exc_info=True)
        return None

async def check_position_status_periodically(rest_client, check_interval_seconds=60):
    """Periodically checks if the target position is still open via REST as a fallback/double-check."""
    logger.info(f"[PeriodicCheck] Starting for {TARGET_INSTRUMENT}, interval: {check_interval_seconds}s.")
    while True:
        try:
            logger.info(f"[PeriodicCheck] Loop iteration started. Waiting for {check_interval_seconds}s...")
            await asyncio.sleep(check_interval_seconds)
            logger.info(f"[PeriodicCheck] Wait finished. Performing REST check for {TARGET_INSTRUMENT}...")
            
            position_details = await get_current_position(rest_client, TARGET_INSTRUMENT)
            
            if position_details is None:
                logger.info(f"[PeriodicCheck] No active position found for {TARGET_INSTRUMENT}. Setting RESTART_TRADING_CYCLE_FLAG.")
                RESTART_TRADING_CYCLE_FLAG.set()
            elif float(position_details.get('total', '0')) == 0:
                logger.info(f"[PeriodicCheck] Position {TARGET_INSTRUMENT} found with size 0. Setting RESTART_TRADING_CYCLE_FLAG.")
                RESTART_TRADING_CYCLE_FLAG.set()
            else:
                logger.info(f"[PeriodicCheck] Position {TARGET_INSTRUMENT} is still open. Size: {position_details.get('total')}. Flag not set.")
        except asyncio.CancelledError:
            logger.info("[PeriodicCheck] Task was cancelled. Exiting loop.")
            break # Exit the loop if the task is cancelled
        except Exception as e:
            logger.error(f"[PeriodicCheck] Error during periodic check: {e}", exc_info=True)
            # Decide if we should continue, break, or wait longer after an error
            logger.info("[PeriodicCheck] Waiting for 30 seconds after error before next attempt.")
            await asyncio.sleep(30) # Wait a bit before retrying to avoid spamming logs on persistent errors

async def main():
    """Main execution function with persistent connections and trading loop."""
    if not all([BITGET_API_KEY, BITGET_SECRET_KEY, BITGET_PASSPHRASE]):
        logger.error("Missing API credentials in .env file. Bot cannot start.")
        return

    rest_client = None
    ws_client_private = None
    ws_client_public = None
    periodic_check_task = None

    loop = asyncio.get_running_loop()

    try:
        # Instantiate REST client
        rest_client = BitgetApi(BITGET_API_KEY, BITGET_SECRET_KEY, BITGET_PASSPHRASE, use_server_time=True, base_url=c.API_URL)
        logger.info(f"REST Client Initialized with base URL: {rest_client.BASE_URL}")

        # Connect WebSockets
        ws_client_private, ws_client_public = await asyncio.gather(
            connect_private_websocket(loop, rest_client), # Pass rest_client
            connect_public_websocket(loop),
            return_exceptions=True # To handle potential connection errors gracefully
        )

        if isinstance(ws_client_private, Exception) or not ws_client_private:
            logger.error(f"Failed to connect private WebSocket: {ws_client_private}. Exiting.")
            return
        if isinstance(ws_client_public, Exception) or not ws_client_public:
            logger.warning(f"Failed to connect public WebSocket: {ws_client_public}. Proceeding without live candle data.")
            # Bot can continue, but analysis will lack live candles

        # Start the periodic position check as a background task
        periodic_check_task = asyncio.create_task(check_position_status_periodically(rest_client))
        logger.info("Periodic position check task started.")

        # Main trading loop
        while True:
            RESTART_TRADING_CYCLE_FLAG.clear()
            logger.info("--- Starting/Restarting Trading Cycle --- ")
            try:
                await run_trading_cycle(rest_client, ws_client_private)
                logger.info("run_trading_cycle completed.")
            except asyncio.CancelledError:
                logger.info("Trading cycle cancelled, will wait for restart flag or shutdown.")
                # If the cycle itself is cancelled, we might want to break or re-evaluate
            except Exception as cycle_exception:
                logger.error(f"Error during run_trading_cycle execution: {cycle_exception}", exc_info=True)
                logger.info("Waiting for 60 seconds after error in run_trading_cycle before checking restart flag.")
                await asyncio.sleep(60)
            
            logger.info("Trading cycle finished or errored. Waiting for RESTART_TRADING_CYCLE_FLAG to be set before next cycle...")
            await RESTART_TRADING_CYCLE_FLAG.wait()
            logger.info("RESTART_TRADING_CYCLE_FLAG is set. Proceeding with new trading cycle.")

    except (KeyboardInterrupt, asyncio.CancelledError) as shutdown_signal:
        logger.info(f"Shutdown signal ({type(shutdown_signal).__name__}) received...")
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
    finally:
        logger.info("Initiating shutdown sequence...")
        if periodic_check_task and not periodic_check_task.done():
            logger.info("Cancelling periodic position check task...")
            periodic_check_task.cancel()
            try:
                await periodic_check_task
            except asyncio.CancelledError:
                logger.info("Periodic position check task successfully cancelled.")
            except Exception as e_cancel_check:
                logger.error(f"Error cancelling periodic check task: {e_cancel_check}")

        logger.info("Closing WebSocket connections...")
        # Attempt to close clients - proper library support needed for clean thread shutdown
        # For BitgetWsClientAsync, we might need a specific close method or rely on task cancellation
        # if they are started as tasks.
        ws_tasks_to_await = []
        for client, name in [(ws_client_private, "PrivateWS"), (ws_client_public, "PublicWS")]:
            if client and hasattr(client, 'stop'): # Assuming a 'stop' method for graceful shutdown
                try:
                    logger.info(f"Stopping {name} client...")
                    # If client.stop() is async, create a task or await directly if not blocking
                    # For now, let's assume it's a synchronous call or handled by its own task cancellation
                    # client.stop() # This is hypothetical, replace with actual stop mechanism
                except Exception as e_ws_stop:
                    logger.error(f"Error stopping {name}: {e_ws_stop}")
            elif client and hasattr(client, '_task') and client._task and not client._task.done(): # If client runs as a task
                logger.info(f"Cancelling {name} client task...")
                client._task.cancel()
                ws_tasks_to_await.append(client._task)
        
        if ws_tasks_to_await:
            results = await asyncio.gather(*ws_tasks_to_await, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, asyncio.CancelledError):
                    logger.info(f"WebSocket client task {i+1} successfully cancelled.")
                elif isinstance(result, Exception):
                    logger.error(f"Error during WebSocket client task {i+1} cancellation/shutdown: {result}")

        logger.info("Trading bot stopped.")


if __name__ == "__main__":
    logger.info("Starting main trading application...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in __main__. Application is shutting down.")
    except asyncio.CancelledError:
        logger.info("Main application run was cancelled in __main__. Application is shutting down.")
    except Exception as e:
        logger.error(f"Unhandled exception from asyncio.run(main()) in __main__: {e}", exc_info=True)
    finally:
        logger.info("Application shutdown complete from __main__.")

TRADE_EVENT_LOG_FILE = "trade_events.log" # Added for detailed event logging

def log_trade_event(event_data):
    """Logs a trade-related event to the main logger and a dedicated event log file."""
    try:
        # Add a timestamp to the event data
        event_data_with_ts = {
            'timestamp': datetime.datetime.now().isoformat(),
            **event_data
        }

        # Log a summary to the main logger
        event_type = event_data.get('event', 'unknown_event')
        cl_ord_id = event_data.get('clOrdId') or event_data.get('client_order_id')
        log_summary = f"Trade Event: {event_type}"
        if cl_ord_id:
            log_summary += f" (ID: {cl_ord_id})"
        
        # Check for exception details to log more informatively
        if 'exception' in event_data:
            logger.error(f"{log_summary} - Exception: {event_data['exception']}", exc_info=False) # exc_info=False as we already have the string
        elif 'error_msg' in event_data:
            logger.error(f"{log_summary} - Error: {event_data['error_msg']} (Code: {event_data.get('error_code', 'N/A')})")
        else:
            logger.info(log_summary)

        # Append the full event data to the trade_events.log file
        # Ensure the directory exists before writing
        event_log_dir = os.path.dirname(TRADE_EVENT_LOG_FILE)
        if event_log_dir: # Check if dirname returned a non-empty string
            os.makedirs(event_log_dir, exist_ok=True)

        with open(TRADE_EVENT_LOG_FILE, 'a') as f:
            json.dump(event_data_with_ts, f)
            f.write('\n') # Newline for each JSON entry

    except Exception as e:
        logger.error(f"Failed to log trade event: {event_data.get('event', 'unknown_event')}. Error: {e}", exc_info=True)


def save_trade_history(client_order_id, side, size, entry_price, stop_loss_price, take_profit_price, status, llm_analysis=None, indicators=None, exchange_order_id=None):
    """Saves trade details and context to a JSON file."""
