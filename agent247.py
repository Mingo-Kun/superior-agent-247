import asyncio
import json
import os
import logging
import time
import datetime
import feedparser
import httpx
import collections # For deque
from dotenv import load_dotenv

# Configure logging before any other imports
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    from bitget.bitget_ws_client import BitgetWsClient, SubscribeReq
    from bitget.exceptions import BitgetAPIException
    from bitget.bitget_api import BitgetApi

# --- Configuration ---
load_dotenv()

# Logging Setup (Reconfiguring to ensure it takes effect)
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) # Re-get logger after reconfig

# API Keys and Credentials
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY","")
BITGET_API_KEY = os.getenv("BITGET_DEMO_API_KEY","")
BITGET_SECRET_KEY = os.getenv("BITGET_DEMO_API_SECRET","")
BITGET_PASSPHRASE = os.getenv("BITGET_DEMO_API_PASSPHRASE","")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free")

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

# Trading Parameters
TARGET_INSTRUMENT = "SBTCSUSDT"  # Demo BTC/USDT futures
# Define Product Type for REST API (V2 uses 'productType' for mix endpoints)
# Common values: 'USDT-FUTURES', 'COIN-FUTURES', 'SUSDT-FUTURES' (for demo)
PRODUCT_TYPE_V2 = "SUSDT-FUTURES"
INST_TYPE_V2 = PRODUCT_TYPE_V2 # Keep for WS compatibility if needed elsewhere

CANDLE_CHANNEL = "candle1H"      # Candle interval (e.g., 1 hour)
MAX_CANDLES = 50                 # Max number of candles to store

# Global storage for candle data
# Using deque for efficient fixed-size storage
candle_data_store = collections.deque(maxlen=MAX_CANDLES)

# --- WebSocket Handlers ---

async def handle_private_message(message):
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
                     logger.info(f"[PrivateWS] Position Update: Inst={pos.get('instId')}, Side={pos.get('holdSide')}, Total={pos.get('total')}, Avail={pos.get('available')}, AvgPx={pos.get('openPriceAvg')}, UPL={pos.get('unrealizedPL')}")
            elif channel == 'orders':
                 for order in data_list:
                    logger.info(f"[PrivateWS] Order Update: ID={order.get('orderId')}, Status={order.get('status')}, FilledSz={order.get('accBaseVolume')}")
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
    logger.error(f"[{client_name}] Connection/Library Error: {err}")

# --- Core Logic ---

async def fetch_news():
    """Fetch news from RSS feeds"""
    news_items = []
    fallback_feeds = [
        "https://news.bitcoin.com/feed/",
        "https://cryptopotato.com/feed/"
    ]
    logger.info("Fetching news from RSS feeds...")
    async with httpx.AsyncClient(timeout=20.0) as client:
        for url in RSS_FEEDS:
            for attempt in range(3):  # Retry up to 3 times
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    feed = feedparser.parse(response.text)
                    news_items.extend({
                        "title": entry.title,
                        "link": entry.link,
                        "published": entry.get("published", "")
                    } for entry in feed.entries[:5])
                    await asyncio.sleep(0.2 * (attempt + 1))  # Exponential backoff
                    break  # Success, exit retry loop
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error fetching {url} (attempt {attempt + 1}): Status {e.response.status_code}")
                    if attempt == 2:  # Last attempt failed, try fallback
                        RSS_FEEDS.extend(fallback_feeds)
                except httpx.RequestError as e:
                    logger.error(f"Request error fetching {url} (attempt {attempt + 1}): {e}")
                except Exception as e:
                    logger.error(f"Error parsing feed {url} (attempt {attempt + 1}): {e}")
    logger.info(f"Fetched {len(news_items)} news items.")
    return news_items

async def get_llm_analysis(news_data, historical_candles):
    """Get trade signal from LLM analysis"""
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not configured")
        return None

    news_titles = [n['title'] for n in news_data[:5]]
    # Format candles for readability in prompt if needed
    formatted_candles = [{"T": c["ts"], "O": c["o"], "H": c["h"], "L": c["l"], "C": c["c"], "V": c["v"]} for c in historical_candles]

    # Add context about the symbol
    symbol_context = f"{TARGET_INSTRUMENT} is the demo trading symbol for Bitcoin (BTC) USDT-margined perpetual futures."

    prompt = f"""
    Analyze the following market data for {TARGET_INSTRUMENT} ({symbol_context}):

    Recent News Headlines:
    {json.dumps(news_titles, indent=2)}

    Recent {CANDLE_CHANNEL} Candles (Timestamp, Open, High, Low, Close, Volume, Turnover):
    {json.dumps(formatted_candles, indent=2)}

    Based ONLY on the provided news and candle data, perform a brief technical analysis with the most fit chart indicators e.g MA,EMA,BOLL,SAR and AVL. and sub-chart e.g VOL,MACD,KDJ,RSI,ROC,CCI and WR. and do sentiment analysis and price action.
    Output ONLY JSON with the following structure:
    {{
      "signal": "Long" or "Short",
      "justification": "Brief reasoning based ONLY on the provided news and candle data. Mention key price levels or patterns if observed.",
      "confidence": "High" or "Medium" or "Low"
    }}
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AITradingBot"
    }

    logger.info("Requesting LLM analysis...")
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            response.raise_for_status()
            result = response.json()

            if not result.get('choices'):
                 logger.error("LLM response missing 'choices'.")
                 return None

            content = result['choices'][0]['message']['content']
            logger.debug(f"Raw LLM Response Content: {content}")

            try:
                content_cleaned = content.strip()
                if content_cleaned.startswith("```json"):
                    content_cleaned = content_cleaned[7:]
                if content_cleaned.endswith("```"):
                    content_cleaned = content_cleaned[:-3]
                content_cleaned = content_cleaned.strip()

                analysis_json = json.loads(content_cleaned)
                if 'signal' not in analysis_json or 'justification' not in analysis_json or 'confidence' not in analysis_json:
                     logger.error(f"LLM JSON missing required keys. Parsed: {analysis_json}")
                     return None
                logger.info("LLM analysis received successfully.")
                return analysis_json

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}\nContent: {content}")
                # Attempt fallback extraction (optional, can be removed if too unreliable)
                try:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        extracted_json_str = content[json_start:json_end]
                        analysis_json = json.loads(extracted_json_str)
                        if 'signal' in analysis_json and 'justification' in analysis_json and 'confidence' in analysis_json:
                            logger.warning("Successfully parsed LLM JSON using fallback extraction.")
                            return analysis_json
                except Exception:
                    pass # Fallback failed
                return None # Parsing failed
            except Exception as e:
                 logger.error(f"Unexpected error processing LLM response: {e}\nContent: {content}")
                 return None

    except httpx.RequestError as e:
        logger.error(f"LLM API request error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during LLM API call: {e}", exc_info=True)
        return None

async def connect_private_websocket(loop):
    """Initialize and connect PRIVATE WebSocket client"""
    if not all([BITGET_API_KEY, BITGET_SECRET_KEY, BITGET_PASSPHRASE]):
        logger.error("[PrivateWS] Connection requires API credentials.")
        return None
    try:
        # --- Wrapper for thread-safe async callback ---
        def private_message_handler_wrapper(message):
            asyncio.run_coroutine_threadsafe(handle_private_message(message), loop)

        client = BitgetWsClient(BITGET_WSS_PRIVATE_URL, need_login=True) \
            .api_key(BITGET_API_KEY) \
            .api_secret_key(BITGET_SECRET_KEY) \
            .passphrase(BITGET_PASSPHRASE) \
            .error_listener(lambda err: asyncio.run_coroutine_threadsafe(handle_ws_error("PrivateWS", err), loop)) \
            .listener(private_message_handler_wrapper) # Use the sync wrapper

        client._BitgetWsClient__loop = loop # Ensure loop is set
        client.build() # Starts connection in a thread

        await asyncio.sleep(3) # Allow time for connection/auth attempt

        if not client.has_connect():
             logger.error("[PrivateWS] Failed to connect initially.")
             return None

        # Subscribe to account and position updates
        channels = [
            SubscribeReq(INST_TYPE_V2, "account", "default"),
            SubscribeReq(INST_TYPE_V2, "positions", "default")
            # SubscribeReq(INST_TYPE_V2, "orders", "default") # Optional
        ]
        # Assuming subscribe uses the listener set above.
        client.subscribe(channels) # Rely on the default listener set earlier

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
        client = BitgetWsClient(BITGET_WSS_PUBLIC_URL, need_login=False) \
            .error_listener(lambda err: asyncio.run_coroutine_threadsafe(handle_ws_error("PublicWS", err), loop)) \
            .listener(public_message_handler_wrapper) # Use the sync wrapper

        client._BitgetWsClient__loop = loop
        client.build()

        await asyncio.sleep(2) # Allow time for connection

        if not client.has_connect():
             logger.error("[PublicWS] Failed to connect initially.")
             return None

        # Subscribe to candle channel
        channels = [
            SubscribeReq(INST_TYPE_V2, CANDLE_CHANNEL, TARGET_INSTRUMENT)
        ]
        # Assuming subscribe uses the listener set earlier.
        client.subscribe(channels) # Rely on the default listener set earlier

        logger.info(f"[PublicWS] Connection initiated and subscribed to {CANDLE_CHANNEL} for {TARGET_INSTRUMENT}.")
        return client

    except Exception as e:
        logger.error(f"[PublicWS] Failed to initialize: {e}", exc_info=True)
        return None


async def place_trade_via_rest(rest_client, instrument, signal, size, order_type="market"):
    """Place trade via REST API"""
    trade_id = f"bot_{int(time.time()*1000)}"
    api_side = None # Initialize

    # Map signal to API side parameter
    pos_side = None # Initialize posSide
    if signal.lower() == "long":
        api_side = "buy"
        pos_side = "long"
    elif signal.lower() == "short":
        api_side = "sell"
        pos_side = "short"
    else:
        logger.error(f"Invalid signal '{signal}' for placing trade.")
        return False

    logger.debug(f"place_trade_via_rest received signal: '{signal}', mapped to api_side: '{api_side}'")

    params = {
        "instId": instrument,
        "productType": PRODUCT_TYPE_V2, # Add the missing productType
        "symbol": "SBTCSUSDT",  # Explicitly set the correct symbol for the pair
        "marginCoin": "SUSDT",  # Actual margin coin (not product type)
        "marginMode": "isolated",
        "posSide": pos_side, # Add posSide based on signal
        "tradeSide": "open", # Explicitly state intent to open position
        "side": api_side,
        "orderType": "market",  # Fixed value for market orders
        "size": str(size),
        "clientOid": trade_id,
        "timeInForce": "GTC",  # Good Till Cancelled
        "leverage": "20"  # Default leverage
    }

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

TRADE_LOG_FILE = "trade_history.json"

def log_trade_event(event_data):
    """Log trade events to JSON file"""
    event_data['timestamp'] = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        try:
            with open(TRADE_LOG_FILE, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        history.append(event_data)
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to log trade event: {e}")

async def run_trading_cycle(rest_client):
    """Execute one full trading cycle"""
    global candle_data_store
    logger.info("--- Starting new trading cycle ---")

    # Fetch news
    news_data = await fetch_news()

    # Get current candle data (convert deque to list for analysis)
    current_candles = list(candle_data_store)
    if not current_candles:
        logger.warning("No candle data available yet for analysis.")
        # Optionally wait or skip cycle if candles are essential
        # await asyncio.sleep(10) # Wait a bit longer for candles
        # return

    # Get LLM analysis
    analysis = await get_llm_analysis(news_data, current_candles)

    if not analysis:
        logger.warning("No analysis received from LLM.")
        return

    signal = analysis.get('signal')
    confidence = analysis.get('confidence', 'Low')
    logger.info(f"LLM Analysis: Signal={signal}, Confidence={confidence}, Justification={analysis.get('justification')}")

    # --- Trade Execution Logic ---
    if signal in ["Long", "Short"] and confidence in ["High", "Medium"]:
        logger.info(f"Executing trade based on signal: {signal}")
        # **TODO: Implement proper risk management and size calculation here**
        logger.debug(f"Analysis details - Signal: {signal}, Confidence: {confidence}, Justification: {analysis.get('justification')}")
        # Example: Calculate size based on equity and risk %
        # equity = get_current_equity() # Need a way to get equity (e.g., from WS or REST)
        # if equity:
        #     risk_amount = equity * (RISK_PER_TRADE_PERCENT / 100.0) # Need RISK_PER_TRADE_PERCENT defined
        #     # Need entry price, stop loss price to calculate size
        #     # trade_size = calculate_position_size(risk_amount, entry_price, stop_loss_price)
        # else:
        #     logger.error("Cannot calculate trade size without equity info.")
        #     trade_size = 0.001 # Fallback to minimal size

        # **INCREASED DEMO SIZE FOR TESTING - ADJUST AS NEEDED**
        trade_size = 0.02  # Using slightly larger placeholder size for demo

        if trade_size > 0:
            success = await place_trade_via_rest(
                rest_client,
                TARGET_INSTRUMENT,
                signal, # Pass 'Long' or 'Short'
                trade_size,
                "MARKET" # Use uppercase for Bitget API order type
            )
            if success:
                 logger.info(f"Trade {signal} for {trade_size} {TARGET_INSTRUMENT} submitted.")
            else:
                 logger.error(f"Failed to submit trade {signal} for {TARGET_INSTRUMENT}.")
        else:
            logger.warning("Trade size calculated as 0, skipping trade.")
    else:
        logger.info(f"No trade executed. Signal: {signal}, Confidence: {confidence}")

async def main():
    """Main execution function"""
    if not all([BITGET_API_KEY, BITGET_SECRET_KEY, BITGET_PASSPHRASE]):
        logger.error("Missing API credentials in .env file")
        return

    rest_client = None
    ws_client_private = None
    ws_client_public = None

    loop = asyncio.get_running_loop() # Get the main event loop

    try:
        # Instantiate REST client
        rest_client = BitgetApi(BITGET_API_KEY, BITGET_SECRET_KEY, BITGET_PASSPHRASE, use_server_time=True)
        logger.info("REST Client Initialized.")

        # Connect WebSockets concurrently, passing the loop
        ws_client_private, ws_client_public = await asyncio.gather(
            connect_private_websocket(loop),
            connect_public_websocket(loop)
        )

        if not ws_client_private:
            logger.error("Failed to connect private WebSocket. Exiting.")
            # Decide if bot can run without private WS
            return
        if not ws_client_public:
            logger.warning("Failed to connect public WebSocket. Proceeding without live candle data.")
            # Bot can continue, but analysis will lack live candles

        logger.info("Starting trading bot loop...")
        while True:
            # Pass only the REST client, cycle fetches candles from global store
            await run_trading_cycle(rest_client)
            logger.info(f"Cycle finished. Waiting for 1 hour...")
            await asyncio.sleep(3600)  # Run every hour

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received...")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
    finally:
        logger.info("Closing WebSocket connections...")
        # Attempt to close clients - proper library support needed for clean thread shutdown
        for client, name in [(ws_client_private, "PrivateWS"), (ws_client_public, "PublicWS")]:
             if client and hasattr(client, '_BitgetWsClient__close'):
                 try:
                     # Direct close from main thread might be problematic
                     logger.warning(f"Attempting to close {name} client (may require manual process stop).")
                     # client._BitgetWsClient__close() # This likely won't work reliably across threads
                 except Exception as e:
                     logger.error(f"Error trying to close {name}: {e}")
        logger.info("Trading bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
