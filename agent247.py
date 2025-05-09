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
import os
import logging
import time
import datetime
import feedparser
import httpx
import collections # For deque
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
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-30b-a3b:free")

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

# Trading Parameters
TARGET_INSTRUMENT = "SBTCSUSDT"  # Demo BTC/USDT futures
# Define Product Type for REST API (V2 uses 'productType' for mix endpoints)
# Common values: 'USDT-FUTURES', 'COIN-FUTURES', 'SUSDT-FUTURES' (for demo)
PRODUCT_TYPE_V2 = "SUSDT-FUTURES" # Use 'umcbl' for USDT-margined futures in V2 API
INST_TYPE_V2 = PRODUCT_TYPE_V2 # Keep for WS compatibility if needed elsewhere

CANDLE_CHANNEL = "candle1H"      # Candle interval (e.g., 1 hour)
MAX_CANDLES = 50                 # Max number of candles to store
# TRADE_SIZE = 0.02                # Fixed trade size (Replaced by dynamic sizing)
STOP_LOSS_PERCENT = 0.01         # 1% stop loss
TAKE_PROFIT_PERCENT = 0.02       # 2% take profit
RISK_PER_TRADE_PERCENT = 0.01    # Risk 1% of equity per trade
MIN_TRADE_SIZE = 0.001           # Minimum allowed trade size (adjust based on exchange)
TRADING_FEE_RATE = 0.0006        # Trading fee rate (e.g., 0.06% = 0.0006). Adjust based on your exchange.
TRADE_HISTORY_FILE = "trade_history.json" # File to store trade history

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
                                         # Need the exit price. This isn't directly in the position update when size is 0.
                                         # We might need to infer it from the last order fill that closed it, 
                                         # or rely on the unrealizedPL just before closure (less accurate).
                                         # For simplicity, let's assume the last UPL before closure reflects PnL.
                                         # A more robust solution involves tracking closing order fills.
                                         
                                         # Placeholder: Use last known UPL as PnL. Needs improvement.
                                         # Ideally, capture the closing fill price from the 'orders' stream.
                                         last_upl = float(upl_str) # UPL just before/during closure
                                         exit_price_approx = float(avg_entry_price_str) + (last_upl / abs(found_trade.get('filled_size', 1))) if found_trade.get('filled_size') else None

                                         update_data = {
                                             'status': 'Closed',
                                             'exit_price': exit_price_approx, # This is an approximation!
                                             'pnl': last_upl # Using UPL as proxy for realized PnL
                                         }
                                         logger.info(f"Updating trade {found_trade.get('client_order_id')} to Closed. Approx Exit: {exit_price_approx}, PnL: {last_upl}")
                                         # Update using the client_order_id stored in the found trade
                                         update_trade_history(found_trade.get('client_order_id'), update_data)
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
async def get_llm_analysis(news_data, historical_candles, indicator_data, learning_insights=None):
#     """Get trade signal from LLM analysis, incorporating calculated indicators and learning insights."""
    # Ensure learning_insights is a dictionary, even if empty, for consistent prompt formatting
    if learning_insights is None:
        learning_insights = {}
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not configured")
        return None

    news_titles = [n['title'] for n in news_data[:5]]
    # Format candles for readability in prompt
    formatted_candles = [
        {"T": c["ts"], "O": c["o"], "H": c["h"], "L": c["l"], "C": c["c"], "V": c["v"]}
        for c in historical_candles
    ]

    # Format indicators for prompt
    formatted_indicators = json.dumps(indicator_data, indent=2) if indicator_data else "Not available"

    # Add context about the symbol
    symbol_context = f"{TARGET_INSTRUMENT} is the demo trading symbol for Bitcoin (BTC) USDT-margined perpetual futures."

    # Format learning insights for prompt
    formatted_learning = json.dumps(learning_insights, indent=2) if learning_insights else "No specific learning insights available."

    prompt = f"""
    Analyze the following market data for {TARGET_INSTRUMENT} ({symbol_context}):

    Recent News Headlines (Consider potential sentiment impact):
    {json.dumps(news_titles, indent=2)}

    Recent {CANDLE_CHANNEL} Candles (Latest {len(formatted_candles)} periods. T=Timestamp, O=Open, H=High, L=Low, C=Close, V=Volume):
    {json.dumps(formatted_candles, indent=2)}

    Calculated Technical Indicators (Latest values):
    {formatted_indicators}

    Learning Insights from Past Trades:
    {formatted_learning}
    Please consider these insights when forming your justification and signal.

    Insticator Key:
    - SMA_20: 20-period Simple Moving Average
    - RSI_14: 14-period Relative Strength Index
    - MACD_12_26_9: MACD Line
    - MACDh_12_26_9: MACD Histogram (MACD Line - Signal Line)
    - MACDs_12_26_9: MACD Signal Line
    - BBL_20_2.0: Lower Bollinger Band (20-period, 2 std dev)
    - BBM_20_2.0: Middle Bollinger Band (SMA_20)
    - BBU_20_2.0: Upper Bollinger Band (20-period, 2 std dev)

    Instructions:
    0.  **Learning Insights:** Consider the provided performance patterns based on past trades. How might this influence the current decision?
    1.  **Sentiment Analysis:** Briefly assess the overall sentiment conveyed by the news headlines (Positive, Negative, Neutral).
    2.  **Technical Analysis:**
        *   Interpret **SMA_20**: Is the price above/below the SMA, suggesting trend direction?
        *   Interpret **RSI_14**: Is it overbought (>70), oversold (<30), or neutral? Any divergences?
        *   Interpret **MACD**: Is the MACD line above/below the signal line (MACDs)? Is the histogram (MACDh) positive/negative and growing/shrinking, indicating momentum?
        *   Interpret **Bollinger Bands (BBL, BBM, BBU)**: Is the price near the upper/lower band (potential reversal or continuation)? Are the bands widening (increasing volatility) or narrowing (decreasing volatility)?
        *   Identify the short-term trend based on recent price action in the candles, considering the indicators.
        *   Note any significant price levels (support/resistance) suggested by the candle data or indicator levels (e.g., Bollinger Bands).
        *   Comment on volume patterns if they appear significant.
    4.  **Synthesize & Signal:** Based *only* on the sentiment, technical analysis (including all indicators), and learning insights above, determine a trade signal.
    5.  **Confidence:** Assign a confidence level based on the clarity and convergence of the analysis.
    6.  **Volatility Assessment:** Based on Bollinger Band width and recent price action, assess the current market volatility.

    Output ONLY JSON with the following structure:
    {{
      "sentiment": "Positive" or "Negative" or "Neutral",
      "technical_summary": "Brief summary integrating trend, key levels, volume, SMA, RSI, MACD, and Bollinger Bands observations.",
      "signal": "Long" or "Short" or "Hold",
      "justification": "Concise reasoning linking sentiment, technical analysis (including all indicators), and learning insights to the signal.",
      "confidence": "High" or "Medium" or "Low",
      "volatility": "High" or "Medium" or "Low"
    }}
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AITradingBot"
    }

    logger.info("Requesting LLM analysis with expanded indicators...")

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

    # Process the response content outside the async with block
    content_cleaned = content.strip()
    if content_cleaned.startswith("```json"):
        content_cleaned = content_cleaned[7:]
    if content_cleaned.endswith("```"):
        content_cleaned = content_cleaned[:-3]
    content_cleaned = content_cleaned.strip()

    try:
        analysis_json = json.loads(content_cleaned)
        # Validate the new structure
        required_keys = ["sentiment", "technical_summary", "signal", "justification", "confidence", "volatility"]
        if not all(key in analysis_json for key in required_keys):
            logger.error(f"LLM JSON missing required keys. Expected: {required_keys}. Parsed: {analysis_json}")
            return None
        logger.info("LLM analysis received successfully.")
        return analysis_json

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}\nContent: {content}")
        # Attempt fallback extraction
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                extracted_json_str = content[json_start:json_end]
                analysis_json = json.loads(extracted_json_str)
                required_keys = ["sentiment", "technical_summary", "signal", "justification", "confidence", "volatility"]
                if all(key in analysis_json for key in required_keys):
                    logger.warning("Successfully parsed LLM JSON using fallback extraction.")
                    return analysis_json
        except Exception as fallback_e:
            logger.error(f"Fallback JSON extraction failed: {fallback_e}")
            pass # Fallback failed
        return None # Parsing and fallback failed
    except Exception as e:
         logger.error(f"Unexpected error processing LLM response: {e}\nContent: {content}")
         return None

    except httpx.RequestError as e:
        logger.error(f"LLM API request error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during LLM API call: {e}", exc_info=True)
        return None

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

async def connect_private_websocket(loop):
    """Initialize and connect PRIVATE WebSocket client"""
    if not all([BITGET_API_KEY, BITGET_SECRET_KEY, BITGET_PASSPHRASE]):
        logger.error("[PrivateWS] Connection requires API credentials.")
        return None

    try:
        # --- Wrapper for thread-safe async callback ---
        def private_message_handler_wrapper(message):
            asyncio.run_coroutine_threadsafe(handle_private_message(message), loop)

        # Initialize client with all necessary parameters
        client = BitgetWsClientAsync(BITGET_WSS_PRIVATE_URL, 
                                   api_key=BITGET_API_KEY, 
                                   api_secret_key=BITGET_SECRET_KEY, 
                                   passphrase=BITGET_PASSPHRASE, 
                                   listener=private_message_handler_wrapper, # Use the sync wrapper
                                   error_listener=lambda err: asyncio.run_coroutine_threadsafe(handle_ws_error("PrivateWS", err), loop))

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
    """Place position TP/SL via REST API using /api/v2/mix/order/place-pos-tpsl."""
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

TRADE_LOG_FILE = "trade_history.json"
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
            json.dump(history, f, indent=4)
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
            json.dump(history, f, indent=4)
        logger.info(f"Successfully updated trade history for {order_id}.")

    except Exception as e:
        logger.error(f"Error updating trade history for {order_id}: {e}", exc_info=True)


# --- Learning Mechanism --- 

def analyze_trade_history_and_learn():
    """Analyzes past trades to identify patterns and generate learning insights."""
    logger.info("--- Analyzing Trade History for Learning --- ")
    if not os.path.exists(TRADE_HISTORY_FILE):
        logger.warning("Trade history file not found. Cannot perform learning analysis.")
        return None # Or return empty insights

    try:
        history = []
        if os.path.exists(TRADE_LOG_FILE):
            with open(TRADE_LOG_FILE, 'r') as f:
                try:
                    history = json.load(f)
                    if not isinstance(history, list):
                        logger.warning(f"{TRADE_LOG_FILE} does not contain a list. Reinitializing.")
                        history = []
                except json.JSONDecodeError:
                    logger.warning(f"Error decoding JSON from {TRADE_LOG_FILE}. Reinitializing.")
                    history = []
        
        with open(TRADE_HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error reading or parsing trade history file: {e}")
        return None

    closed_trades = [t for t in history if t.get('status') == 'Closed' and t.get('pnl') is not None]

    if not closed_trades:
        logger.info("No closed trades with PnL found in history to analyze.")
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

async def get_llm_analysis(news_data, historical_candles, indicator_data, learning_insights=None):
    """Get trade signal from LLM analysis, incorporating calculated indicators and learning insights."""
    # Ensure learning_insights is a dictionary, even if empty, for consistent prompt formatting
    if learning_insights is None:
        learning_insights = {}
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not configured")
        return None

    news_titles = [n['title'] for n in news_data[:5]]
    # Format candles for readability in prompt
    formatted_candles = [
        {"T": c["ts"], "O": c["o"], "H": c["h"], "L": c["l"], "C": c["c"], "V": c["v"]}
        for c in historical_candles
    ]

    # Format indicators for prompt
    formatted_indicators = json.dumps(indicator_data, indent=2) if indicator_data else "Not available"

    # Add context about the symbol
    symbol_context = f"{TARGET_INSTRUMENT} is the demo trading symbol for Bitcoin (BTC) USDT-margined perpetual futures."

    # Format learning insights for prompt
    formatted_learning = json.dumps(learning_insights, indent=2) if learning_insights else "No specific learning insights available."

    prompt = f"""
    Analyze the following market data for {TARGET_INSTRUMENT} ({symbol_context}):

    Recent News Headlines (Consider potential sentiment impact):
    {json.dumps(news_titles, indent=2)}

    Recent {CANDLE_CHANNEL} Candles (Latest {len(formatted_candles)} periods. T=Timestamp, O=Open, H=High, L=Low, C=Close, V=Volume):
    {json.dumps(formatted_candles, indent=2)}

    Calculated Technical Indicators (Latest values):
    {formatted_indicators}

    Learning Insights from Past Trades:
    {formatted_learning}
    Please consider these insights when forming your justification and signal.

    Insticator Key:
    - SMA_20: 20-period Simple Moving Average
    - RSI_14: 14-period Relative Strength Index
    - MACD_12_26_9: MACD Line
    - MACDh_12_26_9: MACD Histogram (MACD Line - Signal Line)
    - MACDs_12_26_9: MACD Signal Line
    - BBL_20_2.0: Lower Bollinger Band (20-period, 2 std dev)
    - BBM_20_2.0: Middle Bollinger Band (SMA_20)
    - BBU_20_2.0: Upper Bollinger Band (20-period, 2 std dev)

    Instructions:
    0.  **Learning Insights:** Consider the provided performance patterns based on past trades. How might this influence the current decision?
    1.  **Sentiment Analysis:** Briefly assess the overall sentiment conveyed by the news headlines (Positive, Negative, Neutral).
    2.  **Technical Analysis:**
        *   Interpret **SMA_20**: Is the price above/below the SMA, suggesting trend direction?
        *   Interpret **RSI_14**: Is it overbought (>70), oversold (<30), or neutral? Any divergences?
        *   Interpret **MACD**: Is the MACD line above/below the signal line (MACDs)? Is the histogram (MACDh) positive/negative and growing/shrinking, indicating momentum?
        *   Interpret **Bollinger Bands (BBL, BBM, BBU)**: Is the price near the upper/lower band (potential reversal or continuation)? Are the bands widening (increasing volatility) or narrowing (decreasing volatility)?
        *   Identify the short-term trend based on recent price action in the candles, considering the indicators.
        *   Note any significant price levels (support/resistance) suggested by the candle data or indicator levels (e.g., Bollinger Bands).
        *   Comment on volume patterns if they appear significant.
    4.  **Synthesize & Signal:** Based *only* on the sentiment, technical analysis (including all indicators), and learning insights above, determine a trade signal.
    5.  **Confidence:** Assign a confidence level based on the clarity and convergence of the analysis.
    6.  **Volatility Assessment:** Based on Bollinger Band width and recent price action, assess the current market volatility.

    Output ONLY JSON with the following structure:
    {{
      "sentiment": "Positive" or "Negative" or "Neutral",
      "technical_summary": "Brief summary integrating trend, key levels, volume, SMA, RSI, MACD, and Bollinger Bands observations.",
      "signal": "Long" or "Short" or "Hold",
      "justification": "Concise reasoning linking sentiment, technical analysis (including all indicators), and learning insights to the signal.",
      "confidence": "High" or "Medium" or "Low",
      "volatility": "High" or "Medium" or "Low"
    }}
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AITradingBot"
    }

    logger.info("Requesting LLM analysis with expanded indicators...")

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
        
        # Extract the message content
        message_content = result['choices'][0]['message']['content']
        logger.debug(f"LLM Raw Response Content: {message_content}")

        # Attempt to parse the JSON content from the message
        try:
            # The response might be a string containing JSON, or already JSON-like.
            # It might also have leading/trailing whitespace or be wrapped in markdown code blocks.
            # Clean common issues before parsing:
            cleaned_content = message_content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:] # Remove ```json
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3] # Remove ```
            cleaned_content = cleaned_content.strip() # Remove any extra whitespace

            analysis_json = json.loads(cleaned_content)
            logger.info(f"Successfully parsed LLM analysis: {analysis_json}")
            return analysis_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LLM response JSON: {e}. Response content: {message_content}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing LLM response: {e}. Response content: {message_content}")
            return None

    return None # Should not be reached if API call is successful and parsed


async def place_order(rest_client, side, equity, entry_price, llm_analysis, indicator_data):
    """Place a market order with dynamic size, stop-loss, and take-profit."""
    if not equity or not entry_price or equity <= 0 or entry_price <= 0:
        logger.error(f"Invalid equity ({equity}) or entry_price ({entry_price}) for placing order.")
        return None

    # --- Dynamic Position Sizing ---
    # Calculate the amount to risk in USDT
    risk_amount_usdt = equity * RISK_PER_TRADE_PERCENT
    # Calculate the distance to the stop loss in price terms
    stop_loss_distance = entry_price * STOP_LOSS_PERCENT
    # Calculate position size in base currency (e.g., BTC)
    # size = risk_amount_usdt / stop_loss_distance
    # Bitget uses contract size (sz) which might be in base currency or contracts. Assuming base currency for now.
    # Let's use a simpler size calculation based on a fraction of equity for now, needs refinement based on contract specs.
    # Example: Use 10x leverage implicitly by sizing based on 10% of equity value at entry price
    # position_value_usdt = equity * 0.10 # Example: Target 10% equity value position
    # size = position_value_usdt / entry_price

    # Revised Dynamic Size based on Risk Amount and Stop Distance:
    if stop_loss_distance <= 0:
        logger.error("Stop loss distance is zero or negative, cannot calculate position size.")
        return None
    size = risk_amount_usdt / stop_loss_distance

    # Ensure minimum trade size
    size = max(size, MIN_TRADE_SIZE)
    # Round size to appropriate precision (e.g., 3 decimal places for BTC)
    size = round(size, 3)

    logger.info(f"Calculated dynamic position size: {size} {TARGET_INSTRUMENT.replace('SUSDT', '')} (Equity: {equity}, Risk %: {RISK_PER_TRADE_PERCENT*100}%, Entry: {entry_price}, SL Dist: {stop_loss_distance})")

    # --- Stop Loss and Take Profit Calculation (Factoring in Fees for TP) ---
    # Fees are applied on both entry and exit, so double the fee rate for TP calculation.
    total_fee_consideration = 2 * TRADING_FEE_RATE

    if TAKE_PROFIT_PERCENT <= total_fee_consideration:
        logger.warning(f"TAKE_PROFIT_PERCENT ({TAKE_PROFIT_PERCENT*100}%) is less than or equal to total fee consideration ({total_fee_consideration*100}%). Profit target may not cover fees.")

    if side == 'buy': # Long position
        stop_loss_price = entry_price * (1 - STOP_LOSS_PERCENT)
        # Adjust TP to ensure profit after fees
        take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENT + total_fee_consideration)
        if take_profit_price <= entry_price * (1 + total_fee_consideration):
            logger.warning(f"Adjusted take profit price ({take_profit_price}) for long is too close to or below entry price + fees. Original TP: {entry_price * (1 + TAKE_PROFIT_PERCENT)}")
            # Fallback or further adjustment logic might be needed here if TP becomes non-sensical
            # For now, we proceed but log the warning.

    elif side == 'sell': # Short position
        stop_loss_price = entry_price * (1 + STOP_LOSS_PERCENT)
        # Adjust TP to ensure profit after fees
        take_profit_price = entry_price * (1 - TAKE_PROFIT_PERCENT - total_fee_consideration)
        if take_profit_price >= entry_price * (1 - total_fee_consideration):
            logger.warning(f"Adjusted take profit price ({take_profit_price}) for short is too close to or above entry price - fees. Original TP: {entry_price * (1 - TAKE_PROFIT_PERCENT)}")
            # Fallback or further adjustment logic might be needed here.

    else:
        logger.error(f"Invalid side '{side}' for SL/TP calculation.")
        return None

    # Round prices to appropriate precision (adjust based on instrument)
    # For SBTCSUSDT, prices need to be a multiple of 0.1, so round to 1 decimal place.
    stop_loss_price = round(stop_loss_price, 1)
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


async def run_trading_cycle(rest_client):
    """Executes one cycle of fetching data, analysis, and potential trading."""
    logger.info("--- Starting Trading Cycle ---")
    try:
        # 0. Check for existing open position for the TARGET_INSTRUMENT
        logger.info(f"Checking for existing position for {TARGET_INSTRUMENT} before proceeding...")
        current_position = await get_current_position(rest_client, TARGET_INSTRUMENT)
        if current_position and float(current_position.get('total', '0')) != 0:
            logger.info(f"Existing position found for {TARGET_INSTRUMENT}: Size = {current_position.get('total')}. LLM analysis will be skipped. Waiting for position to close (handled by WebSocket/periodic check setting RESTART_TRADING_CYCLE_FLAG).")
            return # Skip the rest of the trading cycle, main loop will wait for flag
        else:
            logger.info(f"No existing open position found for {TARGET_INSTRUMENT} or position size is 0. Proceeding with trading cycle.")

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
        learning_insights = analyze_trade_history_and_learn() # Returns a dict or {} if no insights
        if learning_insights is None: # Ensure it's always a dict for the LLM prompt
            learning_insights = {}
        logger.info(f"Learning Insights: {learning_insights}")

        # 5. Get LLM Analysis
        analysis = await get_llm_analysis(news_items, current_candles[-10:], indicator_data, learning_insights)
        if not analysis:
            logger.error("Failed to get LLM analysis.")
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
            # Fetch current equity for position sizing only if trading
            current_equity = await get_account_equity(rest_client)
            if current_equity is None:
                logger.error("Cannot place order: Failed to fetch account equity.")
                return

            # Use latest close price as entry price proxy for calculations
            entry_price_proxy = latest_close_price

            if signal == "Long":
                logger.info("Executing LONG trade based on LLM analysis.")
                await place_order(rest_client, 'buy', current_equity, entry_price_proxy, analysis, indicator_data)
            elif signal == "Short":
                logger.info("Executing SHORT trade based on LLM analysis.")
                await place_order(rest_client, 'sell', current_equity, entry_price_proxy, analysis, indicator_data)
            elif signal == "Hold":
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
            connect_private_websocket(loop),
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
                await run_trading_cycle(rest_client)
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


# --- Placeholder for Trade Size Calculation ---
def calculate_trade_size(equity, risk_percentage, entry_price):
    """Placeholder function to calculate trade size."""
    if equity is None or entry_price is None or entry_price == 0:
        logger.warning("Cannot calculate trade size: Missing equity or entry price.")
        return None
    
    risk_amount = equity * (risk_percentage / 100.0)
    # Simple size calculation (adjust based on contract value/leverage if needed)
    # This is a basic example and might need refinement based on instrument specifics
    size = risk_amount / entry_price 
    # Example: Round to 3 decimal places for BTC
    calculated_size = round(size, 3) 
    logger.info(f"Calculated trade size: {calculated_size} based on equity {equity}, risk {risk_percentage}%, price {entry_price}")
    # Add minimum size check if necessary
    if calculated_size <= 0:
        logger.warning(f"Calculated trade size is zero or negative ({calculated_size}). Returning None.")
        return None
    return calculated_size

# --- Core Trading Logic --- 
async def execute_main_trading_logic(rest_client, ws_client_private, ws_client_public):
    """Executes the main trading cycle: data fetch, analysis, decision, execution."""
    logger.info("Starting main trading logic execution...")

    # Initial Equity Fetch with Delay
    logger.info("Allowing 2 seconds for REST client stabilization...")
    await asyncio.sleep(2)
    initial_equity = await get_account_equity(rest_client)
    if initial_equity is None:
        logger.error("Failed to get initial account equity. Cannot proceed with dynamic sizing.")
        # Decide how to handle this - maybe return or use a default size?
        # For now, we'll log the error and potentially skip trading this cycle
        current_equity = None # Indicate equity is unknown
    else:
        current_equity = initial_equity
        logger.info(f"Initial account equity fetched: {current_equity}")

    # --- Trading Loop (Conceptual - actual loop is in main) ---
    # This function represents one cycle within the main loop
    try:
        # 1. Wait for sufficient data
        min_candles = 26 # For indicators
        if len(candle_data_buffer) < min_candles:
            logger.info(f"Waiting for sufficient candle data ({len(candle_data_buffer)}/{min_candles})...")
            return # Skip this cycle if not enough data

        logger.info("Sufficient candle data available. Proceeding with cycle.")
        local_candle_data = list(candle_data_buffer) # Copy buffer for analysis

        # 2. Fetch Data
        logger.info("Fetching news data...")
        news_items = await fetch_news()
        # Placeholder for social sentiment - replace with actual implementation if available
        social_sentiment_score = 0.0 
        logger.info(f"Using placeholder social sentiment: {social_sentiment_score}")

        # 3. Calculate Indicators
        logger.info("Calculating technical indicators...")
        indicators = calculate_indicators(local_candle_data)
        if indicators is None:
            logger.warning("Failed to calculate indicators. Skipping LLM analysis and trade decision.")
            return

        # 4. Get LLM Analysis
        logger.info("Requesting LLM analysis...")
        llm_analysis = await get_llm_analysis(news_items, local_candle_data[-10:], indicators, social_sentiment_score) # Send last 10 candles

        if llm_analysis is None:
            logger.warning("Failed to get LLM analysis. Skipping trade decision.")
            return
        
        logger.info(f"LLM Analysis Received: {llm_analysis}")
        signal = llm_analysis.get('signal')
        confidence = llm_analysis.get('confidence')

        # 5. Decision Logic & Trade Execution
        if signal in ["Long", "Short"]:
            logger.info(f"LLM Signal: {signal} with {confidence} confidence.")
            
            # Check current position (using global state updated by WS)
            # This assumes handle_private_message updates current_positions correctly
            current_pos_size = current_positions.get(TARGET_INSTRUMENT, 0.0)
            logger.info(f"Current position size for {TARGET_INSTRUMENT}: {current_pos_size}")

            # Basic position management: Avoid opening new trade if already in one
            # (More sophisticated logic could handle adding to positions, etc.)
            if (signal == "Long" and current_pos_size > 0) or \
               (signal == "Short" and current_pos_size < 0):
                logger.info(f"Already in a {signal.lower()} position ({current_pos_size}). Holding.")
            elif current_equity is not None:
                 # Get latest price for size calculation (use last close price)
                last_close_price = float(local_candle_data[-1]['c'])
                
                # Calculate trade size
                trade_size = calculate_trade_size(current_equity, RISK_PER_TRADE_PERCENT, last_close_price)

                if trade_size is not None and trade_size > 0:
                    logger.info(f"Attempting to place {signal} order for {trade_size} {TARGET_INSTRUMENT}...")
                    # Placeholder for SL/TP - implement calculation if needed
                    stop_loss = None 
                    take_profit = None
                    
                    trade_success = await place_trade_via_rest(
                        rest_client,
                        TARGET_INSTRUMENT,
                        signal,
                        trade_size,
                        stop_loss_price=stop_loss,
                        take_profit_price=take_profit
                    )
                    
                    if trade_success:
                        logger.info(f"Successfully placed {signal} order.")
                        # Optionally save trade history here or rely on WS updates
                        # save_trade_history(...) # Consider if needed here
                    else:
                        logger.error(f"Failed to place {signal} order.")
                else:
                    logger.warning("Calculated trade size is invalid or zero. Cannot place order.")
            else:
                 logger.warning("Cannot calculate trade size or place order due to missing equity information.")

        elif signal == "Hold":
            logger.info("LLM Signal: Hold. No trade action taken.")
        else:
            logger.warning(f"Received unexpected signal from LLM: {signal}")

    except Exception as e:
        logger.error(f"Error during main trading logic execution: {e}", exc_info=True)

    logger.info("Finished main trading logic execution cycle.")


TRADE_LOG_FILE = "trade_history.json"
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
