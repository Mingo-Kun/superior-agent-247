import asyncio
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
from sentiment_analyzer import SentimentAnalyzer # Import the new class

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
BITGET_API_KEY = os.getenv("BITGET_API_KEY","")
BITGET_SECRET_KEY = os.getenv("BITGET_SECRET_KEY","")
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE","")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen2.5-vl-72b-instruct:free")

# Instantiate Sentiment Analyzer
sentiment_analyzer = SentimentAnalyzer()
# IMPORTANT: Ensure Twitter API keys (TWITTER_API_KEY, TWITTER_API_SECRET_KEY, 
# TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET, TWITTER_BEARER_TOKEN) 
# are set in your .env file for sentiment analysis to work.

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
BITGET_REST_ACCOUNT_ENDPOINT = '/api/v2/mix/account/account' # Endpoint to get mix account details

# Trading Parameters
TARGET_INSTRUMENT = "SBTCSUSDT"  # Demo BTC/USDT futures
# Define Product Type for REST API (V2 uses 'productType' for mix endpoints)
# Common values: 'USDT-FUTURES', 'COIN-FUTURES', 'SUSDT-FUTURES' (for demo)
PRODUCT_TYPE_V2 = "SUSDT-FUTURES"
INST_TYPE_V2 = PRODUCT_TYPE_V2 # Keep for WS compatibility if needed elsewhere

CANDLE_CHANNEL = "candle1H"      # Candle interval (e.g., 1 hour)
MAX_CANDLES = 50                 # Max number of candles to store
# TRADE_SIZE = 0.02                # Fixed trade size (Replaced by dynamic sizing)
STOP_LOSS_PERCENT = 0.01         # 1% stop loss
TAKE_PROFIT_PERCENT = 0.02       # 2% take profit
RISK_PER_TRADE_PERCENT = 0.01    # Risk 1% of equity per trade
MIN_TRADE_SIZE = 0.001           # Minimum allowed trade size (adjust based on exchange)

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
                    order_id = order.get('orderId')
                    client_order_id = order.get('clOrdId') # Use client order ID if available for matching
                    status = order.get('status')
                    filled_size_str = order.get('accBaseVolume') # Filled size
                    avg_fill_price_str = order.get('avgPx') # Average fill price

                    logger.info(f"[PrivateWS] Order Update: ID={order_id}, ClientID={client_order_id}, Status={status}, FilledSz={filled_size_str}, AvgPx={avg_fill_price_str}")

                    # Update trade history when filled
                    if status == 'filled':
                        try:
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
                        except ValueError:
                             logger.error(f"Could not parse fill price ('{avg_fill_price_str}') or size ('{filled_size_str}') to float for order {order_id}.")
                        except Exception as e:
                            logger.error(f"Error processing filled order update for {order_id}: {e}", exc_info=True)
                    elif status in ['cancelled', 'rejected']:
                         # Optionally update history for failed/cancelled orders
                         update_trade_history(client_order_id, {'status': status.capitalize()})

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

async def get_llm_analysis(news_data, historical_candles, indicator_data, social_sentiment_score):
    """Get trade signal from LLM analysis, incorporating calculated indicators."""
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

    prompt = f"""
    Analyze the following market data for {TARGET_INSTRUMENT} ({symbol_context}):

    Recent News Headlines (Consider potential sentiment impact):
    {json.dumps(news_titles, indent=2)}

    Recent {CANDLE_CHANNEL} Candles (Latest {len(formatted_candles)} periods. T=Timestamp, O=Open, H=High, L=Low, C=Close, V=Volume):
    {json.dumps(formatted_candles, indent=2)}

    Calculated Technical Indicators (Latest values):
    {formatted_indicators}

    Twitter Sentiment Score (Recent Tweets for {TARGET_INSTRUMENT}):
    {social_sentiment_score:.4f} (-1 Negative, 0 Neutral, 1 Positive)

    Learning Insights from Past Trades:
    {formatted_learning}

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
    1.  **Twitter Sentiment:** Consider the provided Twitter sentiment score (-1 Negative to 1 Positive). How does it align with or contradict other signals?
    2.  **Sentiment Analysis:** Briefly assess the overall sentiment conveyed by the news headlines (Positive, Negative, Neutral).
    3.  **Technical Analysis:**
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
                # Validate the new structure
                required_keys = ["sentiment", "technical_summary", "signal", "justification", "confidence", "volatility"]
                if not all(key in analysis_json for key in required_keys):
                     logger.error(f"LLM JSON missing required keys. Expected: {required_keys}. Parsed: {analysis_json}")
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
                        required_keys = ["sentiment", "technical_summary", "signal", "justification", "confidence", "volatility"]
                        if all(key in analysis_json for key in required_keys):
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

async def get_account_equity(rest_client):
    """Fetch account equity using the REST API."""
    logger.info("Fetching account equity via REST...")
    try:
        params = {
            "productType": PRODUCT_TYPE_V2,
            "marginCoin": "SUSDT" # Specify the margin coin
        }
        result = await asyncio.to_thread(
            rest_client.get, BITGET_REST_ACCOUNT_ENDPOINT, params
        )
        logger.debug(f"Account Equity Response: {result}")

        if isinstance(result, dict) and result.get('code') == '0':
            account_data = result.get('data')
            if account_data and 'usdtEquity' in account_data:
                equity = float(account_data['usdtEquity'])
                logger.info(f"Successfully fetched account equity: {equity}")
                return equity
            else:
                logger.error(f"'usdtEquity' not found in account data: {account_data}")
                return None
        else:
            error_msg = result.get('msg', 'Unknown error') if isinstance(result, dict) else str(result)
            logger.error(f"Failed to fetch account equity: {error_msg}")
            return None

    except Exception as e:
        logger.error(f"Error fetching account equity: {e}", exc_info=True)
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


async def place_trade_via_rest(rest_client, instrument, signal, size, stop_loss_price=None, take_profit_price=None, order_type="market"):
    """Place trade via REST API, optionally including SL/TP"""
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

    # Add SL/TP if provided (Using common API parameter names, verify with Bitget docs)
    if stop_loss_price is not None:
        params['stopLossPrice'] = str(stop_loss_price)
    if take_profit_price is not None:
        params['takeProfitPrice'] = str(take_profit_price)

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

# --- Main Trading Logic ---

async def run_trading_cycle(rest_client, private_ws_client, public_ws_client):
    """Main trading logic loop"""
    logger.info("Starting trading cycle...")

    # Ensure we have enough candle data before starting
    while len(candle_data_store) < 10: # Wait for at least 10 candles
        logger.info(f"Waiting for sufficient candle data ({len(candle_data_store)}/{MAX_CANDLES})...")
        await asyncio.sleep(10)

    # Fetch initial equity
    current_equity = await get_account_equity(rest_client)
    if current_equity is None:
        logger.error("Failed to get initial account equity. Cannot proceed with dynamic sizing.")
        # Decide how to handle this: exit, use a default, or retry?
        # For now, let's prevent trading without equity info.
        return

    while True:
        try:
            logger.info("--- New Trading Cycle Iteration ---")
            # 1. Fetch News
            news = await fetch_news()

            # 2. Get Candle Data (already available via WS)
            current_candles = list(candle_data_store) # Get a copy
            if not current_candles:
                logger.warning("No candle data available yet. Skipping cycle.")
                await asyncio.sleep(60) # Wait before retrying
                continue

            # 3. Get LLM Analysis
            analysis = await get_llm_analysis(news, current_candles)

            if analysis:
                logger.info(f"LLM Analysis Result: Signal={analysis.get('signal')}, Confidence={analysis.get('confidence')}, Sentiment={analysis.get('sentiment')}, Summary={analysis.get('technical_summary')}")

                # 4. Decision Making (Example: Trade on High/Medium confidence)
                signal = analysis.get('signal')
                confidence = analysis.get('confidence')

                # Use 'Hold' signal to explicitly avoid trading
                if signal and signal.lower() != 'hold' and confidence in ['High', 'Medium']:
                    logger.info(f"Decision: Proceeding with {signal} signal (Confidence: {confidence})")

                    # --- Dynamic Trade Size Calculation ---
                    # Fetch latest equity before placing trade
                    current_equity = await get_account_equity(rest_client)
                    if current_equity is None:
                        logger.error("Failed to get current account equity before trade. Skipping trade.")
                        await asyncio.sleep(60)
                        continue

                    risk_amount_per_trade = current_equity * RISK_PER_TRADE_PERCENT
                    last_close_price = current_candles[-1]['c']

                    # Calculate SL/TP prices
                    if signal.lower() == "long":
                        stop_loss_price = last_close_price * (1 - STOP_LOSS_PERCENT)
                        take_profit_price = last_close_price * (1 + TAKE_PROFIT_PERCENT)
                    elif signal.lower() == "short":
                        stop_loss_price = last_close_price * (1 + STOP_LOSS_PERCENT)
                        take_profit_price = last_close_price * (1 - TAKE_PROFIT_PERCENT)
                    else:
                        logger.error(f"Invalid signal '{signal}' received from LLM. Skipping trade.")
                        await asyncio.sleep(60)
                        continue

                    stop_loss_distance = abs(last_close_price - stop_loss_price)

                    if stop_loss_distance <= 0:
                        logger.error(f"Stop loss distance is zero or negative ({stop_loss_distance}). Cannot calculate trade size. Skipping trade.")
                        await asyncio.sleep(60)
                        continue

                    trade_size = risk_amount_per_trade / stop_loss_distance

                    # Ensure minimum trade size and round
                    trade_size = max(trade_size, MIN_TRADE_SIZE)
                    # Adjust rounding precision as needed by the exchange (e.g., 3 decimal places for BTC)
                    trade_size = round(trade_size, 3)

                    logger.info(f"Calculated Trade Size: {trade_size} (Equity: {current_equity}, Risk Amount: {risk_amount_per_trade}, SL Distance: {stop_loss_distance})")
                    # --- End Dynamic Trade Size Calculation ---

                    # 5. Place Trade via REST
                    success = await place_trade_via_rest(
                        rest_client,
                        TARGET_INSTRUMENT,
                        signal,
                        trade_size,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price
                    )
                    if success:
                        logger.info(f"Trade placed successfully: {signal} {trade_size} {TARGET_INSTRUMENT}")
                    else:
                        logger.error("Failed to place trade.")
                else:
                    logger.info(f"Decision: No trade action. Signal: {signal}, Confidence: {confidence}")

            else:
                logger.warning("LLM analysis failed or returned no result. Skipping trade decision.")

            # Wait before next cycle (e.g., 1 hour based on candle interval)
            logger.info("Waiting for next trading cycle (approx 1 hour)...")
            await asyncio.sleep(3600) # Adjust based on candle interval

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            await asyncio.sleep(60) # Wait before retrying after an error


# --- Trade History Logging ---

TRADE_HISTORY_FILE = 'trade_history.json'

def save_trade_history(client_order_id, side, size, entry_price, stop_loss_price, take_profit_price, status, llm_analysis=None, sentiment_score=None, indicators=None):
    """Saves trade details and context to a JSON file."""
    try:
        history = []
        if os.path.exists(TRADE_HISTORY_FILE):
            with open(TRADE_HISTORY_FILE, 'r') as f:
                try:
                    history = json.load(f)
                    if not isinstance(history, list):
                        logger.warning(f"'{TRADE_HISTORY_FILE}' does not contain a valid JSON list. Initializing new list.")
                        history = []
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from '{TRADE_HISTORY_FILE}'. Initializing new list.")
                    history = []

        trade_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'client_order_id': client_order_id,
            'symbol': TARGET_INSTRUMENT,
            'side': side,
            'size': size,
            'entry_price_target': entry_price, # The price used for calculation
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'status': status, # e.g., 'Placed', 'PlaceFailed_NoID', 'PlaceFailed_APIExc_...', 'Filled', 'Closed_SL', 'Closed_TP'
            'llm_analysis': llm_analysis, # Full analysis dict
            'twitter_sentiment_score': sentiment_score,
            'indicators_at_trade': indicators, # Full indicators dict
            'actual_entry_price': None, # To be updated on fill
            'exit_price': None, # To be updated on close
            'pnl': None # To be updated on close
        }

        history.append(trade_record)

        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
        logger.info(f"Saved trade record for {client_order_id} with status '{status}'")

    except Exception as e:
        logger.error(f"Error saving trade history for {client_order_id}: {e}", exc_info=True)



def update_trade_history(order_id, update_data):
    """Updates an existing trade record in the history file."""
    try:
        if not os.path.exists(TRADE_HISTORY_FILE):
            logger.warning(f"Trade history file {TRADE_HISTORY_FILE} not found for update.")
            return

        with open(TRADE_HISTORY_FILE, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {TRADE_HISTORY_FILE}. Cannot update.")
                history = [] # Start fresh if file is corrupt

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
    logger.info("Learning analysis complete.")
    return learning_insights


async def place_order(rest_client, side, equity, entry_price, llm_analysis, sentiment_score, indicator_data):
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

    # --- Stop Loss and Take Profit Calculation ---
    if side == 'buy': # Long position
        stop_loss_price = entry_price * (1 - STOP_LOSS_PERCENT)
        take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENT)
    elif side == 'sell': # Short position
        stop_loss_price = entry_price * (1 + STOP_LOSS_PERCENT)
        take_profit_price = entry_price * (1 - TAKE_PROFIT_PERCENT)
    else:
        logger.error(f"Invalid side '{side}' for SL/TP calculation.")
        return None

    # Round prices to appropriate precision (adjust based on instrument)
    stop_loss_price = round(stop_loss_price, 2)
    take_profit_price = round(take_profit_price, 2)

    order_id = f"agent247_{int(time.time() * 1000)}" # Unique client order ID
    params = {
        "symbol": TARGET_INSTRUMENT,
        "productType": PRODUCT_TYPE_V2,
        "marginMode": "isolated", # Or 'cross'
        "marginCoin": "SUSDT",
        "side": side, # 'buy' or 'sell'
        "orderType": "market",
        "size": str(size), # Size must be a string
        "clientOrderId": order_id,
        # Add Stop Loss and Take Profit parameters
        "presetTakeProfitPrice": str(take_profit_price),
        "presetStopLossPrice": str(stop_loss_price)
    }

    logger.info(f"Placing Order: {params}")
    try:
        result = await asyncio.to_thread(
            rest_client.post, BITGET_REST_PLACE_ORDER_ENDPOINT, params
        )
        logger.debug(f"Place Order Response: {result}")

        if isinstance(result, dict) and result.get('code') == '0':
            order_data = result.get('data')
            if order_data and 'orderId' in order_data:
                logger.info(f"Order placed successfully: ID {order_data['orderId']}, Client ID {order_id}")
                # Save trade details with context
                save_trade_history(order_id, side, size, entry_price, stop_loss_price, take_profit_price, "Placed", llm_analysis, sentiment_score, indicator_data)
                return order_data['orderId']
            else:
                logger.error(f"Order placement succeeded but no orderId in response data: {order_data}")
                save_trade_history(order_id, side, size, entry_price, stop_loss_price, take_profit_price, "PlaceFailed_NoID", llm_analysis, sentiment_score, indicator_data)
                return None
        else:
            error_msg = result.get('msg', 'Unknown error') if isinstance(result, dict) else str(result)
            logger.error(f"Failed to place order: {error_msg}")
            save_trade_history(order_id, side, size, entry_price, stop_loss_price, take_profit_price, f"PlaceFailed_{error_msg[:50]}", llm_analysis, sentiment_score, indicator_data) # Save truncated error
            return None

    except BitgetAPIException as e:
        logger.error(f"API Exception placing order: {e}")
        save_trade_history(order_id, side, size, entry_price, stop_loss_price, take_profit_price, f"PlaceFailed_APIExc_{e.message[:50]}", llm_analysis, sentiment_score, indicator_data)
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing order: {e}", exc_info=True)
        save_trade_history(order_id, side, size, entry_price, stop_loss_price, take_profit_price, f"PlaceFailed_Exc_{str(e)[:50]}", llm_analysis, sentiment_score, indicator_data)
        return None


async def run_trading_cycle(rest_client):
    """Executes one cycle of fetching data, analysis, and potential trading."""
    logger.info("--- Starting Trading Cycle ---")
    try:
        # 1. Fetch News
        news_items = await fetch_news()

        # 1.5 Fetch Social Media Sentiment (Twitter)
        # TODO: Make the query configurable or dynamic based on context/ticker
        sentiment_query = f'{TARGET_INSTRUMENT.replace("SUSDT","")} OR ${TARGET_INSTRUMENT.replace("SUSDT","")} -is:retweet lang:en'
        social_sentiment_score = await sentiment_analyzer.fetch_tweets_and_analyze(sentiment_query, max_results=25) # Use await for async function
        logger.info(f"Fetched Twitter sentiment score for '{sentiment_query}': {social_sentiment_score:.4f}")

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

        # 4. Get LLM Analysis
        analysis = await get_llm_analysis(news_items, current_candles[-10:], indicator_data, social_sentiment_score) # Pass sentiment score
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
                await place_order(rest_client, 'buy', current_equity, entry_price_proxy, analysis, social_sentiment_score, indicator_data)
            elif signal == "Short":
                logger.info("Executing SHORT trade based on LLM analysis.")
                await place_order(rest_client, 'sell', current_equity, entry_price_proxy, analysis, social_sentiment_score, indicator_data)
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
            # Removed fixed sleep from main loop, cycle now handles dynamic wait
            # logger.info(f"Cycle finished. Waiting for 1 hour...")
            # await asyncio.sleep(3600)  # Run every hour

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
