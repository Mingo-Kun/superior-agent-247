import os
import time
import json
import asyncio
import requests
import logging
import random
import hmac
import hashlib
from collections import deque
from functools import lru_cache
from dotenv import load_dotenv

###############################################################################
# ENVIRONMENT VARIABLES & SETUP
###############################################################################

load_dotenv()

BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_SECRET_KEY = os.getenv("BITGET_SECRET_KEY")
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE")

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RSS_FEED_URL = os.getenv("RSS_FEED_URL")

RAG_MODEL = os.getenv("RAG_MODEL")            
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL")
CODING_MODEL = os.getenv("CODING_MODEL")      
LLM_MODEL = os.getenv("LLM_MODEL")           

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [AGENT LOG] %(message)s")


###############################################################################
# BITGET API HELPER FUNCTIONS
###############################################################################

def generate_signature(timestamp, method, request_path, body=''):
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    signature = hmac.new(BITGET_SECRET_KEY.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

def get_market_ticker(symbol="SBTCSUSDT"):
    """
    Retrieve market ticker from Bitget demo API.
    Endpoint: GET /api/v2/mix/market/ticker?symbol=...
    """
    timestamp = str(int(time.time() * 1000))
    method = "GET"
    request_path = f"/api/v2/mix/market/ticker?symbol={symbol}"
    signature = generate_signature(timestamp, method, request_path)
    
    headers = {
        "ACCESS-KEY": BITGET_API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-PASSPHRASE": BITGET_PASSPHRASE,
        "ACCESS-TIMESTAMP": timestamp,
        "paptrading": "1",  # Demo mode
        "locale": "en-US",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get("https://api.bitget.com" + request_path, headers=headers)
        response.raise_for_status()
        logging.info("Market ticker fetched successfully.")
        return response.json()
    except Exception as e:
        logging.error("Error fetching market ticker: %s", e)
        return {"data": [{"last": "65000"}]}  # fallback

def get_account_balances():
    """
    Retrieve account balances from Bitget demo API.
    Endpoint: GET /api/v2/account/all-account-balance
    """
    timestamp = str(int(time.time() * 1000))
    method = "GET"
    request_path = "/api/v2/account/all-account-balance"
    signature = generate_signature(timestamp, method, request_path)
    
    headers = {
        "ACCESS-KEY": BITGET_API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-PASSPHRASE": BITGET_PASSPHRASE,
        "ACCESS-TIMESTAMP": timestamp,
        "paptrading": "1",
        "locale": "en-US",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get("https://api.bitget.com" + request_path, headers=headers)
        response.raise_for_status()
        logging.info("Account balances fetched successfully.")
        return response.json()
    except Exception as e:
        logging.error("Error fetching account balances: %s", e)
        return {"data": [{"accountType": "futures", "usdtBalance": "10000"}]}

def place_demo_order(symbol, productType, marginMode, marginCoin, size, price, side, tradeSide, orderType, clientOid):
    """
    Place an order on Bitget demo API.
    Endpoint: POST /api/v2/mix/order/place-order
    """
    timestamp = str(int(time.time() * 1000))
    method = "POST"
    request_path = "/api/v2/mix/order/place-order"
    payload = {
        "symbol": symbol,
        "productType": productType,
        "marginMode": marginMode,
        "marginCoin": marginCoin,
        "size": size,
        "price": price,
        "side": side,
        "tradeSide": tradeSide,
        "orderType": orderType,
        "force": "gtc",
        "clientOid": clientOid,
        "reduceOnly": "NO"
    }
    body = json.dumps(payload)
    signature = generate_signature(timestamp, method, request_path, body)
    
    headers = {
        "ACCESS-KEY": BITGET_API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-PASSPHRASE": BITGET_PASSPHRASE,
        "ACCESS-TIMESTAMP": timestamp,
        "paptrading": "1",
        "locale": "en-US",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post("https://api.bitget.com" + request_path, headers=headers, data=body)
        response.raise_for_status()
        logging.info("Order placed successfully.")
        return response.json()
    except Exception as e:
        logging.error("Error placing order: %s", e)
        return {"error": str(e)}

###############################################################################
# ADAPTIVE BUFFER
###############################################################################

class AdaptiveBuffer:
    def __init__(self, initial_max_size=40, min_size=10, max_size=80):
        self.buffer = deque(maxlen=initial_max_size)
        self.max_size = initial_max_size
        self.min_size = min_size
        self.absolute_max = max_size

    def add_data(self, data):
        self.buffer.append(data)

    def adjust_buffer_size(self, processing_latency):
        # Slightly tweaked thresholds for Codespaces
        high_latency_threshold = 0.6  
        low_latency_threshold = 0.3   

        if processing_latency > high_latency_threshold and self.max_size > self.min_size:
            self.max_size = max(self.min_size, self.max_size - 5)
            self.buffer = deque(self.buffer, maxlen=self.max_size)
            logging.info("High latency (%.3f s). Reducing buffer size to %d.", processing_latency, self.max_size)
        elif processing_latency < low_latency_threshold and self.max_size < self.absolute_max:
            self.max_size = min(self.absolute_max, self.max_size + 5)
            self.buffer = deque(self.buffer, maxlen=self.max_size)
            logging.info("Low latency (%.3f s). Increasing buffer size to %d.", processing_latency, self.max_size)

    def get_data(self):
        return list(self.buffer)

adaptive_buffer = AdaptiveBuffer()

###############################################################################
# OFFLOADED API FUNCTIONS (SENTIMENT, RAG)
###############################################################################

@lru_cache(maxsize=50)
def cached_sentiment_analysis(text):
    sentiment_api_url = f"https://api-inference.huggingface.co/models/{SENTIMENT_MODEL}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": text}
    try:
        response = requests.post(sentiment_api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error("Sentiment analysis error: %s", e)
        return {"error": str(e)}

def offload_sentiment(text):
    return cached_sentiment_analysis(text)

def query_openrouter_api(prompt, model_name, use_quantized=False):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    quant_flag = " [quantized]" if use_quantized else ""
    full_prompt = prompt + quant_flag

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a professional trading AI."},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.7
    }

    start_time = time.time()
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
    latency = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
    else:
        result = {"error": response.status_code, "message": response.text}
    return result, latency

###############################################################################
# RSS FEED FOR NEWS
###############################################################################

def fetch_rss_feed(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error("RSS fetch error: %s", response.status_code)
            return {}
    except Exception as e:
        logging.error("RSS fetch exception: %s", e)
        return {}

def process_rss_news(rss_data):
    headlines = []
    if "items" in rss_data:
        for item in rss_data["items"]:
            headlines.append(item.get("title", ""))
    return headlines

###############################################################################
# AGENT 247
###############################################################################

class Agent247:
    """
    Logs high-level actions (including code rewriting) and
    only trades the 4 supported tickers in Bitget demo:
    SBTCSUSDT, SETHSUSDT, SXRPSUSDT, SEOSSUSDT.
    """
    def __init__(self):
        self.rsi_threshold = 30
        self.max_trade_percent = 20
        self.usdt_balance = 500

    def analyze_rsi(self, ticker):
        # For demonstration, we simulate RSI with random.
        rsi_value = random.randint(10, 60)
        logging.info("RSI for %s: %d", ticker, rsi_value)
        return rsi_value

    def momentum_trade_scenario(self, ticker="SBTCSUSDT"):
        rsi = self.analyze_rsi(ticker)
        if rsi < self.rsi_threshold:
            logging.info("RSI below threshold for %s. Momentum trade scenario...", ticker)
            self.execute_momentum_trade(ticker)
        else:
            logging.info("RSI above threshold for %s. Skipping momentum trade.", ticker)

    def execute_momentum_trade(self, ticker):
        logging.info("Executing momentum trade for %s if under %d%% risk...", ticker, self.max_trade_percent)
        trade_amount = 100
        if trade_amount <= (self.usdt_balance * self.max_trade_percent / 100):
            logging.info("Swapping %d USDT for %s with limit sell & stop-loss.", trade_amount, ticker)
            self.usdt_balance -= trade_amount
            logging.info("New USDT balance: %.2f", self.usdt_balance)
        else:
            logging.info("Trade amount exceeds risk threshold. Trade aborted.")

    def altcoin_pump_scenario(self):
        # Only these 4 tickers are valid in Bitget Demo
        possible_tickers = ["SBTCSUSDT", "SETHSUSDT", "SXRPSUSDT", "SEOSSUSDT"]
        altcoin_ticker = random.choice(possible_tickers)
        logging.info("Identified trending altcoin ticker: %s", altcoin_ticker)
        trade_amount = 50
        if trade_amount <= (self.usdt_balance * self.max_trade_percent / 100):
            logging.info("Swapping %d USDT for %s (pump scenario).", trade_amount, altcoin_ticker)
            self.usdt_balance -= trade_amount
            logging.info("New USDT balance: %.2f", self.usdt_balance)
        else:
            logging.info("Trade exceeds risk threshold. Altcoin pump trade aborted.")

    def run_code_rewrite(self):
        logging.info("Agent 247 rewriting code for improved strategy...")
        new_code = '''
def improved_strategy(self):
    logging.info("Executing improved strategy with enhanced risk mgmt.")
'''
        logging.info("New code snippet generated:\n%s", new_code)

###############################################################################
# MAIN TRADING LOOP (BITGET DEMO)
###############################################################################

async def bitget_demo_trading():
    agent = Agent247()

    while True:
        logging.info("=== Starting Trading Iteration ===")

        # 1. Pick one of the 4 tickers randomly for demonstration
        possible_tickers = ["SBTCSUSDT", "SETHSUSDT", "SXRPSUSDT", "SEOSSUSDT"]
        chosen_ticker = random.choice(possible_tickers)

        # 2. Fetch Market Data for the chosen ticker
        ticker_response = get_market_ticker(chosen_ticker)
        try:
            current_price = float(ticker_response.get("data", [{}])[0].get("last", "65000"))
        except Exception:
            current_price = 65000.0
        market_data = {"symbol": chosen_ticker, "price": current_price, "timestamp": time.time()}
        logging.info("Market Data: %s", market_data)

        # 3. Fetch & Process RSS
        rss_data = fetch_rss_feed(RSS_FEED_URL)
        headlines = process_rss_news(rss_data)
        news_context = " ".join(headlines) if headlines else "No recent news."
        logging.info("News Headlines: %s", headlines)

        # 4. Offload Sentiment
        sample_text = headlines[0] if headlines else "No news"
        sentiment_result = offload_sentiment(sample_text)
        logging.info("Sentiment Result: %s", sentiment_result)

        # 5. Query LLM for decision
        decision_prompt = f"""
Market Data: {json.dumps(market_data)}
News Context: {news_context}
Sentiment: {json.dumps(sentiment_result, indent=2)}
Please provide a JSON decision: {{"action": "BUY"/"SELL"/"HOLD", "confidence": <0-100>}}
        """
        decision_response, decision_latency = query_openrouter_api(decision_prompt, LLM_MODEL, use_quantized=True)
        logging.info("OpenRouter API latency: %.3f s", decision_latency)

        # 6. Adaptive buffer
        adaptive_buffer.adjust_buffer_size(decision_latency)
        adaptive_buffer.add_data(market_data)

        try:
            decision_json = json.loads(decision_response["choices"][0]["message"]["content"])
        except:
            decision_json = {"action": "HOLD", "confidence": 50}
        logging.info("LLM Decision: %s", decision_json)

        # 7. Agent 247's internal scenarios
        agent.momentum_trade_scenario(chosen_ticker)
        agent.altcoin_pump_scenario()

        # 8. Possibly rewrite code
        if random.random() < 0.3:
            agent.run_code_rewrite()

        # 9. Check account balances
        balances = get_account_balances()
        logging.info("Account Balances: %s", balances)
        try:
            available_usdt = 0.0
            for account in balances.get("data", []):
                if account.get("accountType") == "futures":
                    available_usdt = float(account.get("usdtBalance", "0"))
                    break
        except:
            available_usdt = 0.0

        # 10. Position size = 1% of available
        position_size = available_usdt * 0.01
        logging.info("Calculated Position Size: %.2f USDT", position_size)

        # 11. Place an order if decision is BUY/SELL
        if decision_json["action"] in ["BUY", "SELL"]:
            clientOid = str(int(time.time() * 1000))
            order_response = place_demo_order(
                symbol=chosen_ticker,
                productType="susdt-futures",
                marginMode="isolated",
                marginCoin="SUSDT",
                size=str(position_size),
                price=str(current_price),
                side=decision_json["action"].lower(),
                tradeSide="open",
                orderType="limit",
                clientOid=clientOid
            )
            logging.info("Order Response: %s", order_response)
        else:
            logging.info("No order executed (HOLD).")

        logging.info("=== Iteration complete. Waiting 5 minutes. ===\n")
        await asyncio.sleep(300)

async def on_chain_trading():
    logging.info("On-chain trading mode selected. Not implemented here.")

async def bitget_real_trading():
    logging.info("Bitget real trading mode selected. Not implemented here.")

def mode_selection():
    print("Select Trading Mode:")
    print("1. Trade On-Chain")
    print("2. Trade on Bitget Demo")
    print("3. Trade on Bitget Real (Coming Soon)")
    choice = input("Enter your choice (1/2/3): ").strip()
    return choice

async def main():
    choice = mode_selection()
    if choice == "1":
        await on_chain_trading()
    elif choice == "2":
        await bitget_demo_trading()
    elif choice == "3":
        await bitget_real_trading()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    asyncio.run(main())
