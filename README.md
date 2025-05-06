# Superior Agent 247 (Bitget-Demo Trading) - AI-Powered Trading Bot

## Overview
Superior Agent 247 is an advanced, AI-powered trading bot designed for the Bitget exchange (specifically for demo trading). It leverages real-time market data, processes it through a Large Language Model (LLM) for analysis, and automates trade execution based on the generated insights.

## Key Features
- **Real-time Data Processing**: Utilizes WebSocket connections for continuous market data (candles) and private account updates (equity, positions, orders).
- **AI-Driven Analysis**: Employs an LLM (via OpenRouter) to analyze a combination of news sentiment (from RSS feeds), technical indicators, and insights learned from past trade performance (`learning_insights`).
- **Automated Trade Execution**: Places trades automatically on Bitget via their REST API based on the LLM's signals.
- **Dynamic Risk Management**: Implements dynamic trade sizing based on account equity and pre-defined risk percentages. Includes stop-loss and take-profit mechanisms.
- **Comprehensive Logging**: Maintains detailed logs for all operations, including trade history (`trade_history.json`) and general bot activity (`bot.log`).
- **Asynchronous Architecture**: Built with Python's `asyncio` for efficient handling of WebSocket communications and concurrent tasks.
- **Configurable Parameters**: Easily adjustable settings for API keys, trading instruments, candle intervals, risk parameters, and RSS feed sources via a `.env` file and script constants.
- **Learning from Past Trades**: Analyzes `trade_history.json` to generate `learning_insights` (e.g., average PnL, win/loss rates) which are then fed back into the LLM to potentially improve future trading decisions.

## Architecture

```mermaid
graph TD
    A[WebSocket Public Bitget] -->|Candle Data (e.g., 1H SBTCSUSDT)| B[Data Aggregation & TA Calculation]
    A2[WebSocket Private Bitget] -->|Account Updates, Position Info, Order Status| B
    C[RSS News Feeds] -->|News Headlines & Content| D[LLM Analysis Engine]
    B -->|Processed Market Data & Indicators| D
    D -->|Trade Signal (Long/Short/Hold) & Reasoning| E[Trade Execution Logic]
    E -->|Place/Manage Order| F[Bitget REST API]
    F -->|Order Confirmation/Error| E
    E -->|Log Trade Details| G[Trade History (trade_history.json)]
    B -->|Log System Events| H[System Log (bot.log)]
    D -->|Log LLM Interaction| H
```

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Mingo-Kun/superior-agent-247.git
    cd superior-agent-247
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Create a `.env` file** in the root directory of the project by copying `e,g.env` or creating a new one:
    ```ini
    # Bitget API Credentials (Use Demo Account Keys for SBTCSUSDT)
    BITGET_API_KEY=YOUR_BITGET_API_KEY
    BITGET_SECRET_KEY=YOUR_BITGET_SECRET_KEY
    BITGET_PASSPHRASE=YOUR_BITGET_PASSPHRASE

    # OpenRouter API Key for LLM Access
    OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY

    # LLM Model (Ensure it's compatible with OpenRouter and your needs)
    # Example: LLM_MODEL=qwen/qwen2.5-vl-72b-instruct:free
    LLM_MODEL=qwen/qwen3-30b-a3b:free
    ```
    *   **Important**: Replace `YOUR_..._KEY` placeholders with your actual API credentials.
    *   For `SBTCSUSDT` (demo trading), ensure you are using API keys generated from your Bitget demo account.

2.  **Adjust Trading Parameters (Optional, in `agent247.py`):**
    *   `TARGET_INSTRUMENT`: The trading pair (e.g., `SBTCSUSDT` for demo BTC/USDT futures).
    *   `PRODUCT_TYPE_V2`: Product type for Bitget API V2 (e.g., `SUSDT-FUTURES`).
    *   `CANDLE_CHANNEL`: The candle interval for analysis (e.g., `candle1H` for 1-hour candles).
    *   `MAX_CANDLES`: Maximum number of historical candles to store and use for indicator calculations.
    *   `STOP_LOSS_PERCENT`: Percentage for stop-loss orders.
    *   `TAKE_PROFIT_PERCENT`: Percentage for take-profit orders.
    *   `RISK_PER_TRADE_PERCENT`: Percentage of equity to risk per trade, used for dynamic trade sizing.
    *   `MIN_TRADE_SIZE`: Minimum trade size allowed by the exchange for the target instrument.
    *   `RSS_FEEDS`: A list of URLs for RSS feeds to be used for news sentiment analysis.

## Usage

To run the trading bot:
```bash
python agent247.py
```
The bot will start, connect to Bitget WebSockets, and begin its trading cycle.

### Trading Cycle Workflow

The bot operates in a continuous loop, performing the following steps in each cycle:

1.  **Fetch News**: Retrieves the latest news headlines from the configured `RSS_FEEDS`.
2.  **Check Candle Data**: Ensures enough candle data has been collected via WebSocket. If not (e.g., on initial startup or after a disconnection), it waits or logs a warning.
3.  **Calculate Technical Indicators**: If sufficient candle data is available (at least `MAX_CANDLES` or enough for the longest period indicator, typically 26 periods for MACD/Bollinger Bands), it calculates:
    *   Simple Moving Average (SMA)
    *   Relative Strength Index (RSI)
    *   Moving Average Convergence Divergence (MACD)
    *   Bollinger Bands (BBands)
4.  **Fetch Account Equity**: Retrieves the current account equity using the Bitget REST API to be used for dynamic trade sizing.
5.  **LLM Analysis**: Sends the collected news items, latest candle data, and calculated technical indicators to the configured LLM. The LLM analyzes this information based on a predefined prompt to:
    *   Assess overall news sentiment.
    *   Interpret technical indicators.
    *   Consider `learning_insights` derived from analyzing `trade_history.json` (e.g., average PnL, win/loss rates for different conditions).
    *   Provide a trading signal (Long, Short, or Hold) and a confidence score/reasoning.
6.  **Trade Decision & Execution**:
    *   If the LLM provides a 'Long' or 'Short' signal with sufficient confidence (and no existing position or based on strategy rules):
        *   Calculates trade size dynamically based on `RISK_PER_TRADE_PERCENT` and current equity.
        *   Calculates stop-loss and take-profit prices.
        *   Places a market order via the Bitget REST API.
        *   Logs the trade attempt and details in `trade_history.json`.
    *   If the signal is 'Hold' or confidence is low, no trade is executed.
7.  **Wait/Loop**: Waits for a configurable interval (e.g., aligned with candle duration or a fixed polling time) before starting the next cycle. The main loop in the current version is set to break after one full cycle for testing purposes (`MAIN_LOOP_TEST_MODE = True`). Set to `False` for continuous operation.

### WebSocket Data Handling
-   **Public WebSocket (`handle_public_message`)**: Receives and stores candle data for `TARGET_INSTRUMENT` in `candle_data_store`.
-   **Private WebSocket (`handle_private_message`)**: Receives account updates (equity), position updates (entry price, size, PnL), and order updates (fills, cancellations). This data is used for real-time monitoring and updating `trade_history.json` (e.g., marking orders as 'Filled' or 'Closed').

## Core Functions in `agent247.py`

-   `fetch_news()`: Asynchronously fetches and parses news from RSS feeds.
-   `calculate_indicators(candle_data)`: Calculates technical indicators from a Pandas DataFrame of candle data.
-   `analyze_trade_history_and_learn()`: Analyzes `trade_history.json` to calculate performance metrics and generate `learning_insights`.
-   `get_llm_analysis(news_data, historical_candles, indicator_data, learning_insights=None)`: Constructs a prompt with market data, news, and `learning_insights`, sends it to the LLM, and parses the response for a trading signal.
-   `get_account_equity(rest_client)`: Fetches the current account equity via REST API.
-   `place_order(rest_client, side, equity, entry_price, llm_analysis, indicator_data)`: Constructs and places a trade order via the Bitget REST API, including dynamic sizing and SL/TP levels.
-   `save_trade_history(...)`: Saves details of placed, filled, or failed trades to `trade_history.json`.
-   `update_trade_history(client_order_id, update_data)`: Updates existing entries in `trade_history.json`, for example, when an order is filled or a position is closed.
-   `run_trading_cycle(public_ws_client, private_ws_client, rest_client)`: The main loop orchestrating the trading logic.
-   `connect_public_websocket()`, `connect_private_websocket()`: Functions to establish and manage WebSocket connections.

## Customization

To adapt or enhance the trading strategy:

1.  **LLM Prompt Engineering**: Modify the prompt within the `get_llm_analysis` function in `agent247.py` to change how the LLM interprets data or what factors it prioritizes.
2.  **Technical Indicators**: Add new indicators in `calculate_indicators` (using `pandas-ta` or custom calculations) and ensure they are included in the data passed to the LLM and in the LLM's prompt.
3.  **Risk Management Rules**: Adjust `STOP_LOSS_PERCENT`, `TAKE_PROFIT_PERCENT`, `RISK_PER_TRADE_PERCENT`, or the logic in `place_order` for more sophisticated risk strategies.
4.  **Trading Logic**: Modify the decision-making process within `run_trading_cycle` based on the LLM output or other conditions.
5.  **News Sources**: Update the `RSS_FEEDS` list in `agent247.py` to include different or more relevant news sources.

## Monitoring

-   **Trade History**: `trade_history.json` - A JSON file logging details of each trade attempted or executed, including entry/exit prices, status, PnL, and the LLM analysis at the time of the trade.
-   **System Logs**: `bot.log` - Contains detailed operational logs, including WebSocket messages, API calls, errors, and informational messages about the bot's status.

## Troubleshooting

-   **WebSocket Connection Issues**:
    *   Verify that your `BITGET_API_KEY`, `BITGET_SECRET_KEY`, and `BITGET_PASSPHRASE` in `.env` are correct and have the necessary permissions for demo trading (futures/mix account).
    *   Check your internet connectivity.
    *   Ensure Bitget WebSocket services are operational.
-   **LLM Errors / No Analysis**: 
    *   Confirm `OPENROUTER_API_KEY` is correct and has credits/access.
    *   Verify the `LLM_MODEL` specified in `.env` is available on OpenRouter and suitable for the task.
    *   Check `bot.log` for specific error messages from the OpenRouter API.
-   **Trade Failures / Order Rejections**:
    *   Review `bot.log` for error messages from the Bitget API. Common issues include insufficient margin (even in demo), incorrect instrument ID, invalid order size (check `MIN_TRADE_SIZE`), or API permission problems.
    *   Ensure `TARGET_INSTRUMENT` and `PRODUCT_TYPE_V2` are correctly set for demo trading.
-   **"Not enough candle data" Warning**: This is normal on first run or if the bot has been offline. It needs to collect a certain number of candles (e.g., 50, as per `MAX_CANDLES`, or at least 26 for some indicators) before it can calculate all technical indicators. Let it run for a few candle intervals (e.g., a few hours if using `candle1H`).

## Disclaimer
This software is provided for educational and demonstration purposes only, specifically for use with Bitget's demo trading environment. Trading cryptocurrencies involves significant risk of loss. The developers are not responsible for any financial losses incurred through the use of this software. Always test thoroughly in a demo environment before considering any application with real funds. Use at your own risk.
