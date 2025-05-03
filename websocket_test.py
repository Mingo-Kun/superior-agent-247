import websocket
import ssl
import logging
import time

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BITGET_WSS_PRIVATE_URL = "wss://ws.bitget.com/v2/ws/private"

def on_message(ws, message):
    logger.info(f"Received message: {message}")

def on_error(ws, error):
    logger.error(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info(f"### closed ### Status: {close_status_code}, Msg: {close_msg}")

def on_open(ws):
    logger.info("### opened ###")
    # Attempt a basic subscription (optional, just to see if connection works)
    # ws.send('{"op": "subscribe", "args": [{"instType":"SUSDT-FUTURES", "channel":"account", "instId":"default"}]}')

if __name__ == "__main__":
    websocket.enableTrace(True) # Enable detailed library tracing
    logger.info(f"Attempting to connect to: {BITGET_WSS_PRIVATE_URL}")
    try:
        ws = websocket.WebSocketApp(BITGET_WSS_PRIVATE_URL,
                                  on_open=on_open,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)

        # Run with hostname check disabled (matches the setting in BitgetWsClient)
        ws.run_forever(ping_timeout=10, sslopt={"check_hostname": False})
        # Alternative: Run without sslopt
        # ws.run_forever(ping_timeout=10)
    except Exception as e:
        logger.error(f"Failed to initialize or run WebSocketApp: {e}", exc_info=True)

    logger.info("Script finished.")