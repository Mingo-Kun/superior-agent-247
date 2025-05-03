import asyncio
import websockets
import ssl
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BITGET_WSS_PRIVATE_URL = "wss://ws.bitget.com/v2/ws/private"

async def connect_bitget():
    logger.info(f"Attempting to connect to: {BITGET_WSS_PRIVATE_URL} using 'websockets' library")
    # Create a default SSL context, but disable hostname verification
    # Similar to sslopt={"check_hostname": False} in websocket-client
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE # Also disable certificate verification for testing

    try:
        async with websockets.connect(BITGET_WSS_PRIVATE_URL, ssl=ssl_context, ping_interval=10, ping_timeout=10) as websocket:
            logger.info("Connection established successfully!")
            # Keep the connection open for a short time to receive potential messages/errors
            try:
                # Correct way to listen for messages with a timeout
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        logger.info(f"Received message: {message}")
                    except asyncio.TimeoutError:
                        logger.info("No messages received within timeout. Stopping listening.")
                        break # Exit loop after timeout
            except websockets.exceptions.ConnectionClosedOK:
                logger.info("Connection closed normally.")
            except websockets.exceptions.ConnectionClosedError as e:
                logger.error(f"Connection closed with error: {e}")

    except websockets.exceptions.InvalidURI as e:
        logger.error(f"Invalid URI: {e}")
    except websockets.exceptions.InvalidHandshake as e:
        logger.error(f"Handshake failed: {e}")
    except ssl.SSLError as e:
        logger.error(f"SSL Error: {e}")
    except ConnectionRefusedError as e:
        logger.error(f"Connection Refused: {e}")
    except asyncio.TimeoutError:
        logger.error("Connection attempt timed out.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Connection attempt finished.")

if __name__ == "__main__":
    asyncio.run(connect_bitget())
    logger.info("Script finished.")