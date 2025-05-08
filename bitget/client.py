import requests
import json
import logging # Added logging
from . import consts as c, utils, exceptions

# Initialize logger for this module
logger = logging.getLogger(__name__)

class Client(object):

    def __init__(self, api_key, api_secret_key, passphrase, use_server_time=False, first=False, base_url=c.API_URL):

        self.API_KEY = api_key
        self.API_SECRET_KEY = api_secret_key
        self.PASSPHRASE = passphrase
        self.use_server_time = use_server_time
        self.first = first
        self.BASE_URL = base_url # Store the base URL

    def _request(self, method, request_path, params, cursor=False):
        if method == c.GET:
            request_path = request_path + utils.parse_params_to_str(params)
        # url
        url = self.BASE_URL + request_path # Use the stored base URL

        # 获取本地时间 (Get local time)
        timestamp = utils.get_timestamp()

        # sign & header
        if self.use_server_time:
            # 获取服务器时间接口 (Get server time)
            server_timestamp = self._get_timestamp()
            if server_timestamp:
                 timestamp = server_timestamp
                 logger.debug(f"Using server time: {timestamp}")
            else:
                 logger.warning("Failed to get server time, using local time.")
                 # Fallback to local time if server time fetch fails

        body = json.dumps(params) if method == c.POST else ""
        sign = utils.sign(utils.pre_hash(timestamp, method, request_path, str(body)), self.API_SECRET_KEY)
        if c.SIGN_TYPE == c.RSA:
            sign = utils.signByRSA(utils.pre_hash(timestamp, method, request_path, str(body)), self.API_SECRET_KEY)
        header = utils.get_header(self.API_KEY, sign, timestamp, self.PASSPHRASE)

        if self.first:
            logger.debug(f"Request URL: {url}")
            logger.debug(f"Request Method: {method}")
            logger.debug(f"Request Body: {body}")
            logger.debug(f"Request Headers: {header}")
            # print("sign:", sign) # Avoid logging signature
            self.first = False

        # send request
        response = None
        logger.debug(f"Sending {method} request to {url}")
        try:
            if method == c.GET:
                response = requests.get(url, headers=header, timeout=10) # Added timeout
            elif method == c.POST:
                response = requests.post(url, data=body, headers=header, timeout=10) # Added timeout
            elif method == c.DELETE:
                response = requests.delete(url, headers=header, timeout=10) # Added timeout
            else:
                 raise ValueError(f"Unsupported HTTP method: {method}")

            logger.debug(f"Response Status Code: {response.status_code}")
            logger.debug(f"Response Text: {response.text[:500]}...") # Log truncated response

            # exception handle
            if not str(response.status_code).startswith('2'):
                raise exceptions.BitgetAPIException(response)

            try:
                res_header = response.headers
                res_json = response.json() # Parse JSON once

                # Check for API-level status within the JSON response
                if isinstance(res_json, dict):
                    api_code = res_json.get('code')
                    api_msg = res_json.get('msg')
                    if api_code == '00000':
                        # This is a success code, do not log as a warning.
                        # Optionally, could log as INFO if detailed success logging is desired:
                        # logger.info(f"API call successful: code={api_code}, msg={api_msg}")
                        pass
                    else:
                        # This is an actual error code from the API
                        logger.warning(f"API returned error: code={api_code}, msg={api_msg}")
                        # Depending on strictness, could raise exception here:
                        # raise exceptions.BitgetAPIException(response)

                if cursor:
                    r = dict()
                    try:
                        # Use standard header names (case-insensitive access)
                        r['before'] = res_header.get('OK-BEFORE') or res_header.get('ok-before')
                        r['after'] = res_header.get('OK-AFTER') or res_header.get('ok-after')
                    except Exception as e:
                         logger.warning(f"Could not parse cursor headers: {e}")
                    return res_json, r
                else:
                    return res_json

            except ValueError: # Includes JSONDecodeError
                logger.error(f"Invalid JSON Response: {response.text}")
                raise exceptions.BitgetRequestException('Invalid JSON Response: %s' % response.text)
        except requests.exceptions.RequestException as e:
             logger.error(f"HTTP Request failed: {e}")
             raise exceptions.BitgetRequestException(f"Request failed: {e}")


    def _request_without_params(self, method, request_path):
        return self._request(method, request_path, {})

    def _request_with_params(self, method, request_path, params, cursor=False):
        return self._request(method, request_path, params, cursor)

    def _get_timestamp(self):
        """Fetches server time, checking common response structures."""
        url = c.API_URL + c.SERVER_TIMESTAMP_URL
        try:
            response = requests.get(url, timeout=5) # Add timeout
            logger.debug(f"Server time response status: {response.status_code}")
            if response.status_code == 200:
                res_json = response.json()
                logger.debug(f"Server time response JSON: {res_json}")
                # Check common locations for the timestamp
                if isinstance(res_json, dict):
                    data = res_json.get('data')
                    if isinstance(data, dict):
                        if 'serverTime' in data: return data['serverTime'] # Check for 'serverTime'
                        if 'ts' in data: return data['ts']
                        if 'timestamp' in data: return data['timestamp']
                    if 'ts' in res_json: return res_json['ts']
                    if 'timestamp' in res_json: return res_json['timestamp']

                # If not found in expected places, log warning and return None
                logger.warning(f"Timestamp key ('serverTime', 'ts', 'timestamp') not found in expected locations in server time response: {res_json}")
                return None
            else:
                logger.error(f"Failed to fetch server time, status code: {response.status_code}, response: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching server time: {e}")
            return None
        except ValueError: # Includes JSONDecodeError
            logger.error(f"Invalid JSON response from server time endpoint: {response.text}")
            return None
