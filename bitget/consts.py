# Base Url
API_URL = 'https://api.bitget.com'
CONTRACT_WS_URL = 'wss://ws.bitget.com/mix/v1/stream' # Note: This seems like a V1 WS URL, might need updating if V2 WS is used elsewhere directly with this.

# V2 Public Time Endpoint
SERVER_TIMESTAMP_URL = '/api/v2/public/time'

# http header
CONTENT_TYPE = 'Content-Type'
OK_ACCESS_KEY = 'ACCESS-KEY' # Note: Naming convention seems borrowed (OKEx?), standard Bitget headers might differ slightly but library uses these.
OK_ACCESS_SIGN = 'ACCESS-SIGN'
OK_ACCESS_TIMESTAMP = 'ACCESS-TIMESTAMP'
OK_ACCESS_PASSPHRASE = 'ACCESS-PASSPHRASE'
APPLICATION_JSON = 'application/json'

# header key
LOCALE = 'locale'

# method
GET = "GET"
POST = "POST"
DELETE = "DELETE"

# sign type
RSA = "RSA"
SHA256 = "SHA256"
SIGN_TYPE = SHA256 # Default sign type used by the library

# ws auth path (used in V1 style WS auth signing)
REQUEST_PATH = '/user/verify'
