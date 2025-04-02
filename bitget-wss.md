Demo Trading

Demo trading allows you to practice trading and test strategies in a real-market environment using virtual funds, helping you improve your skills and reduce the risk of losses.KYC is needed.

Websocket Demo Coin Trading

Bitget websocket also supports the demo coin trading,please use a real trading API key to make calls for demo symbol
Public channel: wss://ws.bitget.com/v2/ws/public
Private channel: wss://ws.bitget.com/v2/ws/private
Tickers Channel

In websocket subscribe, simply use the demo symbol and demo coin if any
Request Example

{
  "op":"subscribe",
  "args":[
    {
        "instType": "SUSDT-FUTURES",
        "channel": "ticker",
        "instId": "SBTCSUSDT"
    }
  ]
}

Successful Response Example

{
  "event":"subscribe",
  "arg":{
        "instType": "SUSDT-FUTURES",
        "channel": "ticker",
        "instId": "SBTCSUSDT"
  }
}

Push Data Example

{
    "action": "snapshot",
    "arg": {
        "instType": "SUSDT-FUTURES",
        "channel": "ticker",
        "instId": "SBTCSUSDT"
    },
    "data": [
        {
            "instId": "SBTCSUSDT",
            "last": "27000.5",
            "bidPr": "27000",
            "askPr": "27000.5",
            "bidSz": "2.71",
            "askSz": "8.76",
            "open24h": "27000.5",
            "high24h": "30668.5",
            "low24h": "26999.0",
            "priceChangePercent": "-0.00002",
            "fundingRate": "0.000010",
            "nextFundingTime": 1695722400000,
            "markPrice": "27000.0",
            "indexPrice": "25702.4",
            "quantity": "929.502",
            "baseVolume": "368.900",
            "quoteVolume": "10152429.961",
            "openUtc": "27000.5",
            "symbolType": 1,
            "symbol": "SBTCSUSDT",
            "deliveryPrice": "0",
            "ts": 1695715383021
        }
    ],
    "ts": 1695715383039
}

PUBLIC WEBSOKET:

Market Channel
Description

Retrieve the latest traded price, bid price, ask price and 24-hour trading volume of the instruments. When there is a change (deal, buy, sell, issue): 100ms to 300ms.
Request Example

{
    "op": "subscribe",
    "args": [
        {
            "instType": "USDT-FUTURES",
            "channel": "ticker",
            "instId": "BTCUSDT"
        }
    ]
}

Request Parameters
Parameter	Type	Required	Description
op	String	Yes	Operation, subscribe unsubscribe
args	List<Object>	Yes	List of channels to request subscription
> instType	String	Yes	Product type
> channel	String	Yes	Channel name
> instId	String	Yes	Product ID
E.g. ETHUSDT
Response Example

{
    "event": "subscribe",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "ticker",
        "instId": "BTCUSDT"
    }
}

Response Parameters
Parameter	Type	Description
event	String	Event
arg	Object	Subscribed channels
> instType	String	Product type
> channel	String	Channel name
> instId	String	Product ID
E.g. ETHUSDT
code	String	Error code
msg	String	Error message
Push Data

{
    "action": "snapshot",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "ticker",
        "instId": "BTCUSDT"
    },
    "data": [
        {
            "instId": "BTCUSDT",
            "lastPr": "27000.5",
            "bidPr": "27000",
            "askPr": "27000.5",
            "bidSz": "2.71",
            "askSz": "8.76",
            "open24h": "27000.5",
            "high24h": "30668.5",
            "low24h": "26999.0",
            "change24h": "-0.00002",
            "fundingRate": "0.000010",
            "nextFundingTime": "1695722400000",
            "markPrice": "27000.0",
            "indexPrice": "25702.4",
            "holdingAmount": "929.502",
            "baseVolume": "368.900",
            "quoteVolume": "10152429.961",
            "openUtc": "27000.5",
            "symbolType": 1,
            "symbol": "BTCUSDT",
            "deliveryPrice": "0",
            "ts": "1695715383021"
        }
    ],
    "ts": 1695715383039
}

Push Parameters
Parameter	Type	Description
arg	Object	Channels with successful subscription
> instType	String	Product type
> channel	String	Channel name
> instId	String	Product ID
action	String	Push data action, snapshot or update
data	List	Subscription data
> instId	String	Product ID, BTCUSDT
>lastPr	String	Latest price
>askPr	String	Ask price
>bidPr	String	Bid price
>high24h	String	24h high
>low24h	String	24h low
>change24h	String	24h change
>fundingRate	String	Funding rate
>nextFundingTime	String	Next funding rate settlement time, Milliseconds format of timestamp Unix, e.g. 1597026383085
>ts	String	System time, Milliseconds format of current data timestamp Unix, e.g. 1597026383085
>markPrice	String	Mark price
>indexPrice	String	Index price
>holdingAmount	String	Open interest
>baseVolume	String	Trading volume of the coin
>quoteVolume	String	Trading volume of quote currency
>openUtc	String	Price at 00:00 (UTC)
>symbolType	Integer	SymbolType: 1->perpetual 2->delivery
>symbol	String	Trading pair
>deliveryPrice	String	Delivery price of the delivery futures, when symbolType = 1(perpetual) it is always 0
It will be pushed 1 hour before delivery
>bidSz	String	Buying amount
>askSz	String	selling amount
>open24h	String	Entry price of the last 24 hours, The opening time is compared on a 24-hour basis. i.e.: Now it is 7:00 PM of the 2nd day of the month, then the corresponding opening time is 7:00 PM of the 1st day of the month.
Candlestick Channel
Description

Retrieve the candlesticks data of a symbol. Data will be pushed every 500 ms.

The channel will push a snapshot after successful subscribed, later on the updates will be pushed

If intended to query history data in a customized time range, please refer to Get Candle Data
Request Example

{
    "op": "subscribe",
    "args": [
        {
            "instType": "USDT-FUTURES",
            "channel": "candle1m",
            "instId": "BTCUSDT"
        }
    ]
}

Request Parameters
Parameter	Type	Required	Description
op	String	Yes	Operation, subscribe unsubscribe
args	List<Object>	Yes	List of channels to request subscription
> instType	String	Yes	Product type
> channel	String	Yes	Channel name, candle1m (1 minute) candle5m (5 minutes) candle15m (15 minutes) candle30m (30 minutes) candle1H (1 hour) candle4H (4 hours) candle12H (12 hours)
candle1D (1 day) candle1W (1 week) candle6H (6 hours) candle3D (3 days) candle1M (1-month line) candle6Hutc (6-hour line, UTC) candle12Hutc (12-hour line, UTC)
candle1Dutc (1-day line, UTC) candle3Dutc (3-day line, UTC) candle1Wutc (weekly line, UTC) candle1Mutc (monthly line. UTC)
> instId	String	Yes	Product ID
E.g. ETHUSDT
Response Example

{
    "event": "subscribe",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "candle1m",
        "instId": "BTCUSDT"
    }
}

Response Parameters
Parameter	Type	Description
event	String	Yes
Event
arg	Object	Subscribed channels
> channel	String	Channel name, candle1m (1 minute) candle5m (5 minutes) candle15m (15 minutes) candle30m (30 minutes) candle1H (1 hour) candle4H (4 hours) candle12H (12 hours)
candle1D (1 day) candle1W (1 week) candle6H (6 hours) candle3D (3 days) candle1M (1-month line) candle6Hutc (6-hour line, UTC) candle12Hutc (12-hour line, UTC)
candle1Dutc (1-day line, UTC) candle3Dutc (3-day line, UTC) candle1Wutc (weekly line, UTC) candle1Mutc (monthly line. UTC)
> instType	String	Product type
> instId	String	Yes
E.g. ETHUSDT
code	String	Error code, returned only on error
msg	String	Error message
Push Data

{
    "action": "snapshot",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "candle1m",
        "instId": "BTCUSDT"
    },
    "data": [
        [
            "1695685500000",
            "27000",
            "27000.5",
            "27000",
            "27000.5",
            "0.057",
            "1539.0155",
            "1539.0155"
        ]
    ],
    "ts": 1695715462250
}

Push Parameters
Parameter	Type	Description
arg	Object	Channels with successful subscription
> channel	String	Channel name
> instId	String	Product ID
> instType	String	Product type
data	List<String>	Subscription data
> index[0]	String	Start time, milliseconds format of Unix timestamp, e.g.1597026383085
> index[1]	String	Opening price
> index[2]	String	Highest price
> index[3]	String	Lowest price
> index[4]	String	Closing price
> index[5]	String	The value is the trading volume of left coin
> index[6]	String	Trading volume of quote currency
> index[7]	String	Trading volume of USDT
Depth Channel
Description

This is the channel to get the depth data
Default data push frequency for books, books5, books15 is 150ms
Default data push frequency for books1:100ms

    books: All levels of depth. First update pushed is full data: snapshot, and then push the update data: update
    books1: 1st level of depth. Push snapshot each time
    books5: 5 depth levels. Push snapshot each time
    books15: 15 depth levels. Push snapshot each time

Checksum
Calculate Checksum

1. More than 25 levels of bid and ask
A local snapshot of market depth (only 2 levels of the orderbook are shown here, while 25 levels of orderbook should actually be intercepted):
    "bids": [
      [ 43231.1, 4 ],   //bid1
      [ 43231,   6 ]    //bid2
    ]
    "asks": [
      [ 43232.8, 9 ],   //ask1
      [ 43232.9, 8 ]    //ask2
    ]
Build the string to check CRC32:
"43231.1:4:43232.8:9:43231:6:43232.9:8"
The sequence:
"bid1[price:amount]:ask1[price:amount]:bid2[price:amount]:ask2[price:amount]"

2. Less than 25 levels of bid or ask
A local snapshot of market depth:
    "bids": [
      [ 3366.1, 7 ] //bid1
    ]
    "asks": [
      [ 3366.8, 9 ],    //ask1
      [ 3368  , 8 ],    //ask2
      [ 3372  , 8 ]     //ask3
    ]

Build the string to check CRC32:
"3366.1:7:3366.8:9:3368:8:3372:8"
The sequence:
"bid1[price:amount]:ask1[price:amount]:ask2[price:amount]:ask3[price:amount]"

This mechanism can assist users in checking the accuracy of depth(order book) data.

Merging update data into snapshot

After subscribe to the channel (such as books 400 levels) of Order book , user first receive the initial snapshot of market depth. Afterwards the incremental update is subsequently received, user are responsible to update the snapshot from client side.

    If there are any levels with same price from the updates, compare the amount with the snapshot order book:

    If the amount is 0, delete this depth data.

    If the amount changes, replace the depth data.

    If there is no level in the snapshot with same price from the update, insert the update depth information into the snapshot sort by price (bid in descending order, ask in ascending order).

Calculate Checksum

Use the first 25 bids and asks in the local snapshot to build a string (where a colon connects the price and amount in an ask or a bid), and then calculate the CRC32 value (32-bit signed integer).

    When the bid and ask depth data exceeds 25 levels, each of them will intercept 25 levels of data, and the string to be checked is queued in a way that the bid and ask depth data are alternately arranged. Such as: bid1[price:amount]:ask1[price:amount]:bid2[price:amount]:ask2[price:amount]...
    When the bid or ask depth data is less than 25 levels, the missing depth data will be ignored. Such as: bid1[price:amount]:ask1[price:amount]:ask2[price:amount]:ask3[price:amount]...
    If price is '0.5000', DO NOT calculate the checksum by '0.5', please DO use the original value

Request Example

{
    "op": "subscribe",
    "args": [
        {
            "instType": "USDT-FUTURES",
            "channel": "books5",
            "instId": "BTCUSDT"
        }
    ]
}

Request Parameters
Parameter	Type	Required	Description
op	String	Yes	Operation, subscribe unsubscribe
args	List<Object>	Yes	List of channels to request subscription
> instType	String	Yes	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> channel	String	Yes	Channel name: books/books1/books5/books15
> instId	String	Yes	Product ID
Response Example

{
    "event": "subscribe",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "books5",
        "instId": "BTCUSDT"
    }
}

Response Parameters
Parameter	Type	Description
event	String	Event,
arg	Object	Subscribed channels
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> channel	String	Channel name, books/books1/books5/books15
> instId	String	Product ID, e.g. ETHUSDT
msg	String	Error message
code	String	Error code, returned only on error
Push Data

{
    "action": "snapshot",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "books5",
        "instId": "BTCUSDT"
    },
    "data": [
        {
            "asks": [
                [
                    "27000.5",
                    "8.760"
                ],
                [
                    "27001.0",
                    "0.400"
                ]
            ],
            "bids": [
                [
                    "27000.0",
                    "2.710"
                ],
                [
                    "26999.5",
                    "1.460"
                ]
            ],
            "checksum": 0,
            "ts": "1695716059516"
        }
    ],
    "ts": 1695716059516
}

Push Parameters
Parameter	Type	Description
arg	Object	Channels with successful subscription
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> channel	String	Channel name, books/books1/books5/books15
> instId	String	Product ID
action	String	Push data action, Incremental push data or full volume push data
data	List<Object>	Subscription data
> asks	List<String>	Seller depth
> bids	List<String>	Buyer depth
> ts	String	Match engine timestamp(ms), e.g. 1597026383085
> checksum	Long	Testing and
Public Trade Channel
Description

Get the public trade data(taker orders)
Request Example

{
    "op": "subscribe",
    "args": [
        {
            "instType": "USDT-FUTURES",
            "channel": "trade",
            "instId": "BTCUSDT"
        }
    ]
}

Request Parameters
Parameter	Type	Required	Description
op	String	Yes	Operation, subscribe unsubscribe
args	List<Object>	Yes	op list
> instType	String	Yes	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> channel	String	Yes	Channel, trade
> instId	String	Yes	Product ID
e.g: ETHUSDT
Response Example

{
    "event": "subscribe",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "trade",
        "instId": "BTCUSDT"
    }
}

Response Parameters
Parameter	Type	Description
event	String	event
arg	Object	arg list
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> channel	String	Channel, trade
> instId	String	Symbol name
e.g: ETHUSDT
code	String	Error code
msg	String	Error message
Push Data

{
    "action": "snapshot",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "trade",
        "instId": "BTCUSDT"
    },
    "data": [
        {
            "ts": "1695716760565",
            "price": "27000.5",
            "size": "0.001",
            "side": "buy",
            "tradeId": "1111111111"
        },
        {
            "ts": "1695716759514",
            "price": "27000.0",
            "size": "0.001",
            "side": "sell",
            "tradeId": "1111111111"
        }
    ],
    "ts": 1695716761589
}

Push Parameters
Parameter	Type	Description
action	String	action
arg	Object	arg
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> channel	String	Channel, trade
> instId	String	Symbol: ETHUSDT
data	List<Object>	Data
> ts	String	Fill time: 1597026383085
> price	String	Filled price
> size	String	Filled amount
> side	String	Filled side, sell/buy
> tradeId	String	tradeId

PRIVATE WEBSOCKET:

Account channel
Description

Subscribe account channel

Data will be pushed when the following events occurred:

    Transfer balance to Futures account
    Trading voucher deposit
    Open/close orders are filled

Request Example

{
    "op": "subscribe",
    "args": [
        {
            "instType": "USDT-FUTURES",
            "channel": "account",
            "coin": "default"
        }
    ]
}

Request Parameters
Parameter	Type	Required	Description
op	String	Yes	Operation, subscribe unsubscribe
args	List<Object>	Yes	List of channels to request subscription
> instType	String	Yes	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> channel	String	Yes	Channel name
> coin	String	Yes	Coin name，default represents all the coins，Only default is supported now
Response Example

{
    "event": "subscribe",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "account",
        "coin": "default"
    }
}

Response Parameters
Parameter	Type	Description
event	String	Yes
Operation
arg	Object	Subscribed channels
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Futures settled in cryptocurrencies
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo trading
SCOIN-FUTURES Futures settled in cryptocurrencies demo traing
SUSDC-FUTURES USDC professional futures demo trading
> channel	String	Channel name
> coin	String	default
code	String	Error code
msg	String	Error message
Push Data

{
    "action": "snapshot",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "account",
        "coin": "default"
    },
    "data": [
      {
        "marginCoin": "USDT",
        "frozen": "0.00000000",
        "available": "11.98545761",
        "maxOpenPosAvailable": "11.98545761",
        "maxTransferOut": "11.98545761",
        "equity": "11.98545761",
        "usdtEquity": "11.985457617660",
        "crossedRiskRate": "0",
        "unrealizedPL": "0.000000000000"
      }
    ],
    "ts": 1695717225146
}

Push Parameters
Parameter	Type	Description
action	String	snapshot
arg	Object	Channels to request subscription
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Futures settled in cryptocurrencies
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo trading
SCOIN-FUTURES Futures settled in cryptocurrencies demo traing
SUSDC-FUTURES USDC professional futures demo trading
> channel	String	Channel name
> coin	String	default
data	List<Object>	Subscription data
>marginCoin	String	Margin coin
>frozen	String	Locked quantity (margin coin)
>available	String	Currently available assets
>maxOpenPosAvailable	String	Maximum available balance to open positions
>maxTransferOut	String	Maximum transferable amount
>equity	String	Account assets
>usdtEquity	String	Account equity in USD
>crossedRiskRate	String	Risk ratio in cross margin mode
>unrealizedPL	String	Unrealized PnL
Position Channel
Description

Subscribe the position channel

Data will be pushed when the following events occurred:

    Open/Close orders are created
    Open/Close orders are filled
    Orders are canceled

Request Example

{
    "op": "subscribe",
    "args": [
        {
            "instType": "USDT-FUTURES",
            "channel": "positions",
            "instId": "default"
        }
    ]
}

Request Parameters
Parameter	Type	Required	Description
op	String	Yes	Operation, subscribe unsubscribe
args	List<Object>	Yes	List of channels to request subscription
> channel	String	Yes	Channel name: positions
> instType	String	Yes	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	Yes	Symbol name,defaultrepresents all the symbols，Only default is supported now
Response Example

{
    "event": "subscribe",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "positions",
        "instId": "default"
    }
}

Response Parameters
Parameter	Type	Description
event	String	Event
arg	Object	Subscribed channels
> channel	String	Channel name: positions
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	default
code	String	Error code
msg	String	Error message
Push Data

{
    "action": "snapshot",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "positions",
        "instId": "default"
    },
    "data": [
        {
            "posId": "1",
            "instId": "ETHUSDT",
            "marginCoin": "USDT",
            "marginSize": "9.5",
            "marginMode": "crossed",
            "holdSide": "short",
            "posMode": "hedge_mode",
            "total": "0.1",
            "available": "0.1",
            "frozen": "0",
            "openPriceAvg": "1900",
            "leverage": 20,
            "achievedProfits": "0",
            "unrealizedPL": "0",
            "unrealizedPLR": "0",
            "liquidationPrice": "5788.108475905242",
            "keepMarginRate": "0.005",
            "marginRate": "0.004416374196",
            "cTime": "1695649246169",
            "breakEvenPrice": "24778.97",
            "totalFee": "1.45",
            "deductedFee": "0.388",
            "uTime": "1695711602568",
            "autoMargin": "off"
        }
    ],
    "ts": 1695717430441
}

Push Parameters
Parameter	Type	Description
action	String	'snapshot'
arg	Object	Channels with successful subscription
> channel	String	Channel name: positions
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	default
data	List<Object>	Subscription data
> posId	String	Position ID
> instId	String	Product ID,
delivery contract reference：https://www.bitget.com/api-doc/common/release-note
> marginCoin	String	Currency of occupied margin
> marginSize	String	Occupied margin (amount)
> marginMode	String	Margin mode
> holdSide	String	Position direction
> posMode	String	Position mode
> total	String	Open position size
> available	String	Size of positions that can be closed
> frozen	String	Amount of frozen margin
> openPriceAvg	String	Average entry price
> leverage	String	Leverage
> achievedProfits	String	Realized PnL
> unrealizedPL	String	Unrealized PnL
> unrealizedPLR	String	Unrealized ROI
> liquidationPrice	String	Estimated liquidation price
> keepMarginRate	String	Maintenance margin rate
> isolatedMarginRate	String	Actual margin ratio under isolated margin mode
> marginRate	String	Occupancy rate of margin
> breakEvenPrice	String	Position breakeven price
> totalFee	String	Funding fee, the accumulated value of funding fee during the position,The initial value is empty, indicating that no funding fee has been charged yet.
> deductedFee	String	Deducted transaction fees: transaction fees deducted during the position
> cTime	String	Position creation time, milliseconds format of Unix timestamp, e.g.1597026383085
> uTime	String	Lastest position update time, milliseconds format of Unix timestamp, e.g.1597026383085
Fill Channel
Description

Trade details channel

Data will be pushed when order filled.
Request Example

{
    "op": "subscribe",
    "args": [
        {
            "instType": "USDT-FUTURES",
            "channel": "fill",
            "instId": "default"
        }
    ]
}

Request Parameters
Parameter	Type	Required	Description
op	String	Yes	Operation, subscribe unsubscribe
args	List<Object>	Yes	List of channels to request subscription
> channel	String	Yes	Channel name: fill
> instType	String	Yes	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	No	Product ID or default，
delivery contract reference：https://www.bitget.com/api-doc/common/release-note#optimization-of-delivery-futures
Response Example

{
    "event": "subscribe",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "fill",
        "instId": "default"
    }
}

Response Parameters
Parameter	Type	Description
event	String	Event
arg	Object	Subscribed channels
> channel	String	Channel name: fill
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	Product ID or default
delivery contract reference：https://www.bitget.com/api-doc/common/release-note#optimization-of-delivery-futures
code	String	Error code
msg	String	Error message
Push Data

{
    "action":"snapshot",
    "arg":{
        "instType":"USDT-FUTURES",
        "channel":"fill",
        "instId":"default"
    },
    "data":[
        {
            "orderId":"111",
            "tradeId":"222",
            "symbol":"BTCUSDT",
            "side":"buy",
            "orderType":"market",
            "posMode":"one_way_mode",
            "price":"51000.5",
            "baseVolume":"0.01",
            "quoteVolume":"510.005",
            "profit":"0",
            "tradeSide":"open",
            "tradeScope":"taker",
            "feeDetail":[
                {
                    "feeCoin":"USDT",
                    "deduction":"no",
                    "totalDeductionFee":"0",
                    "totalFee":"-0.183717"
                }
            ],
            "cTime":"1703577336606",
            "uTime":"1703577336606"
        }
    ],
    "ts":1703577336700
}

推送数据参数
返回字段	参数类型	字段说明
action	String	snapshot
arg	Object	Channels with successful subscription
> channel	String	Channel name: fill
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	Product ID or default
delivery contract reference：https://www.bitget.com/api-doc/common/release-note#optimization-of-delivery-futures
data	List<Object>	Subscription data
> orderId	String	Order ID
> tradeId	String	Trade ID
> symbol	String	Symbol Name
> side	String	Trade direction
buy: Buy
sell: Sell
Please note, for this channel, in hedge position mode, Open Long and Close Short, the "side" will be buy; Close Long and Open Short, the "side" will be sell
> orderType	String	Order type
limit limit order
market market order
> posMode	String	Hold Mode
one_way_mode
hedge_mode
> price	String	Order price
> baseVolume	String	Amount of base coin
> quoteVolume	String	Amount of denomination coin
> profit	String	Realized PnL
> tradeSide	String	Trade type
close: Close (open and close mode)
open: Open (open and close mode)
reduce_close_long: Liquidate partial long positions for hedge position mode
reduce_close_short：Liquidate partial short positions for hedge position mode
burst_close_long：Liquidate long positions for hedge position mode
burst_close_short：Liquidate short positions for hedge position mode
offset_close_long：Liquidate partial long positions for netting for hedge position mode
offset_close_short：Liquidate partial short positions for netting for hedge position mode
delivery_close_long：Delivery long positions for hedge position mode
delivery_close_short：Delivery short positions for hedge position mode
dte_sys_adl_close_long：ADL close long position for hedge position mode
dte_sys_adl_close_short：ADL close short position for hedge position mode
buy_single：Buy, one way postion mode
sell_single：Sell, one way postion mode
reduce_buy_single：Liquidate partial positions, buy, one way position mode
reduce_sell_single：Liquidate partial positions, sell, one way position mode
burst_buy_single：Liquidate short positions, buy, one way postion mode
burst_sell_single：Liquidate partial positions, sell, one way position mode
delivery_sell_single：Delivery sell, one way position mode
delivery_buy_single：Delivery buy, one way position mode
dte_sys_adl_buy_in_single_side_mode：ADL close position, buy, one way position mode
dte_sys_adl_sell_in_single_side_mode：ADL close position, sell, one way position mode
> tradeScope	String	The liquidity direction
taker
maker
> feeDetail	List<Object>	Transaction fee of the order
  >> deduction	String	deduction
yes
no
  >> totalDeductionFee	String	Fee of deduction
  >> totalFee	String	Fee of all
  >> feeCoin	String	Currency of transaction fee
> cTime	String	Create Time，milliseconds format of Unix timestamp, e.g.1597026383085
> uTime	String	Update Time，milliseconds format of Unix timestamp, e.g.1597026383085
Order Channel
Description

Subscribe the order channel

Data will be pushed when the following events occured:

    Open/Close orders are created
    Open/Close orders are filled
    Orders canceled

Request Example

{
    "op": "subscribe",
    "args": [
        {
            "instType": "USDT-FUTURES",
            "channel": "orders",
            "instId": "default"
        }
    ]
}

Request Parameters
Parameter	Type	Required	Description
op	String	Yes	Operation, subscribe unsubscribe
args	List<Object>	Yes	List of channels to request subscription
> channel	String	Yes	Channel name: orders
> instType	String	Yes	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	No	Trading pair, e.g. BTCUSDT
default: All trading pairs
For settled Futures, it only supports default
Response Example

{
    "event": "subscribe",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "orders",
        "instId": "default"
    }
}

Response Parameters
Parameter	Type	Description
event	String	Event
arg	Object	Subscribed channels
> channel	String	Channel name: orders
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	Product ID
code	String	Error code
msg	String	Error message
Push Data

{
    "action": "snapshot",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "orders",
        "instId": "default"
    },
    "data": [
        {
            "accBaseVolume": "0.01",
            "cTime": "1695718781129",
            "clientOId": "1",
            "feeDetail": [
                {
                    "feeCoin": "USDT",
                    "fee": "-0.162003"
                }
            ],
            "fillFee": "-0.162003",
            "fillFeeCoin": "USDT",
            "fillNotionalUsd": "270.005",
            "fillPrice": "27000.5",
            "baseVolume": "0.01",
            "fillTime": "1695718781146",
            "force": "gtc",
            "instId": "BTCUSDT",
            "leverage": "20",
            "marginCoin": "USDT",
            "marginMode": "crossed",
            "notionalUsd": "270",
            "orderId": "1",
            "orderType": "market",
            "pnl": "0",
            "posMode": "hedge_mode",
            "posSide": "long",
            "price": "0",
            "priceAvg": "27000.5",
            "reduceOnly": "no",
            "stpMode": "cancel_taker",
            "side": "buy",
            "size": "0.01",
            "enterPointSource": "WEB",
            "status": "filled",
            "tradeScope": "T",
            "tradeId": "1111111111",
            "tradeSide": "open",
            "presetStopSurplusPrice": "21.4",
            "totalProfits": "11221.45",
            "presetStopLossPrice": "21.5",
            "uTime": "1695718781146"
        }
    ],
    "ts": 1695718781206
}

Push Parameters
Parameter	Type	Description
arg	Object	Channels with successful subscription
> channel	String	Channel name: orders
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	Product ID
delivery contract reference：https://www.bitget.com/api-doc/common/release-note
data	List<Object>	Subscription data
> orderId	String	Order ID
> clientOId	String	Customized order ID
> price	String	Order price
> size	String	Original order amount in coin
> posMode	String	Position Mode
one_way_mode：one-way mode
hedge-mode: hedge mode
> enterPointSource	String	Order source
WEB: Orders created on the website
API: Orders created on API
SYS: System managed orders, usually generated by forced liquidation logic
ANDROID: Orders created on the Android app
IOS: Orders created on the iOS app
> tradeSide	String	Trade Side trading direction
> notionalUsd	String	Estimated USD value of orders
> orderType	String	Order type
limit: limit order
market: market order
> force	String	Order validity period
> side	String	Order direction
> posSide	String	Position direction
long: hedge-mode, long position
short: hedge-mode, short position
net: one-way-mode position
> marginMode	String	Margin mode
crossed: crossed mode
isolated: isolated mode
> marginCoin	String	Margin coin
> fillPrice	String	Latest filled price
> tradeId	String	Latest transaction ID
> baseVolume	String	Number of latest filled orders
> fillTime	String	Latest transaction time. Unix millisecond timestamp, e.g. 1690196141868
> fillFee	String	Transaction fee of the latest transaction, negative value
> fillFeeCoin	String	Currency of transaction fee of the latest transaction
> tradeScope	String	The liquidity direction of the latest transaction T: taker M maker
> accBaseVolume	String	Total filled quantity
> fillNotionalUsd	String	USD value of filled orders
> priceAvg	String	Average filled price
If the filled size is 0, the field is 0; if the order is not filled, the field is also 0; This field will not be pushed if the order is cancelled
> status	String	Order status
live: New order, waiting for a match in orderbook
partially_filled: Partially filled
filled: All filled
canceled: the order is cancelled
> leverage	String	Leverage
> feeDetail	List<Object>	Transaction fee of the order
>> feeCoin	String	The currency of the transaction fee. The margin is charged.
>> fee	String	Order transaction fee, the transaction fee charged by the platform from the user.
> pnl	String	Profit
> uTime	String	Order update time, Milliseconds format of updated data timestamp Unix, e.g. 1597026383085
> cTime	String	Order creation time, milliseconds format of Unix timestamp, e.g.1597026383085
> reduceOnly	String	Reduce-only
yes: Yes
no: No
> presetStopSurplusPrice	String	Set TP price
> presetStopLossPrice	String	Set SL price
> stpMode	String	STP Mode
none not setting STP
cancel_taker cancel taker order
cancel_maker cancel maker order
cancel_both cancel both of taker and maker orders
> totalProfits	String	Total profits
Trigger Order Channel
Description

Subscribe trigger order channel

Data will be pushed when the trigger plans are opened,cancelled,modified,triggered.
Request Example

{
    "op": "subscribe",
    "args": [
        {
            "instType": "USDT-FUTURES",
            "channel": "orders-algo",
            "instId": "default"
        }
    ]
}

Request Parameters
Parameter	Type	Required	Description
op	String	Yes	Operation, subscribe unsubscribe
args	List<Object>	Yes	List of channels to request subscription
> channel	String	Yes	Channel name:orders-algo
> instType	String	Yes	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	No	Trading pair
Response Example

{
    "event": "subscribe",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "orders-algo",
        "instId": "default"
    }
}

Response Parameters
Parameter	Type	Description
event	String	Event
arg	Object	Subscribed channels
> channel	String	Channel name: orders-algo
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	Product ID
code	String	Error code
msg	String	Error message
Push Data

{
    "action": "snapshot",
    "arg": {
        "instType": "USDT-FUTURES",
        "channel": "orders-algo",
        "instId": "default"
    },
    "data": [
        {
            "instId": "BTCUSDT",
            "orderId": "1",
            "clientOid": "1",
            "triggerPrice": "27000.000000000",
            "triggerType": "fill_price",
            "triggerTime": "1695719197612",
            "planType": "pl",
            "price": "27000.000000000",
            "executePrice": "27000.000000000",
            "size": "0.020000000",
            "actualSize": "0.000000000",
            "orderType": "market",
            "side": "buy",
            "tradeSide": "open",
            "posSide": "long",
            "marginCoin": "USDT",
            "status": "live",
            "posMode": "hedge_mode",
            "enterPointSource": "web",
            "stopSurplusTriggerType": "fill_price",
            "stopLossTriggerType": "fill_price",
            "stpMode": "cancel_taker",
            "cTime": "1695719197612",
            "uTime": "1695719197612"
        }
    ],
    "ts": 1695719197733
}

Push Parameters
Parameter	Type	Description
action	String	Push action
arg	Object	Channels with successful subscription
> channel	String	Channel name: orders-algo
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	Product ID
delivery contract reference：https://www.bitget.com/api-doc/common/release-note
data	List<Object>	Subscription data
> instId	String	Product ID
delivery contract reference：https://www.bitget.com/api-doc/common/release-note
> orderId	String	Bot order ID
> clientOid	String	Customized bot order ID
> triggerPrice	String	Trigger price
> triggerType	String	Trigger type
fill_price: filled price
mark_price: mark price
> triggerTime	String	Trigger time, ms
> planType	String	Websocket trigger order type. Data will be pushed when modify,cancel,open,triggered the plan types below
pl:Default value, trigger order
tp:Partial take profit
sl:Partial stop loss
ptp:Position take profit
psl:Position stop loss
track:Trailing stop
mtpsl:Trailing TP/SL
> price	String	Order price
> executePrice	String	Execute price
> size	String	Original order amount in coin
> actualSize	String	Actual number of orders in coin
> orderType	String	Order type
limit: limit order
market
> side	String	Order direction,
> tradeSide	String	Trade Side trading direction
> posSide	String	Position direction;
> marginCoin	String	Margin coin
> status	String	Order status
live: plan order created
executed: executed
fail_execute: execute failed
cancelled: cancelled
executing: executing
> posMode	String	Position mode
one_way_mode: one-way position mode
hedge_mode: hedge postion mode
> enterPointSource	String	Order source
WEB: Orders created on the website
API: Orders created on API
SYS: System managed orders, usually generated by forced liquidation logic
ANDROID: Orders created on the Android app
IOS: Orders created on the iOS app
> stopSurplusPrice	String	Preset/Partial/Position position take-profit execution price;
1. When planType is pl, it represents the preset take-profit execution price.
2. When planType is tp, it represents the partial take-profit execution price.
3. When planType is ptp, it represents the position take-profit execution price.
> stopSurplusTriggerPrice	String	Preset/Partial/Position take-profit trigger price;
1. When planType is pl, it represents the preset take-profit trigger price.
2. When planType is tp, it represents the partial take-profit trigger price.
3. When planType is ptp, it represents the position take-profit trigger price.
It is empty when there is nothing.
> stopSurplusTriggerType	String	Preset/Partial/Position take-profit trigger type;
1. When planType is pl, it represents the preset take-profit trigger type.
2. When planType is tp, it represents the partial take-profit trigger type.
3. When planType is ptp, it represents the position take-profit trigger type.
It is empty when there is nothing.
> stopLossPrice	String	Preset/Partial/Position stop-loss execution price;
1. When planType is pl, it represents the preset stop-loss execution price.
2. When planType is sl, it represents the partial stop-loss execution price.
3. When planType is psl, it represents the position stop-loss execution price.
It is empty when there is nothing.
> stopLossTriggerPrice	String	Preset/Partial/Position stop-loss trigger price;
1. When planType is pl, it represents the preset stop-loss trigger price.
2. When planType is sl, it represents the partial stop-loss trigger price.
3. When planType is psl, it represents the position stop-loss trigger price.
> stopLossTriggerType	String	Preset/Partial/Position stop-loss trigger type;
1. When planType is pl, it represents the preset stop-loss trigger type.
2. When planType is sl, it represents the partial stop-loss trigger type.
3. When planType is psl, it represents the position stop-loss trigger type.
It is empty when there is nothing.
> uTime	String	Order update time, Milliseconds format of updated data timestamp Unix, e.g. 1597026383085
> stpMode	String	STP Mode
none not setting STP
cancel_taker cancel taker order
cancel_maker cancel maker order
cancel_both cancel both of taker and maker orders
History Position Channel
Description

Subscribe the position channel

Data will be pushed when the position totally closed
Request Example

{
    "args":[
        {
            "channel":"positions-history",
            "instId":"default",
            "instType":"USDT-FUTURES"
        }
    ],
    "op":"subscribe"
}

Request Parameters
Parameter	Type	Required	Description
op	String	Yes	subscribe unsubscribe
args	List<Object>	Yes	List of channels to request subscription
> channel	String	Yes	Channel name: positions-history
> instType	String	Yes	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	Yes	Symbol name,defaultrepresents all the symbols，Only default is supported now
Response Example

{
    "event":"subscribe",
    "arg":{
        "instType":"USDT-FUTURES",
        "channel":"positions-history",
        "instId":"default"
    }
}

Response Parameters
Parameter	Type	Description
event	String	Event
arg	Object	Subscribed channels
> channel	String	Channel name: positions-history
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	default
code	String	Error code
msg	String	Error message
Push Data

{
    "action":"snapshot",
    "arg":{
        "instType":"USDT-FUTURES",
        "channel":"positions-history",
        "instId":"default"
    },
    "data":[
        {
            "posId":"1",
            "instId":"BTCUSDT",
            "marginCoin":"USDT",
            "marginMode":"crossed",
            "holdSide":"short",
            "posMode":"one_way_mode",
            "openPriceAvg":"20000.0",
            "closePriceAvg":"26221.0",
            "openSize":"0.010",
            "closeSize":"0.010",
            "achievedProfits":"-62.21000000",
            "settleFee":"-0.02277989",
            "openFee":"-0.12000000",
            "closeFee":"-0.15732600",
            "cTime":"1696907951177",
            "uTime":"1697090609976"
        }
    ],
    "ts":1697099840122
}

Push Parameters
Parameter	Type	Description
action	String	'snapshot'
arg	Object	Channels with successful subscription
> channel	String	Channel name: positions-history
> instType	String	Product type
USDT-FUTURES USDT professional futures
COIN-FUTURES Mixed futures
USDC-FUTURES USDC professional futures
SUSDT-FUTURES USDT professional futures demo
SCOIN-FUTURES Mixed futures demo
SUSDC-FUTURES USDC professional futures demo
> instId	String	default
data	List<Object>	Subscription data
> posId	String	Position ID
> instId	String	Product ID
delivery contract reference：https://www.bitget.com/api-doc/common/release-note
> marginCoin	String	Currency of occupied margin
> marginMode	String	Margin mode
fixed: isolated mode
crossed: crossed mode
> holdSide	String	Position direction
> posMode	String	Position mode
> openPriceAvg	String	Average entry price
> closePriceAvg	String	Average close price
> openSize	String	Open size
> closeSize	String	Close size
> achievedProfits	String	Realized PnL
> settleFee	String	Settle fee
> openFee	String	Total open fee
> closeFee	String	Total close fee
> cTime	String	Position creation time, milliseconds format of Unix timestamp, e.g.1597026383085
> uTime	String	Lastest position update time, milliseconds format of Unix timestamp, e.g.1597026383085

V2 API Update Guide
Scope of changes
Interface aggregation

In V1, interface changes involving modifications to input parameters are typically addressed by introducing new interfaces, ensuring minimal impact on online users. Therefore, in V2, we have optimized shortcomings such as interface redundancy and confusion in business scenarios. For detailed information on interface aggregation across all business lines, please refer to the V1 and V2 Interface Mapping Tables.
Global symbol request rule changes

In V2, we reduced them to one parameter—symbol—corresponding to symbolName in V1. Additionally, business line notes such as SPBL and UMBCL were removed when passing the symbol value.
Optimization of global query rules

In terms of data retrieval for query interfaces, we have abandoned the pagination method of pageSize and pageNo used in V1. In V2, we replaced it with cursor-based pagination with idLessThan and limit. Based on real-life business scenarios, a time range data query was added to most query interfaces. In addition, user ID can be used as the request parameter in precise query scenarios with some trade record, trade execution, and order-related interfaces. The basic rules and scenarios for id, startTime, endTime, idLessThan, and limit are described as follows:

Basic rule: When querying data, the verification order for returned results is id > startTime + endTime > idLessThan. In other words, it first prioritizes precise queries using the id, then narrows down the data range with startTime and endTime, and finally uses the cursor idLessThan to retrieve a specified number of data entries based on the limit.
Standardization of naming rules for business line interfaces and parameters

In V1, there was a lack of consistency in parameters between business lines. Therefore, in V2, we have standardized the naming and format of parameters with the same meaning for scenarios spanning different business lines (spot, futures, leverage) and interface types (Rest/WebSocket).
Clearer catalog categorization

In V1, the categorization of interface catalogs was too vague, resulting in an excessive number of interfaces in some catalogs. This, in turn, led to difficulties in documentation query and poor user experience. In V2, we have adjusted the categorization and naming of the interface catalogs, making them more detailed, intuitive, thus avoiding readability and retrieval issues.
Accessibility of more in-depth information

For futures and spot trading pairs, we significantly increased the trading pair depth that can be accessed through the interfaces and standardized the tiers across different business lines.
Business Line	Version	Tier
Spot	V1	150/200
Spot	V2	1/5/15/50/max; default: 100. The max is determined by the highest tier available for the designated trading pair.
Future	V1	5/15/50/100
Future	V2	1/5/15/50/max; default: 100.The max is determined by the highest tier available for the designated trading pair.
Merge of futures trigger order and trailing stop-loss

In V2, we combined the trigger order and the trailing stop-loss into one, using the field planType to differentiate the order type.

The order placing conditions of the two are different. Different from the normal trigger order, the trailing stop-loss requires attention to the field callbackRatio, which is used to set the order-triggering percentage. stopSurplusTriggerPrice and stopLossTriggerPrice also require special attention as these two fields are used to determine the trail variance percentage that triggers the trailing stop-loss and take-profit orders.
Position opening/closing in futures order placement

In V2, we improved operations on positions in both one-way and hedging modes.

When placing an order, the field side and tradeSide are combined for parameter entry based on the position mode and direction.

Enumeration values of side and tradeSide:
Field name	Enumeration value	Description
side	buy	Buying
side	sell	selling
tradeSide	open	Opening a position
tradeSide	close	Closing a position

Operation rules of parameter entries in one-way or hedging modes:
Position mode	Parameter entries	Operation	Description
One-way mode	side: buy	Buying	In one-way mode, only one side is needed to indicate whether it is a buying or selling order
One-way mode	side: sell	Selling	In one-way mode, only one side is needed to indicate whether it is a buying or selling order
Hedging mode	side: buy; tradeSide: open	Opening a long position	In hedging mode, both side and tradeSide are needed to determine whether it is opening long/short or closing long/short
Hedging mode	side: sell; tradeSide: open	Opening a short position	In hedging mode, both side and tradeSide are needed to determine whether it is opening long/short or closing long/short
Hedging mode	side: buy; tradeSide: close	Closing a long position	In hedging mode, both side and tradeSide are needed to determine whether it is opening long/short or closing long/short
Hedging mode	side: sell; tradeSide: close	Closing a short position	In hedging mode, both side and tradeSide are needed to determine whether it is opening long/short or closing long/short
Optimization of delivery futures

In V2, we updated the naming rules for symbol in Coin-M delivery futures.

For Coin-M delivery futures, the format of symbol is trading pair + month code + year.

Examples and descriptions:
Symbol	Descriptions
BTCUSDH23	H means March (Q1) and 23 means the year 2023
BTCUSDM23	M means June (Q2) and 23 means the year 2023
BTCUSDU23	U means September (Q3) and 23 means the year 2023
BTCUSDZ23	Z means December (Q4) and 23 means the year 2023

The bolded letters H, M, U, and Z are some of the month codes. Month codes:
Month code	Month	Month code	Month
F	January	N	July
G	February	Q	August
H	March	U	September
J	April	V	October
K	May	X	November
M	June	Z	December
Optimized maximum/minimum order size logic of trading pairs

In V2, we have included parameter descriptions for maximum and minimum order sizes in interfaces that access spot and futures trading pairs. This enhancement enables users to access essential information about trading pairs, including minimum and maximum trading volumes, the maximum number of open orders (considering both trading pairs and products), price precision, amount precision, and other basic information.
Earn interfaces coming soon

To meet users' demand for crypto Earn products, we are about to launch interfaces for Savings and Shark Fin, covering features such as information retrieval, PnL statistics, asset analysis, subscription, and redemption. Both fixed and flexible Savings are supported, enhancing our digital asset management service.
Crypto Loan interfaces coming soon

Catering to the needs of investors seeking more conservative or flexible approaches to grow their wealth, we are about to launch interfaces for Crypto Loan. For users with lower risk tolerance who seek conservative and flexible borrowing solutions, we introduced the interface for Crypto Loan. Crypto Loan is a financial product that allows users to borrow fiat currency or cryptocurrencies by using their crypto assets as collateral.


The Bitget Crypto Loan API aims to help users get additional funds instantly without losing control over their own crypto assets.The whole process includes staking the collateral, obtaining the loan, adding to/withdrawing from the collateral, undergoing liquidation, paying interest, repaying the loan, and redeeming the collateral. It should be noted that, due to the nature of the digital currency market, the crypto market is considerably more volatile than the traditional market. As a result, future fluctuations in coin prices will impact the overall return.
Interface Mapping Tables
Spot
V1 Endpoint	V2 Endpoint
- GET /api/spot/v1/notice/queryAllNotices	- GET /api/v2/public/annoucements
- GET /api/spot/v1/public/time	- GET /api/v2/public/time
- GET /api/spot/v1/public/currencies	- GET /api/v2/spot/public/coins
- GET /api/spot/v1/public/products	- GET /api/v2/spot/public/symbols
- GET /api/spot/v1/public/product	- GET /api/v2/spot/public/symbols
- GET /api/spot/v1/market/ticker	- GET /api/v2/spot/market/tickers
- GET /api/spot/v1/market/tickers	- GET /api/v2/spot/market/tickers
- GET /api/spot/v1/market/fills-history	- GET /api/v2/spot/market/fills-history
- GET /api/spot/v1/market/fills	- GET /api/v2/spot/market/fills
- GET /api/spot/v1/market/candles	- GET /api/v2/spot/market/candles
- GET /api/spot/v1/market/history-candles	- GET /api/v2/spot/market/history-candles
- GET /api/spot/v1/market/depth	- GET /api/v2/spot/market/orderbook
- GET /api/spot/v1/market/merge-depth	- GET /api/v2/spot/market/orderbook
- GET /api/spot/v1/market/spot-vip-level	- GET /api/v2/spot/market/vip-fee-rate
- POST /api/spot/v1/wallet/transfer	- POST /api/v2/spot/wallet/transfer
- POST /api/spot/v1/wallet/transfer-v2	- POST /api/v2/spot/wallet/transfer
- POST /api/spot/v1/wallet/subTransfer	- POST /api/v2/spot/wallet/subaccount-transfer
- POST /api/spot/v1/wallet/withdrawal	- POST /api/v2/spot/wallet/withdrawal
- POST /api/spot/v1/wallet/withdrawal-v2	- POST /api/v2/spot/wallet/withdrawal
- POST /api/spot/v1/wallet/withdrawal-inner	- POST /api/v2/spot/wallet/withdrawal
- POST /api/spot/v1/wallet/withdrawal-inner-v2	- POST /api/v2/spot/wallet/withdrawal
- GET /api/spot/v1/wallet/deposit-address	- GET /api/v2/spot/wallet/deposit-address
- GET /api/spot/v1/wallet/deposit-list	- GET /api/v2/spot/wallet/deposit-records
- GET /api/spot/v1/wallet/withdrawal-list	- GET /api/v2/spot/wallet/withdrawal-records
- GET /api/user/v1/fee/query	- GET /api/v2/public/trade-rate
- GET /api/spot/v1/account/getInfo	- GET /api/v2/spot/account/info
- GET /api/spot/v1/account/assets	- GET /api/v2/spot/account/assets
- GET /api/spot/v1/account/assets-lite	- GET /api/v2/spot/account/assets
- POST /api/spot/v1/account/sub-account-spot-assets	- GET /api/v2/spot/account/subaccount-assets
- POST /api/spot/v1/account/bills	- GET /api/v2/spot/account/bills
- GET /api/spot/v1/account/transferRecords	- GET /api/v2/spot/account/transferRecords
- POST /api/spot/v1/trade/orders	- POST /api/v2/spot/trade/place-order
- POST /api/spot/v1/trade/batch-orders	- POST /api/v2/spot/trade/batch-orders
- POST /api/spot/v1/trade/cancel-order	- POST /api/v2/spot/trade/cancel-order
- POST /api/spot/v1/trade/cancel-order-v2	- POST /api/v2/spot/trade/cancel-order
- POST /api/spot/v1/trade/cancel-symbol-order	- POST /api/v2/spot/trade/cancel-symbol-order
- POST /api/spot/v1/trade/cancel-batch-orders	- POST /api/v2/spot/trade/batch-cancel-order
- POST /api/spot/v1/trade/cancel-batch-orders-v2	- POST /api/v2/spot/trade/batch-cancel-order
- POST /api/spot/v1/trade/orderInfo	- GET /api/v2/spot/trade/orderInfo
- POST /api/spot/v1/trade/open-order	- GET /api/v2/spot/trade/unfilled-orders
- POST /api/spot/v1/trade/history	- GET /api/v2/spot/trade/history-orders
- POST /api/spot/v1/trade/fills	- GET /api/v2/spot/trade/fills
- POST /api/spot/v1/plan/placePlan	- POST /api/v2/spot/trade/place-plan-order
- POST /api/spot/v1/plan/modifyPlan	- POST /api/v2/spot/trade/modify-plan-order
- POST /api/spot/v1/plan/cancelPlan	- POST /api/v2/spot/trade/cancel-plan-order
- POST /api/spot/v1/plan/currentPlan	- GET /api/v2/spot/trade/current-plan-order
- POST /api/spot/v1/plan/historyPlan	- GET /api/v2/spot/trade/history-plan-order
- POST /api/spot/v1/plan/batchCancelPlan	- POST /api/v2/spot/trade/batch-cancel-plan-order
- GET /api/p2p/v1/merchant/merchantList	- GET /api/v2/p2p/merchantList
- GET /api/p2p/v1/merchant/merchantInfo	- GET /api/v2/p2p/merchantInfo
- GET /api/p2p/v1/merchant/advList	- GET /api/v2/p2p/advList
- GET /api/p2p/v1/merchant/orderList	- GET /api/v2/p2p/orderList
- POST /api/user/v1/sub/virtual-create	- POST /api/v2/user/create-virtual-subaccount
- POST /api/user/v1/sub/virtual-modify	- POST /api/v2/user/modify-virtual-subaccount
- POST /api/user/v1/sub/virtual-api-batch-create	- POST /api/v2/user/batch-create-subaccount-and-apikey
- GET /api/user/v1/sub/virtual-list	- GET /api/v2/user/virtual-subaccount-list
- POST /api/user/v1/sub/virtual-api-create	- POST /api/v2/user/create-virtual-subaccount-apikey
- POST /api/user/v1/sub/virtual-api-modify	- POST /api/v2/user/modify-virtual-subaccount-apikey
- GET /api/user/v1/sub/virtual-api-list	- GET /api/v2/user/virtual-subaccount-apikey-list
- GET /api/spot/v1/convert/currencies	- GET /api/v2/convert/currencies
- POST /api/spot/v1/convert/quoted-price	- POST /api/v2/convert/quoted-price
- POST /api/spot/v1/convert/trade	- POST /api/v2/convert/trade
- GET /api/spot/v1/convert/convert-record	- GET /api/v2/convert/convert-record
- GET /api/user/v1/tax/spot-record	- GET /api/v2/tax/spot-record
- GET /api/user/v1/tax/future-record	- GET /api/v2/tax/future-record
- GET /api/user/v1/tax/margin-record	- GET /api/v2/tax/margin-record
- GET /api/user/v1/tax/p2p-record	- GET /api/v2/tax/p2p-record
Future
V1 Endpoint	V2 Endpoint
- GET /api/mix/v1/market/ticker	- GET /api/v2/mix/market/ticker
- GET /api/mix/v1/market/tickers	- GET /api/v2/mix/market/tickers
- GET /api/mix/v1/market/contract-vip-level	- GET /api/v2/mix/market/vip-fee-rate
- GET /api/mix/v1/market/fills	- GET /api/v2/mix/market/fills
- GET /api/mix/v1/market/fills-history	- GET /api/v2/mix/market/fills-history
- GET /api/mix/v1/market/candles	- GET /api/v2/mix/market/candles
- GET /api/mix/v1/market/history-candles	- GET /api/v2/mix/market/history-candles
- GET /api/mix/v1/market/history-index-candles	- GET /api/v2/mix/market/history-index-candles
- GET /api/mix/v1/market/history-mark-candles	- GET /api/v2/mix/market/history-mark-candles
- GET /api/mix/v1/market/funding-time	- GET /api/v2/mix/market/funding-time
- GET /api/mix/v1/market/history-fundRate	- GET /api/v2/mix/market/history-fund-rate
- GET /api/mix/v1/market/current-fundRate	- GET /api/v2/mix/market/current-fund-rate
- GET /api/mix/v1/market/open-interest	- GET /api/v2/mix/market/open-interest
- GET /api/mix/v1/market/queryPositionLever	- GET /api/v2/mix/market/query-position-lever
- GET /api/mix/v1/account/account	- GET /api/v2/mix/account/account
- GET /api/mix/v1/account/accounts	- GET /api/v2/mix/account/accounts
- POST /api/mix/v1/account/sub-account-contract-assets	- GET /api/v2/mix/account/sub-account-assets
- POST /api/mix/v1/account/open-count	- GET /api/v2/mix/account/open-count
- POST /api/mix/v1/account/setLeverage	- POST /api/v2/mix/account/set-leverage
- POST /api/mix/v1/account/setMargin	- POST /api/v2/mix/account/set-margin
- POST /api/mix/v1/account/setMarginMode	- POST /api/v2/mix/account/set-margin-mode
- POST /api/mix/v1/account/setPositionMode	- POST /api/v2/mix/account/set-position-mode
- GET /api/mix/v1/position/singlePosition	- GET /api/v2/mix/position/single-position
- GET /api/mix/v1/position/singlePosition-v2	- GET /api/v2/mix/position/single-position
- GET /api/mix/v1/position/allPosition	- GET /api/v2/mix/position/all-position
- GET /api/mix/v1/position/allPosition-v2	- GET /api/v2/mix/position/all-position
- GET /api/mix/v1/account/accountBill	- GET /api/v2/mix/account/bill
- GET /api/mix/v1/account/accountBusinessBill	- GET /api/v2/mix/account/bill
- GET /api/mix/v1/market/index	- GET /api/v2/mix/market/symbol-price
- GET /api/mix/v1/market/mark-price	- GET /api/v2/mix/market/symbol-price
- GET /api/mix/v1/market/contracts	- GET /api/v2/mix/market/contracts
- GET /api/mix/v1/market/symbol-leverage	- GET /api/v2/mix/market/contracts
- GET /api/mix/v1/market/open-limit	- GET /api/v2/mix/market/contracts
- POST /api/mix/v1/plan/placePlan	- POST /api/v2/mix/order/place-plan-order
- POST /api/mix/v1/plan/placeTrailStop	- POST /api/v2/mix/order/place-plan-order
- POST /api/mix/v1/plan/modifyPlan	- POST /api/v2/mix/order/modify-plan-order
- POST /api/mix/v1/plan/modifyPlanPreset	- POST /api/v2/mix/order/modify-plan-order
- POST /api/mix/v1/plan/cancelPlan	- POST /api/v2/mix/order/cancel-plan-order
- POST /api/mix/v1/plan/cancelSymbolPlan	- POST /api/v2/mix/order/cancel-plan-order
- POST /api/mix/v1/order/cancel-batch-orders	- POST /api/v2/mix/order/batch-cancel-orders
- POST /api/mix/v1/order/cancel-all-orders	- POST /api/v2/mix/order/batch-cancel-orders
- POST /api/mix/v1/order/cancel-symbol-orders	- POST /api/v2/mix/order/batch-cancel-orders
- GET /api/mix/v1/order/current	- GET /api/v2/mix/order/orders-pending
- GET /api/mix/v1/order/marginCoinCurrent	- GET /api/v2/mix/order/orders-pending
- GET /api/mix/v1/order/history	- GET api/v2/mix/order/orders-history
- GET /api/mix/v1/order/historyProductType	- GET api/v2/mix/order/orders-history
- GET /api/mix/v1/order/fills	- GET /api/v2/mix/order/fills
- GET /api/mix/v1/order/allFills	- GET /api/v2/mix/order/fills
- POST /api/mix/v1/order/placeOrder	- POST /api/v2/mix/order/place-order
- POST /api/mix/v1/order/placeOrder	- POST /api/v2/mix/order/click-backhand
- POST /api/mix/v1/order/batch-orders	- POST /api/v2/mix/order/batch-place-order
- POST /api/mix/v1/order/cancel-order	- POST /api/v2/mix/order/cancel-order
- POST /api/mix/v1/order/modifyOrder	- POST /api/v2/mix/order/modify-order
- POST /api/mix/v1/order/close-all-positions	- POST /api/v2/mix/order/close-positions
- GET /api/mix/v1/order/detail	- GET /api/v2/mix/order/detail
N/A	- GET /api/v2/mix/order/orders-plan-pending
N/A	- GET /api/v2/mix/order/orders-plan-history
Margin
V1 Endpoint	V2 Endpoint
- GET /api/margin/v1/cross/public/interestRateAndLimit	- GET /api/v2/margin/cross/interest-rate-and-limit
- GET /api/margin/v1/isolated/public/interestRateAndLimit	- GET /api/v2/margin/isolated/interest-rate-and-limit
- GET /api/margin/v1/cross/public/tierData	- GET /api/v2/margin/cross/tier-data
- GET /api/margin/v1/isolated/public/tierData	- GET /api/v2/margin/isolated/tier-data
- GET /api/margin/v1/public/currencies	- GET /api/v2/margin/currencies
- GET /api/margin/v1/cross/account/assets	- GET /api/v2/margin/cross/account/assets
- GET /api/margin/v1/isolated/account/assets	- GET /api/v2/margin/isolated/account/assets
- POST /api/margin/v1/cross/account/borrow	- POST /api/v2/margin/cross/account/borrow
- POST /api/margin/v1/isolated/account/borrow	- POST /api/v2/margin/isolated/account/borrow
- POST /api/margin/v1/cross/account/repay	- POST /api/v2/margin/cross/account/repay
- GET /api/margin/v1/isolated/account/repay	- POST /api/v2/margin/cross/account/repay
- GET /api/margin/v1/cross/account/riskRate	- GET /api/v2/margin/cross/account/risk-rate
- POST /api/margin/v1/isolated/account/riskRate	- GET /api/v2/margin/cross/account/risk-rate
- POST /api/margin/v1/cross/account/maxBorrowableAmount	- GET /api/v2/margin/cross/account/max-borrowable-amount
- GET /api/margin/v1/isolated/account/maxBorrowableAmount	- GET /api/v2/margin/isolated/account/max-borrowable-amount
- GET /api/margin/v1/cross/account/maxTransferOutAmount	- GET /api/v2/margin/cross/account/max-transfer-out-amount
- GET /api/margin/v1/isolated/account/maxTransferOutAmount	- GET /api/v2/margin/isolated/account/max-transfer-out-amount
- POST /api/margin/v1/isolated/account/flashRepay	- POST /api/v2/margin/isolated/account/flash-repay
- POST /api/margin/v1/isolated/account/queryFlashRepayStatus	- POST /api/v2/margin/isolated/account/query-flash-repay-status
- POST /api/margin/v1/cross/account/flashRepay	- POST /api/v2/margin/cross/account/flash-repay
- POST /api/margin/v1/cross/account/queryFlashRepayStatus	- POST /api/v2/margin/cross/account/flash-repay-status
- POST /api/margin/v1/isolated/order/placeOrder	- POST /api/v2/margin/isolated/place-order
- POST /api/margin/v1/isolated/order/batchPlaceOrder	- POST /api/v2/margin/isolated/batch-place-order
- POST /api/margin/v1/isolated/order/cancelOrder	- POST /api/v2/margin/isolated/cancel-order
- POST /api/margin/v1/isolated/order/batchCancelOrder	- POST /api/v2/margin/isolated/batch-cancel-order
- GET /api/margin/v1/isolated/order/openOrders	- GET /api/v2/margin/isolated/open-orders
- GET /api/margin/v1/isolated/order/history	- GET /api/v2/margin/isolated/history-orders
- GET /api/margin/v1/isolated/order/fills	- GET /api/v2/margin/isolated/fills
- GET /api/margin/v1/isolated/loan/list	- GET /api/v2/margin/isolated/borrow-history
- GET /api/margin/v1/isolated/repay/list	- GET /api/v2/margin/isolated/repay-history
- GET /api/margin/v1/isolated/interest/list	- GET /api/v2/margin/isolated/interest-history
- GET /api/margin/v1/isolated/liquidation/list	- GET /api/v2/margin/isolated/liquidation-history
- GET /api/margin/v1/isolated/fin/list	- GET /api/v2/margin/isolated/financial-records
- POST /api/margin/v1/cross/order/placeOrder	- POST /api/v2/margin/cross/place-order
- POST /api/margin/v1/cross/order/batchPlaceOrder	- POST /api/v2/margin/cross/batch-place-order
- POST /api/margin/v1/cross/order/cancelOrder	- POST /api/v2/margin/cross/cancel-order
- POST /api/margin/v1/cross/order/batchCancelOrder	- POST /api/v2/margin/cross/batch-cancel-order
- GET /api/margin/v1/cross/order/openOrders	- GET /api/v2/margin/cross/open-orders
- GET /api/margin/v1/cross/order/history	- GET /api/v2/margin/cross/history-orders
- GET /api/margin/v1/cross/order/fills	- GET /api/v2/margin/cross/fills
- GET /api/margin/v1/cross/loan/list	- GET /api/v2/margin/cross/borrow-history
- GET /api/margin/v1/cross/repay/list	- GET /api/v2/margin/cross/repay-history
- GET /api/margin/v1/cross/interest/list	- GET /api/v2/margin/cross/interest-history
- GET /api/margin/v1/cross/liquidation/list	- GET /api/v2/margin/cross/liquidation-history
- GET /api/margin/v1/cross/fin/list	- GET /api/v2/margin/cross/financial-records
Broker
V1 Endpoint	V2 Endpoint
- GET /api/broker/v1/account/info	- GET /api/v2/broker/account/info
- POST /api/broker/v1/account/sub-create	- POST /api/v2/broker/account/create-subaccount
- GET /api/broker/v1/account/sub-list	- GET /api/v2/broker/account/subaccount-list
- POST /api/broker/v1/account/sub-modify	- POST /api/v2/broker/account/modify-subaccount
- POST /api/broker/v1/account/sub-modify-email	- POST /api/v2/broker/account/modify-subaccount-email
- GET /api/broker/v1/account/sub-email	- GET /api/v2/broker/account/subaccount-email
- GET /api/broker/v1/account/sub-spot-assets	- GET /api/v2/broker/account/subaccount-spot-assets
- GET /api/broker/v1/account/sub-future-assets	- GET /api/v2/broker/account/subaccount-future-assets
- POST /api/broker/v1/account/sub-address	- POST /api/v2/broker/account/subaccount-address
- POST /api/broker/v1/account/sub-withdrawal	- POST /api/v2/broker/account/subaccount-withdrawal
- POST /api/broker/v1/account/sub-auto-transfer	- POST /api/v2/broker/account/set-subaccount-autotransfer
- POST /api/broker/v1/manage/sub-api-create	- POST /api/v2/broker/manage/create-subaccount-apikey
- GET /api/broker/v1/manage/sub-api-list	- GET /api/v2/broker/manage/subaccount-apikey-list
- POST /api/broker/v1/manage/sub-api-modify	- POST /api/v2/broker/manage/modify-subaccount-apikey
Future Copy Trading
V1 Endpoint	V2 Endpoint
- POST /api/mix/v1/trace/closeTrackOrder	- POST /api/v2/copy/mix-trader/order-close-positions
- POST /api/mix/v1/trace/closeTrackOrderBySymbol	- POST /api/v2/copy/mix-trader/order-close-positions
- GET /api/mix/v1/trace/currentTrack	- GET /api/v2/copy/mix-trader/order-current-track
- GET /api/mix/v1/trace/historyTrack	- GET /api/v2/copy/mix-trader/order-history-track
- POST /api/mix/v1/trace/modifyTPSL	- POST /api/v2/copy/mix-trader/order-modify-tpsl
- GET /api/mix/v1/trace/traderDetail	- GET /api/v2/copy/mix-trader/order-total-detail
- GET /api/mix/v1/trace/summary	- GET /api/v2/copy/mix-trader/profit-summarys
- GET /api/mix/v1/trace/profitSettleTokenIdGroup	- GET /api/v2/copy/mix-trader/profit-summarys
- GET /api/mix/v1/trace/profitDateList	- GET /api/v2/copy/mix-trader/profit-hisotry-details
- GET /api/mix/v1/trace/waitProfitDateList	- GET /api/v2/copy/mix-trader/profit-details
- GET /api/mix/v1/trace/profitDateGroupList	- GET /api/v2/copy/mix-trader/profits-group-coin-date
- GET /api/mix/v1/trace/traderSymbols	- GET /api/v2/copy/mix-trader/config-query-symbols
- POST /api/mix/v1/trace/queryTraderTpslRatioConfig	- GET /api/v2/copy/mix-trader/config-query-symbols
- POST /api/mix/v1/trace/setUpCopySymbols	- POST /api/v2/copy/mix-trader/config-setting-symbols
- POST /api/mix/v1/trace/traderUpdateTpslRatioConfig	- POST /api/v2/copy/mix-trader/config-setting-symbols
- POST /api/mix/v1/trace/traderUpdateConfig	- POST /api/v2/copy/mix-trader/config-settings-base
- GET /api/mix/v1/trace/myFollowerList	- GET /api/v2/copy/mix-trader/config-query-followers
- POST /api/mix/v1/trace/removeFollower	- POST /api/v2/copy/mix-trader/config-remove-follower
- POST /api/mix/v1/trace/followerCloseByTrackingNo	- POST /api/v2/copy/mix-follower/close-positions
- POST /api/mix/v1/trace/followerCloseByAll	- POST /api/v2/copy/mix-follower/close-positions
- GET /api/mix/v1/trace/followerOrder	- GET /api/v2/copy/mix-follower/query-current-orders
- GET /api/mix/v1/trace/followerHistoryOrders	- GET /api/v2/copy/mix-follower/query-history-orders
- POST /api/mix/v1/trace/followerSetTpsl	- POST /api/v2/copy/mix-follower/setting-tpsl
- GET /api/mix/v1/trace/queryTraceConfig	- GET /api/v2/copy/mix-follower/query-settings
- GET /api/mix/v1/trace/public/getFollowerConfig	- GET /api/v2/copy/mix-follower/query-quantity-limit
- POST /api/mix/v1/trace/followerSetBatchTraceConfig	- POST /api/v2/copy/mix-follower/settings
- POST /api/mix/v1/trace/cancelCopyTrader	- POST /api/v2/copy/mix-follower/cancel-trader
- GET /api/mix/v1/trace/myTraderList	- GET /api/v2/copy/mix-follower/query-traders
- GET /api/mix/v1/trace/traderList	- GET /api/v2/copy/mix-broker/query-traders
- GET /api/mix/v1/trace/report/order/historyList	- GET /api/v2/copy/mix-broker/query-history-traces
- GET /api/mix/v1/trace/report/order/currentList	- GET /api/v2/copy/mix-broker/query-current-traces
Spot Copy Trading
V1 Endpoint	V2 Endpoint
- POST /api/spot/v1/trace/profit/totalProfitInfo	- GET /api/v2/copy/spot-trader/profit-summarys
- POST /api/spot/v1/trace/profit/totalProfitList	- GET /api/v2/copy/spot-trader/profit-summarys
- POST /api/spot/v1/trace/profit/profitHisList	- GET /api/v2/copy/spot-trader/profit-summarys
- POST /api/spot/v1/trace/profit/profitHisDetailList	- GET /api/v2/copy/spot-trader/profit-hisotry-details
- POST /api/spot/v1/trace/profit/waitProfitDetailList	- GET /api/v2/copy/spot-trader/profit-details
- POST /api/spot/v1/trace/order/orderCurrentList	- GET /api/v2/copy/spot-trader/order-current-track
- POST /api/spot/v1/trace/order/orderHistoryList	- GET /api/v2/copy/spot-trader/order-history-track
- POST /api/spot/v1/trace/order/updateTpsl	- POST /api/v2/copy/spot-trader/order-modify-tpsl
- POST /api/spot/v1/trace/order/closeTrackingOrder	- POST /api/v2/copy/spot-trader/order-close-tracking
- POST /api/spot/v1/trace/user/getTraderInfo	- GET /api/v2/copy/spot-trader/order-total-detail
- POST /api/spot/v1/trace/config/getTraderSettings	- GET /api/v2/copy/spot-trader/config-query-settings
- POST /api/spot/v1/trace/order/spotInfoList	- GET /api/v2/copy/spot-trader/config-query-settings
- POST /api/spot/v1/trace/config/getRemoveFollowerConfig	- GET /api/v2/copy/spot-trader/config-query-settings
- POST /api/spot/v1/trace/user/myFollowers	- GET /api/v2/copy/spot-trader/config-query-followers
- POST /api/spot/v1/trace/config/setProductCode	- POST /api/v2/copy/spot-trader/config-setting-symbols
- POST /api/spot/v1/trace/user/removeFollower	- POST /api/v2/copy/spot-trader/config-remove-follower
N/A	- GET /api/v2/copy/spot-follower/query-current-orders
N/A	- GET /api/v2/copy/mix-follower/query-history-orders
N/A	- POST /api/v2/copy/spot-follower/setting-tpsl
N/A	- POST /api/v2/copy/spot-follower/order-close-tracking
- POST /api/spot/v1/trace/config/getFollowerSettings	- GET /api/v2/copy/spot-follower/query-settings
- POST /api/spot/v1/trace/user/myTraders	- GET /api/v2/copy/spot-follower/query-traders
- POST /api/spot/v1/trace/config/setFollowerConfig	- POST /api/v2/copy/spot-follower/settings
- POST /api/spot/v1/trace/user/removeTrader	- POST /api/v2/copy/spot-follower/cancel-trader
- POST /api/spot/v1/trace/order/followerEndOrder	- POST /api/v2/copy/spot-follower/stop-order
N/A	- GET /api/v2/copy/spot-follower/query-trader-symbols