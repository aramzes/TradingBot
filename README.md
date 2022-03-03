# trading-bot
This trading bot is developed as my final year project to trade cryptocurrencies with support for centralized and decentralized exchanges

Code Structure in `src`
```
config
|
└───test                   # JSON-formatted config files for unit testing
src  
│
└───feed                   # public data
└───account                # private data
└───strategy               # trading strategies. It may or may not include examples
└───trader                 # executing and keeping track of orders and trades
└───utils                  # utility modules to connect to external modules such as telegram and databases
```

## Description
The expected functionality of this bot:
1. Be able to connect to exchanges using RESTful and websocket API to fetch public and private data
2. Develop strategies to identify trading opportunities
3. Execute and keep track of orders and trades.
4. Provide interfaces to generate trading statistics
5. Notify users of the bot updates using external channels