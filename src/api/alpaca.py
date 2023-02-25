import alpaca_trade_api as tradeapi

# Set up API connection
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
base_url = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading

api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

# Example usage - Get account information
account = api.get_account()
print('Account Information:')
print(f'Account ID: {account.id}')
print(f'Buying Power: ${account.buying_power}')
print(f'Cash Balance: ${account.cash}')
print(f'Equity: ${account.equity}')
print(f'Margin Level: {account.margin_multiplier}')
