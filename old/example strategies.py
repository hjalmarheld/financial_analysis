# EXAMPLE STRATEGIES
import pandas as pd

def mock_decision(prices, ratios):
    return pd.Series({prices['permno'].iloc[0]:.5, prices['permno'].iloc[-1]:.5})

def easy_momentum(prices, ratios, count=50):
    returns = prices.pivot(index='date', values='ret', columns='permno') + 1
    period_returns = returns.cumprod().iloc[-1]
    top_stocks = period_returns.nlargest(count).index
    return pd.Series({stock:1/len(top_stocks) for stock in list(top_stocks)})