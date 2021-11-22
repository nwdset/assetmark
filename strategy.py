# %%
"""
Implementation of strategies as described in ```./papers```
"""
import numpy as np
import pandas as pd

from math import sqrt

sptr = pd.read_csv('./data/sptr.csv', header=None, parse_dates=[0]) \
    .rename({0: 'date', 1: 'sptr'}, axis=1) \
    .set_index('date')['sptr'].rename("sptr")
spx  = pd.read_csv('./data/spx.csv', header=None, parse_dates=[0]) \
    .rename({0: 'date', 1: 'spx'}, axis=1) \
    .set_index('date')['spx'].rename('spx')
tbil = pd.read_csv('./data/tbill.csv', header=None, parse_dates=[0]) \
    .rename({0: 'date', 1: 'tbill'}, axis=1) \
    .set_index('date')['tbill'].rename('tbill')
vix = pd.read_csv('./data/^VIX.csv', parse_dates=['Date']) \
    .set_index('Date') \
    .loc[:, 'Adj Close']
hi_momentum = pd.read_csv('./data/10_Portfolios_Prior_12_2_Daily.csv', parse_dates=['date']) \
    .set_index('date')['Hi PRIOR'] \
    .div(100)
hi_bm = pd.read_csv('./data/Portfolios_Formed_on_BE-ME_Daily.csv', parse_dates=['Date']).dropna() \
    .set_index('Date')['Hi 10'] \
    .div(100)

# %%
# strategy parameters
target_std = .2
cap = 2
floor = .5

# %%
"""Compute returns excess of risk-free
"""
spx_return = spx.pct_change()[1:].sub(tbil.pct_change())
spx_return[1:].dropna().rename('spx_returns').to_csv("./data/spx_return_excess.csv")
# NOTE: change this
# spx_return = hi_momentum

# %%
"""Compute k_scaled ex-post so spx_returns would have annualized 20% vol
"""
spx_std_annual = spx_return.std()*sqrt(252)
k = target_std/spx_std_annual
spx_return_target = spx_return.mul(k)

# %%

# %%
"""Backtest the strategy
"""
strategy_weights = vix.div(100).shift(2).rdiv(target_std).clip(floor, cap)
# NOTE: change `spx_return` to any portfolio to redo everything
strategy_returns = strategy_weights.mul(spx_return).dropna()
strategy_returns_std_annual = strategy_returns.std()*sqrt(252)
k = target_std/strategy_returns_std_annual
strategy_returns = strategy_returns.mul(k)
strategy_returns.to_csv('./data/strategy_return_excess.csv')
strategy_weights.reindex(strategy_returns.index).dropna().to_csv('./data/strategy_weights.csv')

# %%
"""Generating statistics
"""
concatted = pd.concat([
    spx_return_target[1:].rename('SPX'), 
    strategy_returns.rename('SPX 20% Target')
    ], axis=1) \
    .loc['2000':]
        
# %%
mean = concatted.pipe(np.log1p).resample('M').sum().mean().mul(12).rename('Mean')
std = concatted.pipe(np.log1p).std().mul(sqrt(252)).rename('Std')
sharpe_ratio = mean.T.div(std.T).rename('Sharpe Ratio').map("{:.2f}".format)
mean = mean.map("{:.2%}".format)
std = std.map("{:.2%}".format)

# %%
total_returns = concatted.add(1).cumprod().iloc[-1].rename('Total Returns') \
    .map('{:.2%}'.format)

# %%
turnover = (strategy_weights.diff().abs().mean()*252) / (2*strategy_weights.mean())
daily_exposure = strategy_weights.mean()
turnover = pd.DataFrame(
    [[0, target_std/spx_std_annual], [turnover, daily_exposure]], 
    index=['SPX', 'SPX 20% Target'], columns=['Turnover', 'Mean Notional']) \
        .applymap('{:.2f}'.format)


# %%
volofvol = concatted.rolling(30).std().mul(sqrt(252)) \
    .dropna().std().rename('Vol of Vol') \
    .map('{:.2%}'.format)

# %%
bottom_1pct = concatted.quantile(.01)
top_1pct = concatted.quantile(.99)
shortfall = pd.concat([concatted.query('SPX < @bottom_1pct["SPX"]')['SPX'],
                        concatted.query('`SPX 20% Target` < @bottom_1pct["SPX 20% Target"]')['SPX 20% Target']], axis=1) \
    .mean() \
    .rename('Mean Short Fall 1%') \
    .map('{:.2%}'.format)
exceedance = pd.concat([concatted.query('SPX > @top_1pct["SPX"]')['SPX'],
                        concatted.query('`SPX 20% Target` > @top_1pct["SPX 20% Target"]')['SPX 20% Target']], axis=1) \
    .mean() \
    .rename('Mean Exceedance 99%') \
    .map('{:.2%}'.format)


# %%
table_results = pd.concat([
    mean, std, total_returns, sharpe_ratio, 
    turnover, volofvol, shortfall, 
    exceedance], axis=1)

# %%
table_results.T.rename({'SPX 20% Target': 'SPX Target Vol'}, axis=1) \
    .to_csv('./data/SPX_results_table.csv')

# %%
concatted.pipe(np.log1p) \
    .resample('M').sum() \
    .cumsum() \
    .pipe(np.exp) \
    .rename({'SPX 20% Target': 'SPX Target Vol'}, axis=1) \
    .stack() \
    .reset_index() \
    .rename({'level_0': 'date', 'level_1': 'variable', 0: 'PnL'}, axis=1) \
    .to_csv('./data/pnl.csv', index=False)

# %%
concatted.rolling(30).std().dropna().stack().reset_index() \
    .rename({'level_0': 'date', 'level_1': 'variable', 0: '30 Day Vol'}, axis=1) \
    .to_csv('./data/30dayvol.csv')

# %%
pd.concat([concatted.query('SPX < @bottom_1pct["SPX"]')['SPX'],
            concatted.query('`SPX 20% Target` < @bottom_1pct["SPX 20% Target"]')['SPX 20% Target']], axis=1) \
            .rename({'SPX 20% Target': 'SPX_target'}, axis=1) \
            .to_csv('./data/shortfall_returns.csv')
