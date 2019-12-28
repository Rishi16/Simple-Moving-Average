import pandas as pd
from datetime import datetime
import numpy as np

d = 'b'
total_profit = total_loss = price = 0

def decide(r, col, max):
    global d, price
    if r[col + '_sma'] != np.nan:
        if d == 'b' and r[col] < r[col + '_sma']:
            d = 's'
            price = 0 - r[col]
            return pd.Series([0 - r[col], np.nan], index=[col + '_decision', col + '_pnl'])
        elif (d == 's' and r[col] > r[col + '_sma']) or r.name == max - 1:
            d = 'b'
            return pd.Series([r[col], price + r[col]], index=[col + '_decision', col + '_pnl'])
        return pd.Series([np.nan, np.nan], index=[col + '_decision', col + '_pnl'])


def calc_pnl(r, col):
    global profit, loss, total_profit, total_loss
    if r[col + '_pnl'] >= 0:
        profit = profit + r[col + '_pnl']
        total_profit = total_profit + r[col + '_pnl']
    elif r[col + '_pnl'] < 0:
        loss = abs(r[col + '_pnl']) + loss
        total_loss = total_loss + r[col + '_pnl']


df = pd.read_csv(r"close_price.csv")
df['TRADING_DATE'] = pd.to_datetime(df['TRADING_DATE'])
# df.sort_values('TRADING_DATE', ascending=False,  inplace=True)
ma = int(input('Enter the Moving Average Days > '))
sma_df = pd.DataFrame()
stats_df = pd.DataFrame()
pf_df = pd.DataFrame()
sma_df['TRADING_DATE'] = df['TRADING_DATE']
stats_df['stock'] = df.columns[1:]
# individual stock

for col in df.columns[1:]:
    # ot = datetime.now()
    profit = loss = 0
    print(col)
    sma_df[col] = df[col]
    sma_df[col + '_sma'] = df[col].rolling(ma).mean()
    # sma_df.loc[sma_df[col] < sma_df[col+'_sma'], 'decision'] = 0 - sma_df[col]
    # sma_df.loc[sma_df[col] > sma_df[col+'_sma'], 'decision'] = sma_df[col]
    sma_df[[col + '_decision', col + '_pnl']] = sma_df.apply(lambda x: decide(x, col, len(sma_df)), axis=1)
    sma_df.apply(lambda x: calc_pnl(x, col), axis=1)
    pl = sma_df[col + '_decision'].sum()
    tt = len(sma_df[~sma_df[col + '_decision'].isnull()])
    stats_df.index = stats_df.stock
    stats_df = stats_df.set_value(col, 'P/L', pl)
    stats_df = stats_df.set_value(col, 'Total Trades', tt)
    stats_df = stats_df.set_value(col, 'Mean Return Per Trade', pl / tt)
    stats_df = stats_df.set_value(col, 'Profit Factor', profit / abs(loss))

pl = stats_df['P/L'].sum()
tt = stats_df['Total Trades'].sum()
pf_df = pf_df.set_value('Portfolio', 'P/L', pl)
pf_df = pf_df.set_value('Portfolio', 'Total Trades', tt)
pf_df = pf_df.set_value('Portfolio', 'Mean Return Per Trade', pl / tt)
pf_df = pf_df.set_value('Portfolio', 'Profit Factor', total_profit / abs(total_loss))

# combined portfolio


# print((datetime.now()-ot))
