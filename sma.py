import pandas as pd
import math
from datetime import datetime
import numpy as np
import sys
from matplotlib import pyplot


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
    global profit, loss, total_profit, total_loss, total_pnl
    if r[col + '_pnl'] >= 0:
        profit = profit + r[col + '_pnl']
        total_profit = total_profit + r[col + '_pnl']
        r['pnl'] = r['pnl'] + profit
    elif r[col + '_pnl'] < 0:
        loss = abs(r[col + '_pnl']) + loss
        total_loss = total_loss + r[col + '_pnl']
        r['pnl'] = r['pnl'] - loss
    return r['pnl']


def calc_pnl_curve(r):
    global pnl_curve
    # print(pnl_curve, r)
    if math.isnan(r):
        r = 0
    pnl_curve = pnl_curve + r
    return pnl_curve


def progress(i, total, length):
    frac = i / total
    filled = round(frac * length)
    print('\r['+'#' * filled + '-' * (length - filled)+'][{:>7.2%}]'.format(frac), end='')
    sys.stdout.flush()

df = pd.read_csv(r"close_price.csv")
df['TRADING_DATE'] = pd.to_datetime(df['TRADING_DATE'])
# df.sort_values('TRADING_DATE', ascending=False,  inplace=True)
while True:
    ma = int(input('Enter the Moving Average Days > '))
    d = 'b'
    total_profit = 0
    total_loss = 0
    total_pnl = 0
    price = 0
    pnl_curve = 0

    sma_df = pd.DataFrame()
    stats_df = pd.DataFrame()
    pf_df = pd.DataFrame()
    sma_df['TRADING_DATE'] = df['TRADING_DATE']
    stats_df['stock'] = df.columns[1:]
    sma_df['pnl'] = 0
    cols = len(df.columns)-1
    # individual stock
    for i,col in enumerate(df.columns[1:]):
        # ot = datetime.now()
        profit = loss = 0
        sma_df[col] = df[col]
        sma_df[col + '_sma'] = df[col].rolling(ma).mean()
        # sma_df.loc[sma_df[col] < sma_df[col+'_sma'], 'decision'] = 0 - sma_df[col]
        # sma_df.loc[sma_df[col] > sma_df[col+'_sma'], 'decision'] = sma_df[col]
        sma_df[[col + '_decision', col + '_pnl']] = sma_df.apply(lambda x: decide(x, col, len(sma_df)), axis=1)
        sma_df[col + '_pnl_curve'] = 0
        sma_df[col + '_pnl_curve'] = sma_df.apply(lambda x: calc_pnl_curve(x[col+'_pnl']), axis=1)
        sma_df['pnl'] = sma_df.apply(lambda x: calc_pnl(x, col), axis=1)
        pl = sma_df[col + '_decision'].sum()
        tt = len(sma_df[~sma_df[col + '_decision'].isnull()])
        stats_df.index = stats_df.stock
        stats_df.at[col, 'P/L'] = pl
        stats_df.at[col, 'Total Trades'] = pl
        stats_df.at[col, 'Mean Return Per Trade'] = pl/tt
        stats_df.at[col, 'Profit Factor'] = profit / abs(loss)
        pnl_curve = 0
        progress(i+1, cols, 20)

    # combined portfolio
    pl = stats_df['P/L'].sum()
    tt = stats_df['Total Trades'].sum()
    pf_df.at['Portfolio', 'P/L'] = pl
    pf_df.at['Portfolio', 'Total Trades'] = tt
    pf_df.at['Portfolio', 'Mean Return Per Trade'] = pl / tt
    pf_df.at['Portfolio', 'Profit Factor'] = total_profit / abs(total_loss)
    sma_df['pnl_curve'] = sma_df.apply(lambda x: calc_pnl_curve(x['pnl']), axis=1)

    while True:
        option = input('\nView stocks for:\n\t1.Individal Stocks\t2.Portfolio')
        if option == '1':
            is_option = input('')
    pnlc_df = pd.DataFrame()
    pnlc_df['pnl_curve'] = sma_df['pnl_curve']
    pnlc_df.index = sma_df['TRADING_DATE']
    pyplot.figure()
    pnlc_df['pnl_curve'].plot(subplots=True, legend=False, title='PnL Curve for ' + str(ma) + ' Days SMA')
    pyplot.ylabel('PnL')
    pyplot.show()

    if input('\nTry another?(y/n) > ').lower() == 'n':
        exit()


# print((datetime.now()-ot))
