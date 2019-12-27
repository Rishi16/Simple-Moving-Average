import pandas as pd
from datetime import datetime
df = pd.read_csv(r"close_price.csv")
df['TRADING_DATE'] = pd.to_datetime(df['TRADING_DATE'])
# df.sort_values('TRADING_DATE', ascending=False,  inplace=True)
ma = int(input('Enter the Moving Average Days > '))
sma_df = pd.DataFrame()
for col in df.columns[1:]:
    # ot = datetime.now()
    # print(col)
    sma_df[col] = df[col].rolling(ma).mean()
    # print((datetime.now()-ot))

