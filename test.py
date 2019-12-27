import collections
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_df(nrows=100, ncols=3):
    seed = 2018
    np.random.seed(seed)
    df = pd.DataFrame(np.random.randint(10, size=(nrows, ncols)))
    df['athlete_id'] = np.random.randint(10, size=nrows)
    return df

def orig(df, key='athlete_id'):
    columns = list(df.columns.difference([key]))
    result = pd.DataFrame(index=df.index)
    for window in range(2, 4):
        for col in columns:
            colname = 'sum_col{}_winsize{}'.format(col, window)
            result[colname] = df.groupby(key)[col].apply(lambda x: x.rolling(
                center=False, window=window, min_periods=1).sum())
            colname = 'min_col{}_winsize{}'.format(col, window)
            result[colname] = df.groupby(key)[col].apply(lambda x: x.rolling(
                center=False, window=window, min_periods=1).min())
            colname = 'max_col{}_winsize{}'.format(col, window)
            result[colname] = df.groupby(key)[col].apply(lambda x: x.rolling(
                center=False, window=window, min_periods=1).max())
    result = pd.concat([df, result], axis=1)
    return result

def alt(df, key='athlete_id'):
    """
    Call rolling on the whole DataFrame, not each column separately
    """
    columns = list(df.columns.difference([key]))
    result = [df]
    for window in range(2, 4):
        rolled = df.groupby(key, group_keys=False).rolling(
            center=False, window=window, min_periods=1)

        new_df = rolled.sum().drop(key, axis=1)
        new_df.columns = ['sum_col{}_winsize{}'.format(col, window) for col in columns]
        result.append(new_df)

        new_df = rolled.min().drop(key, axis=1)
        new_df.columns = ['min_col{}_winsize{}'.format(col, window) for col in columns]
        result.append(new_df)

        new_df = rolled.max().drop(key, axis=1)
        new_df.columns = ['max_col{}_winsize{}'.format(col, window) for col in columns]
        result.append(new_df)

    df = pd.concat(result, axis=1)
    return df

timing = collections.defaultdict(list)
ncols = [3, 10, 20, 50, 100]
for n in ncols:
    df = make_df(ncols=n)
    timing['orig'].append(timeit.timeit(
        'orig(df)',
        'from __main__ import orig, alt, df',
        number=10))
    timing['alt'].append(timeit.timeit(
        'alt(df)',
        'from __main__ import orig, alt, df',
        number=10))

plt.plot(ncols, timing['orig'], label='using groupby/apply (orig)')
plt.plot(ncols, timing['alt'], label='using groupby/rolling (alternative)')
plt.legend(loc='best')
plt.xlabel('number of columns')
plt.ylabel('seconds')
print(pd.DataFrame(timing, index=pd.Series(ncols, name='ncols')))
plt.show()