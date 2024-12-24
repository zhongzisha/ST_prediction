

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import time

# 获取SPY最近5年的数据
# ticker = 'SPY'
# ticker = 'QQQ'
ticker = '^GSPC'
ticker = '^VIX'

alldata = {}

intervals = ['1d', '1wk', '1mo']
# intervals = ['1wk']

for interval in intervals:
    time.sleep(np.random.randint(5))
    data = yf.download(ticker, start="2019-01-01", end="2024-12-31", interval=interval)
    data.columns = data.columns.droplevel(1)
    alldata[interval] = data

intervals = ['1d', '1wk', '1mo']
# intervals = ['1wk']
for interval in intervals:
    data = alldata[interval]

    # data['Range'] = abs(data[['High','Low']].min(axis=1) - data['Open'])/data['Open']
    data['High_Open'] = abs(data['High'] - data['Open'])/data['Open']
    data['Low_Open'] = abs(data['Low'] - data['Open'])/data['Open']
    data['Range'] = data[['High_Open', 'Low_Open']].max(axis=1)

    data['Range2'] = (data['Close'] - data['Open']) / data['Open']

    data['Range'].hist()
    plt.savefig(f"range_{interval}.png")
    plt.close('all')

    data['Range2'].hist()
    plt.savefig(f"range2_{interval}.png")
    plt.close('all')


# date_objects = [v.astype('datetime64[s]').item().weekday() for v in data.index.values]





