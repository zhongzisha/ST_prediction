

import yfinance as yf
import pandas as pd

# 获取SPY最近5年的数据
# ticker = 'SPY'
# ticker = 'QQQ'
ticker = '^GSPC'
data = yf.download(ticker, period="5y", interval="1wk")

# 计算每周的涨幅 (涨幅 = (本周收盘价 - 上周收盘价) / 上周收盘价)
data['Weekly_Return'] = data['Adj Close'].pct_change()

# 查看数据
print(data[data['Weekly_Return']>0.05])
print(data[data['Weekly_Return']<-0.05])















