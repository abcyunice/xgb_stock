'''
sh sz一起
20150101的hs300

'''

import tushare as ts
import pandas as pd
from sqlalchemy import create_engine
import datetime

from IndexSet import common
import baostock as bs

'''
record the time

'''
start_time = datetime.datetime.now()

''' 
read the data from csv
and get the stock code

'''

token = '82f17f80a6f62681bcbf105689d892a9559a4e65af4045adb472eaf0'
pro = ts.pro_api(token=token)

data_sh = pd.read_csv("./datasets/hs300_2015.csv", encoding="gbk")

'''
get name list

'''
df_last = pd.DataFrame()
code_list = data_sh["code"].tolist()

print(code_list)

'''
get stock data from tushare

'''
mytech = common.TechnicalIndicatorPriceVol4Model()
for i, code in enumerate(code_list):
    df = ts.pro_bar(code, adj='qfq')
    while (df is None):
        df = ts.pro_bar(code, adj='qfq')
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.sort_values(by='trade_date', inplace=True)

    df = mytech.get_index(df)
    df["code"] = code

    df_last = df_last.append(df)
    if i % 10 == 0:
        print(i, code)

df_last.reset_index(drop=True, inplace=True)

'''
SH300 data

'''
lg = bs.login()
rs = bs.query_history_k_data_plus("sh.000300",
                                  "date,code,open,high,low,close,volume",
                                  start_date="2000-01-01", frequency="d")
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
df = pd.DataFrame(data_list, columns=rs.fields)
df.rename({"volume": "vol", "date": "trade_date"}, inplace=True, axis=1)
for col in ["open", "high", "low", "close", "vol"]:
    df[col] = df[col].astype(float)
mytech = common.TechnicalIndicatorPriceVol4Model()
df = mytech.get_index(df)
del df['pct_chg']
df.columns = ["trade_date"] + [c + "_sh" for c in df.columns if c != "trade_date"]
df["trade_date"] = pd.to_datetime(df["trade_date"])

'''
合并

'''
df_last = pd.merge(left=df, right=df_last, on="trade_date", how="inner")
df_last.dropna(inplace=True)
df_last.to_csv('./datasets/stock_20200821.csv', index=False)
print('Finished!')

'''
record the time consumed

'''
end_time = datetime.datetime.now()

print("The time consumed is %s" % (end_time - start_time))
