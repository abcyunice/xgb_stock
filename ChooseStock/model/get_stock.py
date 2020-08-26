import pickle
import pandas as pd
import xgboost
import datetime

with open("./datasets/xgb_0811_sh.m", "rb+") as f:
    myxgb = pickle.load(f)

today_data = pd.read_csv("./datasets/stock_20200811_sh.csv")
today_data = today_data[today_data["trade_date"] == 20200811]

X = today_data.values[:, 1:-2]
y_pred = myxgb.predict(X)

df = pd.DataFrame({"stock": today_data["code"], "y_pred": y_pred})
df.sort_values(by="y_pred", ascending=False, inplace=True)

print(df.head())

with open("./datasets/xgb_0811_sz.m", "rb+") as f:
    myxgb = pickle.load(f)

today_data = pd.read_csv("./datasets/stock_20200811_sz.csv")
today_data = today_data[today_data["trade_date"] == 20200811]

X = today_data.values[:, 1:-2]
y_pred = myxgb.predict(X)

df = pd.DataFrame({"stock": today_data["code"], "y_pred": y_pred})
df.sort_values(by="y_pred", ascending=False, inplace=True)

print(df.head())
