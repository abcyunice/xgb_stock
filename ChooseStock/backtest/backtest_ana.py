import pickle
import pandas as pd
import xgboost
import datetime
import matplotlib.pyplot as plt

df_result = pd.read_csv("../datasets/backtest_new_0825_label.csv")
df_result["date"] = pd.to_datetime(df_result["date"])

df = pd.read_csv("../datasets/stock_20200821.csv")
df["trade_date"] = df["trade_date"].apply(lambda x: str(x))
df["trade_date"] = pd.to_datetime((df["trade_date"]))

df = df.groupby("trade_date")["pct_chg"].mean()

df = pd.merge(df_result, df, left_on="date", right_index=True, how="left")
print(df)

df["nav_my"] = (df["return"] + 1).cumprod()
df["nav_market"] = (df["pct_chg"] + 1).cumprod()

plt.plot(df["date"], df["nav_my"] / (df.iloc[0]["return"] + 1), c="r", label="my")
plt.plot(df["date"], df["nav_market"] / (df.iloc[0]["pct_chg"] + 1), c="b", label="market")

plt.legend()
plt.show()
