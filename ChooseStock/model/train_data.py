import xgboost as xgb
import pandas as pd
import datetime

import pickle

# sh --------------------------------------------------------------------------------
df_sz = pd.read_csv("./datasets/stock_20200811_sh.csv")
df_sz["pct_chg"] = df_sz.groupby("code")["pct_chg"].shift(-2)

df_sz.dropna(inplace=True)

df_sz["trade_date"] = df_sz["trade_date"].apply(lambda x: str(x))
df_sz["trade_date"] = pd.to_datetime(df_sz["trade_date"])

df_sz_train = df_sz[
    (df_sz["trade_date"] < datetime.datetime(2020, 7, 1)) & (df_sz["trade_date"] >= datetime.datetime(2020, 1, 1))]

df_sz_train_X = df_sz_train.values[:, 1:-2]
df_sz_train_y = df_sz_train.values[:, -2]

df_sz_test = df_sz[df_sz["trade_date"] >= datetime.datetime(2020, 7, 1)]
df_sz_test_X = df_sz_test.values[:, 1:-2]
df_sz_test_y = df_sz_test.values[:, -2]

myxgb = xgb.XGBRegressor(n_estimators=10000)
eval_set = [(df_sz_test_X, df_sz_test_y)]
myxgb.fit(df_sz_train_X, df_sz_train_y, eval_metric="rmse", eval_set=eval_set, verbose=True, early_stopping_rounds=100)

with open("./datasets/xgb_0811_sh.m", "wb+") as f:
    pickle.dump(myxgb, f)

df_importance = pd.DataFrame()
df_importance["feature"] = df_sz.columns[1:-2]
df_importance["importance"] = myxgb.feature_importances_

df_importance.sort_values("importance", inplace=True, ascending=False)
print(df_importance)

# sz ----------------------------------------------------------
df_sz = pd.read_csv("./datasets/stock_20200811_sz.csv")

df_sz["pct_chg"] = df_sz.groupby("code")["pct_chg"].shift(-2)

df_sz.dropna(inplace=True)

df_sz["trade_date"] = df_sz["trade_date"].apply(lambda x: str(x))
df_sz["trade_date"] = pd.to_datetime(df_sz["trade_date"])

df_sz_train = df_sz[
    (df_sz["trade_date"] < datetime.datetime(2020, 7, 1)) & (df_sz["trade_date"] >= datetime.datetime(2020, 1, 1))]

df_sz_train_X = df_sz_train.values[:, 1:-2]
df_sz_train_y = df_sz_train.values[:, -2]

df_sz_test = df_sz[df_sz["trade_date"] >= datetime.datetime(2020, 7, 1)]
df_sz_test_X = df_sz_test.values[:, 1:-2]
df_sz_test_y = df_sz_test.values[:, -2]

myxgb = xgb.XGBRegressor(n_estimators=10000)
eval_set = [(df_sz_test_X, df_sz_test_y)]
myxgb.fit(df_sz_train_X, df_sz_train_y, eval_metric="rmse", eval_set=eval_set, verbose=True, early_stopping_rounds=100)

with open("./datasets/xgb_0811_sz.m", "wb+") as f:
    pickle.dump(myxgb, f)
df_importance = pd.DataFrame()
df_importance["feature"] = df_sz.columns[1:-2]
df_importance["importance"] = myxgb.feature_importances_

df_importance.sort_values("importance", inplace=True, ascending=False)
print(df_importance)
