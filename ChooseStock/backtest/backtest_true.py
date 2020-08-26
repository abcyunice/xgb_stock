'''

20200820 : 添加成交量

20200821 ：考虑到过拟合很快，降低learing_rate， 更大训练集， 加sample weight


'''

import xgboost as xgb
import pandas as pd
import datetime

import numpy as np

import pickle

df_all = pd.read_csv("./datasets/stock_20200821.csv")

# 保留指数的pct_chg 和 成交量特征
for ci in df_all.columns:
    if ci.endswith("_sh") and (not ci.startswith("pct_chg") and not ci.startswith("vol")):
        del df_all[ci]

print(df_all.columns)

df_all["pct_chg"] = df_all.groupby("code")["pct_chg"].shift(-2)

df_all.dropna(inplace=True)

df_all["trade_date"] = df_all["trade_date"].apply(lambda x: str(x))
df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])

return_list = []
datelist = []
stock_list = []
score_train_list = []
score_valid_list = []
stock_num_list = []

date_list = list(set(df_all["trade_date"]))
date_list.sort()
date_dict = {di: idx for idx, di in enumerate(date_list)}

for year in range(2015, 2021):
    for month in range(1, 13):
        for i in range(1, 32):
            try:
                today_data = df_all[df_all["trade_date"] == datetime.datetime(year, month, i)]
            except:
                continue

            if len(today_data) == 0:
                continue

            today_date = today_data.iloc[0]["trade_date"]

            test_end_date = date_list[date_dict[today_date] - 1]
            test_start_date = date_list[date_dict[today_date] - 11]
            train_end_date = date_list[date_dict[today_date] - 12]
            train_start_date = date_list[date_dict[today_date] - 52]
            df_sh_train = df_all[
                (df_all["trade_date"] >= train_start_date) & (
                        df_all["trade_date"] <= train_end_date)]
            df_sh_train_X = df_sh_train.values[:, 1:-2]
            df_sh_train_y = df_sh_train.values[:, -2]

            df_sh_test = df_all[
                (df_all["trade_date"] >= test_start_date) & (
                        df_all["trade_date"] < test_end_date)]
            df_sh_test_X = df_sh_test.values[:, 1:-2]
            df_sh_test_y = df_sh_test.values[:, -2]

            delta_days = (df_sh_train["trade_date"].max() - df_sh_train["trade_date"].min()).days
            sample_weight = (df_sh_train["trade_date"] - df_sh_train["trade_date"].min()).apply(
                lambda x: (x.days + delta_days))
            sample_weight = np.array(sample_weight)

            delta_days_eval = (df_sh_test["trade_date"].max() - df_sh_test["trade_date"].min()).days
            sample_weight_eval = (df_sh_test["trade_date"] - df_sh_test["trade_date"].min()).apply(
                lambda x: (x.days + delta_days_eval))
            sample_weight_eval = np.array(sample_weight_eval)

            myxgb = xgb.XGBRegressor(n_estimators=10000, learning_rate=0.01)
            eval_set = [(df_sh_test_X, df_sh_test_y)]
            myxgb.fit(df_sh_train_X, df_sh_train_y, eval_metric="rmse", eval_set=eval_set, verbose=False,
                      early_stopping_rounds=100, sample_weight=sample_weight,
                      sample_weight_eval_set=[sample_weight_eval])

            train_score = myxgb.score(df_sh_train_X, df_sh_train_y)
            valid_score = myxgb.score(df_sh_test_X, df_sh_test_y)

            X = today_data.values[:, 1:-2]
            y_pred = myxgb.predict(X)

            df = pd.DataFrame({"stock": today_data["code"], "y_pred": y_pred})
            df = df[df["y_pred"] > 0.03]
            df.sort_values(by="y_pred", ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)

            tmp = 0
            for stocki in df["stock"]:
                return_ = today_data[today_data["code"] == stocki].values[0, -2]
                tmp += return_

            if len(df) == 0:
                return_list.append(0)
            else:
                return_list.append(tmp / len(df))

            datelist.append(today_data.iloc[0]["trade_date"])
            stock_list.append(df["stock"].tolist())
            score_train_list.append(train_score)
            score_valid_list.append(valid_score)
            stock_num_list.append(len(df))

            print("train score:", train_score)
            print("valid score:", valid_score)
            print(today_data.iloc[0]["trade_date"])
            print(df["stock"].tolist())
            print("return:", return_list[-1])
            print("----------------------------------------")

df_result = pd.DataFrame({"date": datelist, "return": return_list,
                          "stock": stock_list, "stock_num": stock_num_list,
                          "train_score": score_train_list, "test_score": score_valid_list})
print(df_result)
df_result.to_csv("./datasets/backtest_new_0823.csv", index=False)
