import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt

"""
技术指标，考虑ma5，不考虑金叉等0-1变量

"""


class TechnicalIndicator(object):
    '''
    技术指标,只依赖于价格

    macd
    roc
    middleboll upboll downboll
    cci
    rsi
    wr
    mai

    '''

    def getIndex(self, df):
        df_ = df.copy()
        df_.sort_values(by="trade_date", inplace=True)
        df_['macd'] = self.macd(df_['close'])
        df_['roc'] = self.roc(df_['close'])
        df_ = pd.concat([df_, self.boll(df_['close'])], axis=1)
        df_['cci'] = self.cci(df_)
        df_['rsi'] = self.rsi(df_['close'])
        df_['wr'] = self.wr(df_)
        df_['ma5'] = self.ma(df_['close'], 5)
        df_['ma10'] = self.ma(df_['close'], 10)
        df_['ma20'] = self.ma(df_['close'], 20)
        df_['ma60'] = self.ma(df_['close'], 60)

        return df_

    def ma(self, sr, period):
        sr_ = sr.rolling(period).mean()
        sr_.fillna(method='bfill', inplace=True)
        return sr_

    def ema(self, sr, period):
        ema = sr.ewm(span=period).mean()
        return ema

    def macd(self, sr, short_period=12, long_period=26, dea_period=9):
        ema_short = self.ema(sr, short_period)
        ema_long = self.ema(sr, long_period)
        diff_ = ema_short - ema_long
        dea = self.ema(diff_, dea_period)
        macd = 2 * (diff_ - dea)
        return macd

    def roc(self, sr, period=10):
        sr_ = sr.shift(period)
        roc = sr / sr_ - 1
        roc.fillna(value=0, inplace=True)
        return roc

    # 返回布林线
    def boll(self, sr, period=10, width=2):
        middle_ = self.ema(sr, period)
        df_ = pd.DataFrame({'middleboll': middle_})
        df_['upboll'] = middle_.copy()
        df_['downboll'] = middle_.copy()
        std_ = middle_.rolling(period).std()
        std_.fillna(0, inplace=True)
        df_['upboll'] = df_['upboll'] + width * std_
        df_['downboll'] = df_['downboll'] - width * std_
        return df_

    # 需要dataframe
    def cci(self, df, period=14):
        tp = (df['high'] + df['close'] + df['low']) / 3
        ma_ = self.ma(df['close'], period)
        md_ = self.ma((ma_ - df['close']).abs(), period)
        cci = (tp - ma_) / (md_ * 0.015)
        return cci

    def rsi(self, sr, period=14):
        sr_ = sr.diff(1)
        sr_.fillna(value=0, inplace=True)
        up = sr_.copy()
        down = sr_.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.ewm(period - 1).mean()
        roll_down = (down.abs()).ewm(period - 1).mean()
        rs = roll_up / roll_down
        rsi = 100 - 100 / (rs + 1)
        return rsi

    # 需要dataframe
    def wr(self, df, period=30):
        high_period = df['high'].rolling(period).apply(lambda x: max(x), raw=True)
        low_period = df['low'].rolling(period).apply(lambda x: min(x), raw=False)
        wr_ = (high_period - df['close']) / (high_period - low_period) * 100
        wr_.fillna(value=50, inplace=True)
        return wr_

    # 需要df, return (k,d,j)
    def kdj(self, df, period=9):
        rsv = (df['close'] - df['low'].rolling(period).min()) / (
                df['high'].rolling(period).max() - df['low'].rolling(period).min()) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d
        return k, d, j


class TechnicalIndicator4Model(object):
    '''
    为了构建模型，对技术指标，skew等标准化

    '''

    def get_index(self, data):
        myindex = TechnicalIndicator()
        data = myindex.getIndex(data)

        data["pct_chg_1"] = (data["close"] - data["close"].shift(1)) / data["close"].shift(1)
        data["pct_chg_5"] = (data["close"] - data["close"].shift(5)) / data["close"].shift(5)
        data["pct_chg_10"] = (data["close"] - data["close"].shift(10)) / data["close"].shift(10)
        data["pct_chg_20"] = (data["close"] - data["close"].shift(20)) / data["close"].shift(20)
        data["pct_chg_60"] = (data["close"] - data["close"].shift(60)) / data["close"].shift(60)
        data["pct_chg_120"] = (data["close"] - data["close"].shift(120)) / data["close"].shift(120)
        data["pct_chg_250"] = (data["close"] - data["close"].shift(250)) / data["close"].shift(250)

        data["std_5"] = data["close"].rolling(5).std()
        data["std_10"] = data["close"].rolling(10).std()
        data["std_20"] = data["close"].rolling(20).std()
        data["std_60"] = data["close"].rolling(60).std()
        data["std_120"] = data["close"].rolling(120).std()
        data["std_250"] = data["close"].rolling(250).std()

        data["skew_5"] = data["close"].rolling(5).skew()
        data["skew_10"] = data["close"].rolling(10).skew()
        data["skew_20"] = data["close"].rolling(20).skew()

        data["kurt_5"] = data["close"].rolling(5).kurt()
        data["kurt_10"] = data["close"].rolling(10).kurt()
        data["kurt_20"] = data["close"].rolling(20).kurt()

        data["price_buy"] = data["open"]

        data["pct_chg"] = (data["open"] - data["open"].shift(1)) / data["open"].shift(1)

        data.dropna(inplace=True)

        # 要改
        data_standard_cols = ['open', 'high', 'close', 'low', 'ma5', 'ma10', 'ma20',
                              'middleboll', 'upboll', 'downboll', 'rsi', 'wr', "std_5", "std_10",
                              "std_20", "std_60", "std_120", "std_250"]
        for col in data_standard_cols:
            data[col] /= data["close"]

        data["cci"] /= 100
        data["rsi"] /= 100
        data["wr"] /= 100

        del data["close"]

        # 要改
        data = data[['trade_date', 'open', 'high', 'low', 'ma5', 'ma10', 'ma20', 'macd', 'roc',
                     'middleboll', 'upboll', 'downboll', 'cci', 'rsi', 'wr', "pct_chg_1", "pct_chg_5", "pct_chg_10",
                     "pct_chg_20", "pct_chg_60", "pct_chg_120", "pct_chg_250", "std_5", "std_10",
                     "std_20", "std_60", "std_120", "std_250", "skew_5", "skew_10", "skew_20", "kurt_5", "kurt_10",
                     "kurt_20", "pct_chg"]]

        return data


class TechnicalIndicatorPriceVol(TechnicalIndicator):
    '''
    添加量特征
    不怎么删除特征，在TechnicalIndicatorPriceVol4Model里删特征

    '''

    def getIndex(self, df):
        df_ = df.copy()
        df_.sort_values(by="trade_date", inplace=True)

        # 原有的价格
        df_['macd'] = self.macd(df_['close'])
        df_['roc'] = self.roc(df_['close'])
        df_ = pd.concat([df_, self.boll(df_['close'])], axis=1)
        df_['cci'] = self.cci(df_)
        df_['rsi'] = self.rsi(df_['close'])
        df_['wr'] = self.wr(df_)
        df_['ma5'] = self.ma(df_['close'], 5)
        df_['ma10'] = self.ma(df_['close'], 10)
        df_['ma20'] = self.ma(df_['close'], 20)
        df_['ma60'] = self.ma(df_['close'], 60)

        # 量
        df_['volma5'] = self.volma(df_['vol'], 5)
        df_['volma10'] = self.volma(df_['vol'], 10)
        df_['volma20'] = self.volma(df_['vol'], 20)
        df_['volma60'] = self.volma(df_['vol'], 60)

        # 价量
        df_['pricevolma5'] = self.pricevol_ma(df_['close'], df_["vol"], 5)
        df_['pricevolma10'] = self.pricevol_ma(df_['close'], df_["vol"], 10)
        df_['pricevolma20'] = self.pricevol_ma(df_['close'], df_["vol"], 20)
        df_['pricevolma60'] = self.pricevol_ma(df_['close'], df_["vol"], 60)

        return df_

    def pricevol_ma(self, close_sr, vol_sr, period):
        vol_sum = vol_sr.rolling(period).sum()
        close_vol_sr = (close_sr * vol_sr).rolling(period).sum()
        return close_vol_sr / vol_sum

    def volma(self, vol_sr, period):
        return vol_sr.rolling(period).mean()


class TechnicalIndicatorPriceVol4Model(TechnicalIndicator4Model):
    '''
    20200820 版本
    特征说明：
    open high low ：日间形态
    pct_chg_i : 过去涨幅
    std_i : 相对历史的波动率
    macd, roc, middleboll, upboll, downboll, cci, rsi, wr: 技术指标,玄学

    pct_chg_vol_1 : 昨天的成交量变化率
    volma_i : 成交量均线
    vol_std_i: 成交量标准差，即相对历史的波动率

    pricevolma5, pricevolma10, pricevolma20, pricevolma60 : 加权ma

    pct_chg : 未来收益
    说明：
    删去偏度峰度

    '''

    def get_index(self, data):
        myindex = TechnicalIndicatorPriceVol()
        data = myindex.getIndex(data)

        # 价格变化
        data["pct_chg_1"] = (data["close"] - data["close"].shift(1)) / data["close"].shift(1)
        data["pct_chg_5"] = (data["close"] - data["close"].shift(5)) / data["close"].shift(5)
        data["pct_chg_10"] = (data["close"] - data["close"].shift(10)) / data["close"].shift(10)
        data["pct_chg_20"] = (data["close"] - data["close"].shift(20)) / data["close"].shift(20)
        data["pct_chg_60"] = (data["close"] - data["close"].shift(60)) / data["close"].shift(60)
        data["pct_chg_120"] = (data["close"] - data["close"].shift(120)) / data["close"].shift(120)
        data["pct_chg_250"] = (data["close"] - data["close"].shift(250)) / data["close"].shift(250)

        # 市场是否动荡
        data["std_120"] = data["close"].rolling(120).std()

        data["std_5"] = data["close"].rolling(5).std() / data["std_120"]
        data["std_10"] = data["close"].rolling(10).std() / data["std_120"]
        data["std_20"] = data["close"].rolling(20).std() / data["std_120"]
        data["std_60"] = data["close"].rolling(60).std() / data["std_120"]

        data["price_buy"] = data["open"]

        # 量昨天变化
        data["pct_chg_vol_1"] = (data["vol"] - data["vol"].shift(1)) / data["vol"].shift(1)

        # 量标准差
        data["vol_std_120"] = data["vol"].rolling(120).std()

        data["vol_std_5"] = data["vol"].rolling(5).std() / data["vol_std_120"]
        data["vol_std_10"] = data["vol"].rolling(10).std() / data["vol_std_120"]
        data["vol_std_20"] = data["vol"].rolling(20).std() / data["vol_std_120"]
        data["vol_std_60"] = data["vol"].rolling(60).std() / data["vol_std_120"]

        # 开盘价变化幅度
        data["pct_chg"] = (data["open"] - data["open"].shift(1)) / data["open"].shift(1)
        data.dropna(inplace=True)

        # 利用收盘价标准化
        data_standard_cols = ['open', 'high', 'close', 'low', 'pricevolma5', 'pricevolma10', 'pricevolma20',
                              'pricevolma60', 'middleboll', 'upboll', 'downboll', 'rsi', 'wr', "std_5", "std_10",
                              "std_20", "std_60"]

        for col in data_standard_cols:
            data[col] /= data["close"]

        # 利用成交量标准化
        data_standard_cols = ["volma5", "volma10", "volma20", "volma60"]
        for col in data_standard_cols:
            data[col] /= data["vol"]

        # 玄学指标标准化
        data["cci"] /= 100
        data["rsi"] /= 100
        data["wr"] /= 100

        # 要改
        data = data[['trade_date', 'open', 'high', 'low', 'macd', 'roc', "pricevolma5", "pricevolma10", "pricevolma20",
                     "pricevolma60", 'middleboll', 'upboll', 'downboll', 'cci', 'rsi', 'wr', "pct_chg_1", "pct_chg_5",
                     "pct_chg_10", "pct_chg_20", "pct_chg_60", "pct_chg_120", "pct_chg_250", "std_5", "std_10",
                     "std_20", "std_60",
                     "pct_chg_vol_1", "vol_std_5", "vol_std_10", "vol_std_20", "vol_std_60",
                     "volma5", "volma10", "volma20", "volma60",
                     "pct_chg"]]

        return data


if __name__ == '__main__':
    ts.set_token("82f17f80a6f62681bcbf105689d892a9559a4e65af4045adb472eaf0")
    data = ts.pro_bar(ts_code='600000.SH', adj='qfq')

    myindex = TechnicalIndicatorPriceVol4Model()
    print(data.columns)
    data = myindex.get_index(data)
    print(data.max())

    # myindex = TechnicalIndicator()
    # data = myindex.getIndex(data)
    # print(data["close"])
    # print(data["macd"])
    #
    # t4m = TechnicalIndicator4Model()
    # data.sort_values(by="trade_date", inplace=True)
    # data = t4m.get_index(data)
    # print(data)

    # data.to_excel('mydata.xlsx')
    # wr = myindex.rsi(data['close'])
    # print(wr)
    # wr.plot()
    # data['close'].plot()
    # plt.show()
    # boll = myindex.boll(data['close'])
    # print(boll)
