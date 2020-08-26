import pandas as pd
import baostock as bs

lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:' + lg.error_code)
print('login respond  error_msg:' + lg.error_msg)

# 获取沪深300成分股
rs = bs.query_hs300_stocks(date="2015-01-01")
print('query_hs300 error_code:' + rs.error_code)
print('query_hs300  error_msg:' + rs.error_msg)

# 打印结果集
hs300_stocks = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    hs300_stocks.append(rs.get_row_data())
result = pd.DataFrame(hs300_stocks, columns=rs.fields)
result["code"] = result["code"].apply(lambda x: x[3:] + "." + x[:2].upper())
result.to_csv("./datasets/hs300_2015.csv", encoding="gbk", index=False)
print(result)

# 登出系统
bs.logout()

'''
     code
0    000629.SZ
1    000728.SZ
2    000723.SZ
'''
