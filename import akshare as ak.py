import akshare as ak

# ========== 1、居民月度新增开户（仅到2023-08） ==========
print("=====居民月度新增开户=====")
df_account = ak.stock_account_statistics_em()
df_account = df_account.sort_values("数据日期").reset_index(drop=True)
print(df_account[["数据日期", "新增投资者-数量"]].tail(12))

# ========== 2、居民储蓄存款（存款搬家，已经正常） ==========
print("\n=====居民储蓄存款月度数据（存款搬家）=====")
df_deposit = ak.macro_rmb_deposit()
# 新增储蓄存款-数量 负数=存款搬家流出
print(df_deposit[["月份", "新增储蓄存款-数量"]].tail(12))

# ========== 3、沪市两融：先打印全部列名，再取数据 ==========
print("\n=====沪市两融-全部列名=====")
df_sse = ak.stock_margin_sse()
print(df_sse.columns.tolist())
print("沪市两融近5行原始数据：")
print(df_sse.tail(5))

# ========== 4、深市两融：同样先打印列名，不硬编码字段 ==========
print("\n=====深市两融-全部列名=====")
df_sz = ak.stock_margin_szse()
print(df_sz.columns.tolist())
print("深市两融近5行原始数据：")
print(df_sz.tail(5))