#!/usr/bin/env python3
"""均线全多头 + ATR止损 + 回调加仓 策略回测"""
import akshare as ak
import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

# ========== 1. 回测核心 ==========
def backtest_ma_atr(symbol, name="", atr_period=14, atr_mult=3.0):
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date="20180101", end_date="20260628",
            adjust="hfq"
        )
        df.rename(columns={
            "日期": "date", "收盘": "close", "开盘": "open",
            "最高": "high", "最低": "low", "成交量": "volume"
        }, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if len(df) < 250:
            return {"代码": symbol, "名称": name, "备注": "上市时间短，数据不足"}

        # ── 1. 均线 ──
        df["ma5"]   = df["close"].rolling(5).mean()
        df["ma10"]  = df["close"].rolling(10).mean()
        df["ma20"]  = df["close"].rolling(20).mean()
        df["ma60"]  = df["close"].rolling(60).mean()
        df["ma150"] = df["close"].rolling(150).mean()
        df["ma200"] = df["close"].rolling(200).mean()
        df["vol_ma10"] = df["volume"].rolling(10).mean()

        # ── 2. ATR（含跳空） ──
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = abs(df["high"] - df["close"].shift(1))
        df["tr3"] = abs(df["low"]  - df["close"].shift(1))
        df["tr"]  = df[["tr1","tr2","tr3"]].max(axis=1)
        df["atr"] = df["tr"].rolling(atr_period).mean()

        # ── 3. 全多头排列 + 全部向上 ──
        df["bull_arrange"] = (
            (df["ma5"] > df["ma10"]) &
            (df["ma10"] > df["ma20"]) &
            (df["ma20"] > df["ma60"]) &
            (df["ma60"] > df["ma150"]) &
            (df["ma150"] > df["ma200"]) &
            (df["ma5"]   > df["ma5"].shift(1)) &
            (df["ma10"]  > df["ma10"].shift(1)) &
            (df["ma20"]  > df["ma20"].shift(1)) &
            (df["ma60"]  > df["ma60"].shift(1)) &
            (df["ma150"] > df["ma150"].shift(1)) &
            (df["ma200"] > df["ma200"].shift(1))
        )
        df["bull_streak"] = df["bull_arrange"].groupby(
            (~df["bull_arrange"]).cumsum()
        ).cumsum()

        # MA150/MA200 是否仍向上（用于加仓判断）
        df["ma150_up"] = df["ma150"] > df["ma150"].shift(1)
        df["ma200_up"] = df["ma200"] > df["ma200"].shift(1)

        # ── 4. 交易信号 ──
        df["first_buy"] = np.where(
            (df["bull_streak"] >= 2) &
            (df["volume"] > df["vol_ma10"].shift(1) * 1.5),
            1, 0
        )

        # 加仓：均线多头排列被回调破坏后重新恢复(bull_streak==1)，
        #        且 MA150、MA200 都还在向上趋势中
        df["add_buy"] = np.where(
            (df["bull_streak"] == 1) &
            (df["ma150_up"]) &
            (df["ma200_up"]),
            1, 0
        )

        df["sell_1"] = np.where(
            (df["ma5"] < df["ma20"]) & (df["ma10"] < df["ma20"]),
            1, 0
        )
        df["sell_2"] = np.where(
            (df["ma5"] < df["ma60"]) & (df["ma10"] < df["ma60"]) &
            (df["volume"] > df["vol_ma10"].shift(1) * 1.5),
            1, 0
        )
        df["sell_3"] = np.where(
            df["ma200"] < df["ma200"].shift(1), 1, 0
        )

        # ── 5. 逐 K 线模拟 ──
        position = 0.0
        peak_price = 0.0
        atr_stop_price = 0.0
        positions = []
        trade_count = 0

        for i in range(len(df)):
            cur_c = df.loc[i, "close"]
            cur_atr = df.loc[i, "atr"]

            if i >= 1:
                prev = df.loc[i - 1]

                # 优先级 1：ATR 止损
                if position > 0 and atr_stop_price > 0 and cur_c < atr_stop_price:
                    position = 0.0
                    peak_price = 0.0
                    atr_stop_price = 0.0
                    trade_count += 1
                    positions.append(position)
                    continue

                # 优先级 2：MA200 向下清仓
                if prev["sell_3"] == 1 and position > 0:
                    position = 0.0
                    peak_price = 0.0
                    atr_stop_price = 0.0
                    trade_count += 1

                # 优先级 3：二档减仓 → 剩 30%
                elif prev["sell_2"] == 1 and position > 0.3:
                    position = 0.3
                    trade_count += 1

                # 优先级 4：一档减仓 → 剩 50%
                elif prev["sell_1"] == 1 and position > 0.5:
                    position = 0.5
                    trade_count += 1

                # 优先级 5：首仓建仓 50%
                elif prev["first_buy"] == 1 and position == 0:
                    position = 0.5
                    peak_price = df.loc[i, "open"]
                    if pd.notna(cur_atr) and cur_atr > 0:
                        atr_stop_price = peak_price - atr_mult * cur_atr
                    else:
                        atr_stop_price = peak_price * 0.92
                    trade_count += 1

                # 优先级 6：加仓 +10%（上限 100%）
                elif prev["add_buy"] == 1 and 0 < position < 1.0:
                    position = min(position + 0.1, 1.0)
                    trade_count += 1

            # 更新峰值 & ATR 止损（只上移）
            if position > 0:
                if cur_c > peak_price:
                    peak_price = cur_c
                    if pd.notna(cur_atr) and cur_atr > 0:
                        new_stop = peak_price - atr_mult * cur_atr
                        if new_stop > atr_stop_price:
                            atr_stop_price = new_stop

            positions.append(position)

        df["position"] = positions

        # ── 6. 收益与回撤 ──
        df["daily_return"]     = df["close"].pct_change()
        df["strategy_return"]  = df["daily_return"] * df["position"].shift(1)
        df["strategy_cum"]     = (1 + df["strategy_return"]).cumprod()
        df["benchmark_cum"]    = (1 + df["daily_return"]).cumprod()

        def max_drawdown(series):
            peak = series.expanding().max()
            return (series / peak - 1).min()

        return {
            "代码": symbol,
            "名称": name,
            "策略总收益":   df["strategy_cum"].iloc[-1] - 1,
            "持有总收益":   df["benchmark_cum"].iloc[-1] - 1,
            "超额收益":     df["strategy_cum"].iloc[-1] - df["benchmark_cum"].iloc[-1],
            "策略最大回撤": max_drawdown(df["strategy_cum"]),
            "持有最大回撤": max_drawdown(df["benchmark_cum"]),
            "交易次数":     trade_count,
            "首仓信号":     int(df["first_buy"].sum()),
            "加仓信号":     int(df["add_buy"].sum()),
        }
    except Exception as e:
        return {"代码": symbol, "名称": name, "备注": f"失败：{str(e)}"}


# ========== 2. 股票池 ==========
stock_list = [
    ("601899", "紫金矿业"),
    ("600309", "万华化学"),
    ("601088", "中国神华"),
    ("600519", "贵州茅台"),
    ("000858", "五粮液"),
    ("600887", "伊利股份"),
    ("600276", "恒瑞医药"),
    ("603259", "药明康德"),
    ("300760", "迈瑞医疗"),
    ("601012", "隆基绿能"),
    ("300750", "宁德时代"),
    ("600438", "通威股份"),
    ("002371", "北方华创"),
    ("603501", "韦尔股份"),
    ("300308", "中际旭创"),
]

# ========== 3. 批量回测 ==========
results = []
for code, name in stock_list:
    print(f"正在回测：{name}({code})")
    res = backtest_ma_atr(code, name)
    results.append(res)

result_df = pd.DataFrame(results)

print()
print("========== 均线全多头 + ATR止损 回测结果 ==========")
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 300)
disp = result_df.copy()
for col in disp.select_dtypes(include=['float']).columns:
    disp[col] = disp[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) and -100 < x < 100 else (f"{x:.2f}" if pd.notna(x) else "-"))
cols_show = [c for c in disp.columns if c != "备注"]
print(disp[cols_show].to_string(index=False))

# 对比汇总
print()
print("========== 汇总 ==========")
print(f"策略总收益均值：{result_df['策略总收益'].mean():.2%}")
print(f"持有总收益均值：{result_df['持有总收益'].mean():.2%}")
print(f"超额收益均值：{result_df['超额收益'].mean():.2%}")
print(f"策略最大回撤均值：{result_df['策略最大回撤'].mean():.2%}")
print(f"持有最大回撤均值：{result_df['持有最大回撤'].mean():.2%}")
print(f"平均交易次数：{result_df['交易次数'].mean():.1f}")
print()
print("策略规则：")
print("  入场：均线全多头(MA5>10>20>60>150>200 全部向上)持续≥2天 + 放量>1.5倍")
print("  加仓：多头排列被回调破坏后恢复(bull_streak==1) + MA150/MA200 仍向上")
print("  减仓1：MA5<MA20 且 MA10<MA20 → 减至50%")
print("  减仓2：MA5<MA60 且 MA10<MA60 且放量 → 减至30%")
print("  清仓：ATR追踪止损 或 MA200 拐头向下")
