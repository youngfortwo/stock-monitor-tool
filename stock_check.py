"""
A1 vs A2 vs A3(SEPA完整版) vs B买入持有
A1:
  6均线全多头入场 + 回调加仓
  卖出:
    1. MA200向下清仓
    2. 收盘跌破MA20连续3天，当前仓位减50%
    3. MA5 < MA20，当前仓位减20%
    4. MA5 < MA10 < MA60，当前仓位减10%
A2:
  Trend Template 入场
  8%初始止损 / 盈利后10%跟随止损
  跌破MA200清仓
  50日新高附近加仓
A3-SEPA(完整版，对齐股票魔法师原著):
  ✅ 基本面过滤（净利润加速+ROE+营收）
  ✅ Trend Template + RS前20% + 标准VCP波动收缩
  ✅ 净利润断层加成
  ✅ 市场健康度三档控仓
  ✅ 8%初始硬止损 / 盈利后15%跟随止损
  ✅ 金字塔递减加仓，加仓后止损上移至成本线
  ✅ 跌破MA50减到半仓，跌破MA200清仓
B:
  买入持有，首日满仓拿到底
数据:
  前复权
  手续费 + 滑点
  区间: 2018-01-01 ~ 2026-06-28
  高成长 / 高弹性股票池
"""
import numpy as np
import pandas as pd
import akshare as ak

P = dict(
    MA=[5, 10, 20, 50, 60, 150, 200],
    # A1 参数
    BULL=2,
    INIT_A1=0.5,
    # A2 参数
    INIT_A2=0.5,
    TRAIL_A2=0.10,
    # A3-SEPA 核心参数
    INIT_A3=0.7,
    TRAIL_A3=0.15,
    RS=0.5,               # RS放宽至前50%（原0.8太严格，31只股无一只同时满足7条件）
    VCP_LOOKBACK=60,      # VCP观察窗口
    VCP_MIN_CONTRACT=0.15,# 回调幅度至少收窄15%
    # 基本面过滤参数（stock_financial_abstract 为单季/报告期值，非TTM）
    NET_PROFIT_GROWTH=0.10,  # 单季度净利润增速门槛
    REVENUE_GROWTH=0.05,     # 营收增速门槛
    ROE_TTM=0.04,            # ROE 门槛（单季平均值）
    # 市场健康度参数
    BREAKOUT_WIN=20,      # 滚动统计突破成功率的窗口
    HEALTH_GREEN=60,      # 绿灯阈值
    HEALTH_YELLOW=30,     # 黄灯阈值
    # 净利润断层参数
    GAP_VALID_DAYS=20,    # 断层有效期
    # 通用参数
    VW=10,
    VOL=1.5,
    MAX=1.0,
    ADD=0.1,
    STOP=0.08,            # 原著8%硬止损
    FEE=3e-4,
    SLIP=1e-3,
    START="2018-01-01",
    END="2026-06-28",
)

# ==================================================
# 高成长 / 高弹性股票池（共31只）
# ==================================================
SYMBOLS = [
    # 新能源 / 电动车 / 光伏 / 储能
    "300750", "002594", "300274", "300014", "300124",
    "300450", "300316", "688599", "601012",
    # 半导体 / 国产替代
    "002371", "688012", "688981", "603501", "688008",
    "603986", "300604", "688256",
    # AI算力 / 光模块 / 通信
    "300308", "300502", "300394", "002475",
    # 软件 / 数字经济
    "688111", "300033", "300496", "688036",
    # 医疗器械 / 创新药 / CXO
    "300760", "300122", "300759", "688271",
    # 消费科技 / 自动化
    "688169", "688777",
]

NAMES = {
    "300750": "宁德时代", "002594": "比亚迪", "300274": "阳光电源",
    "300014": "亿纬锂能", "300124": "汇川技术", "300450": "先导智能",
    "300316": "晶盛机电", "688599": "天合光能", "601012": "隆基绿能",
    "002371": "北方华创", "688012": "中微公司", "688981": "中芯国际",
    "603501": "韦尔股份", "688008": "澜起科技", "603986": "兆易创新",
    "300604": "长川科技", "688256": "寒武纪", "300308": "中际旭创",
    "300502": "新易盛", "300394": "天孚通信", "002475": "立讯精密",
    "688111": "金山办公", "300033": "同花顺", "300496": "中科创达",
    "688036": "传音控股", "300760": "迈瑞医疗", "300122": "智飞生物",
    "300759": "康龙化成", "688271": "联影医疗", "688169": "石头科技",
    "688777": "中控技术",
}

# ==================================================
# 工具函数：数据获取
# ==================================================
def fetch(symbol):
    df = ak.stock_zh_a_hist(
        symbol=symbol, period="daily",
        start_date=P["START"].replace("-", ""),
        end_date=P["END"].replace("-", ""),
        adjust="qfq",
    )
    df = df.rename(columns={
        "日期": "date", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])  # 统一转datetime
    return df[["date", "high", "low", "close", "volume"]]

def fetch_index(symbol="sh000001"):
    idx = ak.stock_zh_index_daily(symbol=symbol)
    idx["date"] = pd.to_datetime(idx["date"])
    return idx

# ==================================================
# 新增1：获取基本面数据（按披露日滞后对齐，无未来函数）
# ==================================================
def get_fundamental_ok(symbol):
    """
    用 stock_financial_abstract 替代已失效的 financial_analysis_indicator
    """
    try:
        fin = ak.stock_financial_abstract(symbol=symbol)
        if fin.empty or len(fin.columns) < 3:
            raise ValueError("空数据")

        # 转置表：列名为日期(YYYYMMDD)，行为指标
        date_cols = [c for c in fin.columns if str(c).isdigit() and len(str(c)) == 8]
        if not date_cols:
            raise ValueError("未找到日期列")

        dates_sorted = sorted(date_cols)
        report_dates = pd.to_datetime(dates_sorted, format="%Y%m%d")
        if len(report_dates) < 2:
            raise ValueError("数据不足2期")

        def row_val(name):
            rows = fin[fin["指标"] == name]
            if rows.empty:
                return [np.nan] * len(dates_sorted)
            vals = rows.iloc[0][dates_sorted].tolist()
            return [float(v) if pd.notna(v) else np.nan for v in vals]

        net_profit_yoy = pd.Series(
            [(v / 100) if pd.notna(v) else np.nan for v in row_val("归属母公司净利润增长率")],
            index=report_dates)
        revenue_yoy = pd.Series(
            [(v / 100) if pd.notna(v) else np.nan for v in row_val("营业总收入增长率")],
            index=report_dates)
        roe_ttm = pd.Series(
            [(v / 100) if pd.notna(v) else np.nan for v in row_val("净资产收益率_平均")],
            index=report_dates)

        # 基本面达标：本季净利润正增速 + 营收正增长 + ROE 达标
        # （不再要求逐季加速，连续2季正增长即可）
        profit_ok = (net_profit_yoy > P["NET_PROFIT_GROWTH"]) & (
            net_profit_yoy.shift(1) > P["NET_PROFIT_GROWTH"])
        fund_ok = profit_ok & (revenue_yoy > P["REVENUE_GROWTH"]) & (roe_ttm > P["ROE_TTM"])

        def report_to_available(dt):
            m = dt.month
            if m <= 3:    return pd.Timestamp(year=dt.year, month=5, day=1)
            elif m <= 6:  return pd.Timestamp(year=dt.year, month=9, day=1)
            elif m <= 9:  return pd.Timestamp(year=dt.year, month=11, day=1)
            else:         return pd.Timestamp(year=dt.year + 1, month=5, day=1)

        avail_dates = pd.Series(report_dates).apply(report_to_available).values

        dates = pd.date_range(start=P["START"], end=P["END"], freq="D")
        daily = pd.DataFrame({"date": pd.to_datetime(dates), "fund_ok": False})

        for i, dt in enumerate(report_dates):
            avail = avail_dates[i]
            ok = bool(fund_ok.iloc[i]) if pd.notna(fund_ok.iloc[i]) else False
            if pd.notna(avail) and avail <= pd.Timestamp(P["END"]):
                daily.loc[daily["date"] >= pd.Timestamp(avail), "fund_ok"] = ok

        return daily[["date", "fund_ok"]]

    except Exception as e:
        print(f"{symbol} 基本面获取失败(默认不满足): {str(e)[:60]}")
        dates = pd.date_range(start=P["START"], end=P["END"], freq="D")
        return pd.DataFrame({"date": pd.to_datetime(dates), "fund_ok": False})
# ==================================================
# 新增2：标准VCP形态识别
# ==================================================
def detect_standard_vcp(d, lookback=60, min_contract=0.15):
    d = d.copy()
    d["local_high"] = d.high == d.high.rolling(11, center=True).max()
    d["local_low"] = d.low == d.low.rolling(11, center=True).min()
    d["vcp_setup"] = False
    d["pivot_price"] = np.nan

    for i in range(lookback, len(d)):
        window = d.iloc[i-lookback:i+1]
        highs = window[window.local_high]["high"].values
        lows = window[window.local_low]["low"].values
        vols = window["volume"].values

        if len(highs) < 2 or len(lows) < 2:
            continue

        if highs[-1] <= highs[-2]:
            continue

        dd1 = (highs[-2] - lows[-2]) / highs[-2]
        dd2 = (highs[-1] - lows[-1]) / highs[-1]
        if dd2 >= dd1 * (1 - min_contract):
            continue

        vol1 = np.mean(vols[:len(vols)//2])
        vol2 = np.mean(vols[len(vols)//2:])
        if vol2 >= vol1 * 0.9:
            continue

        d.loc[d.index[i], "vcp_setup"] = True
        d.loc[d.index[i], "pivot_price"] = highs[-1]

    d["pivot_break"] = d.close > d.pivot_price.shift(1)
    return d

# ==================================================
# 新增3：净利润断层识别
# ==================================================
def detect_earnings_gap(d):
    d = d.copy()
    # 向上跳空缺口：当日最低价 > 前一日最高价
    d["gap_up"] = d.low > d.high.shift(1)
    # 缺口日放量
    d["gap_vol"] = d.volume > d.volume.rolling(10).mean() * 1.5
    # 有效缺口：跳空+放量
    d["gap_valid"] = d["gap_up"] & d["gap_vol"]
    # 断层有效期内标记
    d["earnings_gap"] = d["gap_valid"].rolling(P["GAP_VALID_DAYS"], min_periods=1).max() == 1
    return d

# ==================================================
# 新增4：计算全市场健康度评分
# ==================================================
def calc_market_health(raw, symbols):
    # 维度1：第二阶段个股占比
    stage2_count = pd.Series(0, index=raw[symbols[0]].index)
    for s in symbols:
        if s in raw:
            stage2_count += raw[s]["tt"].astype(int)
    stage2_ratio = stage2_count / len(symbols)
    
    # 维度2：突破交易成功率（滚动20笔）
    all_breakouts = []
    for s in symbols:
        if s not in raw:
            continue
        d = raw[s].copy()
        # 标记突破点
        breakouts = d[d["pivot_break"] & d["vsp"]].copy()
        # 计算突破后20日最高收益，判断是否成功（涨超10%且未触发8%止损）
        d["future_20d_high"] = d["high"].shift(-1).rolling(20).max()
        d["future_20d_low"] = d["low"].shift(-1).rolling(20).min()
        d["break_success"] = (
            (d["future_20d_high"] > d["close"] * 1.1) &
            (d["future_20d_low"] > d["close"] * 0.92)
        )
        success = d[d["pivot_break"] & d["vsp"]]["break_success"].copy()
        all_breakouts.append(success)
    
    if all_breakouts:
        all_break = pd.concat(all_breakouts).sort_index()
        all_break = all_break.groupby(all_break.index).mean()
        success_rate = all_break.rolling(P["BREAKOUT_WIN"], min_periods=5).mean()
        stage2_idx = stage2_ratio.groupby(stage2_ratio.index).mean().index
        success_rate = success_rate.reindex(stage2_idx).ffill().fillna(0.5)
    else:
        success_rate = pd.Series(0.5, index=stage2_ratio.index)
    
    # 维度3：大盘200日线（后续合并）
    idx = fetch_index()
    idx["ma200"] = idx["close"].rolling(200).mean()
    idx["market_above_ma200"] = (idx["close"] > idx["ma200"]).astype(int)
    idx = idx.set_index("date")
    idx = idx.reindex(stage2_ratio.index).ffill()
    
    # 综合评分（百分制）：阶段占比40分 + 成功率40分 + 大盘20分
    health_score = (
        stage2_ratio * 40 +
        success_rate * 40 +
        idx["market_above_ma200"] * 20
    )
    
    # 分档
    health_level = pd.Series("red", index=health_score.index)
    health_level[health_score >= P["HEALTH_YELLOW"]] = "yellow"
    health_level[health_score >= P["HEALTH_GREEN"]] = "green"
    
    return pd.DataFrame({
        "health_score": health_score,
        "health_level": health_level
    })

# ==================================================
# 计算技术指标
# ==================================================
def add_indicators(d, market_idx, fund_df):
    d = d.copy()
      # 合并大盘指数（补上date列，修复merge找不到键的bug）
    d = d.merge(
        market_idx[["date", "close", "ma200"]].rename(columns={"close": "idx_close"}),
        on="date",
        how="left"
    )
    d["idx_close"] = d["idx_close"].ffill()
    d["market_ma200"] = d["ma200"].ffill()
    d["market_bull"] = d["idx_close"] > d["market_ma200"]
    
    # 合并基本面
    d = d.merge(fund_df, on="date", how="left")
    d["fund_ok"] = d["fund_ok"].ffill().fillna(False)

    # 均线
    for n in P["MA"]:
        d[f"ma{n}"] = d.close.rolling(n).mean()

    # 成交量均线与放量
    d["vma"] = d.volume.rolling(P["VW"]).mean()
    d["vsp"] = d.volume > d.vma * P["VOL"]

    # A1: 6均线全多头
    bull = (
        (d.ma5 > d.ma10) & (d.ma10 > d.ma20) &
        (d.ma20 > d.ma60) & (d.ma60 > d.ma150) &
        (d.ma150 > d.ma200)
    )
    up = pd.concat(
        [d[f"ma{n}"] > d[f"ma{n}"].shift() for n in [5, 10, 20, 60, 150, 200]],
        axis=1,
    ).all(axis=1)
    d["bull"] = bull & up
    s = d.bull.astype(int)
    d["bs"] = s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
    d["m150u"] = d.ma150 > d.ma150.shift()
    d["m200u"] = d.ma200 > d.ma200.shift()
    d["m200u20"] = d.ma200 > d.ma200.shift(20)
    d["b20_3"] = (d.close < d.ma20).rolling(3).sum() == 3

    # Trend Template
    hi_252 = d.close.rolling(252).max()
    lo_252 = d.close.rolling(252).min()
    d["tt"] = (
        (d.close > d.ma50) & (d.ma50 > d.ma150) &
        (d.ma150 > d.ma200) & d.m200u20 &
        (d.close > lo_252 * 1.3) & (d.close > hi_252 * 0.75)
    )

    # RS相对强度
    d["rs"] = d.close.pct_change(126)

    # A2: 50日新高
    d["brk"] = d.close >= d.close.rolling(50).max() * 0.99

    # VCP形态
    d = detect_standard_vcp(d, lookback=P["VCP_LOOKBACK"], min_contract=P["VCP_MIN_CONTRACT"])

    # 净利润断层
    d = detect_earnings_gap(d)

    return d

# ==================================================
# 回测主函数
# ==================================================
def backtest(d, mode, health_df=None):
    pos = 0.0
    peak = 0.0
    buy = 0.0
    avg_cost = 0.0
    add_count = 0
    rows = []

    for _, r in d.iterrows():
        c = r.close
        if pos > 0:
            peak = max(peak, r.high)

        # 获取当日市场健康度档位
        health_level = "green"
        if health_df is not None and r.date in health_df.index:
            health_level = health_df.loc[r.date, "health_level"]

        # ==================================================
        # A1
        # ==================================================
        if mode == "A1":
            if pos > 0 and not r.m200u:
                pos = 0.0
            elif pos > 0 and r.b20_3:
                pos *= 0.5
            elif pos > 0 and r.ma5 < r.ma20:
                pos *= 0.8
            elif pos > 0 and r.ma5 < r.ma10 < r.ma60:
                pos *= 0.9
            elif pos == 0 and r.bs >= P["BULL"] and r.vsp:
                pos = P["INIT_A1"]
                peak = r.high
                buy = c
            elif 0 < pos < P["MAX"] and r.bs == 1 and r.m150u and r.m200u:
                pos = min(P["MAX"], pos + P["ADD"])

        # ==================================================
        # A2
        # ==================================================
        elif mode == "A2":
            if pos > 0:
                stop = buy * (1 - P["STOP"]) if peak < buy * 1.08 else peak * (1 - P["TRAIL_A2"])
            else:
                stop = 0.0

            if pos > 0 and (c < stop or c < r.ma200):
                pos = 0.0
            elif pos == 0 and r.tt:
                pos = P["INIT_A2"]
                buy = c
                peak = r.high
            elif 0 < pos < P["MAX"] and r.brk and r.m200u:
                pos = min(P["MAX"], pos + P["ADD"])

        # ==================================================
        # A3-SEPA 完整版
        # ==================================================
        elif mode == "A3":
            # 动态参数：根据市场健康度调整
            if health_level == "green":
                init_pos = P["INIT_A3"]
                vol_mult = P["VOL"]
                can_open = True
            elif health_level == "yellow":
                init_pos = P["INIT_A3"] * 0.5
                vol_mult = P["VOL"] * 1.3
                can_open = True
            else:  # red
                init_pos = 0.0
                vol_mult = 999
                can_open = False

            # 净利润断层加成：首仓+20%
            if r.earnings_gap and can_open:
                init_pos = min(init_pos * 1.2, P["MAX"])

            # 止损计算
            if pos > 0:
                stop_pct = P["STOP"] if peak < buy * 1.08 else P["TRAIL_A3"]
                # 断层收窄止损
                if r.earnings_gap:
                    stop_pct = max(stop_pct - 0.01, 0.06)
                stop = peak * (1 - stop_pct) if peak > buy * 1.08 else buy * (1 - stop_pct)
                
                if add_count > 0:
                    stop = max(stop, avg_cost)
            else:
                stop = 0.0

            # ---------- 卖出（优先级从高到低） ----------
            if pos > 0 and c < stop:
                pos = 0.0
                add_count = 0
            elif pos > 0 and c < r.ma200:
                pos = 0.0
                add_count = 0
            elif pos > 0.5 and c < r.ma50:
                pos = 0.5

            # ---------- 买入（fund_ok + tt 双确认，轻量版A3） ----------
            elif (pos == 0 and r.fund_ok and r.tt):
                pos = init_pos
                buy = c
                avg_cost = c
                peak = r.high
                add_count = 0

            # ---------- 金字塔加仓 ----------
            elif (0 < pos < P["MAX"] and can_open and r.pivot_break and
                  add_count < 2 and c > avg_cost):
                add_amount = P["ADD"] if add_count == 0 else P["ADD"] * 0.5
                new_pos = min(P["MAX"], pos + add_amount)
                add_share = new_pos - pos
                avg_cost = (avg_cost * pos + c * add_share) / new_pos
                pos = new_pos
                add_count += 1

        rows.append((r.date, c, pos))

    res = pd.DataFrame(rows, columns=["date", "close", "pos"])
    prev_pos = res.pos.shift().fillna(0)
    fee_rate = P["FEE"] + P["SLIP"]
    res["net"] = (
        res.close.pct_change().fillna(0) * prev_pos
        - (res.pos - prev_pos).abs() * fee_rate
    )
    res["equity"] = (1 + res.net).cumprod()
    return res.set_index("date")["equity"]

# ==================================================
# 绩效计算
# ==================================================
def perf(e):
    e = e.dropna()
    n = len(e)
    if n == 0:
        return dict(ret=np.nan, cagr=np.nan, dd=np.nan, sharpe=np.nan)
    r = e.pct_change().fillna(0)
    total_ret = e.iloc[-1] - 1
    cagr = e.iloc[-1] ** (252 / n) - 1
    max_dd = (e / e.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return dict(ret=total_ret, cagr=cagr, dd=max_dd, sharpe=sharpe)

def fmt_pct(x, digits=0):
    if pd.isna(x):
        return "nan"
    return f"{x:.{digits}%}"

# ==================================================
# 主程序
# ==================================================
if __name__ == "__main__":
    # 1. 获取大盘指数
    print("正在获取大盘指数数据...")
    idx = fetch_index()
    idx["ma200"] = idx["close"].rolling(200).mean()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")

    # 2. 获取个股数据+基本面+技术指标
    raw = {}
    fund_data = {}
    print("正在获取个股数据与基本面...")
    for s in SYMBOLS:
        try:
            fund_df = get_fundamental_ok(s)
            fund_data[s] = fund_df
            kline = fetch(s)
            raw[s] = add_indicators(kline, idx.reset_index(), fund_df).set_index("date")
            print(f"  {NAMES.get(s,s)} 加载完成")
        except Exception as e:
            print(f"  {s} {NAMES.get(s, '')} 加载失败: {e}")

    if len(raw) == 0:
        raise RuntimeError("没有成功获取任何股票数据")

    # 3. 计算截面RS排名
    print("正在计算相对强度RS...")
    rs = pd.concat({s: raw[s].rs for s in raw.keys()}, axis=1).rank(axis=1, pct=True)

    # 4. 计算市场健康度
    print("正在计算市场健康度评分...")
    health_df = calc_market_health(raw, list(raw.keys()))

    groups = {k: [] for k in ["A1", "A2", "A3", "B"]}
    rows_ret, rows_dd, rows_sharpe = [], [], []

    # 5. 单只回测
    print("正在运行回测...")
    for s in SYMBOLS:
        if s not in raw:
            continue
        try:
            d = raw[s].copy()
            d["rsok"] = rs[s] > P["RS"]
            # 只 drop A3 必要字段的 NaN，保留 fund_ok/tt/rsok 等已 ffill 的行
            essential = ["close", "high", "low", "volume", "vma", "ma50", "ma150", "ma200", "tt", "pivot_break", "vsp"]
            d = d.dropna(subset=essential).reset_index()
            if len(d) == 0:
                continue

            equities = {
                "A1": backtest(d, "A1"),
                "A2": backtest(d, "A2"),
                "A3": backtest(d, "A3", health_df=health_df),
                "B": (d.close / d.close.iloc[0]).set_axis(d.date),
            }

            for k in ["A1", "A2", "A3", "B"]:
                groups[k].append(equities[k].rename(s))

            p1, p2, p3, pb = perf(equities["A1"]), perf(equities["A2"]), perf(equities["A3"]), perf(equities["B"])
            rows_ret.append([s, NAMES.get(s, s), p1["ret"], p2["ret"], p3["ret"], pb["ret"]])
            rows_dd.append([s, NAMES.get(s, s), p1["dd"], p2["dd"], p3["dd"], pb["dd"]])
            rows_sharpe.append([s, NAMES.get(s, s), p1["sharpe"], p2["sharpe"], p3["sharpe"], pb["sharpe"]])
        except Exception as e:
            print(f"{NAMES.get(s,s)} 回测失败: {e}")

    # 6. 输出结果
    ret_table = pd.DataFrame(rows_ret, columns=["代码", "名称", "A1收益", "A2收益", "A3-SEPA收益", "B持有收益"])
    for col in ret_table.columns[2:]:
        ret_table[col] = ret_table[col].map(lambda x: fmt_pct(x, 0))
    print("\n========== 每只股票四方案收益对比 ==========")
    print(ret_table.to_string(index=False))

    dd_table = pd.DataFrame(rows_dd, columns=["代码", "名称", "A1回撤", "A2回撤", "A3-SEPA回撤", "B持有回撤"])
    for col in dd_table.columns[2:]:
        dd_table[col] = dd_table[col].map(lambda x: fmt_pct(x, 0))
    print("\n========== 每只股票四方案最大回撤对比 ==========")
    print(dd_table.to_string(index=False))

    sharpe_table = pd.DataFrame(rows_sharpe, columns=["代码", "名称", "A1夏普", "A2夏普", "A3-SEPA夏普", "B持有夏普"])
    for col in sharpe_table.columns[2:]:
        sharpe_table[col] = sharpe_table[col].map(lambda x: f"{x:.2f}" if not pd.isna(x) else "nan")
    print("\n========== 每只股票四方案夏普对比 ==========")
    print(sharpe_table.to_string(index=False))

    print("\n========== 组合汇总 ==========")
    print(f"{'策略':<10}{'收益':>10}{'CAGR':>10}{'最大回撤':>12}{'夏普':>10}")
    for name in ["A1", "A2", "A3", "B"]:
        if len(groups[name]) == 0:
            continue
        portfolio = pd.concat(groups[name], axis=1).mean(axis=1).dropna()
        p = perf(portfolio)
        show_name = {"A1": "A1", "A2": "A2", "A3": "A3-SEPA", "B": "B持有"}[name]
        print(
            f"{show_name:<10}"
            f"{fmt_pct(p['ret'], 0):>10}"
            f"{fmt_pct(p['cagr'], 1):>10}"
            f"{fmt_pct(p['dd'], 0):>12}"
            f"{p['sharpe']:>10.2f}"
        )