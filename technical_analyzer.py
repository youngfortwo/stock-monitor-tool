#!/usr/bin/env python3
"""市场技术分析模块 — 全维度指标 + 19种形态检测 + 量价分析 + 交易计划"""
from __future__ import annotations
import os
from datetime import datetime, timedelta

import akshare as ak
import numpy as np
import pandas as pd

PEAK_WINDOW = 5
ATR_STOP_MULTIPLE = 2.0
TARGET_RR_RATIO = 2.0


def get_stock_data(code: str, days: int = 365) -> pd.DataFrame:
    """获取A股前复权日线行情数据（新浪接口，与项目已有数据源一致）"""
    from sepa_stage2_scanner import market_symbol, date_yyyymmdd, today_yyyymmdd
    df = ak.stock_zh_a_daily(
        symbol=market_symbol(code),
        start_date=date_yyyymmdd(520),
        end_date=today_yyyymmdd(),
        adjust="qfq",
    )
    df.rename(columns={
        "open": "open", "high": "high", "low": "low", "close": "close",
        "volume": "volume"
    }, inplace=True)
    if df.empty or len(df) < 120:
        raise RuntimeError(f"股票 {code} 历史数据不足（仅{len(df)}条），无法分析")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    if "pct_chg" not in df.columns:
        df["pct_chg"] = df["close"].pct_change() * 100
    return df.sort_index()


def calc_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    for n in [5, 10, 20, 60, 120, 250]:
        df[f"ma{n}"] = df["close"].rolling(n).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd_bar"] = 2 * (df["dif"] - df["dea"])
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss))
    df["boll_mid"] = df["ma20"]
    df["boll_std"] = df["close"].rolling(20).std()
    df["boll_upper"] = df["boll_mid"] + 2 * df["boll_std"]
    df["boll_lower"] = df["boll_mid"] - 2 * df["boll_std"]
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ma10"] = df["volume"].rolling(10).mean()
    df["price_rank"] = df["close"].rank(pct=True)
    return df.dropna()


def find_peaks_troughs(df: pd.DataFrame, window: int = PEAK_WINDOW, lookback: int = 120) -> tuple:
    df = df.iloc[-lookback:].copy()
    roll = window * 2 + 1
    peaks = df["high"].where(df["high"] == df["high"].rolling(roll, center=True).max(), np.nan)
    troughs = df["low"].where(df["low"] == df["low"].rolling(roll, center=True).min(), np.nan)
    return peaks.dropna(), troughs.dropna()


def get_volume_profile(df: pd.DataFrame, period: int = 60, bins: int = 20) -> dict:
    recent = df.iloc[-period:]
    price_edges = np.linspace(recent["low"].min(), recent["high"].max(), bins + 1)
    vol_profile = []
    for i in range(bins):
        mask = (recent["low"] <= price_edges[i + 1]) & (recent["high"] >= price_edges[i])
        total_vol = recent.loc[mask, "volume"].sum()
        vol_profile.append((price_edges[i], price_edges[i + 1], total_vol))
    max_bin = max(vol_profile, key=lambda x: x[2])
    return {
        "密集区下沿": round(max_bin[0], 2),
        "密集区上沿": round(max_bin[1], 2),
        "密集区中枢": round((max_bin[0] + max_bin[1]) / 2, 2),
        "区间累计成交量": int(max_bin[2])
    }


def calculate_risk_reward(df: pd.DataFrame, signal_type: str = "buy",
                          entry_price: float = None) -> dict:
    latest = df.iloc[-1]
    if entry_price is None:
        entry_price = latest["close"]
    atr = latest["atr"]
    stop_distance = ATR_STOP_MULTIPLE * atr
    if signal_type == "buy":
        stop_loss = entry_price - stop_distance
        target_price = entry_price + TARGET_RR_RATIO * stop_distance
    else:
        stop_loss = entry_price + stop_distance
        target_price = entry_price - TARGET_RR_RATIO * stop_distance
    return {
        "建议入场价": round(entry_price, 2),
        "建议止损价": round(stop_loss, 2),
        "目标价位": round(target_price, 2),
        "预期盈亏比": TARGET_RR_RATIO,
        "单笔止损幅度": f"{abs(entry_price - stop_loss) / entry_price * 100:.2f}%"
    }


def analyze_trend(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    ma_bull = latest["ma5"] > latest["ma10"] > latest["ma20"] > latest["ma60"]
    ma_bear = latest["ma5"] < latest["ma10"] < latest["ma20"] < latest["ma60"]
    above_ma20 = latest["close"] > latest["ma20"]
    above_ma60 = latest["close"] > latest["ma60"]
    above_ma250 = latest["close"] > latest["ma250"] if "ma250" in df.columns else above_ma60
    macd_bull = latest["dif"] > latest["dea"] and latest["dif"] > 0
    score = sum([ma_bull * 2, above_ma20, above_ma60, above_ma250, macd_bull])
    if score >= 5:    trend_level = "强多头趋势"
    elif score >= 3:  trend_level = "多头震荡"
    elif score >= 2:  trend_level = "多空平衡"
    elif score >= 1:  trend_level = "空头震荡"
    else:             trend_level = "强空头趋势"
    return {
        "趋势等级": trend_level,
        "短期趋势(20周期)": "向上" if above_ma20 else "向下",
        "中期趋势(60周期)": "向上" if above_ma60 else "向下",
        "长期趋势(年线)": "向上" if above_ma250 else "向下",
        "均线结构": "多头排列" if ma_bull else ("空头排列" if ma_bear else "杂乱排列"),
        "MACD状态": "多头区域" if macd_bull else "空头区域"
    }


def detect_signals(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    buy_signals = []
    sell_signals = []
    vol_ratio = latest["volume"] / latest["vol_ma5"]
    uptrend_short = latest["close"] > latest["ma20"]
    period_30 = df.iloc[-30:]
    price_near_high = latest["close"] >= period_30["close"].max() * 0.98
    vol_near_high = latest["volume"] >= period_30["volume"].max() * 0.9
    top_divergence = price_near_high and not vol_near_high
    price_near_low = latest["close"] <= period_30["close"].min() * 1.02
    bottom_divergence = price_near_low and latest["volume"] <= period_30["volume"].min() * 1.1

    def calc_strength(is_buy: bool) -> tuple:
        if is_buy:
            if vol_ratio >= 1.3 and uptrend_short and not top_divergence:
                return "强", "放量配合、趋势向上、无顶背离"
            elif 0.8 <= vol_ratio < 1.3 and not top_divergence:
                return "中", "量能正常、无明显背离"
            else:
                reason = []
                if vol_ratio < 0.8: reason.append("缩量无配合")
                if not uptrend_short: reason.append("逆短期趋势")
                if top_divergence: reason.append("存在顶背离")
                return "弱", "、".join(reason)
        else:
            if vol_ratio >= 1.3 and not uptrend_short and not bottom_divergence:
                return "强", "放量确认、趋势向下、无底背离"
            elif 0.8 <= vol_ratio < 1.3 and not bottom_divergence:
                return "中", "量能正常、无明显背离"
            else:
                reason = []
                if vol_ratio < 0.8: reason.append("缩量确认不足")
                if uptrend_short: reason.append("逆短期趋势")
                if bottom_divergence: reason.append("存在底背离")
                return "弱", "、".join(reason)

    if prev["dif"] <= prev["dea"] and latest["dif"] > latest["dea"]:
        level, reason = calc_strength(is_buy=True)
        buy_signals.append({"名称": "MACD金叉", "强度": level, "依据": reason})
    if prev["dif"] >= prev["dea"] and latest["dif"] < latest["dea"]:
        level, reason = calc_strength(is_buy=False)
        sell_signals.append({"名称": "MACD死叉", "强度": level, "依据": reason})

    if prev["ma5"] <= prev["ma10"] and latest["ma5"] > latest["ma10"]:
        level, reason = calc_strength(is_buy=True)
        buy_signals.append({"名称": "5/10日均线金叉", "强度": level, "依据": reason})
    if prev["ma5"] >= prev["ma10"] and latest["ma5"] < latest["ma10"]:
        level, reason = calc_strength(is_buy=False)
        sell_signals.append({"名称": "5/10日均线死叉", "强度": level, "依据": reason})

    if latest["rsi"] < 30:
        level, reason = calc_strength(is_buy=True)
        buy_signals.append({"名称": "RSI进入超卖区间", "强度": level, "依据": reason})
    if latest["rsi"] > 70:
        level, reason = calc_strength(is_buy=False)
        sell_signals.append({"名称": "RSI进入超买区间", "强度": level, "依据": reason})

    if latest["close"] <= latest["boll_lower"]:
        level, reason = calc_strength(is_buy=True)
        buy_signals.append({"名称": "触及布林带下轨支撑", "强度": level, "依据": reason})
    if latest["close"] >= latest["boll_upper"]:
        level, reason = calc_strength(is_buy=False)
        sell_signals.append({"名称": "触及布林带上轨压力", "强度": level, "依据": reason})

    if latest["close"] > prev["close"] and latest["volume"] > latest["vol_ma5"] * 1.2:
        buy_signals.append({"名称": "放量上涨，量价配合", "强度": "强", "依据": "成交量显著放大，资金进场明显"})
    if latest["close"] < prev["close"] and latest["volume"] > latest["vol_ma5"] * 1.2:
        sell_signals.append({"名称": "放量下跌，抛压较重", "强度": "强", "依据": "成交量显著放大，抛压集中释放"})

    if prev["close"] <= prev["ma20"] and latest["close"] > latest["ma20"]:
        level = "强" if vol_ratio >= 1.3 else "中"
        reason = "放量突破" if vol_ratio >= 1.3 else "缩量突破，确认度一般"
        buy_signals.append({"名称": "有效突破20日均线", "强度": level, "依据": reason})
    if prev["close"] >= prev["ma20"] and latest["close"] < latest["ma20"]:
        level = "强" if vol_ratio >= 1.3 else "中"
        reason = "放量破位" if vol_ratio >= 1.3 else "缩量破位，确认度一般"
        sell_signals.append({"名称": "有效跌破20日均线", "强度": level, "依据": reason})

    def score_signals(signals):
        score = 0
        for s in signals:
            if s["强度"] == "强": score += 3
            elif s["强度"] == "中": score += 2
            else: score += 1
        return score

    buy_score = score_signals(buy_signals)
    sell_score = score_signals(sell_signals)
    net_score = buy_score - sell_score

    if net_score >= 4:    overall = "强烈看多"
    elif net_score >= 2:  overall = "偏多"
    elif net_score > -2:  overall = "多空平衡"
    elif net_score > -4:  overall = "偏空"
    else:                 overall = "强烈看空"

    return {
        "买入信号": buy_signals,
        "卖出信号": sell_signals,
        "买入强度总分": buy_score,
        "卖出强度总分": sell_score,
        "净多空强度": net_score,
        "综合信号评级": overall
    }


def detect_vcp(df: pd.DataFrame) -> dict:
    result = {"形态": "VCP波动收缩", "类型": "看涨持续", "是否成立": False, "强度": 0, "说明": ""}
    latest = df.iloc[-1]
    trend_ok = (latest["close"] > latest["ma20"] > latest["ma60"] > latest["ma120"]
                and latest["ma20"] > latest["ma60"])
    if not trend_ok:
        result["说明"] = "不满足上升趋势前提"
        return result
    roll = PEAK_WINDOW * 2 + 1
    df2 = df.copy()
    df2["is_peak"] = df2["high"] == df2["high"].rolling(window=roll, center=True).max()
    df2["is_trough"] = df2["low"] == df2["low"].rolling(window=roll, center=True).min()
    recent = df2.iloc[-60:]
    peaks = recent[recent["is_peak"]]["high"].sort_index()
    troughs = recent[recent["is_trough"]]["low"].sort_index()
    if len(peaks) < 3 or len(troughs) < 2:
        result["说明"] = "极值点不足"
        return result
    waves = []
    peak_list = peaks.tolist()
    peak_dates = peaks.index.tolist()
    trough_list = troughs.tolist()
    trough_dates = troughs.index.tolist()
    for i in range(len(peak_list) - 1):
        post = [t for t in trough_dates if t > peak_dates[i]]
        if not post: continue
        td = post[0]; t_idx = trough_dates.index(td)
        drawdown = (peak_list[i] - trough_list[t_idx]) / peak_list[i]
        period_vol = float(df.loc[peak_dates[i]:td, "volume"].mean())
        waves.append({"peak_price": peak_list[i], "trough_price": trough_list[t_idx],
                        "drawdown": drawdown, "avg_vol": period_vol})
    waves = waves[-4:]
    if len(waves) < 2:
        result["说明"] = "有效回调波段不足"
        return result
    drawdowns = [w["drawdown"] for w in waves]
    vols = [w["avg_vol"] for w in waves]
    is_contracting = all(drawdowns[i] > drawdowns[i + 1] for i in range(len(drawdowns) - 1))
    vol_contracting = all(vols[i] > vols[i + 1] for i in range(len(vols) - 1))
    if is_contracting and vol_contracting:
        cr = drawdowns[-1] / drawdowns[0] if drawdowns[0] else 1
        last_peak = waves[-1]["peak_price"]
        base_height = waves[0]["peak_price"] - waves[0]["trough_price"]
        target_price = last_peak + base_height
        result.update({
            "是否成立": True, "强度": 85 if cr < 0.5 else 70,
            "回调波段数": len(waves),
            "各段回调幅度": [f"{d*100:.2f}%" for d in drawdowns],
            "量能收缩情况": "逐次递减" if vol_contracting else "未同步收缩",
            "波动收缩率": f"{cr*100:.1f}%（末段/首段）",
            "目标位": round(target_price, 2),
            "说明": f"已形成{len(waves)}段收缩结构"
        })
    else:
        result["说明"] = "未呈现逐次收缩"
    return result


def detect_double_bottom(peaks, troughs, latest_price) -> dict:
    result = {"形态": "双底(W底)", "类型": "看涨反转", "是否成立": False, "强度": 0, "说明": ""}
    if len(troughs) < 2 or len(peaks) < 1:
        result["说明"] = "极值点不足"; return result
    t1_price, t2_price = troughs.iloc[-2], troughs.iloc[-1]
    t1_date, t2_date = troughs.index[-2], troughs.index[-1]
    neck = peaks[(peaks.index > t1_date) & (peaks.index < t2_date)]
    if len(neck) == 0:
        result["说明"] = "无明确颈线"; return result
    neck_price = neck.iloc[0]
    bottom_diff = abs(t1_price - t2_price) / t1_price
    if bottom_diff < 0.05 and t2_price >= t1_price * 0.98:
        target_price = neck_price + (neck_price - min(t1_price, t2_price))
        result.update({
            "是否成立": True, "颈线位": round(neck_price, 2), "目标位": round(target_price, 2),
            "强度": int((1 - bottom_diff) * 50 + 50),
            "说明": f"两底价差{bottom_diff*100:.1f}%，颈线{neck_price:.2f}"
        })
    else:
        result["说明"] = "两底不对称"
    return result


def detect_double_top(peaks, troughs, latest_price) -> dict:
    result = {"形态": "双顶(M头)", "类型": "看跌反转", "是否成立": False, "强度": 0, "说明": ""}
    if len(peaks) < 2 or len(troughs) < 1:
        result["说明"] = "极值点不足"; return result
    p1_price, p2_price = peaks.iloc[-2], peaks.iloc[-1]
    p1_date, p2_date = peaks.index[-2], peaks.index[-1]
    neck = troughs[(troughs.index > p1_date) & (troughs.index < p2_date)]
    if len(neck) == 0:
        result["说明"] = "无明确颈线"; return result
    neck_price = neck.iloc[0]
    top_diff = abs(p1_price - p2_price) / p1_price
    if top_diff < 0.05 and p2_price <= p1_price * 1.02:
        target_price = neck_price - (max(p1_price, p2_price) - neck_price)
        result.update({
            "是否成立": True, "颈线位": round(neck_price, 2), "目标位": round(target_price, 2),
            "强度": int((1 - top_diff) * 50 + 50),
            "说明": f"两顶价差{top_diff*100:.1f}%，颈线{neck_price:.2f}"
        })
    else:
        result["说明"] = "两顶不对称"
    return result


def detect_head_shoulders_bottom(peaks, troughs) -> dict:
    result = {"形态": "头肩底", "类型": "看涨反转", "是否成立": False, "强度": 0, "说明": ""}
    if len(troughs) < 3 or len(peaks) < 2:
        result["说明"] = "极值点不足"; return result
    left, head, right = troughs.iloc[-3], troughs.iloc[-2], troughs.iloc[-1]
    if head < left and head < right and abs(left - right) / left < 0.06:
        neckline = (peaks.iloc[-2] + peaks.iloc[-1]) / 2
        target = neckline + (neckline - head)
        result.update({
            "是否成立": True, "颈线位": round(neckline, 2), "目标位": round(target, 2),
            "强度": 85, "说明": f"左肩{left:.2f}/头部{head:.2f}/右肩{right:.2f}"
        })
    else:
        result["说明"] = "结构不满足"
    return result


def detect_head_shoulders_top(peaks, troughs) -> dict:
    result = {"形态": "头肩顶", "类型": "看跌反转", "是否成立": False, "强度": 0, "说明": ""}
    if len(peaks) < 3 or len(troughs) < 2:
        result["说明"] = "极值点不足"; return result
    left, head, right = peaks.iloc[-3], peaks.iloc[-2], peaks.iloc[-1]
    if head > left and head > right and abs(left - right) / left < 0.06:
        neckline = (troughs.iloc[-2] + troughs.iloc[-1]) / 2
        target = neckline - (head - neckline)
        result.update({
            "是否成立": True, "颈线位": round(neckline, 2), "目标位": round(target, 2),
            "强度": 85, "说明": f"左肩{left:.2f}/头部{head:.2f}/右肩{right:.2f}"
        })
    else:
        result["说明"] = "结构不满足"
    return result


def detect_ascending_triangle(peaks, troughs) -> dict:
    result = {"形态": "上升三角形", "类型": "看涨持续", "是否成立": False, "强度": 0, "说明": ""}
    if len(peaks) < 2 or len(troughs) < 2:
        result["说明"] = "极值点不足"; return result
    top_diff = abs(peaks.iloc[-1] - peaks.iloc[-2]) / peaks.iloc[-2]
    if top_diff < 0.04 and troughs.iloc[-1] > troughs.iloc[-2]:
        triangle_h = peaks.iloc[-1] - troughs.iloc[-2]
        target = peaks.iloc[-1] + triangle_h
        result.update({
            "是否成立": True, "压力位": round(peaks.iloc[-1], 2),
            "目标位": round(target, 2), "强度": 75,
            "说明": "顶平底抬，突破后目标位参考"
        })
    else:
        result["说明"] = "不满足顶平底抬"
    return result


def detect_descending_triangle(peaks, troughs) -> dict:
    result = {"形态": "下降三角形", "类型": "看跌持续", "是否成立": False, "强度": 0, "说明": ""}
    if len(peaks) < 2 or len(troughs) < 2:
        result["说明"] = "极值点不足"; return result
    bottom_diff = abs(troughs.iloc[-1] - troughs.iloc[-2]) / troughs.iloc[-2]
    if bottom_diff < 0.04 and peaks.iloc[-1] < peaks.iloc[-2]:
        triangle_h = peaks.iloc[-2] - troughs.iloc[-1]
        target = troughs.iloc[-1] - triangle_h
        result.update({
            "是否成立": True, "支撑位": round(troughs.iloc[-1], 2),
            "目标位": round(target, 2), "强度": 75,
            "说明": "底平顶降，破位后目标位参考"
        })
    else:
        result["说明"] = "不满足底平顶降"
    return result


def detect_box_range(df: pd.DataFrame, window: int = 30) -> dict:
    result = {"形态": "箱体震荡", "类型": "整理形态", "是否成立": False, "强度": 0, "说明": ""}
    recent = df.iloc[-window:]
    high_max = recent["high"].max()
    low_min = recent["low"].min()
    range_pct = (high_max - low_min) / low_min
    if range_pct < 0.15:
        box_h = high_max - low_min
        result.update({
            "是否成立": True,
            "箱顶": round(high_max, 2), "箱底": round(low_min, 2),
            "向上突破目标位": round(high_max + box_h, 2),
            "向下跌破目标位": round(low_min - box_h, 2),
            "强度": int((1 - range_pct / 0.15) * 60 + 40),
            "说明": f"近{window}日振幅{range_pct*100:.1f}%"
        })
    else:
        result["说明"] = "振幅过大非箱体"
    return result


def detect_cup_handle(df, peaks, troughs) -> dict:
    result = {"形态": "杯柄形态", "类型": "看涨持续", "是否成立": False, "强度": 0, "说明": ""}
    if len(troughs) < 2 or len(peaks) < 3:
        result["说明"] = "极值点不足"; return result
    try:
        cup_l = peaks.iloc[-3]; cup_b = troughs.iloc[-2]
        cup_r = peaks.iloc[-2]; handle_l = troughs.iloc[-1]
        cup_peak_diff = abs(cup_l - cup_r) / cup_l
        cup_depth = (cup_l - cup_b) / cup_l
        handle_depth = (cup_r - handle_l) / cup_r
        if cup_peak_diff < 0.06 and 0.12 < cup_depth < 0.35 and handle_depth < cup_depth * 0.5:
            target = cup_l + (cup_l - cup_b)
            result.update({
                "是否成立": True, "突破位": round(cup_l, 2), "目标位": round(target, 2),
                "强度": int((1 - cup_peak_diff) * 30 + 70),
                "说明": f"杯身深度{cup_depth*100:.1f}%，柄部回撤{handle_depth*100:.1f}%"
            })
        else:
            result["说明"] = "不满足杯柄结构"
    except:
        result["说明"] = "结构不完整"
    return result


def detect_triple_bottom(peaks, troughs, latest_price) -> dict:
    result = {"形态": "三重底", "类型": "看涨反转", "是否成立": False, "强度": 0, "说明": ""}
    if len(troughs) < 3 or len(peaks) < 2:
        result["说明"] = "极值点不足"; return result
    b1, b2, b3 = troughs.iloc[-3], troughs.iloc[-2], troughs.iloc[-1]
    max_b = max(b1, b2, b3); min_b = min(b1, b2, b3)
    if (max_b - min_b) / min_b < 0.06 and len(peaks) >= 2:
        n1, n2 = peaks.iloc[-2], peaks.iloc[-1]
        neckline = max(n1, n2)
        if abs(n1 - n2) / n1 < 0.05 and latest_price > min_b:
            target = neckline + (neckline - min_b)
            result.update({
                "是否成立": True, "颈线位": round(neckline, 2), "目标位": round(target, 2),
                "强度": 80, "说明": f"三底价差{(max_b-min_b)/min_b*100:.1f}%"
            })
        else:
            result["说明"] = "颈线不平或未站上"
    else:
        result["说明"] = "底部不对称"
    return result


def detect_triple_top(peaks, troughs, latest_price) -> dict:
    result = {"形态": "三重顶", "类型": "看跌反转", "是否成立": False, "强度": 0, "说明": ""}
    if len(peaks) < 3 or len(troughs) < 2:
        result["说明"] = "极值点不足"; return result
    t1, t2, t3 = peaks.iloc[-3], peaks.iloc[-2], peaks.iloc[-1]
    max_t = max(t1, t2, t3); min_t = min(t1, t2, t3)
    if (max_t - min_t) / min_t < 0.06 and len(troughs) >= 2:
        n1, n2 = troughs.iloc[-2], troughs.iloc[-1]
        neckline = min(n1, n2)
        if abs(n1 - n2) / n1 < 0.05 and latest_price < max_t:
            target = neckline - (max_t - neckline)
            result.update({
                "是否成立": True, "颈线位": round(neckline, 2), "目标位": round(target, 2),
                "强度": 80, "说明": f"三顶价差{(max_t-min_t)/min_t*100:.1f}%"
            })
        else:
            result["说明"] = "颈线不平或未跌破"
    else:
        result["说明"] = "顶部不对称"
    return result


def detect_rounding_bottom(df: pd.DataFrame, period: int = 40) -> dict:
    result = {"形态": "圆弧底", "类型": "看涨反转", "是否成立": False, "强度": 0, "说明": ""}
    if len(df) < period:
        result["说明"] = "数据不足"; return result
    close = df.iloc[-period:]["close"]
    mid = period // 2
    x = np.arange(mid)
    k_left = np.polyfit(x, close.iloc[:mid], 1)[0]
    k_right = np.polyfit(x, close.iloc[mid:], 1)[0]
    range_pct = (close.max() - close.min()) / close.min()
    if k_left < 0 and k_right > 0 and range_pct < 0.2:
        result.update({
            "是否成立": True, "强度": int((1 - range_pct / 0.2) * 50 + 50),
            "底部区间": f"{close.min():.2f} - {close.max():.2f}",
            "说明": f"近{period}日弧形底，振幅{range_pct*100:.1f}%"
        })
    else:
        result["说明"] = "未形成弧形底"
    return result


def detect_rounding_top(df: pd.DataFrame, period: int = 40) -> dict:
    result = {"形态": "圆弧顶", "类型": "看跌反转", "是否成立": False, "强度": 0, "说明": ""}
    if len(df) < period:
        result["说明"] = "数据不足"; return result
    close = df.iloc[-period:]["close"]
    mid = period // 2
    x = np.arange(mid)
    k_left = np.polyfit(x, close.iloc[:mid], 1)[0]
    k_right = np.polyfit(x, close.iloc[mid:], 1)[0]
    range_pct = (close.max() - close.min()) / close.min()
    if k_left > 0 and k_right < 0 and range_pct < 0.2:
        result.update({
            "是否成立": True, "强度": int((1 - range_pct / 0.2) * 50 + 50),
            "顶部区间": f"{close.min():.2f} - {close.max():.2f}",
            "说明": f"近{period}日弧形顶，振幅{range_pct*100:.1f}%"
        })
    else:
        result["说明"] = "未形成弧形顶"
    return result


def detect_broadening_top(peaks, troughs) -> dict:
    result = {"形态": "喇叭形(扩散顶)", "类型": "看跌反转", "是否成立": False, "强度": 0, "说明": ""}
    if len(peaks) < 3 or len(troughs) < 3:
        result["说明"] = "极值点不足"; return result
    if (peaks.iloc[-3] < peaks.iloc[-2] < peaks.iloc[-1]
            and troughs.iloc[-3] > troughs.iloc[-2] > troughs.iloc[-1]):
        result.update({"是否成立": True, "强度": 65, "说明": "高低点同步扩散，多空分歧加剧"})
    else:
        result["说明"] = "未形成扩散结构"
    return result


def detect_diamond_pattern(peaks, troughs) -> dict:
    result = {"形态": "钻石形态(菱形)", "类型": "顶部反转", "是否成立": False, "强度": 0, "说明": ""}
    if len(peaks) < 4 or len(troughs) < 4:
        result["说明"] = "极值点不足"; return result
    expand = (peaks.iloc[-4] < peaks.iloc[-3] and troughs.iloc[-4] > troughs.iloc[-3])
    contract = (peaks.iloc[-3] > peaks.iloc[-2] and troughs.iloc[-3] < troughs.iloc[-2])
    if expand and contract:
        result.update({"是否成立": True, "强度": 65, "说明": "先扩后收形成菱形，预示趋势反转"})
    else:
        result["说明"] = "未形成菱形结构"
    return result


def detect_bull_flag(df, peaks, troughs) -> dict:
    result = {"形态": "上升旗形", "类型": "看涨持续", "是否成立": False, "强度": 0, "说明": ""}
    if len(peaks) < 2 or len(troughs) < 2:
        result["说明"] = "极值点不足"; return result
    latest = df.iloc[-1]
    if not (latest["close"] > latest["ma60"] and latest["ma20"] > latest["ma60"]):
        result["说明"] = "非上升趋势"; return result
    if peaks.iloc[-2] > peaks.iloc[-1] and troughs.iloc[-2] > troughs.iloc[-1]:
        target = peaks.iloc[-2] + peaks.iloc[-2] * 0.1
        result.update({
            "是否成立": True, "突破位": round(peaks.iloc[-2], 2),
            "目标位": round(target, 2), "强度": 70,
            "说明": "上升中向下倾斜整理通道，看涨蓄力"
        })
    else:
        result["说明"] = "不满足旗形结构"
    return result


def detect_bear_flag(df, peaks, troughs) -> dict:
    result = {"形态": "下降旗形", "类型": "看跌持续", "是否成立": False, "强度": 0, "说明": ""}
    if len(peaks) < 2 or len(troughs) < 2:
        result["说明"] = "极值点不足"; return result
    latest = df.iloc[-1]
    if not (latest["close"] < latest["ma60"] and latest["ma20"] < latest["ma60"]):
        result["说明"] = "非下降趋势"; return result
    if peaks.iloc[-2] < peaks.iloc[-1] and troughs.iloc[-2] < troughs.iloc[-1]:
        target = troughs.iloc[-2] - troughs.iloc[-2] * 0.1
        result.update({
            "是否成立": True, "破位位": round(troughs.iloc[-2], 2),
            "目标位": round(target, 2), "强度": 70,
            "说明": "下降中向上倾斜整理通道，看跌中继"
        })
    else:
        result["说明"] = "不满足旗形结构"
    return result


def detect_falling_wedge(df, peaks, troughs) -> dict:
    result = {"形态": "下降楔形", "类型": "看涨持续", "是否成立": False, "强度": 0, "说明": ""}
    if len(peaks) < 3 or len(troughs) < 3:
        result["说明"] = "极值点不足"; return result
    latest = df.iloc[-1]
    if not latest["close"] > latest["ma60"]:
        result["说明"] = "非上升趋势"; return result
    peak_down = peaks.iloc[-3] > peaks.iloc[-2] > peaks.iloc[-1]
    trough_down = troughs.iloc[-3] > troughs.iloc[-2] > troughs.iloc[-1]
    r1 = peaks.iloc[-3] - troughs.iloc[-3]
    r2 = peaks.iloc[-1] - troughs.iloc[-1]
    if peak_down and trough_down and r2 < r1 * 0.7:
        target = peaks.iloc[-2] + r1
        result.update({
            "是否成立": True, "突破位": round(peaks.iloc[-2], 2),
            "目标位": round(target, 2), "强度": 65,
            "说明": "向下收敛楔形整理，向上突破概率高"
        })
    else:
        result["说明"] = "不满足楔形结构"
    return result


def detect_rising_wedge(df, peaks, troughs) -> dict:
    result = {"形态": "上升楔形", "类型": "看跌持续", "是否成立": False, "强度": 0, "说明": ""}
    if len(peaks) < 3 or len(troughs) < 3:
        result["说明"] = "极值点不足"; return result
    latest = df.iloc[-1]
    if not latest["close"] < latest["ma60"]:
        result["说明"] = "非下降趋势"; return result
    peak_up = peaks.iloc[-3] < peaks.iloc[-2] < peaks.iloc[-1]
    trough_up = troughs.iloc[-3] < troughs.iloc[-2] < troughs.iloc[-1]
    r1 = peaks.iloc[-3] - troughs.iloc[-3]
    r2 = peaks.iloc[-1] - troughs.iloc[-1]
    if peak_up and trough_up and r2 < r1 * 0.7:
        target = troughs.iloc[-2] - r1
        result.update({
            "是否成立": True, "破位位": round(troughs.iloc[-2], 2),
            "目标位": round(target, 2), "强度": 65,
            "说明": "向上收敛楔形整理，向下破位概率高"
        })
    else:
        result["说明"] = "不满足楔形结构"
    return result


def analyze_all_patterns(df: pd.DataFrame) -> dict:
    peaks, troughs = find_peaks_troughs(df)
    latest_price = df.iloc[-1]["close"]
    pattern_list = [
        detect_vcp(df), detect_double_bottom(peaks, troughs, latest_price),
        detect_head_shoulders_bottom(peaks, troughs),
        detect_ascending_triangle(peaks, troughs),
        detect_cup_handle(df, peaks, troughs),
        detect_triple_bottom(peaks, troughs, latest_price),
        detect_rounding_bottom(df), detect_bull_flag(df, peaks, troughs),
        detect_falling_wedge(df, peaks, troughs),
        detect_double_top(peaks, troughs, latest_price),
        detect_head_shoulders_top(peaks, troughs),
        detect_descending_triangle(peaks, troughs),
        detect_triple_top(peaks, troughs, latest_price),
        detect_rounding_top(df), detect_broadening_top(peaks, troughs),
        detect_diamond_pattern(peaks, troughs),
        detect_bear_flag(df, peaks, troughs), detect_rising_wedge(df, peaks, troughs),
        detect_box_range(df),
    ]
    bullish, bearish, neutral = [], [], []
    for p in pattern_list:
        if not p.get("是否成立", False): continue
        if p["类型"] in ["看涨反转", "看涨持续"]: bullish.append(p)
        elif p["类型"] in ["看跌反转", "看跌持续", "顶部反转"]: bearish.append(p)
        else: neutral.append(p)
    return {
        "看涨形态": bullish, "看跌形态": bearish, "整理形态": neutral,
        "合计有效形态": len(bullish) + len(bearish) + len(neutral)
    }


def analyze_volume_price(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    result = {}
    vol_ratio_5 = latest["volume"] / latest["vol_ma5"]
    vol_ratio_10 = latest["volume"] / latest["vol_ma10"]
    if vol_ratio_5 >= 2.0:      vol_level = "大幅放量"
    elif vol_ratio_5 >= 1.3:    vol_level = "温和放量"
    elif vol_ratio_5 <= 0.5:    vol_level = "极度缩量"
    elif vol_ratio_5 <= 0.7:    vol_level = "明显缩量"
    else:                       vol_level = "量能正常"
    result["基础量能"] = {
        "量能状态": vol_level,
        "相对5日均量": f"{vol_ratio_5:.2f}倍",
        "相对10日均量": f"{vol_ratio_10:.2f}倍",
        "当日成交量": f"{latest['volume']/10000:.0f}万手"
    }
    recent = df.iloc[-20:]
    price_trend = "上升" if recent["close"].iloc[-1] > recent["close"].iloc[0] else "下降"
    vol_trend = "递增" if recent["volume"].iloc[-5:].mean() > recent["volume"].iloc[:5].mean() else "递减"
    if price_trend == "上升" and vol_trend == "递增":
        cooperation = "量价同步向上，上涨动能充足"; coop_score = 90
    elif price_trend == "上升" and vol_trend == "递减":
        cooperation = "价涨量缩，上涨动能衰减"; coop_score = 40
    elif price_trend == "下降" and vol_trend == "递增":
        cooperation = "价跌量增，抛压较重"; coop_score = 30
    else:
        cooperation = "价跌量缩，抛压逐步衰竭"; coop_score = 60
    result["趋势配合度"] = {
        "价格趋势": price_trend, "量能趋势": vol_trend,
        "配合结论": cooperation, "配合度评分": coop_score
    }
    window = 30
    divergence = {"顶背离": False, "底背离": False, "说明": "无量价背离"}
    if len(df) >= window:
        period = df.iloc[-window:]
        price_new_high = latest["close"] >= period["close"].max() * 0.98
        vol_new_high = latest["volume"] >= period["volume"].max() * 0.9
        price_new_low = latest["close"] <= period["close"].min() * 1.02
        if price_new_high and not vol_new_high:
            divergence["顶背离"] = True
            divergence["说明"] = "股价接近阶段新高但量能未同步放大，上涨动力不足"
        if price_new_low and latest["volume"] <= period["volume"].min() * 1.1:
            divergence["底背离"] = True
            divergence["说明"] = "股价接近阶段新低但量能未同步放大，抛压逐步衰竭"
    result["量价背离"] = divergence
    patterns = []
    if latest["volume"] >= df.iloc[-60:]["volume"].max() * 0.95:
        patterns.append("天量成交（近60日峰值）")
    if latest["volume"] <= df.iloc[-60:]["volume"].min() * 1.05:
        patterns.append("地量成交（近60日地量）")
    result["量能结构"] = patterns if patterns else ["无特殊标志性量能结构"]
    risks = []
    if vol_ratio_5 > 1.5 and abs(latest["pct_chg"]) < 1.0:
        risks.append("放量滞涨：多空分歧剧烈")
    if latest["pct_chg"] > 1.0 and vol_ratio_5 < 0.8:
        risks.append("缩量上涨：缺乏量能支撑")
    result["风险信号"] = risks if risks else ["无明显量价风险信号"]
    score = coop_score
    if divergence["顶背离"]: score -= 20
    if divergence["底背离"]: score += 15
    if risks: score -= len(risks) * 10
    score = min(max(score, 0), 100)
    if score >= 80:      level = "健康"
    elif score >= 60:    level = "基本健康"
    elif score >= 40:    level = "中性偏弱"
    else:                level = "异常/危险"
    result["综合量价健康度"] = {"评分": f"{score}/100", "等级": level}
    return result


def analyze_technical(code: str) -> dict:
    """一站式技术分析入口，返回 JSON 友好的结果"""
    raw_df = get_stock_data(code, 365)
    df = calc_technical_indicators(raw_df)
    latest = df.iloc[-1]
    trend = analyze_trend(df)
    signals = detect_signals(df)
    patterns = analyze_all_patterns(df)
    vol_price = analyze_volume_price(df)
    vp = get_volume_profile(df)
    buy_rr = calculate_risk_reward(df, "buy")
    sell_rr = calculate_risk_reward(df, "sell")

    return {
        "code": code,
        "date": latest.name.strftime("%Y-%m-%d") if hasattr(latest.name, "strftime") else str(latest.name),
        "latest": {
            "close": round(float(latest["close"]), 2),
            "pct_chg": round(float(latest["pct_chg"]), 2),
            "volume_hands": f"{float(latest['volume'])/10000:.0f}万手",
            "turnover": round(float(latest["turnover"]), 2) if "turnover" in df.columns else None,
        },
        "ma": {
            "ma5": round(float(latest["ma5"]), 2),
            "ma10": round(float(latest["ma10"]), 2),
            "ma20": round(float(latest["ma20"]), 2),
            "ma60": round(float(latest["ma60"]), 2),
            "ma120": round(float(latest["ma120"]), 2),
            "ma250": round(float(latest["ma250"]), 2),
        },
        "macd": {
            "dif": round(float(latest["dif"]), 3),
            "dea": round(float(latest["dea"]), 3),
            "macd_bar": round(float(latest["macd_bar"]), 3),
        },
        "rsi": round(float(latest["rsi"]), 1),
        "atr": round(float(latest["atr"]), 2),
        "bollinger": {
            "upper": round(float(latest["boll_upper"]), 2),
            "mid": round(float(latest["boll_mid"]), 2),
            "lower": round(float(latest["boll_lower"]), 2),
        },
        "trend": trend,
        "signals": signals,
        "patterns": patterns,
        "volume_price": vol_price,
        "volume_profile": vp,
        "risk_reward": {"buy": buy_rr, "sell": sell_rr},
    }
