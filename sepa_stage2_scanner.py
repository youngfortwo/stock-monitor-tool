#!/usr/bin/env python3
"""Scan A-share stocks that match SEPA Stage 2 uptrend criteria."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from financial_filter import check_financial_acceleration, load_cache, save_cache


def get_board(code: str) -> str:
    """Map A-share stock code to board/exchange name."""
    c = str(code).zfill(6)
    prefix = c[:3]
    first = c[0]
    if first == "6":
        if prefix == "688":
            return "科创板"
        return "上海主板"
    if first in ("0", "1", "3"):
        if prefix in ("300", "301"):
            return "创业板"
        if prefix in ("000", "001", "002", "003"):
            return "深圳主板"
        return "深圳主板"
    if first == "8":
        return "北交所"
    if first == "4":
        return "新三板"
    return "其他"

# Disable proxies to avoid issues with AkShare
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan A-share stocks by SEPA Stage  trend template."
    )
    parser.add_argument(
        "--output", default="sepa_stage2_candidates.csv", help="Output CSV path."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Only scan first N stocks for testing."
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip first N stocks before scanning."
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Reserved; AkShare daily data is scanned sequentially."
    )
    parser.add_argument(
        "--include-bj", action="store_true", help="Include Beijing Stock Exchange stocks."
    )
    parser.add_argument(
        "--min-history-days", type=int, default=220, help="Minimum usable trading days."
    )
    parser.add_argument(
        "--sleep-seconds", type=float, default=0.05, help="Sleep before each request."
    )
    return parser.parse_args()


def date_yyyymmdd(days_back: int) -> str:
    return (dt.date.today() - dt.timedelta(days=days_back)).strftime("%Y%m%d")


def today_yyyymmdd() -> str:
    return dt.date.today().strftime("%Y%m%d")


def get_stock_pool(include_bj: bool, limit: int, offset: int) -> pd.DataFrame:
    pool = ak.stock_info_a_code_name().rename(columns={"code": "代码", "name": "名称"})
    pool = pool[["代码", "名称"]].copy()
    pool["代码"] = pool["代码"].astype(str).str.zfill(6)
    pool = pool[~pool["名称"].str.contains("ST|退", regex=True, na=False)]

    if not include_bj:
        pool = pool[~pool["代码"].str.startswith(("8", "4"))]
    if offset > 0:
        pool = pool.iloc[offset:]
    if limit > 0:
        pool = pool.head(limit)

    return pool.reset_index(drop=True)


def market_symbol(symbol: str) -> str:
    return f"sh{symbol}" if symbol.startswith(("6", "9")) else f"sz{symbol}"


def fetch_history(
    symbol: str, min_history_days: int, sleep_seconds: float,
    use_cache: bool = True,
) -> pd.DataFrame:
    from daily_cache import load_cached, save_cached

    if use_cache:
        cached = load_cached(symbol, min_days=min_history_days)
        if cached is not None and len(cached) >= min_history_days:
            return cached

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    history = ak.stock_zh_a_daily(
        symbol=market_symbol(symbol),
        start_date=date_yyyymmdd(520),
        end_date=today_yyyymmdd(),
        adjust="qfq",
    )

    if history.empty or len(history) < min_history_days:
        return pd.DataFrame()

    numeric_columns = ["open", "high", "low", "close", "volume", "amount", "turnover"]
    for column in numeric_columns:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")

    result = history.dropna(subset=["close", "volume", "amount"]).reset_index(drop=True)

    if use_cache and not result.empty and len(result) >= min_history_days:
        save_cached(symbol, result)

    return result


def get_industry_map() -> dict[str, str]:
    """Best-effort industry mapping; uses cached data, Eastmoney, or exchange fallbacks."""
    cache_path = Path("industry_map_cache.json")
    now = dt.datetime.now()

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_date = cached.get("_date", "")
            if cached_date == now.strftime("%Y-%m-%d"):
                return cached.get("map", {})
        except Exception:
            pass

    industry_map = load_industry_overrides()

    em_ok = False
    for attempt in range(3):
        try:
            industries = ak.stock_board_industry_name_em()
            em_ok = True
            names = (
                industries["板块名称"]
                if "板块名称" in industries.columns
                else industries.iloc[:, 1]
            )
            for industry in names.dropna().astype(str).tolist():
                try:
                    cons = ak.stock_board_industry_cons_em(symbol=industry)
                    code_column = "代码" if "代码" in cons.columns else "股票代码"
                    for code in cons[code_column].astype(str).str.zfill(6):
                        industry_map.setdefault(code, industry)
                    time.sleep(0.02)
                except Exception:
                    pass
            break
        except Exception:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                print(f"WARN Eastmoney industry retry {attempt + 2}/3...", file=sys.stderr)

    if not em_ok:
        print("WARN Eastmoney industry unavailable, falling back to SZ exchange listing.", file=sys.stderr)
        try:
            sz = ak.stock_info_sz_name_code()
            sz["A股代码"] = sz["A股代码"].astype(str).str.zfill(6)
            for _, row in sz.iterrows():
                code = str(row["A股代码"]).zfill(6)
                raw_industry = str(row.get("所属行业", ""))
                name = raw_industry.strip()
                if name and name != "nan":
                    industry_map.setdefault(code, name)
        except Exception as exc:
            print(f"WARN SZ fallback failed: {exc}", file=sys.stderr)

    try:
        cache_path.write_text(
            json.dumps({"_date": now.strftime("%Y-%m-%d"), "map": industry_map}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass

    return industry_map


def next_report_date() -> dict[str, str]:
    """Return next reporting period name and deadline date."""
    today = dt.date.today()
    deadlines = [
        ("一季报", dt.date(today.year, 4, 30)),
        ("中报", dt.date(today.year, 8, 31)),
        ("三季报", dt.date(today.year, 10, 31)),
        ("年报", dt.date(today.year + 1, 4, 30)),
    ]
    for period, d in deadlines:
        if d >= today:
            return {"period": period, "deadline": d.strftime("%Y-%m-%d")}
    return {"period": "年报", "deadline": deadlines[-1][1].strftime("%Y-%m-%d")}


def build_market_cap_cache() -> dict[str, float]:
    """Pre-build market cap cache using exchange data. Returns {code: market_cap_cny}.
    
    优化4：缓存到本地文件，避免每次启动都重新拉取。
    """
    import json
    cache_file = "market_cap_cache.json"
    
    # 尝试从本地文件加载缓存
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                caps = json.load(f)
            print(f"Loaded market cap cache from {cache_file}: {len(caps)} stocks")
            return caps
    except Exception as e:
        print(f"Failed to load market cap cache: {e}")
    
    caps: dict[str, float] = {}
    print("Building market cap cache...")
    try:
        sz = ak.stock_info_sz_name_code()
        sz["A股代码"] = sz["A股代码"].astype(str).str.zfill(6)
        for _, row in sz.iterrows():
            code = str(row["A股代码"]).zfill(6)
            raw = str(row.get("A股总股本", ""))
            try:
                shares = float(raw.replace(",", ""))
                caps[code] = shares
            except (ValueError, TypeError):
                pass
        print(f"  SZ shares: {len(caps)} stocks")
    except Exception as exc:
        print(f"  SZ failed: {exc}")

    try:
        sh = ak.stock_info_sh_name_code()
        for _, row in sh.iterrows():
            code = str(row["证券代码"]).zfill(6)
            for attempt in range(2):
                try:
                    info = ak.stock_individual_info_em(symbol=code)
                    row_total = info[info["item"] == "总股本"]
                    if not row_total.empty:
                        val = str(row_total["value"].values[0])
                        shares = float(val.replace(",", "").replace("亿", "e8").replace("万", "e4"))
                        caps[code] = shares
                    break
                except Exception:
                    if attempt < 1:
                        time.sleep(1.5)
            time.sleep(0.08)
        print(f"  SH added, total: {len(caps)}")
    except Exception as exc:
        print(f"  SH failed: {exc}")

    # 保存到本地文件
    try:
        with open(cache_file, "w") as f:
            json.dump(caps, f, indent=2)
        print(f"Saved market cap cache to {cache_file}: {len(caps)} stocks")
    except Exception as e:
        print(f"Failed to save market cap cache: {e}")

    return caps


def get_market_cap(code: str, close_price: float, caps: dict[str, float]) -> float:
    """Compute market cap: close × total_shares. Returns 0 if unknown."""
    shares = caps.get(code, 0)
    if shares and close_price:
        return round(shares * close_price, 2)
    return 0


def calculate_pe_ttm(code: str, close_price: float, financial_cache: dict | None = None) -> float | None:
    """计算滚动市盈率 (PE TTM) = 当前股价 / 最近4个季度摊薄每股收益之和。

    从财务缓存中读取每股收益，不需要额外 API 调用。
    返回 None 表示数据不足无法计算。
    """
    if not financial_cache or not close_price or close_price <= 0:
        return None
    raw = financial_cache.get(code)
    if not raw or not isinstance(raw, list) or len(raw) < 3:
        return None
    try:
        import pandas as pd
        df = pd.DataFrame(raw)
        if "摊薄每股收益(元)" not in df.columns:
            return None
        eps_series = pd.to_numeric(df["摊薄每股收益(元)"], errors="coerce").dropna()
        if len(eps_series) < 1:
            return None
        # 中国财报披露的是累计EPS（Q1累Q1，H1累Q1+Q2，9M累Q1-3，全年累全年）
        # 取最近3期的最大值作为年均 EPS 估计
        latest_eps = float(eps_series.tail(3).max())
        if latest_eps <= 0:
            return None
        return round(close_price / latest_eps, 2)
    except Exception:
        return None


def load_industry_overrides(path: str = "industry_overrides.csv") -> dict[str, str]:
    override_path = Path(path)
    if not override_path.exists():
        return {}

    overrides = pd.read_csv(override_path, dtype={"code": str})
    overrides["code"] = overrides["code"].str.zfill(6)
    return dict(zip(overrides["code"], overrides["industry"]))

def detect_cup_handle(data: pd.DataFrame) -> tuple[bool, dict]:
    """
    【重构版】欧奈尔+SEPA标准 杯柄形态检测
    修复核心问题：
    1. 动态识别U型杯身（左沿→杯底→右沿），不再固定切片
    2. 动态定位柄部（从右杯沿后开始），不再强制最后20天
    3. 新增价格突破判定，有量有价才确认形态完成
    4. 新增U型合理性校验，过滤V型尖底
    标准：
    1. 前置：上升趋势（MA200上行 + 股价站上MA200）
    2. U型杯身：左沿→杯底→右沿，深度20-35%，周期约3-5个月
    3. 柄部：右沿后1-4周窄幅回调，深度8-15%，位于杯身上半部
    4. 确认：放量突破柄部上沿（量≥1.5倍20日均量 + 价破柄高）
    """
    if len(data) < 120:
        return False, {"error": "数据长度不足（需≥120日）"}

    # 复用外部已计算均线，避免重复滚动
    df = data.copy()
    if "ma200" not in df.columns:
        df["ma200"] = df["close"].rolling(200).mean()
    recent = df.tail(120).reset_index(drop=True)  # 重置索引，方便定位位置
    close_arr = recent["close"].values
    volume_arr = recent["volume"].values

    # ======================
    # 1. 上升趋势前置校验
    # ======================
    uptrend = (
        (close_arr[-1] > recent["ma200"].iloc[-1])
        and (recent["ma200"].iloc[-1] > recent["ma200"].iloc[-30])
    )
    if not uptrend:
        return False, {"reason": "非上升趋势，无标准杯柄"}

    # ======================
    # 2. 动态识别U型杯身（核心重构）
    # 逻辑：先找杯底 → 左边找左杯沿高点 → 右边找右杯沿高点
    # ======================
    # 预留最后5天作为柄部/突破空间，右杯沿不能太靠后
    search_end = len(recent) - 5
    if search_end < 40:
        return False, {"reason": "有效区间不足，无法构建杯身"}

    # 2.1 找杯底（区间内最低点）
    cup_bottom_idx = int(np.argmin(close_arr[:search_end]))
    cup_bottom_price = float(close_arr[cup_bottom_idx])
    # 杯底不能太靠左/太靠右，保证左右都有空间
    if cup_bottom_idx < 10 or cup_bottom_idx > search_end - 15:
        return False, {"reason": "杯底位置异常，非有效U型"}

    # 2.2 找左杯沿（杯底左边的最高点）
    left_rim_idx = int(np.argmax(close_arr[:cup_bottom_idx]))
    left_rim_price = float(close_arr[left_rim_idx])
    # 左杯沿不能太靠近杯底
    if cup_bottom_idx - left_rim_idx < 10:
        return False, {"reason": "左杯沿到杯底距离过短，非有效U型"}

    # 2.3 找右杯沿（杯底到搜索终点之间的最高点）
    right_rim_idx = cup_bottom_idx + int(np.argmax(close_arr[cup_bottom_idx:search_end]))
    right_rim_price = float(close_arr[right_rim_idx])
    # 右杯沿不能太靠近杯底
    if right_rim_idx - cup_bottom_idx < 10:
        return False, {"reason": "杯底到右杯沿距离过短，非有效U型"}

    # 2.4 杯身核心指标校验
    cup_depth = (1 - cup_bottom_price / left_rim_price) * 100 if left_rim_price > 0 else 0
    valid_cup_depth = 20 <= cup_depth <= 35  # 深度符合标准
    near_rim = right_rim_price >= left_rim_price * 0.9  # 右沿接近左沿高度

    # 2.5 U型合理性校验（过滤V型尖底）
    down_days = cup_bottom_idx - left_rim_idx  # 下跌天数
    up_days = right_rim_idx - cup_bottom_idx  # 回升天数
    u_shape_reasonable = 0.3 <= down_days / up_days <= 3  # 下跌/回升时长比例合理

    # 杯身整体是否合格
    valid_cup = valid_cup_depth and near_rim and u_shape_reasonable

    # ======================
    # 3. 动态定位柄部（从右杯沿之后开始）
    # ======================
    handle_start_idx = right_rim_idx + 1
    handle_end_idx = len(recent) - 1
    handle_days = handle_end_idx - handle_start_idx + 1

    # 柄部周期校验：1-4周（5-20个交易日）
    if not (5 <= handle_days <= 20):
        return False, {"reason": f"柄部周期异常({handle_days}天)，标准1-4周"}

    handle_prices = close_arr[handle_start_idx:handle_end_idx+1]
    handle_high = float(np.max(handle_prices))
    handle_low = float(np.min(handle_prices))
    handle_depth = (1 - handle_low / handle_high) * 100 if handle_high > 0 else 0

    cup_mid_price = (left_rim_price + cup_bottom_price) / 2  # 杯身中线
    valid_handle = (
        8 <= handle_depth <= 15          # 柄部深度符合标准
        and handle_low > cup_mid_price   # 柄部位于杯身上半部
        and handle_high < left_rim_price # 柄部高点不超过杯口
        and handle_high <= right_rim_price * 1.02  # 柄部高点不明显超右沿
    )

    # ======================
    # 4. 量能 + 价格突破校验
    # ======================
    # 4.1 柄部缩量：柄部均量 < 杯身均量 * 0.75
    cup_vol = float(np.mean(volume_arr[left_rim_idx:right_rim_idx+1]))
    handle_vol = float(np.mean(volume_arr[handle_start_idx:handle_end_idx+1]))
    volume_contract = handle_vol < cup_vol * 0.75

    # 4.2 20日均量（支持外部复用）
    if "vol_avg" in df.columns:
        vol_avg_20 = float(df["vol_avg"].iloc[-1])
    else:
        vol_avg_20 = float(df["volume"].rolling(20).mean().iloc[-1])

    # 4.3 突破确认：价破柄高 + 量超1.5倍
    current_close = float(close_arr[-1])
    current_vol = float(volume_arr[-1])
    price_breakout = current_close > handle_high
    volume_breakout = current_vol > vol_avg_20 * 1.5
    breakout_valid = price_breakout and volume_breakout

    # ======================
    # 5. 最终形态判定
    # ======================
    pattern = valid_cup and valid_handle and volume_contract and breakout_valid

    return pattern, {
        "cup_valid": valid_cup,
        "cup_high": round(left_rim_price, 2),       # 左杯沿价格（杯口）
        "cup_low": round(cup_bottom_price, 2),      # 杯底价格
        "cup_depth_pct": round(cup_depth, 2),       # 杯身深度
        "cup_days": right_rim_idx - left_rim_idx,   # 杯身构建天数
        "right_rim_price": round(right_rim_price, 2), # 右杯沿价格
        "handle_valid": valid_handle,
        "handle_high": round(handle_high, 2),
        "handle_low": round(handle_low, 2),
        "handle_depth_pct": round(handle_depth, 2),
        "handle_days": handle_days,
        "volume_contracting": volume_contract,
        "price_breakout": price_breakout,
        "volume_breakout": volume_breakout,
        "near_rim": near_rim,
    }


def detect_vcp(data: pd.DataFrame) -> tuple[bool, dict]:
    """马克·米勒维尼 VCP 波动收缩形态检测（优化版）。

    核心标准：
    1. 前置：上升趋势（MA200上行 + 股价站上MA200 + MA50>MA200）
    2. 2-4轮收缩：每轮振幅逐级收窄（波段峰谷识别），首轮≥12%
    3. 量能同步萎缩：每轮成交量逐级递减
    4. 基底重心平稳：后段高点不低于前段92%
    5. 末端极致缩量+窄幅震荡（作弊区）
    6. 区分「构筑中」和「已突破」两种状态
    """
    if len(data) < 120:
        return False, {"error": "数据长度不足（需≥120日）"}

    needs_copy = "ma200" not in data.columns or "ma50" not in data.columns
    df = data.copy() if needs_copy else data
    if "ma200" not in df.columns:
        df["ma200"] = df["close"].rolling(200).mean()
    if "ma50" not in df.columns:
        df["ma50"] = df["close"].rolling(50).mean()

    uptrend = (
        (df["close"].iloc[-1] > df["ma200"].iloc[-1])
        and (df["ma200"].iloc[-1] > df["ma200"].iloc[-30])
        and (df["ma50"].iloc[-1] > df["ma200"].iloc[-1])
    )
    if not uptrend:
        return False, {"reason": "不满足SEPA上升趋势前提"}

    recent = df.tail(120).reset_index(drop=True)
    close = recent["close"].values
    volume = recent["volume"].values
    n = len(close)

    window = 10
    high_idx = argrelextrema(close, np.greater_equal, order=window)[0]
    low_idx = argrelextrema(close, np.less_equal, order=window)[0]

    pivots = []
    for idx in high_idx:
        pivots.append((idx, "high", float(close[idx])))
    for idx in low_idx:
        pivots.append((idx, "low", float(close[idx])))
    pivots.sort(key=lambda x: x[0])

    clean_pivots = []
    for p in pivots:
        if not clean_pivots:
            clean_pivots.append(p)
            continue
        last_type = clean_pivots[-1][1]
        if p[1] == last_type:
            if (last_type == "high" and p[2] > clean_pivots[-1][2]) or \
               (last_type == "low" and p[2] < clean_pivots[-1][2]):
                clean_pivots[-1] = p
        else:
            clean_pivots.append(p)

    corrections = []
    swing_highs = []
    for i in range(len(clean_pivots) - 1):
        idx1, t1, v1 = clean_pivots[i]
        idx2, t2, v2 = clean_pivots[i + 1]
        if t1 == "high" and t2 == "low" and v1 > v2:
            depth_pct = (1 - v2 / v1) * 100
            vol_avg = float(np.mean(volume[idx1:idx2+1]))
            corrections.append({
                "start_idx": idx1,
                "end_idx": idx2,
                "depth_pct": depth_pct,
                "vol_avg": vol_avg,
                "high_price": v1,
                "low_price": v2
            })
            swing_highs.append(v1)

    if len(corrections) < 2:
        return False, {"reason": "收缩次数不足（需≥2轮）"}
    if corrections[0]["depth_pct"] < 12:
        return False, {"reason": "首轮回调幅度过小，无洗盘意义"}

    depths = [c["depth_pct"] for c in corrections]
    vols = [c["vol_avg"] for c in corrections]

    tighten_count = sum(1 for i in range(1, len(depths)) if depths[i] <= depths[i-1] * 0.88)
    vcp_tighten = tighten_count >= len(depths) - 1

    vol_shrink_count = sum(1 for i in range(1, len(vols)) if vols[i] <= vols[i-1] * 0.88)
    vcp_vol_shrink = vol_shrink_count >= len(vols) - 1

    high_trend_ok = all(swing_highs[i] >= swing_highs[i-1] * 0.92 for i in range(1, len(swing_highs)))

    last_correction = corrections[-1]
    last_correction_range = last_correction["depth_pct"]
    very_tight = last_correction_range < 12

    last_correction_vol = last_correction["vol_avg"]
    base_vol = np.mean(volume[:corrections[0]["end_idx"]])
    vol_dry = last_correction_vol < base_vol * 0.7

    last_swing_high = swing_highs[-1]
    current_close = float(close[-1])
    current_vol = float(volume[-1])
    vol_avg_20 = float(np.mean(volume[-20:]))
    price_breakout = current_close > last_swing_high
    volume_breakout = current_vol > vol_avg_20 * 1.5
    breakout_valid = price_breakout and volume_breakout

    base_vcp = vcp_tighten and vcp_vol_shrink and high_trend_ok and very_tight and vol_dry
    pattern = base_vcp or breakout_valid

    return pattern, {
        "vcp_valid": pattern,
        "vcp_status": "已突破" if breakout_valid else "构筑中",
        "vcp_contractions": len(corrections),
        "vcp_tighten": vcp_tighten,
        "vcp_vol_shrink": vcp_vol_shrink,
        "high_trend_ok": high_trend_ok,
        "vcp_tight_range": round(last_correction_range, 2),
        "vcp_vol_dry_up": vol_dry,
        "vcp_breakout": breakout_valid,
        "vcp_ranges": [round(d, 2) for d in depths],
        "vcp_vols": [round(v, 2) for v in vols],
        "vcp_resistance": round(last_swing_high, 2),
    }


def calc_stage2_core_conditions(data: pd.DataFrame) -> tuple[int, dict]:
    """统一计算 SEPA Stage2 9项核心条件的满足数量与明细。

    返回：(满足个数, 条件明细字典)
    """
    if len(data) < 220:
        return 0, {}

    # 复用调用方已计算好的 MA，避免重复 copy + rolling
    needs_copy = "ma50" not in data.columns or "ma150" not in data.columns or "ma200" not in data.columns
    df = data.copy() if needs_copy else data
    if "ma50" not in df.columns:
        df["ma50"] = df["close"].rolling(50).mean()
    if "ma150" not in df.columns:
        df["ma150"] = df["close"].rolling(150).mean()
    if "ma200" not in df.columns:
        df["ma200"] = df["close"].rolling(200).mean()

    latest = df.iloc[-1]
    ma200_20d_ago = df.iloc[-21]["ma200"]
    high_52w = df.tail(252)["high"].max()
    low_52w = df.tail(252)["low"].min()

    close = float(latest["close"])
    ma50 = float(latest["ma50"])
    ma150 = float(latest["ma150"])
    ma200 = float(latest["ma200"])
    pct_above_low = (close / float(low_52w) - 1) * 100
    pct_below_high = (close / float(high_52w) - 1) * 100

    conditions = {
        "close_gt_ma50": bool(close > ma50),
        "close_gt_ma150": bool(close > ma150),
        "close_gt_ma200": bool(close > ma200),
        "ma50_gt_ma150": bool(ma50 > ma150),
        "ma50_gt_ma200": bool(ma50 > ma200),
        "ma150_gt_ma200": bool(ma150 > ma200),
        "ma200_rising_1m": bool(ma200 > float(ma200_20d_ago)),
        "above_52w_low_30pct": bool(pct_above_low >= 30),
        "within_25pct_52w_high": bool(pct_below_high >= -25),
    }
    count = sum(1 for v in conditions.values() if v)
    return count, conditions


def detect_transition(
    code: str,
    data: pd.DataFrame,
    financial_cache: dict | None = None,
    rps_120: float | None = None,
    core_count_override: int | None = None,
    has_vcp_override: bool | None = None,
    has_cup_override: bool | None = None,
    financial_ready_override: bool | None = None,
) -> tuple[bool, dict]:
    """识别 Stage1 → Stage2 过渡态（临门一脚状态）。

    两级判定：
    1. 基础过渡态：7~8个Stage2核心条件 + 至少1个技术面过渡信号
    2. 高质量过渡态：基础过渡态 + 业绩加速 + 120日RPS进入全市场前30%

    core_count_override / has_vcp_override / has_cup_override 用于调用方（evaluate_stage1）
    传入已计算好的值，避免重复计算。
    """
    if len(data) < 220:
        return False, {"error": "数据不足"}

    core_count = core_count_override if core_count_override is not None else calc_stage2_core_conditions(data)[0]
    if not (7 <= core_count <= 8):
        return False, {"stage2_core_count": core_count}

    # 复用调用方已计算好的 MA，避免重复 copy + rolling
    if "ma200" not in data.columns:
        df = data.copy()
        df["ma200"] = df["close"].rolling(200).mean()
    else:
        df = data
    ma200 = float(df.iloc[-1]["ma200"])
    ma200_20d_ago = float(df.iloc[-21]["ma200"])
    ma200_change_pct = (ma200 / ma200_20d_ago - 1) * 100 if ma200_20d_ago > 0 else 0

    ma_turning_up = 0 < ma200_change_pct <= 5

    recent_20d_vol = df.tail(20)["volume"].mean()
    prior_100d_vol = df.iloc[-120:-20]["volume"].mean()
    recent_5d_vol = df.tail(5)["volume"].mean()
    vol_turning_up = (
        recent_20d_vol < prior_100d_vol * 0.85
        and recent_5d_vol > recent_20d_vol * 1.2
    )

    has_vcp = has_vcp_override if has_vcp_override is not None else detect_vcp(data)[0]
    has_cup = has_cup_override if has_cup_override is not None else detect_cup_handle(data)[0]
    pattern_ready = has_vcp or has_cup

    # 优化2：使用传入的财务加速结果，避免重复计算
    if financial_ready_override is not None:
        financial_ready = financial_ready_override
    else:
        financial_ready = False
        if financial_cache is not None:
            try:
                fin_result = check_financial_acceleration(code, financial_cache)
                financial_ready = fin_result["financial_pass"]
            except Exception:
                financial_ready = False

    rps_ready = False
    if rps_120 is not None:
        rps_ready = rps_120 >= 70

    tech_signal_count = sum([ma_turning_up, vol_turning_up, pattern_ready])
    base_transition = tech_signal_count >= 1
    high_quality_transition = base_transition and financial_ready and rps_ready
    is_transition = base_transition

    return is_transition, {
        "stage2_core_count": core_count,
        "is_base_transition": base_transition,
        "is_high_quality_transition": high_quality_transition,
        "ma200_turning_up": ma_turning_up,
        "volume_turning_up": vol_turning_up,
        "pattern_ready": pattern_ready,
        "financial_ready": financial_ready,
        "rps_120_ready": rps_ready,
        "tech_signal_count": tech_signal_count,
    }


def calc_all_rps(stock_returns: dict[str, float]) -> dict[str, float]:
    """批量计算全市场个股RPS（相对强弱百分位）。

    输入：{code: 120日涨跌幅(%)}
    输出：{code: RPS百分位 0~100}
    """
    codes = list(stock_returns.keys())
    returns = np.array([stock_returns[c] for c in codes])
    ranks = np.argsort(np.argsort(returns))
    rps = (ranks / (len(returns) - 1)) * 100
    return {code: round(float(rps[i]), 2) for i, code in enumerate(codes)}


def detect_pullback(data: pd.DataFrame) -> tuple[bool, dict]:
    if len(data) < 60:
        return False, {}

    recent = data.tail(40).copy()
    close = recent["close"].values
    volume = recent["volume"].values
    ma10 = recent["close"].rolling(10).mean().values
    ma50 = recent["close"].rolling(50).mean().values

    current_close = close[-1]
    current_volume = volume[-1]
    vol_avg_20d = float(volume[-21:-1].mean())

    ma10_pullback = False
    for i in range(-15, -1):
        idx = i if i >= 0 else len(close) + i
        if idx < 60 and ma10[idx] > 0:
            if abs(close[idx] - ma10[idx]) / ma10[idx] < 0.03:
                if volume[idx] < vol_avg_20d * 0.8:
                    ma10_pullback = True
                    break

    ma50_pullback = False
    for i in range(-20, -1):
        idx = i if i >= 0 else len(close) + i
        if idx < 60 and ma50[idx] > 0 and not np.isnan(ma50[idx]):
            if abs(close[idx] - ma50[idx]) / ma50[idx] < 0.03:
                if volume[idx] < vol_avg_20d * 0.8:
                    ma50_pullback = True
                    break

    volume_expanding = current_volume > vol_avg_20d * 1.2
    price_above_ma10 = current_close > ma10[-1] if ma10[-1] > 0 else False

    pullback = (ma10_pullback or ma50_pullback) and volume_expanding and price_above_ma10

    return pullback, {
        "ma10_pullback": ma10_pullback,
        "ma50_pullback": ma50_pullback,
        "volume_expanding": volume_expanding,
        "price_above_ma10": bool(price_above_ma10),
    }


def evaluate_stage2(code: str, name: str, industry: str, history: pd.DataFrame,
                    financial_cache: dict | None = None,
                    market_caps: dict[str, float] | None = None) -> dict:
    if history.empty or len(history) < 220:
        return {
            "code": code,
            "name": name,
            "industry": industry,
            "is_stage2": False,
            "score": 0,
            "error": "Not enough history data to evaluate SEPA Stage 2.",
            "conditions": {},
        }

    data = history.copy()
    data["ma50"] = data["close"].rolling(50).mean()
    data["ma150"] = data["close"].rolling(150).mean()
    data["ma200"] = data["close"].rolling(200).mean()

    latest = data.iloc[-1]
    high_52w = data.tail(252)["high"].max()
    low_52w = data.tail(252)["low"].min()

    close = float(latest["close"])
    ma50 = float(latest["ma50"])
    ma150 = float(latest["ma150"])
    ma200 = float(latest["ma200"])
    pct_above_low = (close / float(low_52w) - 1) * 100
    pct_below_high = (close / float(high_52w) - 1) * 100
    return_20d = (close / float(data.iloc[-21]["close"]) - 1) * 100
    return_120d = (close / float(data.iloc[-121]["close"]) - 1) * 100 if len(data) >= 121 else 0

    cup_handle_pattern, cup_handle_details = detect_cup_handle(data)
    vcp_pattern, vcp_details = detect_vcp(data)
    pullback, pullback_details = detect_pullback(data)

    core_count, core_conditions = calc_stage2_core_conditions(data)
    conditions = {
        **core_conditions,
        "cup_handle_pattern": cup_handle_pattern,
        "vcp_pattern": vcp_pattern,
        "pullback_confirm": pullback,
    }

    score = 0
    score += 20 if conditions["close_gt_ma50"] else 0
    score += 20 if conditions["ma50_gt_ma150"] and conditions["ma150_gt_ma200"] else 0
    score += 20 if conditions["ma200_rising_1m"] else 0
    score += min(20, pct_above_low / 3)
    score += max(0, 20 + pct_below_high)
    score += 10 if cup_handle_pattern else 0
    score += 10 if vcp_pattern else 0
    score += 10 if pullback else 0

    fin_pass = False
    fin_score = 0
    fin_rev = []
    fin_profit = []
    fin_margin = []
    if financial_cache is not None:
        try:
            fin = check_financial_acceleration(code, financial_cache)
            fin_pass = fin["financial_pass"]
            fin_score = fin["financial_score"]
            fin_rev = fin["rev_growth"]
            fin_profit = fin["profit_growth"]
            fin_margin = fin["profit_margin"]
            score += fin_score
        except Exception:
            pass

    is_stage2 = core_count == 9

    # 优化9：根据实际满足的条件动态生成 reasons，而非固定文案
    reasons = []
    if conditions["close_gt_ma50"] and conditions["close_gt_ma150"] and conditions["close_gt_ma200"]:
        reasons.append("price above MA50/150/200")
    if conditions["ma50_gt_ma150"] and conditions["ma150_gt_ma200"]:
        reasons.append("MA50>MA150>MA200")
    if conditions["ma200_rising_1m"]:
        reasons.append("MA200 rising")
    if conditions["above_52w_low_30pct"] and conditions["within_25pct_52w_high"]:
        reasons.append("price strong vs 52w range")
    if cup_handle_pattern:
        reasons.append("杯柄形态突破")
    if vcp_pattern:
        reasons.append("VCP 波动收缩")
    if pullback:
        reasons.append("pullback to MA with volume confirmation")
    if fin_pass:
        reasons.append("acceleration: rev+profit up, margin improving")

    return {
        "code": code,
        "name": name,
        "industry": industry,
        "board": get_board(code),
        "is_stage2": is_stage2,
        "date": str(latest["date"]),
        "close": round(close, 2),
        "ma50": round(ma50, 2),
        "ma150": round(ma150, 2),
        "ma200": round(ma200, 2),
        "52w_high": round(float(high_52w), 2),
        "52w_low": round(float(low_52w), 2),
        "pct_above_52w_low": round(float(pct_above_low), 2),
        "pct_below_52w_high": round(float(pct_below_high), 2),
        "return_20d_pct": round(float(return_20d), 2),
        "return_120d_pct": round(float(return_120d), 2),
        "amount_cny": round(float(latest["amount"]), 2),
        "volume": round(float(latest["volume"]), 2),
        "score": round(float(score), 2),
        "conditions": conditions,
        "cup_handle_details": cup_handle_details,
        "vcp_details": vcp_details,
        "pullback_details": pullback_details,
        "financial_pass": fin_pass,
        "financial_score": fin_score,
        "market_cap_cny": get_market_cap(code, close, market_caps) if market_caps else 0,
        "pe_ttm": calculate_pe_ttm(code, close, financial_cache) if financial_cache else None,
        "next_report_period": next_report_date()["period"],
        "next_report_deadline": next_report_date()["deadline"],
        "rev_growth": json.dumps(fin_rev[-3:] if len(fin_rev) >= 3 else fin_rev, ensure_ascii=False) if fin_rev else "",
        "profit_growth": json.dumps(fin_profit[-3:] if len(fin_profit) >= 3 else fin_profit, ensure_ascii=False) if fin_profit else "",
        "profit_margin": json.dumps(fin_margin[-3:] if len(fin_margin) >= 3 else fin_margin, ensure_ascii=False) if fin_margin else "",
        "matched_reason": "SEPA Stage 2: " + "; ".join(reasons),
    }


def analyze_stage2(
    code: str, name: str, industry: str, history: pd.DataFrame,
    financial_cache: dict | None = None,
    market_caps: dict[str, float] | None = None,
) -> dict | None:
    result = evaluate_stage2(code, name, industry, history,
                             financial_cache=financial_cache, market_caps=market_caps)
    if not result.get("is_stage2"):
        return None
    return result


def scan_one(
    row: pd.Series, industry_map: dict[str, str], args: argparse.Namespace,
    financial_cache: dict | None = None,
    market_caps: dict[str, float] | None = None,
) -> dict | None:
    code = row["代码"]
    name = row["名称"]
    history = fetch_history(code, args.min_history_days, args.sleep_seconds)
    return analyze_stage2(code, name, industry_map.get(code, "Unknown"), history,
                          financial_cache=financial_cache, market_caps=market_caps)


def main() -> int:
    args = parse_args()
    if args.workers != 1:
        print("WARN --workers is currently forced to 1 because AkShare daily data is not thread-safe.")

    pool = get_stock_pool(args.include_bj, args.limit, args.offset)
    industry_map = get_industry_map()
    financial_cache = load_cache()
    # market_caps 仅用于展示市值，批量扫描时不构建以节省时间
    market_caps = None
    print(
        f"Scanning {len(pool)} stocks for SEPA Stage 2, "
        f"offset={args.offset}, limit={args.limit}, industry mappings={len(industry_map)}..."
    )

    matches: list[dict] = []
    for index, row in pool.iterrows():
        try:
            result = scan_one(row, industry_map, args, financial_cache=financial_cache, market_caps=market_caps)
            if result:
                matches.append(result)
                print(f"MATCH {result['code']} {result['name']} {result['industry']} score={result['score']}")
        except Exception as exc:
            print(f"WARN failed {row['代码']} {row['名称']}: {exc}")

        if (index + 1) % 100 == 0:
            print(f"Progress: {index + 1}/{len(pool)}, matches={len(matches)}")

    save_cache(financial_cache)
    result_df = pd.DataFrame(matches)
    if not result_df.empty:
        result_df = result_df.sort_values(["score", "amount_cny"], ascending=[False, False])

    output = Path(args.output)
    result_df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result_df)} SEPA Stage 2 candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
