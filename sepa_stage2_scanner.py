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
    """简化版杯柄检测：近120日寻找U型回调+回升+缩量形态。"""
    if len(data) < 120:
        return False, {"error": "数据不足"}
    close = data["close"].values
    high = data["high"].values
    low = data["low"].values
    volume = data["volume"].values
    n = len(close)
    lookback = min(120, n)
    recent_close = close[-lookback:]
    recent_high = high[-lookback:]
    recent_low = low[-lookback:]
    recent_vol = volume[-lookback:]
    peak_idx = int(np.argmax(recent_high[:int(lookback * 0.75)]))
    peak_price = float(recent_high[peak_idx])
    search_start = peak_idx + 5
    if search_start >= len(recent_close) - 20:
        return False, {"reason": "无足够空间形成杯底"}
    bottom_idx = search_start + int(np.argmin(recent_low[search_start:-10]))
    bottom_price = float(recent_low[bottom_idx])
    cup_depth = (1 - bottom_price / peak_price) * 100
    if not (8 <= cup_depth <= 40):
        return False, {"reason": f"杯深{cup_depth:.0f}%不在8-40%"}
    rim_search = recent_high[bottom_idx + 10:]
    if len(rim_search) < 5:
        return False, {"reason": "无右杯沿"}
    rim_idx = bottom_idx + 10 + int(np.argmax(rim_search))
    rim_price = float(recent_high[rim_idx])
    if rim_price < peak_price * 0.7:
        return False, {"reason": "右杯沿回升不足"}
    handle_search = recent_high[rim_idx + 3:]
    if len(handle_search) < 3:
        return False, {"reason": "无柄部空间"}
    handle_low_idx = rim_idx + 3 + int(np.argmin(recent_low[rim_idx + 3:min(rim_idx + 30, len(recent_low))]))
    handle_low = float(recent_low[handle_low_idx])
    handle_depth = (1 - handle_low / rim_price) * 100 if rim_price > 0 else 0
    pre_peak_vol = np.mean(recent_vol[max(0, peak_idx-20):peak_idx+1])
    cup_vol = np.mean(recent_vol[peak_idx:rim_idx+1])
    vol_contract = cup_vol < pre_peak_vol * 1.2 and cup_vol > 0
    current_close = float(close[-1])
    current_vol = float(volume[-1])
    vol_ma20 = np.mean(volume[-20:])
    price_near_rim = current_close > rim_price * 0.9
    pattern = (
        cup_depth >= 10 and cup_depth <= 40 and
        rim_price >= peak_price * 0.75 and
        handle_depth >= 2 and handle_depth <= 20 and
        handle_low >= bottom_price * 1.02 and
        vol_contract and price_near_rim
    )
    return pattern, {
        "cup_valid": cup_depth >= 8 and cup_depth <= 40 and rim_price >= peak_price * 0.7,
        "cup_high": round(peak_price, 2),
        "cup_low": round(bottom_price, 2),
        "cup_depth_pct": round(cup_depth, 2),
        "cup_days": rim_idx - peak_idx,
        "right_rim_price": round(rim_price, 2),
        "handle_valid": handle_depth >= 2 and handle_depth <= 20,
        "handle_high": round(rim_price, 2),
        "handle_low": round(handle_low, 2),
        "handle_depth_pct": round(handle_depth, 2),
        "handle_days": handle_low_idx - rim_idx - 3,
        "volume_contracting": vol_contract,
        "price_breakout": current_close > rim_price,
        "volume_breakout": current_vol > vol_ma20 * 1.2,
        "near_rim": price_near_rim,
    }
def detect_vcp(data: pd.DataFrame) -> tuple[bool, dict]:
    """简化版VCP：近120日波动率收窄+量萎缩。"""
    if len(data) < 120:
        return False, {"error": "数据不足"}
    recent = data.tail(120)
    close = recent["close"].values
    high = recent["high"].values
    low = recent["low"].values
    volume = recent["volume"].values
    if "ma200" not in data.columns:
        data["ma200"] = data["close"].rolling(200).mean()
    ma200_now = float(data["ma200"].iloc[-1])
    def amp(arr):
        return (np.max(arr) / np.min(arr) - 1) * 100 if np.min(arr) > 0 else 100
    a1, a2, a3 = amp(close[-20:]), amp(close[-40:-20]), amp(close[-60:-40])
    tight = a1 < max(a2, a3) * 0.9 if max(a2, a3) > 0 and a1 < 25 else False
    v20 = np.mean(volume[-20:])
    v60 = np.mean(volume[-80:-20])
    vol_dry = v20 < v60 * 0.95
    current_close = float(close[-1])
    current_vol = float(volume[-1])
    high_20 = float(np.max(high[-20:]))
    price_break = current_close > high_20 * 0.95
    vol_break = current_vol > np.mean(volume[-20:]) * 1.2
    up = current_close > ma200_now and float(data["ma200"].iloc[-1]) > float(data["ma200"].iloc[-31])
    pattern = up and (tight or vol_dry) and price_break
    return pattern, {
        "vcp_valid": pattern,
        "vcp_status": "已突破" if (price_break and vol_break) else "构筑中",
        "vcp_contractions": sum(1 for v in [a1, a2, a3] if v > 3),
        "vcp_tighten": tight,
        "vcp_vol_shrink": vol_dry,
        "high_trend_ok": up,
        "vcp_tight_range": round(a1, 2),
        "vcp_vol_dry_up": vol_dry,
        "vcp_breakout": price_break and vol_break,
        "vcp_ranges": [round(a1, 2), round(a2, 2), round(a3, 2)],
        "vcp_vols": [round(v20, 0), round(v60, 0)],
        "vcp_resistance": round(high_20, 2),
    }


def calc_stage2_core_conditions(data: pd.DataFrame, rps_120: float | None = None) -> tuple[int, dict]:
    """计算 SEPA Stage2 5项核心条件（去掉MA50/MA150，增加RS强度）。"""
    if len(data) < 220:
        return 0, {}

    needs_copy = "ma200" not in data.columns
    df = data.copy() if needs_copy else data
    if "ma200" not in df.columns:
        df["ma200"] = df["close"].rolling(200).mean()

    latest = df.iloc[-1]
    ma200_20d_ago = df.iloc[-21]["ma200"]
    high_52w = df.tail(252)["high"].max()
    low_52w = df.tail(252)["low"].min()

    close = float(latest["close"])
    ma200 = float(latest["ma200"])
    pct_above_low = (close / float(low_52w) - 1) * 100
    pct_below_high = (close / float(high_52w) - 1) * 100

    conditions = {
        "close_gt_ma200": bool(close > ma200),
        "ma200_rising_1m": bool(ma200 > float(ma200_20d_ago)),
        "above_52w_low_30pct": bool(pct_above_low >= 30),
        "within_25pct_52w_high": bool(pct_below_high >= -25),
        "rps_120_strong": bool(rps_120 is not None and rps_120 >= 80),
    }
    count = sum(1 for v in conditions.values() if v)
    return count, conditions


def calc_all_rps(stock_returns: dict[str, float]) -> dict[str, float]:
    """批量计算全市场个股RPS（相对强弱百分位）。"""
    codes = list(stock_returns.keys())
    returns = np.array([stock_returns[c] for c in codes])
    ranks = np.argsort(np.argsort(returns))
    rps = (ranks / (len(returns) - 1)) * 100
    return {code: round(float(rps[i]), 2) for i, code in enumerate(codes)}


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
    """识别 Stage1 → Stage2 过渡态。"""
    if len(data) < 220:
        return False, {"error": "数据不足"}

    core_count = core_count_override if core_count_override is not None else calc_stage2_core_conditions(data, rps_120)[0]
    if not (3 <= core_count <= 4):
        return False, {"stage2_core_count": core_count}

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
    vol_turning_up = recent_20d_vol < prior_100d_vol * 0.85 and recent_5d_vol > recent_20d_vol * 1.2

    has_vcp = has_vcp_override if has_vcp_override is not None else detect_vcp(data)[0]
    has_cup = has_cup_override if has_cup_override is not None else detect_cup_handle(data)[0]
    pattern_ready = has_vcp or has_cup

    if financial_ready_override is not None:
        financial_ready = financial_ready_override
    else:
        try:
            fin = check_financial_acceleration(code, financial_cache)
            financial_ready = fin["financial_pass"]
        except Exception:
            financial_ready = False

    rps_ready = rps_120 is not None and rps_120 >= 70
    base_transition = ma_turning_up or vol_turning_up or pattern_ready
    high_quality_transition = base_transition and financial_ready and rps_ready

    return high_quality_transition or base_transition, {
        "is_transition": high_quality_transition or base_transition,
        "is_high_quality": high_quality_transition,
        "stage2_core_count": core_count,
        "ma_turning_up": ma_turning_up,
        "vol_turning_up": vol_turning_up,
        "pattern_ready": pattern_ready,
        "financial_ready": financial_ready,
        "rps_120_ready": rps_ready,
        "ma200_change_pct": round(ma200_change_pct, 2),
    }


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


def _to_json_safe(obj):
    """递归转换 numpy 类型为原生 Python 类型，确保 json.dumps 可用。"""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return _to_json_safe(obj.tolist())
    return obj


def evaluate_stage2(code: str, name: str, industry: str, history: pd.DataFrame,
                    financial_cache: dict | None = None,
                    market_caps: dict[str, float] | None = None,
                    rps_120: float | None = None) -> dict:
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
    data["ma200"] = data["close"].rolling(200).mean()

    latest = data.iloc[-1]
    high_52w = data.tail(252)["high"].max()
    low_52w = data.tail(252)["low"].min()

    close = float(latest["close"])
    ma200 = float(latest["ma200"])
    pct_above_low = (close / float(low_52w) - 1) * 100
    pct_below_high = (close / float(high_52w) - 1) * 100
    return_20d = (close / float(data.iloc[-21]["close"]) - 1) * 100
    return_120d = (close / float(data.iloc[-121]["close"]) - 1) * 100 if len(data) >= 121 else 0

    cup_handle_pattern, cup_handle_details = detect_cup_handle(data)
    vcp_pattern, vcp_details = detect_vcp(data)
    pullback, pullback_details = detect_pullback(data)

    core_count, core_conditions = calc_stage2_core_conditions(data, rps_120)
    conditions = {
        **core_conditions,
        "cup_handle_pattern": cup_handle_pattern,
        "vcp_pattern": vcp_pattern,
        "pullback_confirm": pullback,
    }

    score = 0
    score += 25 if conditions["close_gt_ma200"] else 0
    score += 30 if conditions["rps_120_strong"] else 0
    score += 25 if conditions["ma200_rising_1m"] else 0
    score += max(0, 10 + pct_below_high)
    score += min(10, pct_above_low / 5)
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

    is_stage2 = core_count == 5

    # 杯柄/VCP合并评分: 0=无, 1=满足一项, 2=两项都满足
    cup_vcp_score = (1 if cup_handle_pattern else 0) + (1 if vcp_pattern else 0)

    # 提取 ROE（净资产收益率）
    roe = None
    try:
        fin_abs = ak.stock_financial_abstract(symbol=code)
        if not fin_abs.empty:
            roe_row = fin_abs[fin_abs["指标"] == "净资产收益率_平均" if "净资产收益率_平均" in fin_abs["指标"].values else "净资产收益率(ROE)"]
            # Try 净资产收益率_平均 first, fallback to 净资产收益率(ROE)
            if roe_row.empty:
                roe_row = fin_abs[fin_abs["指标"] == "净资产收益率(ROE)"]
            if not roe_row.empty:
                date_cols = [c for c in fin_abs.columns if str(c).isdigit() and len(str(c)) == 8]
                if date_cols:
                    latest_col = sorted(date_cols)[-1]
                    val = roe_row.iloc[0][latest_col]
                    if pd.notna(val):
                        roe = round(float(val), 2)
    except Exception:
        pass

    reasons = []
    if conditions["close_gt_ma200"]:
        reasons.append("price above MA200")
    if conditions["rps_120_strong"]:
        reasons.append(f"RS强度{rps_120:.0f}(前20%)")
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
        "ma200": round(ma200, 2),
        "52w_high": round(float(high_52w), 2),
        "52w_low": round(float(low_52w), 2),
        "pct_above_52w_low": round(float(pct_above_low), 2),
        "pct_below_52w_high": round(float(pct_below_high), 2),
        "return_20d_pct": round(float(return_20d), 2),
        "return_120d_pct": round(float(return_120d), 2),
        "rps_120": round(rps_120, 1) if rps_120 is not None else None,
        "amount_cny": round(float(latest["amount"]), 2),
        "volume": round(float(latest["volume"]), 2),
        "score": round(float(score), 2),
        "conditions": json.dumps(_to_json_safe(conditions), ensure_ascii=False),
        "cup_vcp_score": cup_vcp_score,
        "roe": roe,
        "cup_handle_details": json.dumps(_to_json_safe(cup_handle_details), ensure_ascii=False),
        "vcp_details": json.dumps(_to_json_safe(vcp_details), ensure_ascii=False),
        "pullback_details": json.dumps(_to_json_safe(pullback_details), ensure_ascii=False),
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
    rps_120: float | None = None,
) -> dict | None:
    result = evaluate_stage2(code, name, industry, history,
                             financial_cache=financial_cache, market_caps=market_caps,
                             rps_120=rps_120)
    if not result.get("is_stage2"):
        return None
    return result


def scan_one(
    row: pd.Series, industry_map: dict[str, str], args: argparse.Namespace,
    financial_cache: dict | None = None,
    market_caps: dict[str, float] | None = None,
    rps_map: dict[str, float] | None = None,
) -> dict | None:
    code = row["代码"]
    name = row["名称"]
    history = fetch_history(code, args.min_history_days, args.sleep_seconds)
    rps_120 = rps_map.get(code) if rps_map else None
    return analyze_stage2(code, name, industry_map.get(code, "Unknown"), history,
                          financial_cache=financial_cache, market_caps=market_caps,
                          rps_120=rps_120)


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

    # ── 批量计算 120 日 RPS ──
    print("Calculating 120-day RPS for all stocks...")
    rps_returns: dict[str, float] = {}
    for index, row in pool.iterrows():
        code = row["代码"]
        hist = fetch_history(code, 130, args.sleep_seconds)
        if hist is not None and len(hist) >= 121:
            rps_returns[code] = (hist["close"].iloc[-1] / hist["close"].iloc[-121] - 1) * 100
    rps_map = calc_all_rps(rps_returns) if rps_returns else {}
    print(f"RPS calculated for {len(rps_map)} stocks")

    matches: list[dict] = []
    for index, row in pool.iterrows():
        try:
            result = scan_one(row, industry_map, args, financial_cache=financial_cache,
                              market_caps=market_caps, rps_map=rps_map)
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
    result_df.to_csv(output, index=False, encoding="utf-8")
    print(f"Saved {len(result_df)} SEPA Stage 2 candidates to {output}")

    # 输出全市场 RPS 缓存，供手动评估查询
    rps_out = Path("rps_all.csv")
    pd.DataFrame({"code": list(rps_map.keys()), "rps_120": list(rps_map.values())}).to_csv(
        rps_out, index=False, encoding="utf-8")
    print(f"Saved {len(rps_map)} RPS records to {rps_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
