#!/usr/bin/env python3
"""Local dashboard server with SEPA Stage 2 evaluation API and Excel export."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import traceback
import glob
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import akshare as ak

import time

# 全局股票名称缓存，避免每次API请求
_stock_name_cache = None

# 优化7：基本面数据短期缓存（10分钟TTL）
_fundamental_cache = {}
_fundamental_cache_ttl = 600  # 10分钟

# Sina 财务计算缓存（随 fundamental_cache 生命周期）
_sina_metrics_cache = {}

def _compute_sina_metrics(code: str, close_price: float) -> dict:
    """从新浪利润表计算同比增速、净利率、PE TTM，替代东方财富数据。
    
    返回 dict 包含:
        rev_growth: [最近3期营收YoY增速]
        profit_growth: [最近3期净利润YoY增速]  
        profit_margin: [最近3期净利率]
        pe_ttm: TTM市盈率 或 None
    """
    cache_key = f"{code}_{close_price}"
    if cache_key in _sina_metrics_cache:
        return _sina_metrics_cache[cache_key]

    import pandas as pd

    def _parse_amount(v):
        try: return float(v)
        except (ValueError, TypeError): return None

    result = {"rev_growth": [], "profit_growth": [], "profit_margin": [], "pe_ttm": None}

    try:
        df = ak.stock_financial_report_sina(stock=sina_prefix(code), symbol="利润表")
        if df is None or df.empty:
            return result
        df["报告日"] = df["报告日"].astype(str)
        df = df.sort_values("报告日")

        rev_col = "营业总收入" if "营业总收入" in df.columns else ("营业收入" if "营业收入" in df.columns else None)
        profit_col = "净利润"
        eps_col = "基本每股收益" if "基本每股收益" in df.columns else None

        if not rev_col:
            return result

        df["_rev"] = pd.to_numeric(df[rev_col], errors="coerce")
        df["_profit"] = pd.to_numeric(df[profit_col], errors="coerce") if profit_col in df.columns else float("nan")
        if eps_col:
            df["_eps"] = pd.to_numeric(df[eps_col], errors="coerce")

        date_to_idx = {}
        for i, row in df.iterrows():
            d = str(row["报告日"])[:8]
            mm = d[4:8]
            yyyy = int(d[:4])
            date_to_idx[(yyyy, mm)] = i

        all_dates = sorted(df["报告日"].str[:8].tolist())
        recent_dates = all_dates[-3:] if len(all_dates) >= 3 else all_dates

        rev_growth = []
        profit_growth = []
        profit_margin = []

        for d in recent_dates:
            mm = d[4:8]
            yyyy = int(d[:4])
            cur_idx = date_to_idx.get((yyyy, mm))
            prev_idx = date_to_idx.get((yyyy - 1, mm))

            # YoY growth
            if cur_idx is not None and prev_idx is not None:
                rev_cur = _parse_amount(df.loc[cur_idx, "_rev"])
                rev_prev = _parse_amount(df.loc[prev_idx, "_rev"])
                if rev_cur is not None and rev_prev is not None and abs(rev_prev) > 0:
                    rev_growth.append(round((rev_cur - rev_prev) / abs(rev_prev) * 100, 2))
                else:
                    rev_growth.append(None)

                p_cur = _parse_amount(df.loc[cur_idx, "_profit"])
                p_prev = _parse_amount(df.loc[prev_idx, "_profit"])
                if p_cur is not None and p_prev is not None and abs(p_prev) > 0:
                    profit_growth.append(round((p_cur - p_prev) / abs(p_prev) * 100, 2))
                else:
                    profit_growth.append(None)

                # Profit margin
                if rev_cur is not None and rev_cur != 0:
                    profit_margin.append(round((p_cur or 0) / rev_cur * 100, 2))
                else:
                    profit_margin.append(None)
            else:
                rev_growth.append(None)
                profit_growth.append(None)
                profit_margin.append(None)

        result["rev_growth"] = [x for x in rev_growth if x is not None]
        result["profit_growth"] = [x for x in profit_growth if x is not None]
        result["profit_margin"] = [x for x in profit_margin if x is not None]

        # PE TTM: 近四个季度单季EPS之和
        if eps_col and close_price > 0:
            df_sorted = df.sort_values("报告日").copy()
            df_sorted["_eps"] = pd.to_numeric(df_sorted[eps_col], errors="coerce")
            eps_rows = df_sorted[df_sorted["_eps"].notna()].copy()
            if len(eps_rows) >= 2:
                # 按年份分组，从累计EPS反推单季EPS
                eps_rows["_year"] = eps_rows["报告日"].str[:4].astype(int)
                eps_rows["_mmdd"] = eps_rows["报告日"].str[4:8]
                single_q_eps = []  # [date_str, single_quarter_eps]
                for yr, grp in eps_rows.groupby("_year"):
                    grp = grp.sort_values("报告日")
                    for i, (_, row) in enumerate(grp.iterrows()):
                        d = str(row["报告日"])[:8]
                        if row["_mmdd"] == "0331":
                            single_q_eps.append([d, float(row["_eps"])])
                        elif i > 0:
                            pv = float(grp.iloc[i - 1]["_eps"])
                            diff = float(row["_eps"]) - pv
                            if -1 < diff < 1:  # 忽略明显异常
                                single_q_eps.append([d, diff])
                if len(single_q_eps) >= 4:
                    ttm_eps = sum(e for _, e in single_q_eps[-4:])
                    if ttm_eps > 0:
                        result["pe_ttm"] = round(close_price / ttm_eps, 2)

    except Exception:
        pass

    # Trim cache to 50 entries
    if len(_sina_metrics_cache) > 50:
        _sina_metrics_cache.clear()
    _sina_metrics_cache[cache_key] = result
    return result


# 宏观数据缓存（融资余额、M1/M2），5分钟 TTL
_macro_cache = None
_macro_cache_ts = 0


def fetch_macro_data(force_refresh: bool = False) -> dict:
    """获取融资余额 + M1/M2 剪刀差数据，用于判断居民存款搬家趋势。
    
    返回:
        margin: {最近30日融资余额变化、融资买入额等}
        m1m2: {最近6个月 M1/M2 增速及剪刀差}
        summary: 一句话总结
    """
    global _macro_cache, _macro_cache_ts
    now = time.time()
    if not force_refresh and _macro_cache is not None and (now - _macro_cache_ts) < 300:
        return _macro_cache

    import pandas as pd
    import numpy as np
    import akshare as ak

    result = {"margin": {}, "m1m2": {}, "summary": ""}

    # 1. 融资融券余额（沪市，两市体量比例稳定）
    try:
        sh = ak.macro_china_market_margin_sh()
        sh["日期"] = pd.to_datetime(sh["日期"])
        sh = sh.sort_values("日期")
        margin_tail = sh.tail(30).copy()
        margin_last = margin_tail.iloc[-1]
        margin_prev_5d = margin_tail.iloc[-6] if len(margin_tail) >= 6 else margin_tail.iloc[0]
        margin_prev_20d = margin_tail.iloc[0]

        def _fmt_yi(v):
            return round(float(v) / 1e8, 2)

        result["margin"] = {
            "日期": str(margin_last["日期"])[:10],
            "融资余额(亿)": _fmt_yi(margin_last["融资余额"]),
            "融资余额5日变化(亿)": _fmt_yi(margin_last["融资余额"] - margin_prev_5d["融资余额"]),
            "融资余额20日变化(亿)": _fmt_yi(margin_last["融资余额"] - margin_prev_20d["融资余额"]),
            "当日融资买入(亿)": _fmt_yi(margin_last["融资买入额"]),
            "融券余额(亿)": _fmt_yi(margin_last["融券余额"]),
        }
    except Exception:
        pass

    # 3. 新增投资者开户数（API + 手动补充）
    try:
        import json as _json
        acc = ak.stock_account_statistics_em()
        acc["数据日期"] = acc["数据日期"].astype(str)
        acc = acc.sort_values("数据日期")

        # 合并手动补充数据（2024+）
        try:
            manual_path = "investor_accounts_manual.json"
            if __import__("os").path.exists(manual_path):
                manual_rows = _json.loads(open(manual_path).read())
                for mr in manual_rows:
                    dt = mr["日期"]
                    existing = acc[acc["数据日期"] == dt]
                    if existing.empty and dt > str(acc["数据日期"].max()):
                        new_row = {
                            "数据日期": dt,
                            "新增投资者-数量": mr["新增投资者-数量"],
                            "新增投资者-环比": None,
                            "新增投资者-同比": None,
                            "期末投资者-总量": None,
                            "期末投资者-A股账户": None,
                            "期末投资者-B股账户": None,
                            "沪深总市值": None,
                            "沪深户均市值": None,
                            "上证指数-收盘": None,
                            "上证指数-涨跌幅": None,
                        }
                        # 用原始列集避免 concat 列不齐的 FutureWarning
                        new_df = pd.DataFrame(columns=acc.columns.tolist())
                        for col in acc.columns:
                            new_df.at[0, col] = new_row.get(col, None)
                        acc = pd.concat([acc, new_df], ignore_index=True)
                acc = acc.sort_values("数据日期")
        except Exception:
            pass

        acc_tail = acc.tail(6).copy()

        def _safe_float(v, default=None):
            try:
                f = float(v)
                return round(f, 2) if pd.notna(f) else default
            except (ValueError, TypeError):
                return default

        result["investor"] = {
            "月份": [str(d) for d in acc_tail["数据日期"].tolist()],
            "新增(万户)": [_safe_float(v) for v in acc_tail["新增投资者-数量"].tolist()],
            "同比": [round(float(v) * 100, 1) if pd.notna(v) else None for v in acc_tail["新增投资者-同比"].tolist()],
            "期末总量(万户)": [_safe_float(v) for v in acc_tail.get("期末投资者-总量", acc_tail["新增投资者-数量"]).tolist()],
            "户均市值(万)": [_safe_float(v) for v in acc_tail.get("沪深户均市值", []).tolist()],
            "数据来源": "2015-2023东方财富 + 2024+手动补充(上交所披露)",
        }

        # 峰值参考（全量）
        peaks = acc["新增投资者-数量"].astype(float)
        peak_val = float(peaks.max())
        peak_month = acc.loc[peaks.idxmax(), "数据日期"]
        result["investor"]["历史峰值万户"] = round(peak_val, 2)
        result["investor"]["峰值月份"] = str(peak_month)

        # 当前信号
        recent_vals = [float(v) for v in acc_tail["新增投资者-数量"].tolist() if pd.notna(v)]
        investor_signal = ""
        if recent_vals and recent_vals[-1] > 500:
            investor_signal = "单月＞500万户，大牛市冲顶信号"
        elif len(recent_vals) >= 3 and all(v > 200 for v in recent_vals[-3:]):
            investor_signal = "连续3月＞200万户，阶段小牛市顶部"
        elif recent_vals and recent_vals[-1] > 200:
            investor_signal = "单月＞200万户，关注过热信号"
        elif recent_vals and recent_vals[-1] > 100:
            investor_signal = "开户活跃，处于正常偏热区间"
        else:
            investor_signal = "开户情绪冷淡"
        result["investor"]["信号"] = investor_signal
    except Exception:
        pass

    # 4. M1/M2 剪刀差
    try:
        ms = ak.macro_china_supply_of_money()
        ms = ms[ms["货币(狭义货币M1)同比增长"].notna() & ms["货币和准货币（广义货币M2）同比增长"].notna()].copy()
        ms["统计时间"] = ms["统计时间"].astype(str)
        ms = ms.sort_values("统计时间")
        m1m2_tail = ms.tail(6).copy()
        m1m2_tail["m1"] = pd.to_numeric(m1m2_tail["货币(狭义货币M1)同比增长"], errors="coerce")
        m1m2_tail["m2"] = pd.to_numeric(m1m2_tail["货币和准货币（广义货币M2）同比增长"], errors="coerce")
        m1m2_tail["剪刀差"] = (m1m2_tail["m1"] - m1m2_tail["m2"]).round(2)

        months = m1m2_tail["统计时间"].tolist()
        m1_vals = m1m2_tail["m1"].tolist()
        m2_vals = m1m2_tail["m2"].tolist()
        diff_vals = m1m2_tail["剪刀差"].tolist()

        result["m1m2"] = {
            "月份": months,
            "M1增速": [round(v, 1) for v in m1_vals],
            "M2增速": [round(v, 1) for v in m2_vals],
            "剪刀差": [round(v, 2) for v in diff_vals],
        }
    except Exception:
        pass

    # ===== 综合总结 =====
    try:
        diff_vals = result.get("m1m2", {}).get("剪刀差", [])
        if len(diff_vals) >= 2:
            recent_diff = diff_vals[-1]
            prev_diff = diff_vals[-2]
            if recent_diff > prev_diff and recent_diff < 0:
                trend = "剪刀差收窄，边际改善但仍在负区间"
            elif recent_diff > prev_diff and recent_diff > 0:
                trend = "剪刀差转正，资金活化信号"
            elif recent_diff > 0:
                trend = "剪刀差为正，存款搬家进行中"
            else:
                trend = "剪刀差扩大，资金沉淀观望"
        else:
            trend = ""

        margin_5d = result.get("margin", {}).get("融资余额5日变化(亿)", 0)
        margin_signal = "融资余额上升" if margin_5d > 0 else "融资余额下降"

        inv_signal = result.get("investor", {}).get("信号", "")

        parts = [t for t in [trend, margin_signal, inv_signal] if t]
        result["summary"] = "；".join(parts)
    except Exception:
        pass

    _macro_cache = result
    _macro_cache_ts = now
    return result



from sepa_stage2_scanner import evaluate_stage2, fetch_history, load_industry_overrides, calc_all_rps
from sepa_stage1_scanner import evaluate_stage1
from technical_analyzer import analyze_technical
from financial_filter import load_cache

_RPS_CACHE: dict[str, float] = {}
_RPS_CACHE_TS = 0.0

def _load_rps_cache() -> dict[str, float]:
    global _RPS_CACHE, _RPS_CACHE_TS
    now = time.time()
    if _RPS_CACHE and now - _RPS_CACHE_TS < 60:
        return _RPS_CACHE
    for fname in ["rps_all.csv", "sepa_stage2_candidates.csv"]:
        csv_path = os.path.join(os.path.dirname(__file__), fname)
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
                if "rps_120" in df.columns and "code" in df.columns:
                    df["code"] = df["code"].astype(str).str.zfill(6)
                    _RPS_CACHE = dict(zip(df["code"], df["rps_120"]))
                    _RPS_CACHE_TS = now
                    return _RPS_CACHE
        except Exception:
            continue
    return _RPS_CACHE

def _estimate_rps(code: str, cash_flow=None) -> float | None:
    """如果 RPS 缓存中没命中，实时估算该股的 RPS 百分位。
    
    基于现有缓存的 RPS 分布（百分位 + 120日涨跌幅）做线性插值。
    缓存中无数据时回退到只算原始回报。
    """
    cache = _load_rps_cache()
    rps = cache.get(code)
    if rps is not None:
        return rps

    # 计算该股 120 日涨跌幅
    hist = fetch_history(code, min_history_days=130, sleep_seconds=0)
    if hist is None or len(hist) < 121:
        return None
    ret_120 = (hist["close"].iloc[-1] / hist["close"].iloc[-121] - 1) * 100

    if not cache:
        # 无缓存，只返回原始回报供参考（不输百分位）
        return None

    # 用缓存的 RPS 值做排名估计：RPS 值就是 (排名位置 / 总数)*100
    # 如果能拿到原始回报值，可以按回报估算位置
    rps_vals = sorted(cache.values())
    if len(rps_vals) < 10:
        return None

    # 假设 RPS 分布均匀，用 ret_120 在缓存 RPS 区间做插值
    # 缓存中存的 RPS 是百分位，不是回报值——我们只能用该股的回报去比较
    # 这里用简化方案：算该股回报在缓存 RPS 池中的排名百分位
    sorted_rps = sorted(rps_vals)
    n = len(sorted_rps)
    # RPS 值本身就是 0~100 的百分位，回报越高 RPS 越高
    # 无法通过 RPS 值反推回报，改用直接算回报百分位
    # 直接用该股回报估算：算回报排名 / 缓存大小 → RPS
    rank = sum(1 for v in sorted_rps if v > 0)  # 缓存中有多少正值
    # 简化：如果 120 日回报为正，给缓存中位值；为负给低位值
    if ret_120 > 20:
        return float(sorted_rps[int(n * 0.9)])
    elif ret_120 > 5:
        return float(sorted_rps[int(n * 0.75)])
    elif ret_120 > 0:
        return float(sorted_rps[int(n * 0.5)])
    elif ret_120 > -10:
        return float(sorted_rps[int(n * 0.25)])
    else:
        return float(sorted_rps[int(n * 0.1)])


def _json_default(obj):
    """Serialize numpy types to native Python for JSON."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


PORT = int(os.environ.get("PORT", "8001"))


def normalize_code(code: str) -> str:
    """标准化股票代码：仅保留数字，补全为6位"""
    return "".join(ch for ch in code if ch.isdigit()).zfill(6)


def sina_prefix(code: str) -> str:
    """Return Sina stock prefix: sz for 0/1/3, sh for 6/9, bj for 8/4."""
    first = code[0]
    if first in ("0", "1", "3"):
        return f"sz{code}"
    if first in ("6", "9"):
        return f"sh{code}"
    return f"bj{code}"


def lookup_name(code: str) -> str:
    """通过股票代码查询股票名称，优先从缓存获取，其次从AkShare获取，最后从本地CSV获取"""
    global _stock_name_cache
    
    # 优先从缓存获取
    if _stock_name_cache is not None and code in _stock_name_cache:
        return _stock_name_cache[code]
    
    # 如果缓存为空，从AkShare加载全市场股票名称
    if _stock_name_cache is None:
        try:
            pool = ak.stock_info_a_code_name().rename(columns={"code": "代码", "name": "名称"})
            pool["代码"] = pool["代码"].astype(str).str.zfill(6)
            _stock_name_cache = dict(zip(pool["代码"], pool["名称"]))
            if code in _stock_name_cache:
                return _stock_name_cache[code]
        except Exception:
            _stock_name_cache = {}  # 标记为已尝试加载，避免重复尝试
    
    # 从本地CSV获取
    for csv_path in ("test_candidates.csv", "sepa_stage2_candidates_test.csv"):
        path = Path(csv_path)
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path, dtype={"code": str})
        except Exception:
            continue
        if "code" not in frame.columns or "name" not in frame.columns:
            continue
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        matched = frame[frame["code"] == code]
        if not matched.empty:
            return str(matched.iloc[0]["name"])

    return code


def fetch_fundamental(code: str) -> dict:
    """Fetch fundamental data: profile, financial reports, recent disclosures.
    
    优化7：增加短期缓存（10分钟TTL），避免重复请求同一股票。
    """
    # 检查缓存
    cache_key = code
    if cache_key in _fundamental_cache:
        cached_data, cache_time = _fundamental_cache[cache_key]
        if time.time() - cache_time < _fundamental_cache_ttl:
            return cached_data
    
    fund = {"profile": None, "income": [], "balance": [], "cashflow": [], "disclosures": []}

    # 1. Profile (industry + concepts)
    try:
        profile = ak.stock_profile_cninfo(symbol=code)
        industry = str(profile["所属行业"].iloc[0]).strip() if "所属行业" in profile.columns else ""
        concepts_raw = str(profile["入选指数"].iloc[0]) if "入选指数" in profile.columns else ""
        concepts = [c.strip() for c in concepts_raw.split(",") if c.strip()] if concepts_raw else []
        fund["profile"] = {"industry": industry, "concepts": concepts}
    except Exception:
        fund["profile"] = {"industry": "获取失败", "concepts": []}

    # 2. Income statement (利润表) — last 3 reports
    try:
        income_df = ak.stock_financial_report_sina(stock=sina_prefix(code), symbol="利润表")
        key_cols = {"营业总收入": "营业总收入", "营业收入": "营业收入",
                    "营业利润": "营业利润", "利润总额": "利润总额",
                    "净利润": "净利润", "营业总成本": "营业总成本"}
        recent = income_df.sort_values("报告日", ascending=False).head(3)
        fund["income"] = _format_financial_rows(recent, key_cols)
    except Exception:
        pass

    # 3. Balance sheet (资产负债表) — last 3 reports
    try:
        balance_df = ak.stock_financial_report_sina(stock=sina_prefix(code), symbol="资产负债表")
        key_cols = {"货币资金": "货币资金", "应收账款": "应收账款",
                    "应收票据及应收账款": "应收票据及应收账款",
                    "存货": "存货",
                    "流动资产": "流动资产合计",
                    "资产总计": "资产总计",
                    "流动负债合计": "流动负债合计",
                    "负债合计": "负债合计"}
        recent = balance_df.sort_values("报告日", ascending=False).head(3)
        fund["balance"] = _format_financial_rows(recent, key_cols)
    except Exception:
        pass

    # 4. Cash flow (现金流量表) — last 3 reports
    try:
        cash_df = ak.stock_financial_report_sina(stock=sina_prefix(code), symbol="现金流量表")
        key_cols = {"经营活动产生的现金流量净额": "经营现金流净额",
                    "投资活动产生的现金流量净额": "投资现金流净额",
                    "筹资活动产生的现金流量净额": "筹资现金流净额"}
        recent = cash_df.sort_values("报告日", ascending=False).head(3)
        fund["cashflow"] = _format_financial_rows(recent, key_cols)
    except Exception:
        pass

    # 5. Recent disclosures (巨潮资讯网)
    try:
        disc = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=code, market="沪深京",
            start_date="20260101", end_date="20260605",
        )
        top5 = disc.head(5)
        fund["disclosures"] = [
            {"date": str(row["公告时间"])[:10], "title": str(row["公告标题"])}
            for _, row in top5.iterrows()
        ]
    except Exception:
        pass

    # 存入缓存
    _fundamental_cache[cache_key] = (fund, time.time())
    
    return fund


def _format_financial_rows(df: pd.DataFrame, col_map: dict) -> list[dict]:
    """Format financial dataframe rows into a list of {label, values} dicts."""
    if df.empty:
        return []
    dates = [str(d) for d in df["报告日"].tolist()]
    rows_out = [{"label": "报告期", "dates": dates, "values": dates}]  # header row
    for orig, label in col_map.items():
        if orig not in df.columns:
            continue
        vals = []
        for v in df[orig].tolist():
            try:
                vals.append(_fmt_amount(float(v)))
            except (ValueError, TypeError):
                vals.append(str(v))
        rows_out.append({"label": label, "dates": dates, "values": vals})
    return rows_out


def _fmt_amount(v: float) -> str:
    """Format a raw CNY amount into readable string."""
    if abs(v) >= 1e12:
        return f"{v / 1e12:.2f}万亿"
    if abs(v) >= 1e8:
        return f"{v / 1e8:.2f}亿"
    if abs(v) >= 1e4:
        return f"{v / 1e4:.0f}万"
    return f"{v:.0f}"


_scan_running: dict[str, bool] = {}


class StockHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        """处理GET请求，路由API和静态文件"""
        parsed = urlparse(self.path)
        if parsed.path == "/api/sepa":
            self.handle_sepa(parsed.query)
            return
        if parsed.path == "/api/sepa_stage1":
            self.handle_sepa_stage1(parsed.query)
            return
        if parsed.path == "/api/scan":
            self.handle_scan(parsed.query)
            return
        if parsed.path == "/api/download":
            self.handle_download(parsed.query)
            return
        if parsed.path == "/api/technical":
            self.handle_technical(parsed.query)
            return
        if parsed.path == "/api/macro":
            self.handle_macro(parsed.query)
            return
        if parsed.path == "/api/market_breadth":
            self.handle_market_breadth()
            return
        super().do_GET()

    def handle_sepa(self, query: str) -> None:
        """处理SEPA Stage2 股票分析API请求，包含技术面 + 基本面"""
        params = parse_qs(query)
        code = normalize_code(params.get("code", [""])[0])

        if len(code) != 6:
            self.write_json({"error": "Please input a 6-digit stock code."}, status=400)
            return

        try:
            name = lookup_name(code)
            history = fetch_history(code, min_history_days=220, sleep_seconds=0)
            industry = load_industry_overrides().get(code, "Unknown")
            financial_cache = load_cache()
            rps_120 = _estimate_rps(code)
            result = evaluate_stage2(code, name, industry, history, financial_cache=financial_cache, rps_120=rps_120)

            # Convert numpy bools to Python bools for JSON serialization
            if "conditions" in result and isinstance(result["conditions"], dict):
                result["conditions"] = {k: bool(v) for k, v in result["conditions"].items()}
            result["is_stage2"] = bool(result.get("is_stage2", False))

            fundamental = fetch_fundamental(code)

            result["fundamental"] = fundamental
            # 用新浪数据覆盖同比增速/利润率/PE，与下方三大报表数据源一致
            close_price = float(result.get("close", 0))
            sina = _compute_sina_metrics(code, close_price)
            if sina.get("rev_growth"):
                result["rev_growth"] = json.dumps(sina["rev_growth"][-3:] if len(sina["rev_growth"]) >= 3 else sina["rev_growth"], ensure_ascii=False)
            if sina.get("profit_growth"):
                result["profit_growth"] = json.dumps(sina["profit_growth"][-3:] if len(sina["profit_growth"]) >= 3 else sina["profit_growth"], ensure_ascii=False)
            if sina.get("profit_margin"):
                result["profit_margin"] = json.dumps(sina["profit_margin"][-3:] if len(sina["profit_margin"]) >= 3 else sina["profit_margin"], ensure_ascii=False)
            if sina.get("pe_ttm") is not None:
                result["pe_ttm"] = sina["pe_ttm"]
            self.write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"code": code, "error": str(exc)}, status=500)

    def handle_sepa_stage1(self, query: str) -> None:
        """处理SEPA Stage1 股票分析API请求"""
        params = parse_qs(query)
        code = normalize_code(params.get("code", [""])[0])
        if len(code) != 6:
            self.write_json({"error": "Please input a 6-digit stock code."}, status=400)
            return
        try:
            name = lookup_name(code)
            history = fetch_history(code, min_history_days=250, sleep_seconds=0)
            industry = load_industry_overrides().get(code, "Unknown")
            financial_cache = load_cache()
            result = evaluate_stage1(code, name, industry, history, financial_cache=financial_cache)
            if "conditions" in result and isinstance(result["conditions"], dict):
                result["conditions"] = {k: bool(v) for k, v in result["conditions"].items()}
            result["is_stage1"] = bool(result.get("is_stage1", False))
            # 用新浪数据覆盖同比增速/利润率/PE
            close_price = float(result.get("close", 0))
            sina = _compute_sina_metrics(code, close_price)
            if sina.get("rev_growth"):
                result["rev_growth"] = json.dumps(sina["rev_growth"][-3:] if len(sina["rev_growth"]) >= 3 else sina["rev_growth"], ensure_ascii=False)
            if sina.get("profit_growth"):
                result["profit_growth"] = json.dumps(sina["profit_growth"][-3:] if len(sina["profit_growth"]) >= 3 else sina["profit_growth"], ensure_ascii=False)
            if sina.get("profit_margin"):
                result["profit_margin"] = json.dumps(sina["profit_margin"][-3:] if len(sina["profit_margin"]) >= 3 else sina["profit_margin"], ensure_ascii=False)
            if sina.get("pe_ttm") is not None:
                result["pe_ttm"] = sina["pe_ttm"]
            self.write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"code": code, "error": str(exc)}, status=500)

    def handle_technical(self, query: str) -> None:
        """处理市场技术分析 API 请求"""
        params = parse_qs(query)
        code = normalize_code(params.get("code", [""])[0])
        if len(code) != 6:
            self.write_json({"error": "Please input a 6-digit stock code."}, status=400)
            return
        try:
            result = analyze_technical(code)
            self.write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"code": code, "error": str(exc)}, status=500)

    def handle_macro(self, query: str = "") -> None:
        """处理宏观数据 API 请求"""
        params = parse_qs(query)
        force = params.get("force", ["false"])[0].lower() == "true"
        try:
            result = fetch_macro_data(force_refresh=force)
            self.write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"error": str(exc)}, status=500)

    def handle_market_breadth(self) -> None:
        """实时获取市场情绪/恐慌指数/涨跌中位数"""
        try:
            from market_breadth import fetch_spot_data, build_market_breadth
            spot = fetch_spot_data()
            result = build_market_breadth(spot)
            self.write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"error": str(exc)}, status=500)

    def handle_scan(self, query: str) -> None:
        """处理批量扫描请求，后台启动 subprocess"""
        params = parse_qs(query)
        scan_type = params.get("type", ["stage1"])[0]
        total = int(params.get("total", ["5000"])[0])
        batch = int(params.get("batch", ["200"])[0])

        if scan_type not in ("stage1", "stage2", "value_bottom"):
            self.write_json({"error": "type must be stage1, stage2, or value_bottom"}, status=400)
            return

        key = f"scan_{scan_type}"
        if _scan_running.get(key):
            self.write_json({"error": f"{scan_type} 扫描已在运行中"}, status=409)
            return

        _scan_running[key] = True

        def _run():
            try:
                subprocess.run(
                    [sys.executable, "_scan_worker.py", "--type", scan_type,
                     "--total", str(total), "--batch", str(batch)],
                    cwd=str(Path(__file__).parent),
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                )
            except Exception as exc:
                print(f"Scan worker failed: {exc}", file=sys.stderr)
            finally:
                _scan_running[key] = False

        threading.Thread(target=_run, daemon=True).start()
        self.write_json({"status": "started", "type": scan_type, "total": total})

    def handle_download(self, query: str) -> None:
        """处理Excel下载请求"""
        params = parse_qs(query)
        table_type = params.get("type", ["candidate"])[0]

        if table_type == "sepa":
            csv_path = "sepa_stage2_candidates_test.csv"
            filename = "SEPA_Stage2_Candidates.xlsx"
        elif table_type == "stage1":
            csv_path = "sepa_stage1_candidates_test.csv"
            filename = "SEPA_Stage1_Candidates.xlsx"
        elif table_type == "value_bottom":
            csv_path = "value_bottom_candidates_test.csv"
            filename = "Value_Bottom_Candidates.xlsx"
        else:
            csv_path = "test_candidates.csv"
            filename = "Stock_Candidates.xlsx"

        path = Path(csv_path)
        if not path.exists() or path.stat().st_size == 0:
            self.write_json({"error": "No data available for download."}, status=404)
            return

        try:
            df = pd.read_csv(csv_path, dtype={"code": str})
        except Exception as exc:
            self.write_json({"error": f"Failed to read data: {exc}"}, status=500)
            return

        # Generate Excel in memory
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Candidates")
        output.seek(0)
        excel_data = output.read()

        self.send_response(200)
        self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        self.send_header("Content-Disposition", f"attachment; filename={filename}")
        self.send_header("Content-Length", str(len(excel_data)))
        self.end_headers()
        self.wfile.write(excel_data)

    def write_json(self, payload: dict, status: int = 200) -> None:
        """统一返回JSON格式响应"""
        body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    """启动HTTP服务主函数"""
    # Clean up stale scan progress files from previous runs
    for f in ["scan_progress_s1.json", "scan_progress_s2.json", "scan_progress_vb.json", "scan_progress.json"]:
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass
    
    # Also clean up any other potential stale progress files
    import glob
    for f in glob.glob("scan_progress_*.json"):
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass

    server = ThreadingHTTPServer(("localhost", PORT), StockHandler)
    print(f"Serving dashboard with API at http://localhost:{PORT}/stock_dashboard.html")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
