#!/usr/bin/env python3
"""Local dashboard server with SEPA Stage 2 evaluation API and Excel export."""
from __future__ import annotations
import json
import os
import re
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

# 巴菲特指数缓存：每季度更新一次（GDP 按季度发布，避免市值日内波动干扰长期估值判断）
_buffett_cache = None
_buffett_cache_quarter = None


def _current_quarter_label() -> str:
    """返回当前季度标签，如 '2026Q3'。1-3月=Q1, 4-6月=Q2, 7-9月=Q3, 10-12月=Q4。"""
    import datetime as _dt
    now = _dt.datetime.now()
    q = (now.month - 1) // 3 + 1
    return f"{now.year}Q{q}"


def _compute_buffett_index() -> dict:
    """计算巴菲特指数（A股总市值 / GDP × 100%）。失败时返回空 dict。"""
    import datetime as _dt

    # 获取 A 股总市值（亿元）
    total_mv_yi = 0.0
    mv_source = ""

    # 方案1: 上交所 + 全市场缓存
    try:
        sse_df = ak.stock_sse_summary()
        sse_row = sse_df[sse_df["项目"] == "总市值"]
        if not sse_row.empty:
            sh_mv = float(sse_row.iloc[0]["股票"])
            from industry_analyzer import _ensure_cache
            cache = _ensure_cache()
            all_mv = sum(s.get("market_cap", 0) for s in cache.get("stocks", {}).values()) / 1e8
            if all_mv > sh_mv:
                total_mv_yi = all_mv
                mv_source = "全市场缓存"
            else:
                total_mv_yi = sh_mv
                mv_source = "上交所"
    except Exception as exc:
        print(f"[macro] 沪深总市值获取失败: {exc}", file=sys.stderr)

    # 方案2: 全市场缓存
    if total_mv_yi == 0:
        try:
            from industry_analyzer import _ensure_cache
            cache = _ensure_cache()
            total_mv_yi = sum(s.get("market_cap", 0) for s in cache.get("stocks", {}).values()) / 1e8
            mv_source = "全市场缓存"
        except Exception:
            pass

    # 获取 GDP（亿元）— 优先使用最近全年 GDP，无则用 TTM
    gdp_yi = 0.0
    gdp_label = ""
    try:
        gdp_df = ak.macro_china_gdp()
        for _, row in gdp_df.iterrows():
            quarter_str = str(row["季度"])
            if "第1-4季度" in quarter_str:
                gdp_yi = float(row["国内生产总值-绝对值"])
                gdp_label = quarter_str.replace("第1-4季度", "全年")
                break
        if gdp_yi == 0:
            cumulative_list = []
            for _, row in gdp_df.iterrows():
                quarter_str = str(row["季度"])
                cumulative = float(row["国内生产总值-绝对值"])
                cumulative_list.append((quarter_str, cumulative))
            if len(cumulative_list) >= 5:
                latest_q, latest_cum = cumulative_list[0]
                import re
                m = re.match(r"(\d{4})年第1-(\d)季度", latest_q)
                if m:
                    year = int(m.group(1))
                    q_num = int(m.group(2))
                    prev_year_q = f"{year-1}年第1-{q_num}季度"
                    prev_cum = None
                    for q, c in cumulative_list:
                        if q == prev_year_q:
                            prev_cum = c
                            break
                    prev_year_full = f"{year-1}年第1-4季度"
                    prev_full_cum = None
                    for q, c in cumulative_list:
                        if q == prev_year_full:
                            prev_full_cum = c
                            break
                    if prev_cum is not None and prev_full_cum is not None:
                        gdp_yi = prev_full_cum + (latest_cum - prev_cum)
                        gdp_label = f"近4季度（{year-1}Q{q_num+1}-{year}Q{q_num}）"
    except Exception as exc:
        print(f"[macro] GDP 获取失败: {exc}", file=sys.stderr)

    if total_mv_yi > 0 and gdp_yi > 0:
        ratio = round(total_mv_yi / gdp_yi * 100, 2)
        if ratio < 60:
            signal = "严重低估"
        elif ratio < 80:
            signal = "合理偏低"
        elif ratio < 100:
            signal = "合理"
        elif ratio < 120:
            signal = "偏高"
        else:
            signal = "严重高估"

        return {
            "ratio": ratio,
            "total_market_value_yi": round(total_mv_yi, 2),
            "gdp_yi": round(gdp_yi, 2),
            "gdp_period": gdp_label,
            "mv_source": mv_source,
            "signal": signal,
            "note": "巴菲特指数 = A股总市值 / GDP，A股权重阈值：<60%低估，60-80%合理偏低，80-100%合理，100-120%偏高，>120%高估",
            "quarter": _current_quarter_label(),
        }
    return {}


def _get_buffett_index() -> dict:
    """获取巴菲特指数，季度级缓存：同一季度内返回固定值。"""
    global _buffett_cache, _buffett_cache_quarter
    cur_q = _current_quarter_label()
    if _buffett_cache is not None and _buffett_cache_quarter == cur_q:
        return _buffett_cache
    try:
        result = _compute_buffett_index()
        if result:  # 仅当成功时才写入缓存
            _buffett_cache = result
            _buffett_cache_quarter = cur_q
            return result
    except Exception as exc:
        print(f"[macro] 巴菲特指数计算失败: {exc}", file=sys.stderr)
    # 失败时若有旧缓存，返回旧缓存（即使跨季度），否则空 dict
    return _buffett_cache if _buffett_cache is not None else {}


_spread_cache = None
_spread_cache_ts = 0.0


def _compute_equity_bond_spread() -> dict:
    """计算沪深300股债利差（FED 模型）。

    股债利差 = 沪深300盈利收益率(1/PE-TTM) - 10年期国债收益率
    阈值：>6% 股市底部（机会）；3%~6% 正常震荡区；<3% 股市顶部（风险）。
    """
    # 1) 沪深300 PE(TTM)：乐咕乐股
    pe = None
    pe_date = ""
    try:
        pe_df = ak.stock_index_pe_lg()  # 默认沪深300，"指数"列为点位
        if pe_df is not None and not pe_df.empty:
            row = pe_df.iloc[-1]
            pe = float(row["滚动市盈率"])
            pe_date = str(row["日期"])[:10]
    except Exception as exc:
        print(f"[spread] 沪深300 PE 获取失败: {exc}", file=sys.stderr)

    # 2) 10年期国债收益率
    bond_yield = None
    bond_date = ""
    try:
        bond_df = ak.bond_zh_us_rate()
        if bond_df is not None and not bond_df.empty:
            bond_df = bond_df.dropna(subset=["中国国债收益率10年"])
            row = bond_df.iloc[-1]
            bond_yield = float(row["中国国债收益率10年"])
            bond_date = str(row["日期"])[:10]
    except Exception as exc:
        print(f"[spread] 10年期国债收益率获取失败: {exc}", file=sys.stderr)

    if pe is None or bond_yield is None or pe <= 0:
        return {}

    earnings_yield = 100.0 / pe          # 盈利收益率 %
    spread = earnings_yield - bond_yield  # 股债利差 %

    if spread > 6:
        signal, advice = "股市底部", "股相对债极便宜：可逐步加仓/加大定投"
    elif spread >= 3:
        signal, advice = "正常震荡区", "维持原有定投，仓位中性，不激进也不恐慌"
    else:
        signal, advice = "股市顶部", "股相对债偏贵：谨慎追高，注意减仓保护收益"

    return {
        "pe_ttm": round(pe, 2),
        "earnings_yield": round(earnings_yield, 2),
        "bond_yield_10y": round(bond_yield, 2),
        "spread": round(spread, 2),
        "signal": signal,
        "advice": advice,
        "pe_date": pe_date,
        "bond_date": bond_date,
        "note": "股债利差 = 沪深300盈利收益率(1/PE-TTM) - 10年期国债收益率；阈值：>6% 股市底部，3%~6% 正常震荡区，<3% 股市顶部",
    }


def _get_equity_bond_spread(force_refresh: bool = False) -> dict:
    """获取股债利差，10 分钟内存缓存（与估值指标缓存时长一致）。"""
    global _spread_cache, _spread_cache_ts
    now = time.time()
    if not force_refresh and _spread_cache is not None and (now - _spread_cache_ts) < 600:
        return _spread_cache
    try:
        result = _compute_equity_bond_spread()
        if result:
            _spread_cache = result
            _spread_cache_ts = now
            return result
    except Exception as exc:
        print(f"[spread] 股债利差计算失败: {exc}", file=sys.stderr)
    return _spread_cache if _spread_cache is not None else {}


def _fetch_m1m2_data():
    """获取 M1/M2 货币供应量数据。

    优先使用 akshare；失败时回退到东方财富 datacenter 接口。
    返回 DataFrame: 列 [月份, M1, M2]，M1/M2 为同比增速（%）。
    """
    import pandas as pd

    # 方案1: akshare
    try:
        ms = ak.macro_china_supply_of_money()
        ms = ms[ms["货币(狭义货币M1)同比增长"].notna() & ms["货币和准货币（广义货币M2）同比增长"].notna()].copy()
        ms["统计时间"] = ms["统计时间"].astype(str)
        ms["M1"] = pd.to_numeric(ms["货币(狭义货币M1)同比增长"], errors="coerce")
        ms["M2"] = pd.to_numeric(ms["货币和准货币（广义货币M2）同比增长"], errors="coerce")
        ms = ms.rename(columns={"统计时间": "月份"})[["月份", "M1", "M2"]]
        if not ms.empty:
            return ms
    except Exception as e:
        print(f"[macro] akshare M1/M2 failed: {e}", file=sys.stderr)

    # 方案2: 东方财富 datacenter (datacenter-web)
    try:
        import requests
        url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://data.eastmoney.com/cjsj/hbgyl.html",
        }
        params = {
            "sortColumns": "REPORT_DATE",
            "sortTypes": -1,
            "pageSize": 50,
            "pageNumber": 1,
            "reportName": "RPT_ECONOMY_CURRENCY_SUPPLY",
            "columns": "REPORT_DATE,TIME,BASIC_CURRENCY_SAME,CURRENCY_SAME",
        }
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()
        if data.get("result") and data["result"].get("data"):
            rows = []
            for item in data["result"]["data"]:
                # TIME 字段如 "2026年05月份"，标准化为 "2026年05月"
                month_label = str(item.get("TIME", "")).replace("份", "")
                m1 = item.get("CURRENCY_SAME")        # M1 同比
                m2 = item.get("BASIC_CURRENCY_SAME")  # M2 同比
                if m1 is not None and m2 is not None:
                    rows.append({
                        "月份": month_label,
                        "M1": float(m1),
                        "M2": float(m2),
                    })
            if rows:
                return pd.DataFrame(rows)
    except Exception as e:
        print(f"[macro] eastmoney M1/M2 failed: {e}", file=sys.stderr)

    # 方案3: 东方财富 cjsj HTML 页面解析（最可靠）
    try:
        import requests
        from bs4 import BeautifulSoup
        import re
        url = "https://data.eastmoney.com/cjsj/hbgyl.html"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.content, "html.parser")
        # 找表格行
        rows = []
        for tr in soup.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 6:
                # 第一列是月份，如 "2026年05月份"
                month_text = tds[0].get_text(strip=True)
                if not re.match(r"\d{4}年\d{2}月", month_text):
                    continue
                # M2 同比、M1 同比分别在 td[2]、td[5]
                # 列结构: 月份 | M2数量 | M2同比 | M2环比 | M1数量 | M1同比 | M1环比 | M0数量 | M0同比 | M0环比
                try:
                    m2_yoy_str = tds[2].get_text(strip=True).replace("%", "")
                    m1_yoy_str = tds[5].get_text(strip=True).replace("%", "")
                    m2_yoy = float(m2_yoy_str) if m2_yoy_str else None
                    m1_yoy = float(m1_yoy_str) if m1_yoy_str else None
                    if m1_yoy is not None and m2_yoy is not None:
                        rows.append({
                            "月份": month_text,
                            "M1": m1_yoy,
                            "M2": m2_yoy,
                        })
                except (ValueError, IndexError):
                    continue
        if rows:
            return pd.DataFrame(rows)
    except Exception as e:
        print(f"[macro] eastmoney HTML M1/M2 failed: {e}", file=sys.stderr)

    return None


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

    result = {
        "margin": {},
        "m1m2": {},
        "summary": "",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

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
        import datetime as _dt
        acc = ak.stock_account_statistics_em()
        acc["数据日期"] = acc["数据日期"].astype(str)
        acc = acc.sort_values("数据日期")

        # 合并手动补充数据（2024+）
        manual_months_set = set()
        try:
            manual_path = "investor_accounts_manual.json"
            if __import__("os").path.exists(manual_path):
                manual_rows = _json.loads(open(manual_path).read())
                for mr in manual_rows:
                    dt = mr["日期"]
                    manual_months_set.add(dt)
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

        # 上交所官方 API 自动补齐（commonQuery.do，覆盖手动补充之后/遗漏的月份；
        # 上交所仅公布沪市 A 股新开户，口径与手动补充一致）
        sse_months_set = set()
        try:
            import requests as _rq
            import re as _re
            _now = _dt.datetime.now()
            for _y in sorted({_now.year - 1, _now.year}):
                # MDATE 不能晚于已发布月份：当年用当前月探测，空则回退上月
                _mdates = [f"{_y}{_now.month:02d}"] if _y == _now.year else [f"{_y}12"]
                if _y == _now.year and _now.month > 1:
                    _mdates.append(f"{_y}{_now.month - 1:02d}")
                _payload = None
                for _md in _mdates:
                    rr = _rq.get(
                        "https://query.sse.com.cn/commonQuery.do",
                        params={
                            "jsonCallBack": "jp",
                            "sqlId": "COMMON_SSE_TZZ_M_ALL_ACCT_C",
                            "isPagination": "false",
                            "MDATE": _md,
                        },
                        headers={
                            "Referer": "https://www.sse.com.cn/aboutus/publication/monthly/investor/",
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                        },
                        timeout=8,
                    )
                    _body = rr.text.strip()
                    _m = _re.match(r"^[\w$]+\((.*)\)$", _body, _re.S)
                    if _m:
                        _body = _m.group(1)
                    _payload = _json.loads(_body)
                    if _payload.get("result"):
                        break
                if not _payload or not _payload.get("result"):
                    continue
                for _row in _payload.get("result") or []:
                    _term = str(_row.get("TERM", ""))
                    _val = _row.get("A_ACCT")
                    if not _re.match(r"^\d{4}\.\d{2}$", _term) or _val in (None, "", "0.00"):
                        continue  # 跳过合计/累计/未发布月份（0.00）
                    _month = f"{_term[:4]}-{_term[5:7]}"
                    sse_months_set.add(_month)
                    if acc[acc["数据日期"] == _month].empty:
                        new_df = pd.DataFrame(columns=acc.columns.tolist())
                        for col in acc.columns:
                            new_df.at[0, col] = None
                        new_df.at[0, "数据日期"] = _month
                        new_df.at[0, "新增投资者-数量"] = float(_val)
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

        # 判断最新月份的数据源
        latest_month_str = str(acc_tail["数据日期"].iloc[-1])
        latest_is_manual = latest_month_str in manual_months_set
        latest_is_sse = (not latest_is_manual) and (latest_month_str in sse_months_set)

        # 滞后检测：最新月份距今超过 45 天视为滞后
        days_lag = 0
        is_stale = False
        try:
            # 兼容 "2026-06" 和 "2026-06-08" 两种格式
            if len(latest_month_str) == 7:
                latest_date = _dt.datetime.strptime(latest_month_str + "-01", "%Y-%m-%d")
            else:
                latest_date = _dt.datetime.strptime(latest_month_str[:10], "%Y-%m-%d")
            days_lag = (_dt.datetime.now() - latest_date).days
            is_stale = days_lag > 45
        except Exception:
            pass

        result["investor"] = {
            "月份": [str(d) for d in acc_tail["数据日期"].tolist()],
            "新增(万户)": [_safe_float(v) for v in acc_tail["新增投资者-数量"].tolist()],
            "同比": [round(float(v) * 100, 1) if pd.notna(v) else None for v in acc_tail["新增投资者-同比"].tolist()],
            "期末总量(万户)": [_safe_float(v) for v in acc_tail.get("期末投资者-总量", acc_tail["新增投资者-数量"]).tolist()],
            "户均市值(万)": [_safe_float(v) for v in acc_tail.get("沪深户均市值", []).tolist()],
            "数据来源": "2015-2023东方财富 + 2024+上交所API自动更新",
            "最新月份": latest_month_str,
            "最新数据源": "手动维护" if latest_is_manual else ("上交所API" if latest_is_sse else "东方财富"),
            "滞后天数": days_lag,
            "数据滞后": is_stale,
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
        ms = _fetch_m1m2_data()
        if ms is not None and not ms.empty:
            ms = ms[ms["M1"].notna() & ms["M2"].notna()].copy()
            ms = ms.sort_values("月份")
            m1m2_tail = ms.tail(6).copy()
            m1m2_tail["剪刀差"] = (m1m2_tail["M1"] - m1m2_tail["M2"]).round(2)

            months = m1m2_tail["月份"].tolist()
            m1_vals = m1m2_tail["M1"].tolist()
            m2_vals = m1m2_tail["M2"].tolist()
            diff_vals = m1m2_tail["剪刀差"].tolist()

            result["m1m2"] = {
                "月份": months,
                "M1增速": [round(v, 1) for v in m1_vals],
                "M2增速": [round(v, 1) for v in m2_vals],
                "剪刀差": [round(v, 2) for v in diff_vals],
            }
    except Exception:
        pass

    # 5. 巴菲特指数 = 沪深总市值 / GDP（季度级缓存，每季度只更新一次）
    try:
        result["buffett"] = _get_buffett_index()
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
# 扫描机查询服务地址（sepa_query_server，deploy.sh 注册为 8010 常驻）
SCANNER_QUERY_BASE = os.environ.get("SEPA_SCANNER_BASE", "http://192.168.31.43:8010")


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


# 行业板块排行缓存（按天缓存，同一天内不重新请求）
_industry_rank_cache: dict = {}

# 股票整体分析缓存（按天缓存：全市场约5500只，单次拉取约2秒）
_stock_overview_cache: dict = {}


def _latest_trading_date() -> str:
    """返回最近的 A 股交易日（YYYY-MM-DD）。
    周六回退到周五，周日回退到上一个周五；工作日返回当天。
    注：未考虑法定节假日，若节假日访问会保存节假日文件，但不影响历史积累。
    """
    import datetime as _dt
    now = _dt.datetime.now()
    wd = now.weekday()  # 周一=0, 周日=6
    if wd == 5:  # 周六 → 周五
        now = now - _dt.timedelta(days=1)
    elif wd == 6:  # 周日 → 上周五
        now = now - _dt.timedelta(days=2)
    return now.strftime("%Y-%m-%d")


def _fetch_net_inflow_direct() -> float | None:
    """获取沪深两市最新主力净流入合计（元）。
    通过 akshare stock_market_fund_flow 获取（走 push2his 域名）。
    带缓存：成功获取后缓存，东财限流时返回上次缓存值。
    """
    # 缓存：5分钟TTL，失败时返回上次成功值
    cache_key = "net_inflow"
    cache_ts = _net_inflow_cache.get("ts", 0)
    now_ts = time.time()
    if now_ts - cache_ts < 300 and _net_inflow_cache.get("value") is not None:
        return _net_inflow_cache["value"]

    try:
        fund_df = ak.stock_market_fund_flow()
        if fund_df is None or fund_df.empty:
            return _net_inflow_cache.get("value")
        latest = fund_df.iloc[-1]
        nf = latest.get("主力净流入-净额")
        if nf is not None:
            val = round(float(nf), 2)
            _net_inflow_cache["value"] = val
            _net_inflow_cache["ts"] = now_ts
            return val
    except Exception as exc:
        print(f"[net_inflow] akshare 调用失败: {exc}", file=sys.stderr)
    return _net_inflow_cache.get("value")


_net_inflow_cache: dict = {}

# 行业板块历史快照存储目录（每日一份 JSON，用于20天排名计算）
_INDUSTRY_RANK_HISTORY_DIR = Path(__file__).parent / "industry_rank_history"


def _industry_rank_history_dir() -> Path:
    """确保历史快照目录存在并返回。"""
    _INDUSTRY_RANK_HISTORY_DIR.mkdir(exist_ok=True)
    return _INDUSTRY_RANK_HISTORY_DIR


def _save_industry_rank_snapshot(date_str: str, boards: list[dict], source: str) -> None:
    """保存当日板块快照到 industry_rank_history/YYYY-MM-DD.json。"""
    try:
        path = _industry_rank_history_dir() / f"{date_str}.json"
        payload = {
            "date": date_str,
            "source": source,
            "boards": boards,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"[industry_rank] 保存快照失败 {date_str}: {exc}", file=sys.stderr)


def _load_industry_rank_snapshots(days: int = 20) -> list[dict]:
    """加载最近 N 天的板块快照（按日期升序）。
    返回 [{"date": "YYYY-MM-DD", "source": "...", "boards": [...]}, ...]
    """
    try:
        files = sorted(_industry_rank_history_dir().glob("*.json"))
    except Exception:
        return []
    if not files:
        return []
    recent = files[-days:] if len(files) > days else files
    out: list[dict] = []
    for f in recent:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("boards"):
                out.append(d)
        except Exception:
            continue
    return out


def _compute_board_avg_rank(snapshots: list[dict]) -> dict:
    """计算每个板块在给定快照中的平均排名（行业/概念通用）。

    对每个快照按 pct_chg 降序排名（涨幅最高=1），
    板块的平均排名 = 出现天数内名次之和 / 出现天数。
    返回 {board_key: {"avg_rank": float, "days": int, "name": str, "code": str}}
    board_key 用 name 作为主键（新浪/东财代码体系不同，但名称基本一致）。
    """
    if not snapshots:
        return {}
    # 板块名 → 名次列表
    name_ranks: dict[str, list[int]] = {}
    name_meta: dict[str, dict] = {}
    for snap in snapshots:
        boards = snap.get("boards") or []
        # 按 pct_chg 降序排名
        sorted_boards = sorted(boards, key=lambda x: x.get("pct_chg") if x.get("pct_chg") is not None else -1e9, reverse=True)
        for idx, b in enumerate(sorted_boards, start=1):
            name = b.get("name", "").strip()
            if not name:
                continue
            name_ranks.setdefault(name, []).append(idx)
            # 保留最近一次的 code/source 用于链接生成
            name_meta[name] = {
                "name": name,
                "code": b.get("code", ""),
                "source": snap.get("source", ""),
            }
    result: dict[str, dict] = {}
    for name, ranks in name_ranks.items():
        avg = round(sum(ranks) / len(ranks), 2) if ranks else None
        meta = name_meta.get(name, {})
        result[name] = {
            "avg_rank": avg,
            "days": len(ranks),
            "name": name,
            "code": meta.get("code", ""),
            "source": meta.get("source", ""),
        }
    return result


def _fetch_industry_board_rank() -> dict:
    """获取行业板块当日行情并保存快照。

    数据源（双源回退）：
    1. 东方财富 push2 clist 接口（fs=m:90 t:2 f:!50），字段全（含上涨/下跌家数、换手率）
    2. 新浪行业板块接口，字段少但稳定（仅涨跌幅、领涨股），东财被限流时回退

    按天缓存：同一天内只拉取一次，避免频繁请求触发东财限流。
    拉取成功后保存到 industry_rank_history/YYYY-MM-DD.json，用于20天排名计算。
    """
    import requests as _req

    today = _latest_trading_date()
    cached = _industry_rank_cache.get("data")
    cached_date = _industry_rank_cache.get("date", "")
    if cached and cached_date == today:
        return cached

    session = _req.Session()
    session.trust_env = False
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com/",
    })

    # ── 数据源1：东财 push2 ──
    boards = _fetch_industry_rank_eastmoney(session)

    # ── 数据源2：新浪（东财失败时回退）──
    source = "eastmoney"
    if not boards:
        session.headers.update({"Referer": "https://finance.sina.com.cn/"})
        boards = _fetch_industry_rank_sina(session)
        source = "sina"

    if not boards:
        raise RuntimeError("行业板块接口不可用（东财 push2 与新浪均失败）")

    boards.sort(key=lambda x: x["pct_chg"])

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "total": len(boards),
        "boards": boards,
        "date": today,
    }
    _industry_rank_cache["data"] = result
    _industry_rank_cache["date"] = today
    # 保存当日快照（用于20天排名计算）
    _save_industry_rank_snapshot(today, boards, source)
    return result


def _fetch_tencent_div_yields(codes: list) -> dict:
    """从腾讯行情接口批量获取标准TTM股息率（过去12个月已实施分红/现价）。

    口径说明：东财 push2 f133 含未实施分红预案（如南山铝业含"10派2.65"预案时
    显示13.40%），腾讯字段64 仅计已实施分红（南山铝业 8.83%），与主流
    股息率(TTM)口径一致。接口为 GBK 编码，单次请求约支持100只。
    """
    import requests as _req

    session = _req.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    result: dict = {}
    batch_size = 100
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        q = ",".join(("sh" if c.startswith("6") else "sz") + c for c in batch)
        try:
            r = session.get("https://qt.gtimg.cn/q=" + q, timeout=10)
            r.raise_for_status()
            text = r.content.decode("gbk", errors="ignore")
        except Exception:
            continue
        for m in re.finditer(r'v_(?:sh|sz|bj)(\d{6})="([^"]*)"', text):
            parts = m.group(2).split("~")
            if len(parts) > 64 and parts[64]:
                try:
                    result[m.group(1)] = float(parts[64])
                except ValueError:
                    pass
    return result


def _fetch_stock_overview() -> dict:
    """拉取全市场 A 股整体数据（排除北交所）。

    数据源：东方财富 push2 clist 接口（push2delay 延时行情域名优先，
    push2 实时域名常被限流 RemoteDisconnected，作为回退）。

    fs 市场过滤：沪深主板 + 创业板 + 科创板（不含北交所 m:0 t:81 s:2048）。
    字段映射：f12代码 f14名称 f2最新价 f3涨跌幅 f20总市值 f133股息率(含预案口径，
    仅作腾讯接口失败时的回退；主口径为腾讯字段64的标准TTM股息率)。

    按天缓存：全市场约5500只、单次拉取约8秒（含腾讯股息率56个批量请求），
    同一天内直接返回缓存。
    """
    import requests as _req

    today = _latest_trading_date()
    cached = _stock_overview_cache.get("data")
    cached_date = _stock_overview_cache.get("date", "")
    if cached and cached_date == today:
        return cached

    session = _req.Session()
    session.trust_env = False
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com/",
    })

    hosts = [
        "https://push2delay.eastmoney.com",
        "https://82.push2.eastmoney.com",
        "https://push2.eastmoney.com",
    ]
    fields = "f12,f14,f2,f3,f20,f133"
    # 沪深主板+创业板+科创板，天然排除北交所
    fs = "m:1+t:2,m:0+t:6,m:0+t:80,m:1+t:23"

    base_url = None
    for host in hosts:
        try:
            r = session.get(
                f"{host}/api/qt/clist/get",
                params={"pn": 1, "pz": 5, "po": 1, "np": 1,
                        "fltt": 2, "invt": 2, "fid": "f12",
                        "fs": fs, "fields": fields},
                timeout=8,
            )
            if r.status_code == 200 and r.json().get("data"):
                base_url = f"{host}/api/qt/clist/get"
                break
        except Exception:
            continue

    if not base_url:
        raise RuntimeError("股票整体行情接口不可用（push2 各域名均失败）")

    all_rows: list[dict] = []
    page = 1
    page_size = 100  # clist 单页上限100
    total = None
    while True:
        try:
            r = session.get(base_url, params={
                "pn": page, "pz": page_size, "po": 1, "np": 1,
                "fltt": 2, "invt": 2, "fid": "f12",
                "fs": fs, "fields": fields,
            }, timeout=12)
        except Exception:
            break
        if r.status_code != 200:
            break
        data = r.json()
        if not data.get("data"):
            break
        if total is None:
            total = data["data"].get("total", 0)
        diff = data["data"].get("diff") or []
        if not diff:
            break
        for x in diff:
            all_rows.append({
                "code": x.get("f12", ""),
                "name": x.get("f14", ""),
                "price": x.get("f2"),
                "pct_chg": x.get("f3"),
                "market_cap": x.get("f20"),   # 总市值（元）
                "div_yield": x.get("f133"),   # 股息率(TTM, %)
            })
        if len(all_rows) >= (total or 0):
            break
        page += 1

    if not all_rows:
        raise RuntimeError("股票整体行情拉取失败（无数据返回）")

    # 二次保险：过滤北交所代码（4/8/9 开头）
    all_rows = [x for x in all_rows if x["code"][:1] not in ("4", "8", "9")]

    # 股息率改用腾讯TTM口径覆盖：东财 f133 含未实施分红预案，数值偏大
    try:
        tencent_yields = _fetch_tencent_div_yields([x["code"] for x in all_rows])
    except Exception:
        tencent_yields = {}
    for row in all_rows:
        if row["code"] in tencent_yields:
            row["div_yield"] = tencent_yields[row["code"]]

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "eastmoney+tencent_div",
        "date": today,
        "total": len(all_rows),
        "stocks": all_rows,
    }
    _stock_overview_cache["data"] = result
    _stock_overview_cache["date"] = today
    return result


def _fetch_industry_rank_history(days: int = 20, date_filter: str = "", industry_filter: str = "") -> dict:
    """获取最近 N 天行业板块排行数据 + 20天平均排名。

    参数：
        days: 取最近多少个交易日的快照（默认20）
        date_filter: 指定日期 "YYYY-MM-DD"，只返回该日数据；为空返回全部
        industry_filter: 行业名称关键词，模糊匹配过滤

    返回结构：
    {
        "generated_at": "...",
        "days_available": int,        # 实际可用快照数
        "days_requested": int,        # 请求的 days
        "date_filter": str,           # 日期过滤
        "industry_filter": str,       # 行业过滤
        "avg_ranks": { name: {"avg_rank": float, "days": int, "code": str, "source": str} },
        "rows": [                     # 展平后的行（每板块每日一行）
            {"date": "...", "name": "...", "code": "...", "pct_chg": ..., ..., "avg_rank": float}
        ]
    }
    """
    # 触发当日快照拉取（若当天尚未拉取）
    try:
        _fetch_industry_board_rank()
    except Exception as exc:
        print(f"[industry_rank_history] 当日快照拉取失败: {exc}", file=sys.stderr)

    snapshots = _load_industry_rank_snapshots(days)
    if not snapshots:
        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "days_available": 0,
            "days_requested": days,
            "date_filter": date_filter,
            "industry_filter": industry_filter,
            "avg_ranks": {},
            "rows": [],
        }

    avg_ranks = _compute_board_avg_rank(snapshots)

    # 展平为行
    rows: list[dict] = []
    for snap in snapshots:
        snap_date = snap.get("date", "")
        if date_filter and snap_date != date_filter:
            continue
        snap_source = snap.get("source", "")
        for b in snap.get("boards", []):
            name = b.get("name", "").strip()
            if industry_filter and industry_filter not in name:
                continue
            row = dict(b)
            row["date"] = snap_date
            row["source"] = snap_source
            meta = avg_ranks.get(name)
            row["avg_rank"] = meta["avg_rank"] if meta else None
            row["rank_days"] = meta["days"] if meta else 0
            rows.append(row)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "days_available": len(snapshots),
        "days_requested": days,
        "date_filter": date_filter,
        "industry_filter": industry_filter,
        "date_list": [s.get("date", "") for s in snapshots],
        "avg_ranks": avg_ranks,
        "rows": rows,
    }


def _fetch_industry_rank_eastmoney(session, fs: str = "m:90 t:2 f:!50") -> list[dict]:
    """东财 push2 clist 接口获取板块行情。失败返回空列表。

    fs 参数区分板块类型：
      行业板块: m:90 t:2 f:!50
      概念板块: m:90 t:3 f:!50
    """
    hosts = [
        "http://82.push2.eastmoney.com",
        "https://82.push2.eastmoney.com",
        "https://19.push2.eastmoney.com",
        "https://push2.eastmoney.com",
    ]
    fields = "f2,f3,f4,f8,f12,f14,f104,f105,f128,f136"

    base_url = None
    for host in hosts:
        try:
            r = session.get(
                f"{host}/api/qt/clist/get",
                params={"pn": 1, "pz": 10, "po": 1, "np": 1,
                        "fltt": 2, "invt": 2, "fid": "f3",
                        "fs": fs, "fields": fields},
                timeout=8,
            )
            if r.status_code == 200 and r.json().get("data"):
                base_url = f"{host}/api/qt/clist/get"
                break
        except Exception:
            continue

    if not base_url:
        return []

    all_items: list[dict] = []
    page = 1
    page_size = 200
    total = None
    while True:
        try:
            r = session.get(base_url, params={
                "pn": page, "pz": page_size, "po": 1, "np": 1,
                "fltt": 2, "invt": 2, "fid": "f3",
                "fs": fs, "fields": fields,
            }, timeout=12)
        except Exception:
            break
        if r.status_code != 200:
            break
        data = r.json()
        if not data.get("data"):
            break
        if total is None:
            total = data["data"].get("total", 0)
        diff = data["data"].get("diff") or []
        if not diff:
            break
        all_items.extend(diff)
        if len(all_items) >= (total or 0):
            break
        page += 1
        time.sleep(0.1)

    boards: list[dict] = []
    for it in all_items:
        pct = it.get("f3")
        if pct is None or pct == "-":
            continue
        boards.append({
            "code": str(it.get("f12", "")),
            "name": str(it.get("f14", "")),
            "close": it.get("f2"),
            "pct_chg": round(float(pct), 2),
            "chg_amount": it.get("f4"),
            "turnover": it.get("f8"),
            "up_count": it.get("f104", 0),
            "down_count": it.get("f105", 0),
            "leader_name": str(it.get("f128", "")),
            "leader_pct": it.get("f136"),
        })
    return boards


def _fetch_industry_rank_sina(session, url: str = "https://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php") -> list[dict]:
    """新浪板块接口获取行情。东财被限流时的回退数据源。

    可通过 url 参数切换行业板块与概念板块：
      行业板块: https://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php （49个粗分类）
      概念板块: https://money.finance.sina.com.cn/q/view/newFLJK.php?param=class （175个概念）
    新浪无上涨/下跌家数和换手率，但数据稳定。
    字段格式: "code,名称,成分股数,均价,涨跌额,涨跌幅%,成交量,成交额,领涨股代码,领涨股涨幅%,领涨股价,领涨股涨跌额,领涨股名称"
    """
    import json as _json
    import re as _re
    try:
        r = session.get(url, timeout=10)
        if r.status_code != 200:
            return []
        m = _re.search(r"=\s*(\{.*\})", r.text, _re.S)
        if not m:
            return []
        data = _json.loads(m.group(1))
    except Exception:
        return []

    boards: list[dict] = []
    for key, val in data.items():
        f = val.split(",")
        if len(f) < 13:
            continue
        try:
            pct = float(f[5])
        except (ValueError, IndexError):
            continue
        try:
            leader_pct = float(f[9]) if f[9] else None
        except (ValueError, IndexError):
            leader_pct = None
        boards.append({
            "code": str(f[0]),
            "name": str(f[1]),
            "close": None,
            "pct_chg": round(pct, 2),
            "chg_amount": None,
            "turnover": None,
            "up_count": None,
            "down_count": None,
            "leader_name": str(f[12]) if len(f) > 12 else "",
            "leader_pct": round(leader_pct, 2) if leader_pct is not None else None,
        })
    return boards


# ── 概念板块排行（与行业板块并列，独立历史快照） ─────────────────────
_concept_rank_cache: dict = {}

_CONCEPT_RANK_HISTORY_DIR = Path(__file__).parent / "concept_rank_history"


def _concept_rank_history_dir() -> Path:
    """确保概念板块历史快照目录存在并返回。"""
    _CONCEPT_RANK_HISTORY_DIR.mkdir(exist_ok=True)
    return _CONCEPT_RANK_HISTORY_DIR


def _save_concept_rank_snapshot(date_str: str, boards: list[dict], source: str) -> None:
    """保存当日概念板块快照到 concept_rank_history/YYYY-MM-DD.json。"""
    try:
        path = _concept_rank_history_dir() / f"{date_str}.json"
        payload = {
            "date": date_str,
            "source": source,
            "boards": boards,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"[concept_rank] 保存快照失败 {date_str}: {exc}", file=sys.stderr)


def _load_concept_rank_snapshots(days: int = 20) -> list[dict]:
    """加载最近 N 天的概念板块快照（按日期升序）。"""
    try:
        files = sorted(_concept_rank_history_dir().glob("*.json"))
    except Exception:
        return []
    if not files:
        return []
    recent = files[-days:] if len(files) > days else files
    out: list[dict] = []
    for f in recent:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("boards"):
                out.append(d)
        except Exception:
            continue
    return out


def _fetch_concept_board_rank() -> dict:
    """获取概念板块当日行情并保存快照。

    数据源优先级：
      1. 东方财富 push2 clist（fs=m:90 t:3 f:!50）—— 有上涨/下跌家数和换手率
      2. 新浪概念板块（newFLJK.php?param=class）—— 175个概念，无上涨/下跌家数
    按天缓存：同一天内只拉取一次。
    """
    import requests as _req

    today = _latest_trading_date()
    cached = _concept_rank_cache.get("data")
    cached_date = _concept_rank_cache.get("date", "")
    if cached and cached_date == today:
        return cached

    session = _req.Session()
    session.trust_env = False
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com/",
    })

    # 方案1: 东财 push2
    boards = _fetch_industry_rank_eastmoney(session, fs="m:90 t:3 f:!50")
    source = "eastmoney"

    # 方案2: 新浪概念板块 fallback
    if not boards:
        boards = _fetch_industry_rank_sina(
            session,
            url="https://money.finance.sina.com.cn/q/view/newFLJK.php?param=class",
        )
        source = "sina"

    if not boards:
        raise RuntimeError("概念板块接口不可用（东财 push2 与新浪均失败）")

    boards.sort(key=lambda x: x["pct_chg"])

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "total": len(boards),
        "boards": boards,
        "date": today,
    }
    _concept_rank_cache["data"] = result
    _concept_rank_cache["date"] = today
    _save_concept_rank_snapshot(today, boards, source)
    return result


def _fetch_concept_rank_history(days: int = 20, date_filter: str = "", concept_filter: str = "") -> dict:
    """获取最近 N 天概念板块排行数据 + 20天平均排名。

    参数：
        days: 取最近多少个交易日的快照（默认20）
        date_filter: 指定日期 "YYYY-MM-DD"，只返回该日数据；为空返回全部
        concept_filter: 概念名称关键词，模糊匹配过滤
    """
    # 触发当日快照拉取（若当天尚未拉取）
    today_error = ""
    today_source = ""
    try:
        result = _fetch_concept_board_rank()
        today_source = result.get("source", "")
        # 东财失败但新浪 fallback 成功时，提示数据源降级
        if today_source == "sina":
            today_error = "东财 push2 限流，已降级到新浪数据源（无上涨/下跌家数和换手率）"
    except Exception as exc:
        today_error = str(exc)
        print(f"[concept_rank_history] 当日快照拉取失败: {exc}", file=sys.stderr)

    snapshots = _load_concept_rank_snapshots(days)
    if not snapshots:
        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "days_available": 0,
            "days_requested": days,
            "date_filter": date_filter,
            "concept_filter": concept_filter,
            "avg_ranks": {},
            "rows": [],
            "today_fetch_error": today_error,
        }

    avg_ranks = _compute_board_avg_rank(snapshots)

    # 展平为行
    rows: list[dict] = []
    for snap in snapshots:
        snap_date = snap.get("date", "")
        if date_filter and snap_date != date_filter:
            continue
        snap_source = snap.get("source", "")
        for b in snap.get("boards", []):
            name = b.get("name", "").strip()
            if concept_filter and concept_filter not in name:
                continue
            row = dict(b)
            row["date"] = snap_date
            row["source"] = snap_source
            meta = avg_ranks.get(name)
            row["avg_rank"] = meta["avg_rank"] if meta else None
            row["rank_days"] = meta["days"] if meta else 0
            rows.append(row)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "days_available": len(snapshots),
        "days_requested": days,
        "date_filter": date_filter,
        "concept_filter": concept_filter,
        "date_list": [s.get("date", "") for s in snapshots],
        "avg_ranks": avg_ranks,
        "rows": rows,
        "today_fetch_error": today_error,
    }


def fetch_fundamental(code: str, force_refresh: bool = False) -> dict:
    """Fetch fundamental data: profile, financial reports, recent disclosures.
    
    优化7：增加短期缓存（10分钟TTL），避免重复请求同一股票。
    增加 force_refresh 参数，用于强制刷新缓存。
    """
    # 检查缓存
    cache_key = code
    if not force_refresh and cache_key in _fundamental_cache:
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
        balance_sorted = balance_df.sort_values("报告日", ascending=False)
        # 近6期历史提取（货币资金/应收/存货），用于前端风险检验（现金同比、应收占比、存货暴增）
        history_fields = {
            "货币资金": "cash_history",
            "应收账款": "receivables_history",
            "应收票据及应收账款": "receivables_history",
            "存货": "inventory_history",
        }
        for col_name, fund_key in history_fields.items():
            if col_name not in balance_sorted.columns or fund_key in fund:
                continue
            h_dates, h_vals = [], []
            for _, row in balance_sorted.head(6).iterrows():
                try:
                    h_dates.append(str(row["报告日"])[:8])
                    h_vals.append(float(row[col_name]))
                except (ValueError, TypeError):
                    continue
            if len(h_dates) >= 2:
                fund[fund_key] = {"dates": h_dates, "values": h_vals}
        recent = balance_sorted.head(3)
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

    # 4.5 财务摘要（东方财富）：扣非净利润、归母净利润、净现比
    try:
        abstract_df = ak.stock_financial_abstract(symbol=code)
        if abstract_df is not None and not abstract_df.empty:
            date_cols = sorted(
                (c for c in abstract_df.columns if c not in ("选项", "指标")),
                reverse=True,
            )
            use_cols = date_cols[:6]

            def _row_vals(metric: str) -> list:
                row = abstract_df[abstract_df["指标"] == metric]
                if row.empty:
                    return []
                vals = []
                for c in use_cols:
                    try:
                        vals.append(float(row.iloc[0][c]))
                    except (ValueError, TypeError):
                        vals.append(None)
                return vals

            np_parent = _row_vals("归母净利润")
            np_deducted = _row_vals("扣非净利润")
            ocf = _row_vals("经营现金流量净额")
            cash_ratio = _row_vals("经营活动净现金/归属母公司的净利润")
            # 非经常性损益 = 归母净利润 - 扣非净利润
            non_recurring = [
                round(p - d, 2) if (p is not None and d is not None) else None
                for p, d in zip(np_parent, np_deducted)
            ]
            if any(v is not None for v in np_deducted):
                fund["fin_abstract"] = {
                    "dates": use_cols,
                    "np_parent": np_parent,
                    "np_deducted": np_deducted,
                    "non_recurring": non_recurring,
                    "ocf": ocf,
                    "cash_ratio": cash_ratio,
                }
    except Exception:
        pass

    # 5. Recent disclosures (巨潮资讯网)
    try:
        # 动态获取当前日期作为end_date
        today = time.strftime("%Y%m%d")
        disc = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=code, market="沪深京",
            start_date="20260101", end_date=today,
        )
        top5 = disc.head(5)
        fund["disclosures"] = [
            {
                "date": str(row["公告时间"])[:10],
                "title": str(row["公告标题"]),
                "url": str(row["公告链接"]) if "公告链接" in disc.columns else ""
            }
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
        if parsed.path == "/api/sepa/stage2/remote/dates":
            self.handle_stage2_remote("dates")
            return
        if parsed.path == "/api/sepa/stage2/remote/load":
            self.handle_stage2_remote("load", parsed.query)
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
        if parsed.path == "/api/equity_bond_spread":
            self.handle_equity_bond_spread(parsed.query)
            return
        if parsed.path == "/api/market_breadth":
            self.handle_market_breadth()
            return
        if parsed.path == "/api/index_deviation":
            self.handle_index_deviation()
            return
        if parsed.path == "/api/industry":
            self.handle_industry(parsed.query)
            return
        if parsed.path == "/api/industry_map":
            self.handle_industry_map()
            return
        if parsed.path == "/api/stock_lookup":
            self.handle_stock_lookup(parsed.query)
            return
        if parsed.path == "/api/industry_rank":
            self.handle_industry_rank(parsed.query)
            return
        if parsed.path == "/api/concept_rank":
            self.handle_concept_rank(parsed.query)
            return
        if parsed.path == "/api/concepts":
            self.handle_concepts(parsed.query)
            return
        if parsed.path == "/api/stock_overview":
            self.handle_stock_overview(parsed.query)
            return
        if parsed.path == "/api/investor/update":
            self.handle_investor_update(parsed.query)
            return
        super().do_GET()

    def do_POST(self) -> None:
        """处理POST请求（当前仅扫描机上报接口）"""
        parsed = urlparse(self.path)
        if parsed.path == "/api/sepa/stage2/upload":
            self.handle_stage2_upload()
            return
        self.send_error(404, "Not Found")

    def handle_stage2_upload(self) -> None:
        """接收扫描机（局域网另一台机器 sepa_stage2_job.py）上报的 Stage2 候选结果。

        流程：校验 token → JSON 重建 DataFrame → 写服务器 SQLite（历史归档）
        → 原子覆写 sepa_stage2_candidates_test.csv（前端页面/下载链路零改动）→ 覆写 rps_all.csv。
        """
        # 可选 token：环境变量 SEPA_UPLOAD_TOKEN 设置后强制校验
        expected_token = os.environ.get("SEPA_UPLOAD_TOKEN", "")
        if expected_token and self.headers.get("X-Upload-Token") != expected_token:
            self.write_json({"ok": False, "error": "invalid upload token"}, status=403)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 200 * 1024 * 1024:  # 上限 200MB
            self.write_json({"ok": False, "error": "invalid body size"}, status=400)
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except Exception as exc:
            self.write_json({"ok": False, "error": f"invalid JSON: {exc}"}, status=400)
            return

        columns = payload.get("columns") or []
        rows = payload.get("rows") or []
        scan_date = str(payload.get("scan_date") or "")[:10]
        generated_at = str(payload.get("generated_at") or "")[:19]
        if not columns or not isinstance(rows, list) or not scan_date:
            self.write_json({"ok": False, "error": "columns/rows/scan_date required"}, status=400)
            return

        try:
            df = pd.DataFrame(rows, columns=columns)
            if "code" in df.columns:
                df["code"] = df["code"].astype(str).str.zfill(6)
            if not generated_at:
                generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            if "scanned_at" not in df.columns:
                df["scanned_at"] = generated_at

            # 1) 服务器侧 SQLite 历史归档（同 scan_date+code 覆盖）
            from sepa_db import save_candidates
            saved = save_candidates(df, "sepa_stage2.db", scan_date)

            # 2) 原子覆写 CSV：写临时文件后 os.replace，前端 fetch 不会读到半截文件
            csv_path = Path("sepa_stage2_candidates_test.csv")
            tmp_path = csv_path.with_suffix(".csv.tmp")
            df.to_csv(tmp_path, index=False, encoding="utf-8")
            os.replace(tmp_path, csv_path)

            # 3) 覆写 rps_all.csv（RPS 全量缓存，供 /api/sepa 个股评估使用）
            rps_csv = payload.get("rps_csv") or ""
            if rps_csv.strip():
                rps_path = Path("rps_all.csv")
                rps_tmp = rps_path.with_suffix(".csv.tmp")
                rps_tmp.write_text(rps_csv, encoding="utf-8")
                os.replace(rps_tmp, rps_path)

            self.write_json({
                "ok": True,
                "count": len(df),
                "scan_date": scan_date,
                "sqlite_saved": saved,
            })
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"ok": False, "error": str(exc)}, status=500)

    def handle_stage2_remote(self, action: str, query: str = "") -> None:
        """中转代理：浏览器 → 本服务 → 扫描机 sepa_query_server（默认 8010）。

        浏览器直连扫描机会受本机代理（如 Clash）拦截（502/Failed to fetch），
        改由服务端拉取：trust_env=False 绕过代理环境变量，直连局域网扫描机。
        扫描机地址可用环境变量 SEPA_SCANNER_BASE 覆盖。
        """
        url = f"{SCANNER_QUERY_BASE}/api/sepa/dates" if action == "dates" \
            else f"{SCANNER_QUERY_BASE}/api/sepa/data"
        if action == "load":
            date = (parse_qs(query).get("date", [""])[0] or "").strip()[:10]
            if date:
                url += f"?date={date}"
        try:
            import requests as _req
            session = _req.Session()
            session.trust_env = False  # 绕过本机 Clash 等代理，直连局域网扫描机
            r = session.get(url, timeout=15)
            try:
                payload = r.json()
            except ValueError:
                self.write_json({"ok": False,
                                "error": f"扫描机响应非 JSON（HTTP {r.status_code}）"}, status=502)
                return
            self.write_json(payload, status=r.status_code)
        except Exception as exc:  # 连接失败/超时等
            self.write_json({"ok": False,
                            "error": f"无法连接扫描机 {SCANNER_QUERY_BASE}（{exc.__class__.__name__}）。"
                                     "请确认扫描机已部署 sepa_query_server（deploy.sh 自动注册）且在线"},
                            status=502)

    def handle_sepa(self, query: str) -> None:
        """处理SEPA Stage2 股票分析API请求，包含技术面 + 基本面 + 估值分析"""
        params = parse_qs(query)
        code = normalize_code(params.get("code", [""])[0])
        refresh_param = params.get("refresh", [""])[0]
        force_refresh = refresh_param == "1"

        if len(code) != 6:
            self.write_json({"error": "Please input a 6-digit stock code."}, status=400)
            return

        try:
            name = lookup_name(code)
            history = fetch_history(code, min_history_days=220, sleep_seconds=0)
            industry = load_industry_overrides().get(code, "Unknown")
            # 优先使用东财细分行业（三级分类）
            try:
                from industry_analyzer import _ensure_cache
                cache = _ensure_cache()
                stock_info = cache.get("stocks", {}).get(code, {})
                em_sub = stock_info.get("em_sub_industry", "") or stock_info.get("sub_industry", "")
                if em_sub:
                    industry = em_sub
            except Exception:
                pass
            financial_cache = load_cache()
            rps_120 = _estimate_rps(code)
            result = evaluate_stage2(code, name, industry, history, financial_cache=financial_cache, rps_120=rps_120)

            # Convert numpy bools to Python bools for JSON serialization
            if "conditions" in result and isinstance(result["conditions"], dict):
                result["conditions"] = {k: bool(v) for k, v in result["conditions"].items()}
            result["is_stage2"] = bool(result.get("is_stage2", False))

            fundamental = fetch_fundamental(code, force_refresh=force_refresh)

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

            # 五步法估值分析
            try:
                from valuation_analyzer import five_step_valuation

                # 从已有数据构建 fallback（当 akshare API 限流时使用）
                profit_growth_list = json.loads(result.get("profit_growth", "[]"))
                fallback_data = {
                    "name": name,
                    "industry": industry,
                    "pe_ttm": result.get("pe_ttm"),
                    "roe": result.get("roe"),
                    "close_price": close_price,
                }
                # 本地行业缓存中的市值（亿元）兜底
                try:
                    from industry_analyzer import _ensure_cache
                    mc = _ensure_cache().get("stocks", {}).get(code, {}).get("market_cap", 0)
                    if mc:
                        fallback_data["total_mv_yi"] = round(float(mc) / 1e8, 2)
                except Exception:
                    pass
                if profit_growth_list:
                    fallback_data["avg_growth"] = profit_growth_list[-1]

                valuation_result = five_step_valuation(
                    code, safety_margin=0.75, close_price=close_price,
                    fallback_data=fallback_data,
                )
                result["valuation"] = valuation_result
            except Exception as ve:
                traceback.print_exc()
                result["valuation_error"] = str(ve)

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

    def handle_equity_bond_spread(self, query: str = "") -> None:
        """沪深300股债利差（FED 模型）：盈利收益率 - 10年期国债收益率。"""
        params = parse_qs(query)
        force = params.get("force", ["false"])[0].lower() == "true"
        try:
            result = _get_equity_bond_spread(force_refresh=force)
            if not result:
                self.write_json({"error": "数据源暂不可用，请稍后重试"}, status=502)
                return
            result = dict(result)
            result["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"error": str(exc)}, status=500)

    def handle_market_breadth(self) -> None:
        """实时获取市场情绪/恐慌指数/涨跌中位数/总成交额/主力净流入"""
        try:
            from market_breadth import fetch_spot_data, build_market_breadth
            spot = fetch_spot_data()
            result = build_market_breadth(spot)
            # 若 spot 源未提供主力净流入，用东财 ulist.np 接口补充
            if result.get("net_inflow") is None:
                try:
                    result["net_inflow"] = _fetch_net_inflow_direct()
                except Exception as exc:
                    print(f"[market_breadth] 主力净流入补充失败: {exc}", file=sys.stderr)
            # 成功时保存到文件，作为下次失败时的缓存
            try:
                import json as _json
                from pathlib import Path as _Path
                cache_path = _Path(__file__).parent / "market_breadth.json"
                cache_path.write_text(_json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            self.write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"error": str(exc)}, status=500)

    def handle_index_deviation(self) -> None:
        """三大指数负乖离率判断：上证指数、深圳成指、创业板指。

        计算 (收盘价 - MA250) / MA250 × 100%，负值表示价格低于年线。
        阈值：
            上证指数 负乖离率 > 15% → 反弹预警
            深圳成指 负乖离率 > 20% → 反弹预警
            创业板指 负乖离率 > 25% → 反弹预警
        """
        indices = [
            {"code": "000001", "symbol": "sh000001", "name": "上证指数", "threshold": 15},
            {"code": "399001", "symbol": "sz399001", "name": "深圳成指", "threshold": 20},
            {"code": "399006", "symbol": "sz399006", "name": "创业板指", "threshold": 25},
        ]
        results = []
        try:
            for idx in indices:
                try:
                    df = ak.stock_zh_index_daily(symbol=idx["symbol"])
                    if df is None or df.empty or len(df) < 250:
                        results.append({
                            "name": idx["name"],
                            "close": None,
                            "ma250": None,
                            "deviation": None,
                            "alert": False,
                            "error": "数据不足",
                        })
                        continue
                    df["ma250"] = df["close"].rolling(250).mean()
                    latest = df.iloc[-1]
                    close = float(latest["close"])
                    ma250 = float(latest["ma250"])
                    if pd.isna(ma250) or ma250 <= 0:
                        results.append({
                            "name": idx["name"],
                            "close": round(close, 2),
                            "ma250": None,
                            "deviation": None,
                            "alert": False,
                            "error": "MA250 不可用",
                        })
                        continue
                    deviation = round((close - ma250) / ma250 * 100, 2)
                    alert = deviation < 0 and abs(deviation) > idx["threshold"]
                    results.append({
                        "name": idx["name"],
                        "close": round(close, 2),
                        "ma250": round(ma250, 2),
                        "deviation": deviation,
                        "threshold": idx["threshold"],
                        "alert": alert,
                        "alert_msg": f"负乖离率 {abs(deviation):.1f}% 超过阈值 {idx['threshold']}%，注意反弹！" if alert else "",
                    })
                except Exception as e:
                    results.append({
                        "name": idx["name"],
                        "close": None,
                        "ma250": None,
                        "deviation": None,
                        "alert": False,
                        "error": str(e)[:100],
                    })
            self.write_json({"indices": results, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"error": str(exc)}, status=500)

    def handle_industry_rank(self, query: str) -> None:
        """行业板块涨幅排行（最近 N 个交易日，含20天平均排名）。

        查询参数：
            days: 取最近多少个交易日快照（默认20）
            date: 指定日期 "YYYY-MM-DD"，只返回该日数据；为空返回全部
            industry: 行业名称关键词，模糊匹配过滤

        数据源：东方财富 push2 / 新浪行业板块接口（双源回退）。
        返回字段：日期、板块代码、名称、涨跌幅、涨跌额、换手率、上涨/下跌家数、领涨股、20天平均排名。
        """
        params = parse_qs(query)
        days = int(params.get("days", ["20"])[0] or "20")
        date_filter = params.get("date", [""])[0].strip()
        industry_filter = params.get("industry", [""])[0].strip()
        try:
            result = _fetch_industry_rank_history(
                days=days,
                date_filter=date_filter,
                industry_filter=industry_filter,
            )
            self.write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"error": str(exc)}, status=500)

    def handle_concept_rank(self, query: str) -> None:
        """概念板块涨幅排行（最近 N 个交易日，含20天平均排名）。

        查询参数：
            days: 取最近多少个交易日快照（默认20）
            date: 指定日期 "YYYY-MM-DD"，只返回该日数据；为空返回全部
            concept: 概念名称关键词，模糊匹配过滤

        数据源：东方财富 push2（概念板块无新浪回退）。
        返回字段：日期、板块代码、名称、涨跌幅、涨跌额、换手率、上涨/下跌家数、领涨股、20天平均排名。
        """
        params = parse_qs(query)
        days = int(params.get("days", ["20"])[0] or "20")
        date_filter = params.get("date", [""])[0].strip()
        concept_filter = params.get("concept", [""])[0].strip()
        try:
            result = _fetch_concept_rank_history(
                days=days,
                date_filter=date_filter,
                concept_filter=concept_filter,
            )
            self.write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"error": str(exc)}, status=500)

    def handle_stock_overview(self, query: str) -> None:
        """股票整体分析：全市场 A 股（排除北交所）基础行情。

        返回字段：代码、名称、最新价、当日涨跌幅、总市值、股息率(TTM)。
        排序在前端完成；服务端按天缓存，同一天内不重新拉取。
        支持 refresh=1 强制刷新缓存。
        """
        params = parse_qs(query)
        refresh = params.get("refresh", [""])[0] == "1"
        if refresh:
            _stock_overview_cache["data"] = None
            _stock_overview_cache["date"] = ""
        try:
            result = _fetch_stock_overview()
            self.write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"error": str(exc)}, status=500)

    def handle_industry(self, query: str) -> None:
        """处理行业分析 API 请求"""
        params = parse_qs(query)
        code = normalize_code(params.get("code", [""])[0])
        if len(code) != 6:
            self.write_json({"error": "请输入6位股票代码"}, status=400)
            return
        try:
            from industry_analyzer import find_stock_industry
            result = find_stock_industry(code)
            self.write_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"code": code, "error": str(exc)}, status=500)

    def handle_industry_map(self) -> None:
        """返回轻量行业映射: {code: {ths, sub, display, concepts}} 用于前端丰富行业列

        display 优先级: 东财细分行业(em_sub) > ths > sub 组合 > ths_industry
        """
        try:
            from industry_analyzer import _ensure_cache, _classify_sub_industry
            cache = _ensure_cache()
            stocks = cache.get("stocks", {})
            mapping = {}
            for code, s in stocks.items():
                ths = s.get("ths_industry", "")
                sub = s.get("sub_industry", "")
                em_sub = s.get("em_sub_industry", "")
                concepts = s.get("concepts", "")
                if not sub and ths:
                    sub = _classify_sub_industry(ths, s.get("main_business", ""), s.get("name", ""))
                # 东财细分行业优先作为 display
                if em_sub:
                    display = em_sub
                elif sub and ths and sub != ths:
                    display = f"{ths} > {sub}"
                else:
                    display = ths
                if display or concepts:
                    mapping[code] = {
                        "ths": ths,
                        "sub": sub or em_sub,
                        "display": display,
                        "concepts": concepts,
                    }
            self.write_json(mapping)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"error": str(exc)}, status=500)

    def handle_stock_lookup(self, query: str) -> None:
        """模糊搜索股票名称，返回匹配的 [code, name] 列表

        GET /api/stock_lookup?q=药明
        """
        try:
            # query 可能包含 UTF-8 编码的中文，需要正确解码
            from urllib.parse import unquote
            query_decoded = unquote(query, encoding='utf-8')
            params = parse_qs(query_decoded)
            q = params.get("q", [""])[0].strip()
            if not q:
                self.write_json([])
                return
            from industry_analyzer import _ensure_cache
            cache = _ensure_cache()
            stocks = cache.get("stocks", {})
            results = []
            for code, s in stocks.items():
                name = s.get("name", "")
                if name and q in name:
                    results.append([code, name])
            # 限制最多返回 20 条
            results = results[:20]
            self.write_json(results)
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"error": str(exc)}, status=500)

    def handle_concepts(self, query: str) -> None:
        """返回个股真正的概念板块（东财datacenter，BOARD_TYPE is None）

        GET /api/concepts?codes=300373,603061
        返回: {"300373": ["机器人","5G概念",...], "603061": [...]}
        """
        params = parse_qs(query)
        codes_param = params.get("codes", [""])[0]
        codes = [c.strip() for c in codes_param.split(",") if c.strip()]
        if not codes:
            self.write_json({})
            return
        try:
            from industry_analyzer import _em_concept_map
            result = _em_concept_map(codes)
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

    def handle_investor_update(self, query: str) -> None:
        """更新手动维护的投资者开户数据 JSON。

        GET /api/investor/update?month=2026-07&count=286

        - month: 格式 YYYY-MM，必填
        - count: 新增投资者数量（万户），必填，正数
        - 行为：存在该月则覆盖，不存在则按序追加；写回 investor_accounts_manual.json
        - 同时清空宏观缓存，使下次 /api/macro 拉取能读到新数据
        """
        params = parse_qs(query)
        month = params.get("month", [""])[0].strip()
        count_str = params.get("count", [""])[0].strip()

        import re
        if not re.match(r"^\d{4}-\d{2}$", month):
            self.write_json({"ok": False, "error": "month 格式应为 YYYY-MM，例如 2026-07"}, status=400)
            return
        try:
            count = float(count_str)
            if count <= 0:
                raise ValueError("count 必须为正数")
        except ValueError:
            self.write_json({"ok": False, "error": "count 必须为正数（万户）"}, status=400)
            return

        manual_path = Path(__file__).parent / "investor_accounts_manual.json"
        try:
            if manual_path.exists():
                rows = json.loads(manual_path.read_text(encoding="utf-8"))
            else:
                rows = []
            if not isinstance(rows, list):
                rows = []

            # 规范化每条记录
            cleaned = []
            for r in rows:
                if isinstance(r, dict) and "日期" in r and "新增投资者-数量" in r:
                    cleaned.append({"日期": str(r["日期"]), "新增投资者-数量": float(r["新增投资者-数量"])})
            rows = cleaned

            # 更新或追加
            action = "updated"
            existing_idx = next((i for i, r in enumerate(rows) if r["日期"] == month), None)
            new_row = {"日期": month, "新增投资者-数量": round(count, 2)}
            if existing_idx is not None:
                rows[existing_idx] = new_row
                action = "updated"
            else:
                rows.append(new_row)
                action = "added"

            # 按月份升序排序
            rows.sort(key=lambda r: r["日期"])

            manual_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # 清空宏观缓存，使下次拉取反映新数据
            global _macro_cache, _macro_cache_ts
            _macro_cache = None
            _macro_cache_ts = 0

            self.write_json({
                "ok": True,
                "action": action,
                "month": month,
                "count": new_row["新增投资者-数量"],
                "total_records": len(rows),
                "latest_month": rows[-1]["日期"] if rows else None,
            })
        except Exception as exc:
            traceback.print_exc()
            self.write_json({"ok": False, "error": str(exc)}, status=500)

    def write_json(self, payload: dict, status: int = 200) -> None:
        """统一返回JSON格式响应"""
        body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # 客户端已断开，忽略写入错误


def main() -> int:
    """启动HTTP服务主函数"""
    # 强制停止所有残留的扫描 worker 进程（服务器重启 = 全新开始）
    try:
        subprocess.run(["pkill", "-9", "-f", "_scan_worker.py"], timeout=5)
        print("[server] 已停止所有残留扫描 worker 进程")
    except Exception:
        pass

    # Clean up stale scan progress files from previous runs
    for f in ["scan_progress_s1.json", "scan_progress_s2.json", "scan_progress_vb.json", "scan_progress.json"]:
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass
    # 清理残留的分批结果文件
    import glob as _glob
    for f in _glob.glob("batch_results/*.csv"):
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass
    for f in _glob.glob("scan_progress_*.json"):
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass

    # 绑定 0.0.0.0：允许局域网内另一台扫描机（sepa_stage2_job.py）上报数据
    server = ThreadingHTTPServer(("0.0.0.0", PORT), StockHandler)
    print(f"Serving dashboard with API at http://0.0.0.0:{PORT}/stock_dashboard.html")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
