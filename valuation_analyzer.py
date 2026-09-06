#!/usr/bin/env python3
"""五步法估值分析模块

流程：
  Step 1：定性判断 —— 获取关键指标，判断公司类型标签
  Step 2：推荐估值方法 —— 根据行业/增速/ROE 选择 PE/PB/PS
  Step 3：可比公司估值中位数 —— 同行业前15家
  Step 4：溢价/折价调整 —— 龙头/增速/风险/ROE
  Step 5：安全边际 → 建议买入价
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

# 数据源均为国内站点（东财/新浪），直连即可；本地代理失效时会导致请求被劫持失败
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

warnings.filterwarnings("ignore")

# ─── 缓存 ─────────────────────────────────────────────────────────
_indicator_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_indicator_cache_ttl = 600  # 10分钟

_info_cache: dict[str, tuple[float, dict]] = {}
_info_cache_ttl = 3600  # 1小时

_finance_cache: dict[str, tuple[float, dict]] = {}
_finance_cache_ttl = 3600  # 1小时

# 最新收盘价缓存（新浪日线）
_price_cache: dict[str, tuple[float, float]] = {}
_price_cache_ttl = 600  # 10分钟

# 同行业可比公司估值缓存：{行业名: (timestamp, result)}
_peer_cache: dict[str, tuple[float, dict]] = {}
_peer_cache_ttl = 3600  # 1小时

_CACHE_MAX = 200

# 本地行业分类缓存（避免东财限流时 step3 完全失败）
_local_industry_cache: dict | None = None


def _trim_cache(cache: dict, ttl: float) -> None:
    """清理过期缓存项，控制内存"""
    now = time.time()
    expired = [k for k, (ts, _) in cache.items() if now - ts > ttl]
    for k in expired:
        del cache[k]


def _load_local_industry_cache() -> dict:
    """加载本地行业分类缓存（industry_classification_cache.json）"""
    global _local_industry_cache
    if _local_industry_cache is not None:
        return _local_industry_cache
    import json
    from pathlib import Path
    cache_file = Path(__file__).parent / "industry_classification_cache.json"
    if not cache_file.exists():
        _local_industry_cache = {}
        return _local_industry_cache
    try:
        _local_industry_cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        _local_industry_cache = {}
    return _local_industry_cache


def _get_peers_from_local_cache(industry: str, top_n: int = 15) -> list[dict]:
    """从本地缓存获取同行业股票（按市值降序，取前N家）。

    多字段匹配：industry / sub_industry / em_sub_industry / ths_industry
    匹配策略：完全包含 或 关键词（前4字）命中任一字段
    """
    data = _load_local_industry_cache()
    stocks = data.get("stocks", {})
    if not stocks or not industry:
        return []

    # 提取行业关键词（前4字，过滤掉"制造业"等通用词）
    stop_words = {"制造业", "工业", "业", "其他", "及"}
    keywords = [industry[:4], industry[:2]]
    keywords = [k for k in keywords if k and k not in stop_words]

    peers = []
    for code, info in stocks.items():
        if not isinstance(info, dict):
            continue
        # 收集所有行业相关字段
        fields = [
            info.get("industry", ""),
            info.get("sub_industry", ""),
            info.get("em_sub_industry", ""),
            info.get("ths_industry", ""),
        ]
        fields_text = " ".join(fields)

        # 匹配：完整包含 或 关键词命中
        matched = (industry in fields_text
                   or any(f and f in industry for f in fields)
                   or any(kw in fields_text for kw in keywords))
        if matched:
            peers.append({
                "code": str(code).zfill(6),
                "name": info.get("name", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("market_cap", 0),  # 元
            })

    # 去重 + 按市值降序，取前N
    seen = set()
    unique_peers = []
    for p in peers:
        if p["code"] not in seen:
            seen.add(p["code"])
            unique_peers.append(p)
    unique_peers.sort(key=lambda x: x.get("market_cap", 0), reverse=True)
    return unique_peers[:top_n]


def _get_indicator(code: str) -> pd.DataFrame | None:
    """获取个股估值指标（PE/PB/PS/总市值），带缓存。

    数据源优先级：
      1. 东财 stock_a_indicator_lg（逐个）
      2. 东财 stock_zh_a_spot_em 全市场快照（批量，仅初始化一次）
    """
    _trim_cache(_indicator_cache, _indicator_cache_ttl)
    now = time.time()
    cached = _indicator_cache.get(code)
    if cached and now - cached[0] < _indicator_cache_ttl:
        return cached[1]

    # 方案1: 东财逐个获取
    try:
        import akshare as ak
        df = ak.stock_a_indicator_lg(symbol=code)
        if df is not None and not df.empty:
            _indicator_cache[code] = (now, df)
            return df
    except Exception:
        pass

    # 方案2: 从全市场快照中查（批量，只拉一次）
    spot = _get_market_spot()
    if spot is not None:
        row = spot[spot["代码"].astype(str).str.zfill(6) == code]
        if not row.empty:
            r = row.iloc[0]
            pe_val = r.get("市盈率-动态")
            pb_val = r.get("市净率")
            mv_val = r.get("总市值")
            df_fallback = pd.DataFrame([{
                "trade_date": pd.Timestamp.now().strftime("%Y%m%d"),
                "pe_ttm": float(pe_val) if pd.notna(pe_val) and pe_val != "-" else None,
                "pb": float(pb_val) if pd.notna(pb_val) and pb_val != "-" else None,
                "ps_ttm": None,
                "total_mv": float(mv_val) / 1e4 if pd.notna(mv_val) else None,  # 元→万元
            }])
            _indicator_cache[code] = (now, df_fallback)
            return df_fallback
    return None


# 全市场快照缓存（一次性拉取，避免逐个调用）
_market_spot_cache: pd.DataFrame | None = None
_market_spot_ts: float = 0


def _get_market_spot() -> pd.DataFrame | None:
    """获取全市场行情快照（含 PE/PB/总市值），5分钟缓存"""
    global _market_spot_cache, _market_spot_ts
    now = time.time()
    if _market_spot_cache is not None and now - _market_spot_ts < 300:
        return _market_spot_cache
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            _market_spot_cache = df
            _market_spot_ts = now
            return df
    except Exception:
        pass
    return None


def _get_info(code: str) -> dict:
    """获取个股基本信息（名称/行业），带缓存"""
    _trim_cache(_info_cache, _info_cache_ttl)
    now = time.time()
    cached = _info_cache.get(code)
    if cached and now - cached[0] < _info_cache_ttl:
        return cached[1]
    try:
        import akshare as ak

        info = ak.stock_individual_info_em(symbol=code)
        d = dict(zip(info["item"], info["value"]))
        _info_cache[code] = (now, d)
        return d
    except Exception:
        return {}


def _get_finance(code: str) -> dict | None:
    """获取财务分析指标（ROE/资产负债率/净利润增速/每股净资产），带缓存。

    数据源：东方财富财务摘要 stock_financial_abstract（行式：指标为行、日期为列）。
    新浪 stock_financial_analysis_indicator 已返回空数据，不再使用。
    """
    _trim_cache(_finance_cache, _finance_cache_ttl)
    now = time.time()
    cached = _finance_cache.get(code)
    if cached and now - cached[0] < _finance_cache_ttl:
        return cached[1]
    try:
        import akshare as ak

        df = ak.stock_financial_abstract(symbol=code)
        if df is None or df.empty:
            return None
        date_cols = sorted(
            (c for c in df.columns if c not in ("选项", "指标")), reverse=True
        )
        if not date_cols:
            return None

        def _row_vals(metric: str) -> list:
            row = df[df["指标"] == metric]
            if row.empty:
                return []
            vals = []
            for c in date_cols:
                try:
                    v = pd.to_numeric(row.iloc[0][c], errors="coerce")
                    vals.append(float(v) if pd.notna(v) else None)
                except Exception:
                    vals.append(None)
            return vals

        result = {
            "dates": date_cols,
            "roe_vals": _row_vals("净资产收益率(ROE)"),
            "debt_vals": _row_vals("资产负债率"),
            "growth_vals": _row_vals("归属母公司净利润增长率"),
            "bvps_vals": _row_vals("每股净资产"),  # 每股净资产，用于 PB 计算
        }
        if any(v is not None for v in result["roe_vals"] + result["debt_vals"]):
            _finance_cache[code] = (now, result)
            return result
    except Exception:
        pass
    return None


def _get_latest_price(code: str) -> float | None:
    """获取最新收盘价（新浪日线，直连可用），带缓存"""
    _trim_cache(_price_cache, _price_cache_ttl)
    now = time.time()
    cached = _price_cache.get(code)
    if cached and now - cached[0] < _price_cache_ttl:
        return cached[1]
    try:
        import akshare as ak
        import datetime as dt

        end = dt.date.today().strftime("%Y%m%d")
        start = (dt.date.today() - dt.timedelta(days=15)).strftime("%Y%m%d")
        prefix = "sh" if str(code).startswith(("6", "9")) else (
            "bj" if str(code).startswith(("4", "8")) else "sz"
        )
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", start_date=start, end_date=end)
        if df is not None and not df.empty:
            price = float(df.iloc[-1]["close"])
            _price_cache[code] = (now, price)
            return price
    except Exception:
        pass
    return None


# ============================================================
# Step 1：定性判断 —— 这是家什么公司？
# ============================================================
def step1_qualitative_judge(code: str, fallback: dict | None = None) -> dict:
    """获取关键指标，辅助定性判断公司类型

    Args:
        code: 股票代码
        fallback: 预获取的回退数据 dict，当 akshare API 不可用时使用。
            可包含 name, industry, pe_ttm, pb, roe, avg_growth, debt_ratio,
            total_mv_yi, close_price
    """
    fallback = fallback or {}

    info = _get_info(code)
    name = info.get("股票简称", "") or fallback.get("name", "")
    industry = info.get("行业", "") or fallback.get("industry", "")

    # 估值与盈利指标
    ind_df = _get_indicator(code)
    pe_ttm = pb = ps_ttm = total_mv = None
    if ind_df is not None and not ind_df.empty:
        latest = ind_df.iloc[-1]
        pe_val = latest.get("pe_ttm")
        pb_val = latest.get("pb")
        ps_val = latest.get("ps_ttm")
        mv_val = latest.get("total_mv")
        pe_ttm = float(pe_val) if pd.notna(pe_val) else None
        pb = float(pb_val) if pd.notna(pb_val) else None
        ps_ttm = float(ps_val) if pd.notna(ps_val) else None
        # total_mv 单位是万元，转为亿元
        total_mv = float(mv_val) / 10000 if pd.notna(mv_val) else None

    # 回退到预获取数据
    if pe_ttm is None and fallback.get("pe_ttm") is not None:
        pe_ttm = float(fallback["pe_ttm"])
    if pb is None and fallback.get("pb") is not None:
        pb = float(fallback["pb"])
    if total_mv is None and fallback.get("total_mv_yi") is not None:
        total_mv = float(fallback["total_mv_yi"])

    # 近4期净利润增速（判断成长性）+ ROE + 资产负债率 + 每股净资产
    avg_growth = roe = debt_ratio = None
    fin = _get_finance(code)
    if fin:
        # 净利润同比增长率（近4期均值）
        growth_vals = [v for v in (fin.get("growth_vals") or []) if v is not None]
        if growth_vals:
            avg_growth = float(np.mean(growth_vals[:4]))

        # ROE（最新报告期）
        for v in (fin.get("roe_vals") or []):
            if v is not None:
                roe = float(v)
                break

        # 资产负债率（最新报告期）
        for v in (fin.get("debt_vals") or []):
            if v is not None:
                debt_ratio = float(v)
                break

    # 回退到预获取数据
    if avg_growth is None and fallback.get("avg_growth") is not None:
        avg_growth = float(fallback["avg_growth"])
    if roe is None and fallback.get("roe") is not None:
        roe = float(fallback["roe"])
    if debt_ratio is None and fallback.get("debt_ratio") is not None:
        debt_ratio = float(fallback["debt_ratio"])

    # PB 兜底：最新股价 ÷ 每股净资产（东财 push2 限流时 stock_zh_a_spot_em 不可用）
    if pb is None and fin:
        bvps = next((v for v in (fin.get("bvps_vals") or []) if v is not None), None)
        price = fallback.get("close_price")
        if price is None:
            price = _get_latest_price(code)
        if bvps and price:
            pb = round(float(price) / float(bvps), 2)

    # 判断公司类型标签
    tags = []
    if avg_growth and avg_growth > 20:
        tags.append("成长股")
    elif avg_growth and avg_growth < 5:
        tags.append("价值股")

    if roe and roe > 15:
        tags.append("高ROE")
    elif roe and roe < 5:
        tags.append("低盈利")

    if pe_ttm and pe_ttm > 80:
        tags.append("高估值")
    elif pe_ttm and pe_ttm < 15:
        tags.append("低估值")

    return {
        "name": name,
        "industry": industry,
        "total_mv_yi": round(total_mv, 2) if total_mv else None,
        "pe_ttm": round(pe_ttm, 2) if pe_ttm else None,
        "pb": round(pb, 2) if pb else None,
        "ps_ttm": round(ps_ttm, 2) if ps_ttm else None,
        "roe": round(roe, 2) if roe is not None else None,
        "avg_growth": round(avg_growth, 2) if avg_growth is not None else None,
        "debt_ratio": round(debt_ratio, 2) if debt_ratio is not None else None,
        "tags": tags,
    }


# ============================================================
# Step 2：根据公司特征推荐估值方法
# ============================================================
def step2_pick_valuation_method(q: dict) -> str:
    """自动推荐适用的估值方法"""

    industry = q.get("industry", "")
    growth = q.get("avg_growth") or 0
    roe = q.get("roe") or 0
    pe = q.get("pe_ttm") or 0

    # 金融地产 → PB
    finance_industries = ["银行", "保险", "证券", "房地产", "多元金融"]
    if any(f in industry for f in finance_industries):
        return "PB"

    # 强周期 → PB + 周期位置
    cycle_industries = ["钢铁", "煤炭", "有色", "化工", "石油", "航运", "船舶"]
    if any(c in industry for c in cycle_industries):
        return "PB(周期股)"

    # 高速成长但未盈利 / 盈利波动大 → PS
    if growth > 30 and (pe > 60 or pe == 0):
        return "PS/PEG"

    # 盈利稳定 → PE（默认）
    if growth > 0 and roe > 8:
        return "PE"

    return "PE(参考为主)"


# ============================================================
# Step 3：找可比公司，算估值中位数
# ============================================================
def step3_peer_comparison(code: str, industry: str, method: str) -> dict:
    """获取同行业可比公司，计算估值中位数。

    数据源优先级：
      1. 东财板块成分股 API（stock_board_industry_cons_em）
      2. 本地行业分类缓存（industry_classification_cache.json）作为 fallback
    """

    if not industry:
        return {"error": "行业信息缺失，无法找可比公司"}

    # 检查缓存（同行业1小时内只算一次）
    _trim_cache(_peer_cache, _peer_cache_ttl)
    now = time.time()
    cached = _peer_cache.get(industry)
    cached_result = None
    if cached and now - cached[0] < _peer_cache_ttl:
        cached_result = cached[1]

    if cached_result:
        result = dict(cached_result)
    else:
        peer_codes: list[str] = []
        board_name = industry
        data_source = "东财板块"

        # 方案1: 东财板块成分股
        try:
            import akshare as ak
            board = ak.stock_board_industry_name_em()
            matched = board[board["板块名称"].str.contains(industry[:2], na=False)]
            if len(matched) > 0:
                board_name = matched.iloc[0]["板块名称"]
                stocks = ak.stock_board_industry_cons_em(symbol=board_name)
                if "总市值" in stocks.columns:
                    stocks["总市值"] = pd.to_numeric(stocks["总市值"], errors="coerce")
                    stocks = stocks.dropna(subset=["总市值"]).sort_values("总市值", ascending=False)
                peer_codes = stocks.head(15)["代码"].astype(str).str.zfill(6).tolist()
        except Exception:
            peer_codes = []  # 东财失败，走 fallback

        # 方案2: 本地缓存 fallback
        if not peer_codes:
            # 先查本地缓存中该股票自身的行业（避免东财与证监会行业名不一致）
            local_data = _load_local_industry_cache()
            stock_info = local_data.get("stocks", {}).get(code, {})
            local_industry = ""
            if isinstance(stock_info, dict):
                # 优先用证监会行业，其次东财子行业
                local_industry = (stock_info.get("industry", "")
                                  or stock_info.get("em_sub_industry", "")
                                  or stock_info.get("sub_industry", ""))

            # 用本地行业名或东财行业名找 peers
            search_industry = local_industry or industry
            local_peers = _get_peers_from_local_cache(search_industry, top_n=15)
            if local_peers:
                peer_codes = [p["code"] for p in local_peers]
                board_name = search_industry
                data_source = "本地缓存"
            else:
                return {"error": f"东财限流且本地缓存无匹配行业: {industry}（本地:{local_industry}）"}

        pe_list, pb_list, ps_list, mv_list = [], [], [], []

        for peer_code in peer_codes:
            try:
                ind = _get_indicator(peer_code)
                if ind is None or ind.empty:
                    continue
                latest = ind.iloc[-1]

                pe = latest.get("pe_ttm")
                pb = latest.get("pb")
                ps = latest.get("ps_ttm")
                mv = latest.get("total_mv")

                if pd.notna(pe) and 0 < float(pe) < 200:
                    pe_list.append(float(pe))
                if pd.notna(pb) and 0 < float(pb) < 50:
                    pb_list.append(float(pb))
                if pd.notna(ps) and 0 < float(ps) < 100:
                    ps_list.append(float(ps))
                if pd.notna(mv):
                    mv_list.append(float(mv) / 10000)  # 万元→亿元
            except Exception:
                continue

        # 如果东财估值指标也拉不到（全面限流），用本地缓存的市值作为补充
        if not mv_list and data_source == "本地缓存":
            local_peers = _get_peers_from_local_cache(industry, top_n=15)
            mv_list = [p["market_cap"] / 1e8 for p in local_peers if p["market_cap"] > 0]

        result = {
            "peer_industry": board_name,
            "peer_count": max(len(pe_list), len(mv_list)),
            "pe_median": round(float(np.median(pe_list)), 2) if pe_list else None,
            "pb_median": round(float(np.median(pb_list)), 2) if pb_list else None,
            "ps_median": round(float(np.median(ps_list)), 2) if ps_list else None,
            "avg_mv_yi": round(float(np.mean(mv_list)), 2) if mv_list else None,
            "data_source": data_source,
        }

        _peer_cache[industry] = (now, dict(result))

    # 选对应方法的中位数作为基准估值
    # 注意匹配顺序：PS → PB → PE，避免 "PEG" 里的 "PE" 误匹配
    if "PS" in method:
        result["base_valuation"] = result.get("ps_median")
        result["metric"] = "PS"
    elif "PB" in method:
        result["base_valuation"] = result.get("pb_median")
        result["metric"] = "PB"
    else:
        result["base_valuation"] = result.get("pe_median")
        result["metric"] = "PE"

    return result


# ============================================================
# Step 4：溢价 / 折价调整
# ============================================================
def step4_adjust_premium(q: dict, peer: dict) -> dict:
    """根据龙头地位、增速、风险给出溢价/折价系数"""

    premium = 1.0
    adjustments = []

    # 1. 龙头溢价：市值超过行业均值的2倍 → 龙头溢价25%
    avg_mv = peer.get("avg_mv_yi", 0) or 0
    stock_mv = q.get("total_mv_yi") or 0
    if avg_mv and stock_mv > avg_mv * 2:
        premium *= 1.25
        adjustments.append("龙头地位 +25%")
    elif avg_mv and stock_mv > avg_mv * 1.5:
        premium *= 1.15
        adjustments.append("细分龙头 +15%")

    # 2. 增速溢价
    growth = q.get("avg_growth") or 0
    if growth > 30:
        premium *= 1.20
        adjustments.append("高增速(>30%) +20%")
    elif growth > 20:
        premium *= 1.10
        adjustments.append("中高增速(>20%) +10%")
    elif growth < 5:
        premium *= 0.90
        adjustments.append("低增速(<5%) -10%")

    # 3. 风险折价：负债率过高
    debt_ratio = q.get("debt_ratio") or 0
    if debt_ratio and debt_ratio > 70:
        premium *= 0.85
        adjustments.append("高负债风险 -15%")
    elif debt_ratio and debt_ratio > 50:
        premium *= 0.95
        adjustments.append("负债偏高 -5%")

    # 4. ROE折价/溢价
    roe = q.get("roe") or 0
    if roe > 20:
        premium *= 1.10
        adjustments.append("高ROE(>20%) +10%")
    elif roe < 8:
        premium *= 0.90
        adjustments.append("低ROE(<8%) -10%")

    base = peer.get("base_valuation")
    adjusted_val = base * premium if base else None

    return {
        "premium": round(premium, 3),
        "adjustments": adjustments,
        "fair_valuation": round(adjusted_val, 2) if adjusted_val else None,
        "metric": peer.get("metric", ""),
    }


# ============================================================
# Step 5：安全边际 → 最终买入价
# ============================================================
def step5_safety_margin(
    code: str,
    adjusted: dict,
    close_price: float | None = None,
    margin: float = 0.75,
) -> dict:
    """计算安全边际后的买入价格"""

    ind_df = _get_indicator(code)
    pe_ttm = pb = ps_ttm = None
    if ind_df is not None and not ind_df.empty:
        latest = ind_df.iloc[-1]
        pe_val = latest.get("pe_ttm")
        pb_val = latest.get("pb")
        ps_val = latest.get("ps_ttm")
        pe_ttm = float(pe_val) if pd.notna(pe_val) else None
        pb = float(pb_val) if pd.notna(pb_val) else None
        ps_ttm = float(ps_val) if pd.notna(ps_val) else None

    # 获取当前股价
    current_price = close_price
    if current_price is None:
        try:
            import akshare as ak

            spot = ak.stock_zh_a_spot_em()
            row = spot[spot["代码"] == code]
            if not row.empty:
                current_price = float(row.iloc[0]["最新价"])
        except Exception:
            pass

    if current_price is None or current_price <= 0:
        return {"error": "无法获取当前股价"}

    metric = adjusted.get("metric", "")
    fair_metric = adjusted.get("fair_valuation")

    # 根据估值指标倒推合理股价
    fair_price = None
    if fair_metric and fair_metric > 0:
        if metric == "PE" and pe_ttm and pe_ttm > 0:
            eps = current_price / pe_ttm
            fair_price = eps * fair_metric
        elif metric == "PB" and pb and pb > 0:
            bvps = current_price / pb
            fair_price = bvps * fair_metric
        elif metric == "PS" and ps_ttm and ps_ttm > 0:
            sps = current_price / ps_ttm
            fair_price = sps * fair_metric

    buy_price = fair_price * margin if fair_price else None

    premium_rate = None
    if fair_price and fair_price > 0:
        premium_rate = round((current_price / fair_price - 1) * 100, 1)

    return {
        "current_price": round(current_price, 2),
        "fair_price": round(fair_price, 2) if fair_price else None,
        "margin": f"{int(margin * 100)}折",
        "buy_price": round(buy_price, 2) if buy_price else None,
        "premium_rate": premium_rate,
    }


# ============================================================
# 主流程：五步法一条龙
# ============================================================
def five_step_valuation(
    code: str,
    safety_margin: float = 0.75,
    close_price: float | None = None,
    fallback_data: dict | None = None,
) -> dict:
    """完整五步法估值流水线

    Args:
        code: 股票代码（6位数字）
        safety_margin: 安全边际折扣（0.75 = 75折，留25%安全边际）
        close_price: 当前股价（可选，不传则自动获取）
        fallback_data: 预获取的回退数据 dict，当 akshare API 不可用时使用。
            可包含 name, industry, pe_ttm, pb, roe, avg_growth, debt_ratio,
            total_mv_yi, close_price

    Returns:
        dict: 包含 step1 ~ step5 的完整估值结果
    """
    code = str(code).zfill(6)
    fallback_data = fallback_data or {}

    # 将 close_price 也放入 fallback，供 step5 使用
    if close_price is not None:
        fallback_data = {**fallback_data, "close_price": close_price}

    # Step 1：定性判断
    q = step1_qualitative_judge(code, fallback=fallback_data)

    # Step 2：推荐估值方法
    method = step2_pick_valuation_method(q)

    # Step 3：可比公司估值中位数
    peer = step3_peer_comparison(code, q.get("industry", ""), method)

    if "error" in peer:
        return {
            "code": code,
            "step1": q,
            "step2_method": method,
            "step3_error": peer["error"],
        }

    # Step 4：溢价/折价调整
    adj = step4_adjust_premium(q, peer)

    # Step 5：安全边际 → 买入价
    final = step5_safety_margin(code, adj, close_price, safety_margin)

    return {
        "code": code,
        "step1": q,
        "step2_method": method,
        "step3": peer,
        "step4": adj,
        "step5": final,
    }


# 兼容旧接口：保留 analyze_valuation 入口，内部转调五步法
def analyze_valuation(code: str, **kwargs) -> dict:
    """兼容旧接口，转调 five_step_valuation

    旧参数（name/industry/pe_ttm/pb/roe 等）作为 fallback_data 传入，
    当 akshare API 不可用时作为回退数据使用。
    """
    close_price = kwargs.get("close_price")
    safety_margin = kwargs.get("safety_margin", 0.75)
    # 从 kwargs 提取有价值的回退数据
    fallback_keys = ("name", "industry", "pe_ttm", "pb", "roe", "avg_growth",
                     "debt_ratio", "total_mv_yi", "profit_growth", "rev_growth")
    fallback_data = {k: v for k, v in kwargs.items() if k in fallback_keys and v is not None}
    # profit_growth 映射到 avg_growth
    if "profit_growth" in fallback_data and "avg_growth" not in fallback_data:
        fallback_data["avg_growth"] = fallback_data.pop("profit_growth")
    return five_step_valuation(code, safety_margin=safety_margin,
                                close_price=close_price, fallback_data=fallback_data)


if __name__ == "__main__":
    import json

    result = five_step_valuation("600519", safety_margin=0.75)
    print(json.dumps(result, ensure_ascii=False, indent=2))
