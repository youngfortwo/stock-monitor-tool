#!/usr/bin/env python3
"""行业分析模块 — 基于 cninfo 行业分类 + 腾讯行情市值。

功能:
1. 查询个股所属行业（cninfo 证监会行业分类）
2. 全市场股票行业分类 + 市值缓存
3. 按总市值排名，输出行业龙1/龙2/龙3
4. 补充同花顺概念板块标签

数据源:
- cninfo (webapi) → 行业分类、主营业务、入选指数
- 腾讯 (qt.gtimg.cn) → 总市值（批量查询）
- 新浪 (stock_zh_a_spot) → 全市场股票代码列表
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import py_mini_racer
import requests

# ─── 常量 ─────────────────────────────────────────────────────────────────────
_CACHE_FILE = Path(__file__).parent / "industry_classification_cache.json"
_CACHE_EXPIRE_DAYS = 7
_CNINFO_WORKERS = 20
_TENCENT_BATCH_SIZE = 80

# ─── 内存缓存 ─────────────────────────────────────────────────────────────────
_industry_map: dict[str, dict] | None = None   # {code: {industry, name, main_business, concepts}}
_market_cap_map: dict[str, float] | None = None  # {code: market_cap_yuan}
_mcode: str | None = None
_session: requests.Session | None = None


def _clear_proxy_env():
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "REQUESTS_CA_BUNDLE"):
        os.environ.pop(key, None)


# ─── cninfo 认证 ─────────────────────────────────────────────────────────────
def _get_mcode() -> str:
    global _mcode
    if _mcode:
        return _mcode
    from akshare.datasets import get_ths_js
    js_code = py_mini_racer.MiniRacer()
    with open(get_ths_js("cninfo.js"), encoding="utf-8") as f:
        js_code.eval(f.read())
    _mcode = js_code.call("getResCode1")
    return _mcode


def _get_session() -> requests.Session:
    global _session
    if _session:
        return _session
    _clear_proxy_env()
    _session = requests.Session()
    _session.headers.update({
        "Accept": "*/*",
        "Host": "webapi.cninfo.com.cn",
        "Accept-Enckey": _get_mcode(),
        "Origin": "https://webapi.cninfo.com.cn",
        "Referer": "https://webapi.cninfo.com.cn/",
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "Mozilla/5.0",
    })
    return _session


# ─── cninfo 查询 ─────────────────────────────────────────────────────────────
def _cninfo_query_one(code: str) -> dict:
    """查询单只股票的 cninfo 信息。"""
    session = _get_session()
    try:
        r = session.get(
            "https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1133",
            params={"scode": code},
            timeout=8,
        )
        data = r.json()
        if data.get("records"):
            rec = data["records"][0]
            return {
                "industry": rec.get("F032V", "") or "",
                "name": rec.get("ASECNAME", "") or "",
                "main_business": rec.get("F015V", "") or "",
                "concepts": rec.get("F044V", "") or "",
            }
    except Exception:
        pass
    return {"industry": "", "name": "", "main_business": "", "concepts": ""}


def _cninfo_batch_query(codes: list[str], progress_cb=None) -> dict[str, dict]:
    """并发批量查询 cninfo 行业分类。"""
    results = {}
    with ThreadPoolExecutor(max_workers=_CNINFO_WORKERS) as pool:
        futures = {pool.submit(_cninfo_query_one, c): c for c in codes}
        done_count = 0
        for f in as_completed(futures):
            done_count += 1
            c = futures[f]
            try:
                results[c] = f.result()
            except Exception:
                results[c] = {"industry": "", "name": "", "main_business": "", "concepts": ""}
            if progress_cb and done_count % 100 == 0:
                progress_cb(done_count, len(codes))
    return results


# ─── 腾讯市值 ─────────────────────────────────────────────────────────────────
def _tencent_market_caps(codes: list[str]) -> dict[str, float]:
    """批量获取总市值（单位：元）。腾讯API返回的市值单位是亿元。"""
    _clear_proxy_env()
    mv_map: dict[str, float] = {}

    # 构造腾讯代码
    qq_codes = []
    code_map = {}  # qq_code -> pure code
    for c in codes:
        c = str(c).zfill(6)
        if c.startswith("6") or c.startswith("9"):
            qq = f"sh{c}"
        else:
            qq = f"sz{c}"
        qq_codes.append(qq)
        code_map[qq] = c

    # 分批请求
    for i in range(0, len(qq_codes), _TENCENT_BATCH_SIZE):
        batch = qq_codes[i: i + _TENCENT_BATCH_SIZE]
        url = f"https://qt.gtimg.cn/q={','.join(batch)}"
        try:
            r = requests.get(url, timeout=10)
            for line in r.text.strip().split(";"):
                line = line.strip()
                if "~" not in line:
                    continue
                parts = line.split("~")
                if len(parts) > 45:
                    pure_code = parts[2]
                    try:
                        mv_yi = float(parts[44]) if parts[44] else 0
                        if mv_yi > 0:
                            mv_map[pure_code] = mv_yi * 1e8  # 亿 → 元
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass

    return mv_map


# ─── 全市场股票列表 ────────────────────────────────────────────────────────────
def _get_all_stock_codes() -> list[str]:
    """获取全市场 A 股代码列表。"""
    _clear_proxy_env()
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot()
        codes = []
        for _, row in df.iterrows():
            raw = str(row["代码"]).strip()
            # 新浪格式: sz000001, sh600000, bj920000
            pure = raw.replace("sz", "").replace("sh", "").replace("bj", "")
            if len(pure) == 6 and pure.isdigit():
                codes.append(pure)
        return codes
    except Exception as e:
        print(f"[industry] 获取全市场股票列表失败: {e}", file=sys.stderr)
        return []


# ─── 缓存管理 ──────────────────────────────────────────────────────────────────
def _load_cache() -> dict | None:
    if _CACHE_FILE.exists():
        try:
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            cache_date = data.get("date", "")
            today = time.strftime("%Y-%m-%d")
            # 检查是否过期
            if cache_date == today:
                return data
            # 检查天数
            try:
                from datetime import datetime
                d1 = datetime.strptime(cache_date, "%Y-%m-%d")
                d2 = datetime.strptime(today, "%Y-%m-%d")
                if (d2 - d1).days < _CACHE_EXPIRE_DAYS:
                    return data
            except Exception:
                pass
        except Exception:
            pass
    return None


def _save_cache(data: dict):
    try:
        _CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        print(f"[industry] 缓存保存失败: {e}", file=sys.stderr)


def _em_sub_industry_batch(codes: list[str]) -> dict[str, str]:
    """批量查询东财 datacenter 三级细分行业。

    通过 RPT_F10_CORETHEME_BOARDTYPE 接口，filter=(BOARD_TYPE="行业") 分页
    获取全市场行业分类。同一股票返回 一级/二级/三级 三条记录（按级别排序），
    取最后一条即最细分级。
    返回: {code: 三级行业名称} 如 {"300373": "分立器件"}
    """
    result: dict[str, str] = {}
    if not codes:
        return result

    _clear_proxy_env()
    codes_set = set(codes)
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    page = 1
    total_pages = 1
    while page <= total_pages:
        params = {
            "sortColumns": "SECURITY_CODE",
            "sortTypes": 1,
            "pageSize": 500,
            "pageNumber": page,
            "reportName": "RPT_F10_CORETHEME_BOARDTYPE",
            "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,BOARD_NAME,BOARD_TYPE",
            "filter": '(BOARD_TYPE="行业")',
        }
        try:
            r = requests.get(url, params=params, timeout=15)
        except Exception as exc:
            print(f"[industry] 东财第 {page} 页查询失败: {exc}", file=sys.stderr)
            time.sleep(0.5)
            page += 1
            continue
        if r.status_code != 200:
            print(f"[industry] 东财第 {page} 页 HTTP {r.status_code}", file=sys.stderr)
            page += 1
            continue
        try:
            data = r.json()
        except Exception:
            page += 1
            continue
        if not data.get("result") or not data["result"].get("data"):
            break
        if page == 1:
            total_pages = data["result"].get("pages", 1)
            print(f"[industry] 东财行业分类共 {total_pages} 页, "
                  f"{data['result'].get('count', 0)} 条", file=sys.stderr)
        for item in data["result"]["data"]:
            code = item.get("SECURITY_CODE", "")
            board_name = item.get("BOARD_NAME", "")
            if code and board_name and code in codes_set:
                # 三级细分行业名称通常比一级/二级更长，取最长的
                if code not in result or len(board_name) > len(result[code]):
                    result[code] = board_name
        page += 1
        time.sleep(0.1)

    return result


def _em_concept_map(codes: list[str]) -> dict[str, list[str]]:
    """查询东财 datacenter 概念板块（并发按股票代码查询）。

    通过 RPT_F10_CORETHEME_BOARDTYPE 接口，filter=(SECURITY_CODE="xxx")，
    获取 BOARD_TYPE 为 None 的概念板块（如"机器人"、"5G概念"、"半导体概念"）。
    过滤掉指数类、地区板块类、风格类等非概念板块。
    返回: {code: [概念1, 概念2, ...]}

    本地文件缓存 1 天，避免每次页面加载都逐个查询东财。
    """
    result: dict[str, list[str]] = {}
    if not codes:
        return result

    # ── 本地文件缓存 ──────────────────────────────────────
    # 概念板块不会每天变化，缓存到 concept_map_cache.json，TTL=1天
    import json as _json
    cache_file = Path(__file__).parent / "concept_map_cache.json"
    _cache_data: dict = {}
    _cache_loaded = False
    _cache_dirty = False

    def _load_cache():
        nonlocal _cache_data, _cache_loaded
        if _cache_loaded:
            return
        _cache_loaded = True
        try:
            if cache_file.exists():
                import time as _time
                age = _time.time() - cache_file.stat().st_mtime
                if age < 86400:  # 1天
                    _cache_data = _json.loads(cache_file.read_text(encoding="utf-8"))
                    if not isinstance(_cache_data, dict):
                        _cache_data = {}
        except Exception:
            _cache_data = {}

    def _save_cache():
        nonlocal _cache_dirty
        if not _cache_dirty:
            return
        try:
            cache_file.write_text(_json.dumps(_cache_data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # 先从缓存读取，过滤出未缓存的代码
    _load_cache()
    need_query: list[str] = []
    for c in codes:
        if c in _cache_data:
            if _cache_data[c]:
                result[c] = list(_cache_data[c])
        else:
            need_query.append(c)

    if not need_query:
        return result

    _clear_proxy_env()
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"

    # 概念板块过滤：排除指数类、地区板块、风格类、市场类
    exclude_keywords = (
        "中证", "国证", "深证", "深成", "上证", "沪股通", "深股通", "创业板综",
        "创业成份", "MSCI", "富时罗素", "融资融券", "板块", "成长", "价值",
        "大盘", "中盘", "小盘", "百元股", "低价股", "中证A", "注册制",
        "标准普尔", "HS300", "茅指数", "昨日", "趋势股", "题材股", "风格",
        "先进制造", "央国企", "国企改革", "深圳特区",
    )

    def is_real_concept(name: str) -> bool:
        if not name:
            return False
        return not any(kw in name for kw in exclude_keywords)

    def _query_one(code: str) -> list[str]:
        params = {
            "sortColumns": "SECURITY_CODE",
            "sortTypes": 1,
            "pageSize": 200,
            "pageNumber": 1,
            "reportName": "RPT_F10_CORETHEME_BOARDTYPE",
            "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,BOARD_NAME,BOARD_TYPE",
            "filter": f'(SECURITY_CODE="{code}")',
        }
        try:
            r = requests.get(url, params=params, timeout=5)
        except Exception:
            return []
        if r.status_code != 200:
            return []
        try:
            data = r.json()
        except Exception:
            return []
        if not data.get("result") or not data["result"].get("data"):
            return []
        concepts = []
        for item in data["result"]["data"]:
            board_name = item.get("BOARD_NAME", "")
            board_type = item.get("BOARD_TYPE", "")
            if board_name and board_type is None and is_real_concept(board_name):
                if board_name not in concepts:
                    concepts.append(board_name)
        return concepts

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {pool.submit(_query_one, c): c for c in need_query}
        for f in as_completed(futures):
            c = futures[f]
            try:
                concepts = f.result()
                if concepts:
                    result[c] = concepts
                # 无论有无结果都写入缓存（空列表也缓存，避免重复查询）
                _cache_data[c] = concepts
                _cache_dirty = True
            except Exception:
                pass

    _save_cache()
    return result


def _build_full_cache(progress_cb=None) -> dict:
    """构建全市场行业分类 + 市值缓存。"""
    print("[industry] 开始构建全市场行业分类缓存...", file=sys.stderr)
    t0 = time.time()

    # 1. 获取全市场股票代码
    codes = _get_all_stock_codes()
    if not codes:
        return {"date": "", "stocks": {}}
    print(f"[industry] 全市场 {len(codes)} 只股票", file=sys.stderr)

    # 2. 并发查询 cninfo 行业分类
    cninfo_data = _cninfo_batch_query(codes, progress_cb)
    print(f"[industry] cninfo 查询完成: {time.time()-t0:.1f}s", file=sys.stderr)

    # 3. 批量查询腾讯市值
    mv_data = _tencent_market_caps(codes)
    print(f"[industry] 市值查询完成: {len(mv_data)} 只", file=sys.stderr)

    # 4. 批量查询东财 datacenter 三级细分行业（覆盖 cninfo 粗粒度行业）
    em_sub_map = _em_sub_industry_batch(codes)
    print(f"[industry] 东财细分行业查询完成: {len(em_sub_map)} 只, "
          f"耗时 {time.time()-t0:.1f}s", file=sys.stderr)

    # 5. 组装缓存
    stocks = {}
    for code in codes:
        info = cninfo_data.get(code, {})
        em_sub = em_sub_map.get(code, "")
        if info.get("industry") or em_sub:
            ths_ind = _classify_ths_industry(
                info.get("main_business", ""), info.get("name", ""))
            sub_ind = _classify_sub_industry(
                ths_ind, info.get("main_business", ""), info.get("name", ""))
            # 东财细分行业优先于关键词分类
            if em_sub:
                sub_ind = em_sub
                if not ths_ind:
                    ths_ind = em_sub
            stocks[code] = {
                "industry": info.get("industry", "") or em_sub,
                "name": info.get("name", ""),
                "main_business": info.get("main_business", ""),
                "concepts": info.get("concepts", ""),
                "market_cap": mv_data.get(code, 0),
                "ths_industry": ths_ind,
                "sub_industry": sub_ind,
                "em_sub_industry": em_sub,
            }

    cache = {"date": time.strftime("%Y-%m-%d"), "stocks": stocks}
    _save_cache(cache)
    print(f"[industry] 全市场缓存构建完成: {len(stocks)} 只有行业数据, "
          f"总耗时 {time.time()-t0:.1f}s", file=sys.stderr)
    return cache


def _ensure_cache() -> dict:
    """确保缓存可用，不可用则重建。"""
    global _industry_map, _market_cap_map

    cache = _load_cache()
    if cache is None:
        cache = _build_full_cache()

    stocks = cache.get("stocks", {})
    if not stocks:
        cache = _build_full_cache()
        stocks = cache.get("stocks", {})

    _industry_map = stocks
    _market_cap_map = {c: s.get("market_cap", 0) for c, s in stocks.items()}
    return cache


# ─── 格式化 ────────────────────────────────────────────────────────────────────
def _fmt_market_cap(cap: float) -> str:
    if cap >= 1e12:
        return f"{cap / 1e12:.2f}万亿"
    if cap >= 1e8:
        return f"{cap / 1e8:.2f}亿"
    if cap >= 1e4:
        return f"{cap / 1e4:.0f}万"
    return f"{cap:.0f}"


# ─── RPS 计算 ────────────────────────────────────────────────────────────────────
def _get_120d_return(code: str) -> float | None:
    """从 daily_cache 读取 120 日前和最新的收盘价，计算收益率(%)。"""
    csv_path = Path(__file__).parent / "daily_cache" / f"{code}.csv"
    if not csv_path.exists():
        return None
    try:
        import csv
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if len(rows) < 120:
            return None
        # 最新价
        latest_close = float(rows[-1]["close"])
        # 120日前
        target_idx = max(0, len(rows) - 121)
        price_120d_ago = float(rows[target_idx]["close"])
        if price_120d_ago <= 0:
            return None
        return (latest_close / price_120d_ago - 1) * 100
    except Exception:
        return None


# ─── 行业5天对比 ──────────────────────────────────────────────────────────────────
# 行业名 → 代表性ETF代码映射（按成交量和代表性优选）
_INDUSTRY_ETF_MAP: dict[str, str] = {
    "有色金属": "512400",   # 有色金属ETF南方
    "工业金属": "512400",
    "贵金属": "159812",
    "小金属": "512400",
    "能源金属": "159610",
    "半导体": "512480",     # 半导体ETF
    "消费电子": "159732",
    "光学光电子": "159951",
    "元件": "159836",
    "银行": "512800",       # 银行ETF
    "证券": "512880",       # 证券ETF
    "保险": "512070",
    "医药生物": "512010",   # 医药ETF（东财一级分类）
    "医药": "512010",
    "医疗器械": "159883",
    "医疗服务": "512010",
    "化学制药": "512010",
    "生物制品": "159837",
    "中药": "159643",
    "白酒": "512690",       # 酒ETF
    "饮料制造": "512690",
    "食品加工制造": "515170",
    "电力": "159611",
    "电网设备": "159775",
    "光伏设备": "515790",   # 光伏ETF
    "风电设备": "516190",
    "电池": "159755",
    "煤炭开采加工": "515220",  # 煤炭ETF
    "煤炭": "515220",
    "钢铁": "515210",       # 钢铁ETF
    "化工": "516220",
    "化学原料": "516220",
    "化学制品": "516220",
    "房地产开发": "512200",
    "房地产": "512200",
    "军工": "512660",       # 军工ETF
    "国防军工": "512660",
    "汽车整车": "516110",
    "汽车零部件": "516110",
    "传媒": "512980",
    "游戏": "159869",
    "通信设备": "515880",
    "通信服务": "515880",
    "计算机": "512720",
    "软件开发": "515230",
    "IT服务": "515230",
    "电子": "159997",
    "机械设备": "516270",
    "通用设备": "516270",
    "专用设备": "516270",
    "家用电器": "159996",
    "家电": "159996",
    "农林牧渔": "159825",
    "农业": "159825",
    "环保": "512580",
    "建筑材料": "159745",
    "建材": "159745",
    "建筑装饰": "159749",
    "交通运输": "159666",
    "物流": "516950",
    "零售": "516630",
    "纺织服饰": "513080",
    "石油开采": "161129",
    "石油石化": "161129",
    "燃气": "159548",
    "房地产服务": "512200",
}


def _get_5d_return_from_cache(code: str) -> tuple[float, float] | None:
    """从 daily_cache 读取最近6个交易日收盘价，计算5日涨跌幅(%)。

    返回 (5日涨跌幅%, 最新收盘价) 或 None。
    """
    csv_path = Path(__file__).parent / "daily_cache" / f"{code}.csv"
    if not csv_path.exists():
        return None
    try:
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 6:
            return None
        # 取最近6个交易日：5天前的close作为基准，今天的close作为最新
        close_5d_ago = float(rows[-6]["close"])
        latest_close = float(rows[-1]["close"])
        if close_5d_ago <= 0:
            return None
        ret = (latest_close / close_5d_ago - 1) * 100
        return (round(ret, 2), latest_close)
    except Exception:
        return None


def _get_etf_5d_return(etf_code: str) -> tuple[float, float, str] | None:
    """获取ETF的5日涨跌幅。

    优先用 fund_etf_hist_em；失败回退到东财push2 K线接口。
    返回 (5日涨跌幅%, 最新价, 数据源) 或 None。
    """
    import datetime as _dt
    end_date = _dt.date.today().strftime("%Y%m%d")
    start_date = (_dt.date.today() - _dt.timedelta(days=15)).strftime("%Y%m%d")

    # 方案1: akshare fund_etf_hist_em
    try:
        import akshare as ak
        df = ak.fund_etf_hist_em(symbol=etf_code, period="daily",
                                  start_date=start_date, end_date=end_date, adjust="")
        if df is not None and len(df) >= 6:
            close_5d_ago = float(df.iloc[-6]["收盘"])
            latest = float(df.iloc[-1]["收盘"])
            if close_5d_ago > 0:
                ret = (latest / close_5d_ago - 1) * 100
                return (round(ret, 2), round(latest, 3), "akshare")
    except Exception as e:
        print(f"[etf_5d] akshare failed for {etf_code}: {e}", file=sys.stderr)

    # 方案2: 东财push2 K线接口
    try:
        import requests
        # ETF的secid前缀：沪市1，深市0
        prefix = "1" if etf_code.startswith("5") else "0"
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": f"{prefix}.{etf_code}",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56",
            "klt": 101, "fqt": 0,
            "beg": start_date, "end": end_date,
        }
        r = requests.get(url, params=params, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0"})
        kdata = r.json().get("data")
        if kdata and kdata.get("klines"):
            klines = kdata["klines"]
            if len(klines) >= 6:
                # 格式: 日期,开,收,高,低,成交量
                parts_5d = klines[-6].split(",")
                parts_now = klines[-1].split(",")
                close_5d_ago = float(parts_5d[2])
                latest = float(parts_now[2])
                if close_5d_ago > 0:
                    ret = (latest / close_5d_ago - 1) * 100
                    return (round(ret, 2), round(latest, 3), "eastmoney")
    except Exception as e:
        print(f"[etf_5d] eastmoney failed for {etf_code}: {e}", file=sys.stderr)

    return None


def _calc_industry_today_comparison(code: str, peer_codes: list[str],
                                     industry_name: str) -> dict:
    """计算个股 vs 行业 vs 行业ETF 的【当日】涨跌对比（实时数据）。

    返回:
        stock_today: 个股当日涨跌幅%
        stock_close: 个股最新价
        industry_today_median: 行业成分股当日涨跌幅中位数%
        industry_today_mean: 行业成分股当日涨跌幅均值%
        industry_up_count: 行业上涨家数
        industry_down_count: 行业下跌家数
        etf_code: 行业ETF代码（若有）
        etf_name: 行业ETF名称
        etf_today: ETF当日涨跌幅%
        etf_close: ETF最新价
        outperform_industry: 个股是否跑赢行业
        outperform_etf: 个股是否跑赢ETF
    """
    result = {
        "stock_today": None, "stock_close": None,
        "industry_today_median": None, "industry_today_mean": None,
        "industry_up_count": 0, "industry_down_count": 0,
        "etf_code": None, "etf_name": "", "etf_today": None, "etf_close": None,
        "outperform_industry": None, "outperform_etf": None,
    }

    # 1. 个股 + 成分股当日涨跌幅（实时快照，30秒缓存复用）
    all_codes = list(dict.fromkeys([code] + peer_codes))
    spot_map = _fetch_peer_spot(all_codes)

    stock_spot = spot_map.get(code)
    if stock_spot:
        if stock_spot["pct_chg"]:
            result["stock_today"] = stock_spot["pct_chg"]
        if stock_spot["close"]:
            result["stock_close"] = stock_spot["close"]

    # 2. 行业成分股当日涨跌（中位数 + 均值）
    peer_pcts = []
    for c in peer_codes:
        s = spot_map.get(c)
        if s and s["pct_chg"]:
            peer_pcts.append(s["pct_chg"])
    if peer_pcts:
        import statistics
        result["industry_today_median"] = round(statistics.median(peer_pcts), 2)
        result["industry_today_mean"] = round(statistics.mean(peer_pcts), 2)
        result["industry_up_count"] = sum(1 for r in peer_pcts if r > 0)
        result["industry_down_count"] = sum(1 for r in peer_pcts if r < 0)

    # 3. 行业ETF 当日涨跌（东财push2实时行情）
    etf_code = _INDUSTRY_ETF_MAP.get(industry_name)
    if etf_code:
        result["etf_code"] = etf_code
        result["etf_name"] = industry_name + "ETF"
        etf_ret = _get_etf_today(etf_code)
        if etf_ret:
            result["etf_today"] = etf_ret[0]
            result["etf_close"] = etf_ret[1]

    # 4. 跑赢判断
    if result["stock_today"] is not None and result["industry_today_median"] is not None:
        result["outperform_industry"] = result["stock_today"] > result["industry_today_median"]
    if result["stock_today"] is not None and result["etf_today"] is not None:
        result["outperform_etf"] = result["stock_today"] > result["etf_today"]

    return result


# ─── 实时行情快照缓存（单次 find_stock_industry 调用内复用） ─────────────────
_spot_snapshot = None        # pd.DataFrame
_spot_snapshot_ts: float = 0.0
_etf_spot_df = None          # pd.DataFrame (akshare fund_etf_spot_em)
_etf_spot_ts: float = 0.0


def _get_spot_snapshot():
    """获取全市场实时快照（30 秒内复用，避免重复网络请求）。"""
    global _spot_snapshot, _spot_snapshot_ts
    now = time.time()
    if _spot_snapshot is not None and (now - _spot_snapshot_ts) < 30:
        return _spot_snapshot
    try:
        from market_breadth import fetch_spot_data
        _spot_snapshot = fetch_spot_data()
        _spot_snapshot_ts = now
    except Exception as e:
        print(f"[industry] fetch_spot_data failed: {e}", file=sys.stderr)
    return _spot_snapshot


def _get_etf_today(etf_code: str) -> tuple[float, float] | None:
    """获取ETF当日涨跌幅和最新价（akshare fund_etf_spot_em，60秒缓存）。
    返回 (pct_chg%, close) 或 None。
    """
    global _etf_spot_df, _etf_spot_ts
    now = time.time()
    if _etf_spot_df is None or (now - _etf_spot_ts) > 60:
        try:
            import akshare as ak
            _etf_spot_df = ak.fund_etf_spot_em()
            _etf_spot_ts = now
        except Exception as e:
            print(f"[etf_today] fund_etf_spot_em failed: {e}", file=sys.stderr)
    if _etf_spot_df is None or _etf_spot_df.empty:
        return None
    try:
        row = _etf_spot_df[_etf_spot_df["代码"].astype(str).str.zfill(6) == etf_code]
        if row.empty:
            return None
        r = row.iloc[0]
        pct = float(r["涨跌幅"])
        close = float(r["最新价"])
        return (round(pct, 2), round(close, 3))
    except Exception:
        return None


def _get_pct_chg_from_cache(code: str) -> float | None:
    """从 daily_cache 计算最近交易日的涨跌幅(%)。

    取最近两个交易日的 close 计算 (今日/昨日 - 1) * 100。
    """
    csv_path = Path(__file__).parent / "daily_cache" / f"{code}.csv"
    if not csv_path.exists():
        return None
    try:
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 2:
            return None
        prev_close = float(rows[-2]["close"])
        latest_close = float(rows[-1]["close"])
        if prev_close <= 0:
            return None
        return round((latest_close / prev_close - 1) * 100, 2)
    except Exception:
        return None


def _fetch_peer_spot(peer_codes: list[str]) -> dict[str, dict]:
    """获取成分股的实时价格和涨跌幅。

    单次调用 fetch_spot_data() 获取全市场行情快照，提取 peer_codes 对应的
    涨跌幅（pct_chg）和最新价（close）。当实时数据缺失时，从 daily_cache 兜底。

    返回: {code: {"close": float, "pct_chg": float}}
    """
    result: dict[str, dict] = {}
    if not peer_codes:
        return result

    # 1. 全市场实时快照（含 涨跌幅；可能含 最新价）——30秒缓存复用
    spot_df = _get_spot_snapshot()

    pct_map: dict[str, float] = {}
    close_map: dict[str, float] = {}
    if spot_df is not None and not spot_df.empty:
        codes_series = spot_df["代码"].astype(str).str.zfill(6)
        pct_series = spot_df["涨跌幅"]
        has_close_col = "最新价" in spot_df.columns
        close_series = spot_df["最新价"] if has_close_col else None
        for i in range(len(spot_df)):
            code = str(codes_series.iloc[i])
            try:
                pct_map[code] = float(pct_series.iloc[i])
            except (ValueError, KeyError, IndexError):
                pass
            if has_close_col and close_series is not None:
                try:
                    cv = float(close_series.iloc[i])
                    if cv > 0:
                        close_map[code] = cv
                except (ValueError, KeyError, IndexError):
                    pass

    # 2. 兜底：从 daily_cache 补 close 和 pct_chg
    for code in peer_codes:
        pct_val = pct_map.get(code)
        close_val = close_map.get(code)
        if close_val is None:
            ret = _get_5d_return_from_cache(code)
            if ret:
                close_val = ret[1]
        if pct_val is None:
            pct_val = _get_pct_chg_from_cache(code)
        if close_val is not None or pct_val is not None:
            result[code] = {
                "close": round(close_val, 2) if close_val is not None else 0,
                "pct_chg": pct_val if pct_val is not None else 0,
            }
    return result


def _calc_industry_rps(codes: list[str]) -> dict[str, float]:
    """计算同行业内各股票的 120 日 RPS 百分位排名 (0-100)。

    返回值: {code: rps_percentile}
    """
    returns_map: dict[str, float] = {}
    for c in codes:
        ret = _get_120d_return(c)
        if ret is not None:
            returns_map[c] = ret

    if len(returns_map) < 2:
        return {}

    # 排序计算百分位
    sorted_codes = sorted(returns_map.keys(), key=lambda c: returns_map[c])
    n = len(sorted_codes)
    rps_map: dict[str, float] = {}
    for i, c in enumerate(sorted_codes):
        rps_map[c] = round(i / (n - 1) * 100, 1)
    return rps_map


# ─── 同花顺行业分类 ─────────────────────────────────────────────────────────────
_THS_KEYWORD_MAP: dict[str, list[str]] = {
    # 电子/半导体
    "半导体": ["半导体", "芯片", "集成电路", "分立器件", "晶圆", "封装测试", "功率器件"],
    "消费电子": ["消费电子", "智能手机", "可穿戴", "TWS"],
    "光学光电子": ["光学", "光电子", "LED", "显示面板", "液晶", "OLED"],
    "元件": ["元件", "电容", "电阻", "PCB", "印制电路", "连接器", "电感"],
    "电子化学品": ["电子化学品", "光刻胶", "蚀刻液", "电子材料"],
    "其他电子": ["电子", "元器件"],
    "军工电子": ["军工电子", "雷达", "电子对抗", "军用通信"],
    # 计算机
    "软件开发": ["软件", "SaaS", "ERP", "信息系统", "软件开发", "平台软件"],
    "IT服务": ["IT服务", "信息技术服务", "系统集成", "IT外包"],
    "计算机设备": ["计算机", "服务器", "PC", "打印机", "存储"],
    # 通信
    "通信设备": ["通信设备", "5G", "基站", "光纤光缆", "光通信"],
    "通信服务": ["通信服务", "电信", "移动网络"],
    # 医药
    "化学制药": ["化学制药", "原料药", "化学药"],
    "生物制品": ["生物制品", "疫苗", "抗体", "血液制品"],
    "中药": ["中药", "中成药", "中草药", "中药材"],
    "医疗器械": ["医疗器械", "影像设备", "诊断设备", "CT"],
    "医疗服务": ["医疗服务", "医院", "体检", "诊所"],
    "医药商业": ["医药商业", "药店", "医药流通", "医药批发"],
    # 食品饮料
    "白酒": ["白酒", "酿酒", "浓香", "酱香"],
    "饮料制造": ["饮料", "牛奶", "乳制品", "矿泉水", "啤酒"],
    "食品加工制造": ["食品加工", "调味品", "烘焙", "休闲食品"],
    # 电力/能源
    "电力": ["电力", "发电", "火电", "水电", "核电", "电力供应"],
    "电网设备": ["电网", "变压器", "输电", "配电", "智能电网"],
    "光伏设备": ["光伏", "太阳能", "光伏电池", "硅片"],
    "风电设备": ["风电", "风力", "风机", "风电叶片"],
    "电池": ["电池", "锂电池", "储能", "蓄电池", "动力电池"],
    "其他电源设备": ["电源", "UPS", "逆变器", "充电桩"],
    "燃气": ["燃气", "天然气", "LNG", "城市燃气"],
    "煤炭开采加工": ["煤炭", "煤矿", "焦煤", "洗煤"],
    "油气开采及服务": ["油气", "石油开采", "油田服务", "钻井"],
    # 汽车
    "汽车整车": ["汽车整车", "乘用车", "商用车", "新能源车"],
    "汽车零部件": ["汽车零部件", "发动机", "变速箱", "底盘"],
    "汽车服务及其他": ["汽车服务", "4S店", "二手车"],
    # 家电
    "白色家电": ["空调", "冰箱", "洗衣机", "白色家电", "家电"],
    "小家电": ["小家电", "厨房电器", "扫地机器人"],
    "黑色家电": ["电视", "音响", "黑色家电"],
    "厨卫电器": ["厨电", "油烟机", "热水器", "集成灶"],
    # 房地产/建筑
    "房地产": ["房地产", "地产开发", "房产", "住宅"],
    "建筑装饰": ["建筑装饰", "装修", "工程施工", "幕墙"],
    "建筑材料": ["建材", "水泥", "玻璃", "陶瓷", "防水材料"],
    # 金融
    "银行": ["银行"],
    "保险": ["保险", "人寿保险", "财产保险"],
    "证券": ["证券", "券商", "证券经纪"],
    "多元金融": ["信托", "期货", "金融租赁"],
    # 军工
    "军工装备": ["军工", "国防", "武器装备", "航空装备", "航天", "船舶制造"],
    # 机械/设备
    "工程机械": ["工程机械", "挖掘机", "起重机", "混凝土"],
    "通用设备": ["通用设备", "泵", "阀门", "压缩机", "轴承"],
    "专用设备": ["专用设备", "半导体设备", "锂电设备", "光伏设备"],
    "自动化设备": ["自动化", "机器人", "工控", "伺服"],
    "环保设备": ["环保设备", "水处理设备", "除尘"],
    # 化工
    "化学原料": ["化学原料", "化工", "纯碱", "烧碱", "特种气体"],
    "化学制品": ["化学制品", "涂料", "日化", "胶粘剂"],
    "化学纤维": ["化学纤维", "化纤", "涤纶", "氨纶"],
    "农化制品": ["农化", "农药", "化肥", "复合肥"],
    # 金属/材料
    "钢铁": ["钢铁", "钢材", "冶炼", "不锈钢"],
    "工业金属": ["工业金属", "铝", "铜", "锌"],
    "小金属": ["小金属", "钨", "锑", "稀土", "钴"],
    "贵金属": ["贵金属", "黄金", "白银"],
    "能源金属": ["能源金属", "锂矿", "锂盐"],
    "金属新材料": ["金属新材料", "合金", "特种钢", "高温合金"],
    # 交通运输
    "港口航运": ["港口", "航运", "船运", "集装箱"],
    "公路铁路运输": ["公路", "运输", "高速公路", "铁路"],
    "机场航运": ["机场", "航空", "航空公司"],
    "物流": ["物流", "快递", "仓储", "供应链管理"],
    # 传媒/互联网
    "文化传媒": ["文化传媒", "出版", "广告"],
    "影视院线": ["影视", "院线", "电影"],
    "游戏": ["游戏", "手游", "端游", "网络游戏"],
    "互联网电商": ["电商", "互联网", "互联网平台"],
    # 消费
    "美容护理": ["美容", "化妆品", "护肤", "医美"],
    "旅游及酒店": ["旅游", "酒店", "景区", "旅行社"],
    "服装家纺": ["服装", "家纺", "服饰", "鞋帽"],
    "家居用品": ["家居", "家具", "卫浴", "定制家居"],
    "零售": ["零售", "商超", "百货", "连锁"],
    "教育": ["教育", "培训", "在线教育"],
    # 其他
    "环境治理": ["环境治理", "污水处理", "固废处理", "垃圾焚烧"],
    "造纸": ["造纸", "纸浆", "包装纸"],
    "包装印刷": ["包装", "印刷", "纸包装"],
    "塑料制品": ["塑料", "塑料薄膜", "管材"],
    "橡胶制品": ["橡胶", "轮胎"],
    "非金属材料": ["非金属", "石墨", "碳纤维", "石英"],
    "农产品加工": ["农产品加工", "粮食", "食用油"],
    "养殖业": ["养殖", "猪", "鸡", "畜牧", "水产养殖"],
    "种植业与林业": ["种植", "林业", "种子", "苗木"],
    "纺织制造": ["纺织", "面料", "棉纺", "织造"],
    "石油加工贸易": ["石油加工", "石化", "炼油"],
    "贸易": ["贸易", "进出口", "外贸"],
    "电机": ["电机", "马达", "电动机"],
    "轨交设备": ["轨交", "铁路装备", "地铁车辆"],
    "综合": ["综合", "多元化"],
    "其他社会服务": ["社会服务"],
}


# ─── 子行业分类（同花顺行业内进一步细分）────────────────────────────────────────
# 格式: { ths_industry: { sub_name: [keywords] } }
_SUB_INDUSTRY_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "半导体": {
        # 封装测试优先匹配，避免封装公司被误分到功率器件
        "封装测试": ["封测", "封装", "封裝", "封装测试"],
        "存储": ["存储", "闪存", "DRAM", "NAND", "EEPROM", "储存", "存储芯片", "存储器"],
        "功率器件": ["功率", "MOSFET", "IGBT", "二极管", "整流"],
        "半导体设备": ["半导体设备", "刻蚀", "薄膜沉积", "离子注入", "检测设备", "光刻机"],
        "半导体材料": ["半导体材料", "硅片", "电子特气", "光刻胶", "抛光液", "靶材"],
        "模拟芯片": ["模拟芯片", "模拟IC", "电源管理", "ADC", "DAC", "运放"],
        "射频": ["射频", "RF", "PA", "射频前端", "滤波器"],
        "晶圆代工": ["晶圆代工", "代工", "晶圆制造", "Foundry"],
        "传感器": ["传感器", "MEMS", "CIS", "图像传感器", "CMOS图像"],
        "SoC设计": ["SoC", "FPGA", "GPU", "ASIC", "芯片设计"],
    },
    # 可扩展其他同花顺行业的子行业
}


def _classify_sub_industry(ths_industry: str, main_business: str, name: str = "") -> str:
    """在同花顺行业内进一步细分子行业。返回空字符串表示无子行业分类。"""
    sub_map = _SUB_INDUSTRY_KEYWORDS.get(ths_industry)
    if not sub_map or not main_business or main_business == "nan":
        return ""
    # 从主营业务匹配
    for sub_name, keywords in sub_map.items():
        if any(kw in main_business for kw in keywords):
            return sub_name
    # 从公司名匹配（兜底）
    for sub_name, keywords in sub_map.items():
        if any(kw in name for kw in keywords):
            return sub_name
    return ""


def _classify_ths_industry(main_business: str, name: str = "") -> str:
    """根据主营业务和公司名，匹配同花顺细分行业。"""
    if not main_business or main_business == "nan":
        return ""

    # 先从主营业务匹配（优先级更高）
    for ths_name, keywords in _THS_KEYWORD_MAP.items():
        if any(kw in main_business for kw in keywords):
            return ths_name

    # 再从公司名匹配（兜底）
    for ths_name, keywords in _THS_KEYWORD_MAP.items():
        if any(kw in name for kw in keywords):
            return ths_name

    return ""


# ─── 主入口 ────────────────────────────────────────────────────────────────────
def find_stock_industry(code: str) -> dict:
    """查询个股所属行业，并返回同行业成分股排名。

    返回:
        {
            "code": "300373",
            "name": "扬杰科技",
            "industry": "计算机、通信和其他电子设备制造业",
            "ths_industry": "半导体",
            "main_business": "...",
            "industry_stock_count": 120,
            "my_rank": 15,
            "industry_stocks": [...],
            "leaders": [...],
            "concepts": [...],
        }
    """
    code = str(code).zfill(6)

    # 确保缓存可用
    _ensure_cache()
    if _industry_map is None:
        return {
            "code": code, "name": code, "industry": "未找到",
            "industry_stocks": [], "concepts": [], "leaders": [],
            "error": "无法构建行业分类缓存",
        }

    # 1. 查找目标股票信息
    target = _industry_map.get(code)
    if not target or not target.get("industry"):
        # 尝试实时查询 cninfo
        info = _cninfo_query_one(code)
        if not info["industry"]:
            return {
                "code": code, "name": info.get("name", code), "industry": "未找到",
                "industry_stocks": [], "concepts": [], "leaders": [],
                "error": f"无法查询到股票 {code} 的行业信息",
            }
        # 获取市值
        mv = _tencent_market_caps([code])
        target = {
            "industry": info["industry"],
            "name": info["name"],
            "main_business": info["main_business"],
            "concepts": info["concepts"],
            "market_cap": mv.get(code, 0),
        }

    target_industry = target["industry"]
    stock_name = target["name"]
    main_biz = target.get("main_business", "")
    concepts_raw = target.get("concepts", "")

    # 2. 确定同花顺细分行业
    ths_target = target.get("ths_industry", "") or _classify_ths_industry(main_biz, stock_name)
    # 2.1 确定子行业
    sub_target = target.get("sub_industry", "") or _classify_sub_industry(ths_target, main_biz, stock_name)
    # 2.2 东财三级细分行业（最权威，覆盖最全，如"有色金属"82只）
    em_target = target.get("em_sub_industry", "")

    # 3. 筛选同行业股票
    # 优先级：em_sub_industry（东财三级，最准）> sub_industry（子行业）> ths_industry（同花顺）
    # em_sub 覆盖完整时直接用它筛选；否则回退到 ths/sub 双匹配逻辑
    em_peers = []
    sub_peers = []
    ths_peers = []
    for c, s in _industry_map.items():
        peer_em = s.get("em_sub_industry", "")
        peer_ths = s.get("ths_industry", "") or _classify_ths_industry(
            s.get("main_business", ""), s.get("name", ""))

        # 优先用 em_sub_industry 匹配（东财权威分类，如"有色金属"覆盖82只）
        if em_target and peer_em == em_target:
            pass  # 命中 em 匹配
        elif ths_target:
            if peer_ths != ths_target:
                continue
        else:
            if s.get("industry") != target["industry"]:
                continue

        mcap = s.get("market_cap", 0)
        if mcap == 0:
            mv = _tencent_market_caps([c])
            mcap = mv.get(c, 0)
        peer_entry = {
            "code": c,
            "name": s.get("name", ""),
            "industry": s.get("industry", ""),
            "market_cap": mcap,
            "market_cap_display": _fmt_market_cap(mcap),
            "sub_industry": s.get("sub_industry", ""),
            "em_sub_industry": peer_em,
            "close": 0,
            "pct_chg": 0,
        }
        ths_peers.append(peer_entry)
        if em_target and peer_em == em_target:
            em_peers.append(peer_entry)
        # 子行业也匹配
        if sub_target:
            peer_sub = s.get("sub_industry", "") or _classify_sub_industry(
                peer_ths, s.get("main_business", ""), s.get("name", ""))
            if peer_sub == sub_target:
                sub_peers.append(peer_entry)

    # 决定使用哪个层级作为同行业池：
    # 优先 em_sub_industry（东财三级，覆盖最全，如"有色金属"82只）
    # 其次子行业（sub_industry >=5 只时用）
    # 最后回退到 ths 同花顺行业
    if em_target and len(em_peers) >= 5:
        peers = em_peers
        active_industry = em_target
        use_em = True
        use_sub = False
    elif sub_target and len(sub_peers) >= 5:
        peers = sub_peers
        active_industry = f"{ths_target} > {sub_target}"
        use_em = False
        use_sub = True
    else:
        peers = ths_peers
        active_industry = ths_target or target["industry"]
        use_em = False
        use_sub = False

    # 4. 计算行业 RPS (120日相对强弱)
    peer_codes = [p["code"] for p in peers]
    rps_map = _calc_industry_rps(peer_codes)
    for p in peers:
        p["rps_120"] = rps_map.get(p["code"])

    # 5. 按 RPS 降序排列（无 RPS 数据的排最后）
    ranked = sorted(
        [s for s in peers if s["market_cap"] > 0],
        key=lambda x: (x["rps_120"] if x["rps_120"] is not None else -1,
                       x["market_cap"]),
        reverse=True,
    )
    for i, s in enumerate(ranked):
        s["rank"] = i + 1

    # 5.5 填充实时行情（close + pct_chg）——单次拉取全市场快照
    peer_spot = _fetch_peer_spot([p["code"] for p in peers])
    for p in peers:
        spot = peer_spot.get(p["code"])
        if spot:
            p["close"] = spot["close"]
            p["pct_chg"] = spot["pct_chg"]

    # 6. 取龙1/龙2/龙3（按 RPS 排名）
    leaders = []
    for i, s in enumerate(ranked[:3]):
        leaders.append({
            "rank": i + 1,
            "code": s["code"],
            "name": s["name"],
            "market_cap": s["market_cap"],
            "market_cap_display": s["market_cap_display"],
            "rps_120": s["rps_120"],
            "close": s["close"],
            "pct_chg": s["pct_chg"],
        })

    # 7. 找到该股排名
    my_rank = None
    for s in ranked:
        if s["code"] == code:
            my_rank = s["rank"]
            break

    # 8. 概念标签（真正的概念板块，从东财datacenter获取，如"机器人"、"5G概念"）
    concepts = []
    try:
        em_concepts = _em_concept_map([code])
        if em_concepts.get(code):
            concepts = em_concepts[code]
    except Exception:
        pass
    # 回退到 cninfo 指数概念（仅在东财查询失败时使用）
    if not concepts and concepts_raw and concepts_raw != "nan":
        concepts = [c.strip() for c in concepts_raw.split(",") if c.strip()][:5]

    # 9. RPS 有数据的股票数量
    rps_count = len([p for p in peers if p.get("rps_120") is not None])

    # 10. 显示行业名（优先 em_sub > 子行业 > 同花顺行业 > 证监会行业）
    if use_em:
        display_industry = em_target
    elif use_sub:
        # 当 ths 和 sub 相同时只显示一个，避免 "X > X" 重复
        display_industry = sub_target if sub_target == ths_target else active_industry
    else:
        display_industry = ths_target if ths_target else target_industry

    # 11. 当日涨跌对比（个股 vs 行业成分股 vs 行业ETF，实时数据）
    comparison_today = _calc_industry_today_comparison(
        code, [p["code"] for p in peers], display_industry
    )

    return {
        "code": code,
        "name": stock_name,
        "industry": target_industry,
        "ths_industry": ths_target,
        "sub_industry": sub_target,
        "display_industry": display_industry,
        "main_business": main_biz if main_biz != "nan" else "",
        "industry_stock_count": len(ranked),
        "rps_count": rps_count,
        "my_rank": my_rank,
        "industry_stocks": ranked,
        "leaders": leaders,
        "concepts": concepts,
        "comparison_today": comparison_today,
    }


def rebuild_cache(progress_cb=None) -> dict:
    """手动触发重建缓存。"""
    global _industry_map, _market_cap_map
    _industry_map = None
    _market_cap_map = None
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()
    return _build_full_cache(progress_cb)
