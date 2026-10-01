#!/usr/bin/env python3
"""房地产数据分析：NBS 月度销售数据 + 70 城二手房价格指数（多城市）。

数据源：
- 国家统计局 monthly 公报「全国房地产市场基本情况」（www.stats.gov.cn/sj/zxfb/），
  正文文本解析出商品房销售面积/销售额、二手房网签面积等累计值与同比。
- akshare macro_china_new_house_price（东方财富，底层为统计局 70 城口径），
  按城市对取数，二手住宅环比累乘成链式指数。

月度公报一年只更新一次内容，因此解析结果持久化到 real_estate_cache.json，
增量抓取：列表里出现新月份才去请求对应文章。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_FILE = PROJECT_ROOT / "real_estate_cache.json"

_UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}

# 代表城市：一线 + 强二线，覆盖东财 70 城接口
CITIES = ["北京", "上海", "深圳", "广州", "杭州", "南京", "武汉", "成都", "天津", "厦门"]

# 链式指数展示窗口（月）；接口历史最早到 2011-01，不足部分按可用长度截断
CHAIN_WINDOW = 240

_LIST_URL = "https://www.stats.gov.cn/sj/zxfb/{page}"
_TITLE_RE = re.compile(r"(20\d\d)年1—(\d{1,2})月份全国房地产市场基本情况")
_HREF_RE = re.compile(
    r"href=['\"]([^'\"]*t\d{8}_\d+\.html)['\"][^>]*>\s*(20\d\d年1—\d{1,2}月份全国房地产市场基本情况)"
)


def _get(url: str, timeout: int = 20) -> str | None:
    try:
        r = requests.get(url, headers=_UA, timeout=timeout)
        if r.status_code != 200:
            return None
        r.encoding = "utf-8"
        return r.text
    except Exception:
        return None


def _load_cache() -> dict[str, Any]:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def _parse_article(html: str) -> dict[str, Any] | None:
    """从公报正文提取累计值与同比（下降为负）。"""
    plain = re.sub(r"<[^>]+>", "", html)
    plain = plain.replace("，", "，").replace("；", ";")

    def _signed(word: str, num: str) -> float:
        return -float(num) if word == "下降" else float(num)

    out: dict[str, Any] = {}
    m = re.search(r"(?:新建)?商品房销售面积([\d.]+)万平方米，?同比(下降|增长)([\d.]+)%", plain)
    if m:
        out["sales_area_wan"] = float(m.group(1))
        out["sales_area_yoy"] = _signed(m.group(2), m.group(3))
    m = re.search(r"(?:新建)?商品房销售额([\d.]+)亿元，?(?:同比)?(下降|增长)([\d.]+)%", plain)
    if m:
        out["sales_value_yi"] = float(m.group(1))
        out["sales_value_yoy"] = _signed(m.group(2), m.group(3))
    m = re.search(r"二手房交易网签面积[^。]{0,20}?([\d.]+)万平方米，?同比(增长|下降)([\d.]+)%", plain)
    if m:
        out["secondhand_area_wan"] = float(m.group(1))
        out["secondhand_area_yoy"] = _signed(m.group(2), m.group(3))
    m = re.search(r"(?:房屋)新开工面积([\d.]+)万平方米，?(?:同比)?(下降|增长)([\d.]+)%", plain)
    if m:
        out["newstart_yoy"] = _signed(m.group(2), m.group(3))
    m = re.search(r"房屋竣工面积([\d.]+)万平方米，?(?:同比)?(下降|增长)([\d.]+)%", plain)
    if m:
        out["completion_yoy"] = _signed(m.group(2), m.group(3))
    m = re.search(r"房地产开发投资([\d.]+)亿元，?(?:同比)?(下降|增长)([\d.]+)%", plain)
    if m:
        out["invest_yoy"] = _signed(m.group(2), m.group(3))
    return out if "sales_area_yoy" in out else None


def _discover_releases(max_pages: int = 30) -> dict[str, str]:
    """扫描公报列表页，返回 {YYYY-MM: 文章绝对URL}。"""
    found: dict[str, str] = {}
    for i in range(max_pages):
        page = "index.html" if i == 0 else f"index_{i}.html"
        html = _get(_LIST_URL.format(page=page))
        if not html:
            break
        hits = list(_HREF_RE.finditer(html))
        if not hits:
            continue
        newest_seen = False
        for hit in hits:
            href, title = hit.group(1), hit.group(2)
            m = _TITLE_RE.search(title)
            if not m:
                continue
            key = f"{m.group(1)}-{int(m.group(2)):02d}"
            if key not in found:
                newest_seen = True
                url = href if href.startswith("http") else \
                    "https://www.stats.gov.cn/sj/zxfb/" + href.lstrip("./")
                found[key] = url
        # 列表按时间倒序，若整页没有新月份可提前结束
        if not newest_seen and i > 3:
            break
    return found


def fetch_nbs_monthly(force: bool = False) -> dict[str, Any]:
    """增量抓取 NBS 月度公报，返回 {YYYY-MM: 指标dict}（含历史缓存）。"""
    cache = _load_cache()
    nbs: dict[str, Any] = cache.get("nbs") or {}
    releases = _discover_releases()
    changed = False
    for key in sorted(releases, reverse=True):
        if not force and key in nbs:
            continue
        html = _get(releases[key], timeout=25)
        parsed = _parse_article(html) if html else None
        if parsed:
            nbs[key] = parsed
            changed = True
        time.sleep(0.5)
    if changed or not cache.get("nbs"):
        cache["nbs"] = nbs
        _save_cache(cache)
    return nbs


def fetch_city_price_index() -> dict[str, list[tuple[str, float, float]]]:
    """akshare 70 城接口，返回 {city: [(YYYY-MM, 二手环比, 二手同比), ...]}。"""
    import akshare as ak

    rows: dict[str, dict[str, tuple[float, float]]] = {c: {} for c in CITIES}
    for i in range(0, len(CITIES) - 1, 2):
        try:
            df = ak.macro_china_new_house_price(CITIES[i], CITIES[i + 1])
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            city = str(r.get("城市", "")).strip()
            if city not in rows:
                continue
            try:
                mom = float(r["二手住宅价格指数-环比"])
            except (ValueError, TypeError, KeyError):
                continue
            if mom != mom:
                continue
            try:
                yoy = float(r["二手住宅价格指数-同比"])
            except (ValueError, TypeError, KeyError):
                yoy = float("nan")
            month = str(r["日期"])[:7]
            rows[city][month] = (mom, yoy)
    return {c: sorted(v.items()) for c, v in rows.items() if v}


# ── 上海每日网签（搜狐“今日房产”账号日更转载“网上房地产”数据）──
_SH_PROFILE = ("https://mp.sohu.com/profile?xpt="
               "MUJGMDZBRjZENTY3RUNGMDI1OEZCNjY0NURDNzQwOUNAcXEuc29odS5jb20=")
_SH_TITLE_RE = re.compile(r"上海楼市(\d{1,2})月(\d{1,2})日成交量出炉")


def _parse_sh_calendar(html: str) -> dict[str, dict[str, int]]:
    """解析文章内成交日历表，返回 {YYYY-MM-DD: {new, second}}。"""
    m = re.search(r"(20\d{2})\.(\d{2})\.01-", html)
    if not m:
        return {}
    year, month = int(m.group(1)), int(m.group(2))
    out: dict[str, dict[str, int]] = {}
    cell_re = re.compile(
        r"<td>\s*<p><strong><span>(\d{1,2})</span></strong></p>\s*"
        r"<p><strong><span>(\d+|-)</span></strong></p>\s*"
        r"<p><strong><span>(\d+|-)</span></strong></p>\s*</td>"
    )
    for day, new, second in cell_re.findall(html):
        day = int(day)
        entry: dict[str, int] = {}
        if new.isdigit():
            entry["new"] = int(new)
        if second.isdigit():
            entry["second"] = int(second)
        if entry:
            out[f"{year}-{month:02d}-{day:02d}"] = entry
    return out


def fetch_shanghai_daily(force: bool = False) -> dict[str, Any]:
    """找最新一篇日更文章并解析当月日历；结果按天并入磁盘缓存。"""
    cache = _load_cache()
    sh: dict[str, Any] = cache.get("shanghai") or {}
    latest_cached = max(sh) if sh else ""
    html = _get(_SH_PROFILE)
    if not html:
        return sh
    ids: list[str] = []
    for aid in re.findall(r"href=\"//www\.sohu\.com/a/(\d+)_353578", html):
        if aid not in ids:
            ids.append(aid)
    for aid in ids[:12]:
        art = _get(f"https://www.sohu.com/a/{aid}_353578", timeout=25)
        if not art:
            continue
        if not _SH_TITLE_RE.search(art):
            continue
        parsed = _parse_sh_calendar(art)
        newest = max(parsed) if parsed else ""
        if parsed and newest > latest_cached:
            sh.update(parsed)
            cache["shanghai"] = sh
            _save_cache(cache)
        break
    return sh


def _chain_index(series: list[tuple[str, float, float]], window: int) -> dict[str, float]:
    """环比累乘成链式指数，窗口起点归一为 100。"""
    tail = series[-window:]
    if not tail:
        return {}
    level = 100.0
    out: dict[str, float] = {}
    for i, (month, (mom, _yoy)) in enumerate(tail):
        if i > 0:
            level *= mom / 100.0
        out[month] = round(level, 2)
    return out


def analyze_real_estate(force: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ── NBS 月度销售 ──
    try:
        nbs = fetch_nbs_monthly(force=force)
        keys = sorted(nbs)[-24:] if nbs else []
        months = [k for k in keys]
        result["nbs"] = {
            "months": months,
            "sales_area_yoy": [nbs[k].get("sales_area_yoy") for k in months],
            "sales_value_yoy": [nbs[k].get("sales_value_yoy") for k in months],
            "secondhand_area_yoy": [nbs[k].get("secondhand_area_yoy") for k in months],
            "invest_yoy": [nbs[k].get("invest_yoy") for k in months],
            "latest": nbs[keys[-1]] | {"period": keys[-1]} if keys else None,
        }
    except Exception as exc:
        result["nbs"] = {"error": f"国统局公报抓取失败：{exc}"}

    # ── 70 城二手房价格 ──
    try:
        city_data = fetch_city_price_index()
        chains = {c: _chain_index(s, CHAIN_WINDOW) for c, s in city_data.items()}
        all_months = sorted({m for ch in chains.values() for m in ch})
        city_rows = []
        for c, s in city_data.items():
            if not s:
                continue
            month, mom, yoy = s[-1][0], s[-1][1][0], s[-1][1][1]
            ch = chains[c]
            peak = max(ch.values())
            city_rows.append({
                "city": c,
                "month": month,
                "mom": mom,
                "yoy": None if yoy != yoy else yoy,
                "chain_latest": ch.get(month),
                "chain_peak_drawdown": round((ch.get(month, 0) / peak - 1) * 100, 1) if peak else None,
            })
        result["cities"] = {
            "months": all_months,
            "chains": {c: [ch.get(m) for m in all_months] for c, ch in chains.items()},
            "latest": sorted(
                city_rows,
                key=lambda r: (r["yoy"] is not None, r["yoy"] if r["yoy"] is not None else 0),
                reverse=True,
            ),
            "window_months": CHAIN_WINDOW,
        }
    except Exception as exc:
        result["cities"] = {"error": f"70 城价格接口失败：{exc}"}

    # ── 上海每日网签 ──
    try:
        sh = fetch_shanghai_daily(force=force)
        dates = sorted(sh)[-60:] if sh else []
        monthly: dict[str, dict[str, int]] = {}
        for d, v in sh.items():
            mk = d[:7]
            agg = monthly.setdefault(mk, {"new": 0, "second": 0, "days": 0})
            agg["new"] += v.get("new", 0)
            agg["second"] += v.get("second", 0)
            agg["days"] += 1
        result["shanghai"] = {
            "dates": dates,
            "new_daily": [sh[d].get("new") for d in dates],
            "second_daily": [sh[d].get("second") for d in dates],
            "monthly": [
                {"month": k, **v} for k, v in sorted(monthly.items()) if k >= "2025-01"
            ],
            "latest": {"date": max(sh), **sh[max(sh)]} if sh else None,
        }
    except Exception as exc:
        result["shanghai"] = {"error": f"上海每日网签抓取失败：{exc}"}

    # ── 城镇化率（世界银行年度序列，1960 起）──
    try:
        from demographics_analyzer import _fetch_indicator

        urb = _fetch_indicator("SP.URB.TOTL.IN.ZS")
        years = sorted(urb)
        rates = [round(urb[y], 2) for y in years]
        recent = [
            {
                "year": y,
                "rate": round(urb[y], 2),
                "delta": round(urb[y] - urb[y - 1], 2) if y - 1 in urb else None,
            }
            for y in years[-10:]
        ]
        result["urbanization"] = {
            "years": [str(y) for y in years],
            "rates": rates,
            "latest_year": years[-1] if years else None,
            "latest": rates[-1] if rates else None,
            "recent": recent,
            "note": (
                "世界银行/联合国口径的城镇人口占比，与统计局公布的常住口径逐年数值"
                "约差 0.3~1 个百分点（如 2024 年世行 66.1% vs 统计局 67.0%），"
                "世行序列胜在可回溯到 1960 年，用于观察长期趋势。"
            ),
        }
    except Exception as exc:
        result["urbanization"] = {"error": f"城镇化率取数失败：{exc}"}

    return result


if __name__ == "__main__":
    import pprint

    pprint.pprint(analyze_real_estate(force=True))
