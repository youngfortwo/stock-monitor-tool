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

from sepa_stage2_scanner import evaluate_stage2, fetch_history, load_industry_overrides
from sepa_stage1_scanner import evaluate_stage1
from technical_analyzer import analyze_technical


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
            result = evaluate_stage2(code, name, industry, history)

            # Convert numpy bools to Python bools for JSON serialization
            if "conditions" in result and isinstance(result["conditions"], dict):
                result["conditions"] = {k: bool(v) for k, v in result["conditions"].items()}
            result["is_stage2"] = bool(result.get("is_stage2", False))

            fundamental = fetch_fundamental(code)

            result["fundamental"] = fundamental
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
            result = evaluate_stage1(code, name, industry, history)
            if "conditions" in result and isinstance(result["conditions"], dict):
                result["conditions"] = {k: bool(v) for k, v in result["conditions"].items()}
            result["is_stage1"] = bool(result.get("is_stage1", False))
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
