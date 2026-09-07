#!/usr/bin/env python3
"""SEPA Stage2 候选股 SQLite 存储层（扫描机与服务器共用）。

设计要点：
- 表 stage2_candidates，主键 (scan_date, code)：同日重跑自动覆盖去重，历史按日累积
- 列按 DataFrame 动态创建，scanner 增删字段无需改本模块（新列 ALTER TABLE 补齐）
- 数值列建为 REAL，其余 TEXT；WAL 模式保证写入与页面读取互不阻塞
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

TABLE = "stage2_candidates"


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _sqlite_type(series: pd.Series) -> str:
    return "REAL" if pd.api.types.is_numeric_dtype(series) else "TEXT"


def save_candidates(df: pd.DataFrame, db_path: str | Path, scan_date: str) -> int:
    """写入一批候选股（同 scan_date 整日快照替换），返回写入行数。

    同一天重跑时先清空该日旧行、再整批写入（同一事务，WAL 下读取方不会看到
    半截状态）：仅 INSERT OR REPLACE 会残留已落选的旧行——同日先后两次扫描
    （如开机一次、晚间一次）的数据会混在同一 scan_date 下，出现价格日期
    不一致的"僵尸行"。空 DataFrame 表示"今日无候选"，同样清空该日。
    """
    if df is None:
        return 0
    if not df.empty and "code" not in df.columns:
        return 0  # 结构异常，不动库里已有数据
    data = df.copy()
    if not data.empty:
        data["code"] = data["code"].astype(str).str.zfill(6)
    data.insert(0, "scan_date", str(scan_date))

    cols = list(data.columns)
    with _connect(db_path) as conn:
        # 表已存在：整日替换（DELETE 与后续 INSERT 同事务提交）
        if conn.execute(
            f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{TABLE}'"
        ).fetchone():
            conn.execute(f'DELETE FROM {TABLE} WHERE scan_date = ?', (str(scan_date),))
        if data.empty:
            return 0
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS {TABLE} ('
            + ", ".join(
                f'"{c}" ' + ("TEXT NOT NULL" if c in ("scan_date", "code") else _sqlite_type(data[c]))
                for c in cols
            )
            + ', PRIMARY KEY ("scan_date", "code"))'
        )
        # scanner 未来新增列：对已有表补齐（ALTER ADD 默认 TEXT，SQLite 动态类型不影响写入）
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({TABLE})")}
        for c in cols:
            if c not in existing:
                conn.execute(f'ALTER TABLE {TABLE} ADD COLUMN "{c}" TEXT')

        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(f'"{c}"' for c in cols)
        conn.executemany(
            f'INSERT OR REPLACE INTO {TABLE} ({col_names}) VALUES ({placeholders})',
            data.where(pd.notna(data), None).values.tolist(),
        )
    return len(data)


def load_candidates(db_path: str | Path, scan_date: str | None = None) -> pd.DataFrame:
    """读取候选股。scan_date=None 时返回最新一个扫描日的数据。"""
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    with _connect(path) as conn:
        if scan_date is None:
            row = conn.execute(
                f"SELECT MAX(scan_date) FROM {TABLE}"
            ).fetchone()
            if not row or not row[0]:
                return pd.DataFrame()
            scan_date = row[0]
        df = pd.read_sql_query(
            f'SELECT * FROM {TABLE} WHERE scan_date = ? ORDER BY CAST(score AS REAL) DESC',
            conn, params=(scan_date,),
        )
    if not df.empty and "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def load_history(db_path: str | Path, code: str | None = None, days: int = 60) -> pd.DataFrame:
    """读取近 N 天历史（可选按代码过滤），用于连续入选等分析。"""
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    sql = (
        f"SELECT * FROM {TABLE} WHERE scan_date >= date('now', ?)"
    )
    params: list = [f"-{int(days)} days"]
    if code:
        sql += " AND code = ?"
        params.append(str(code).zfill(6))
    with _connect(path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def available_dates(db_path: str | Path) -> list[str]:
    """返回所有扫描日期（升序）。"""
    path = Path(db_path)
    if not path.exists():
        return []
    with _connect(path) as conn:
        rows = conn.execute(f"SELECT DISTINCT scan_date FROM {TABLE} ORDER BY scan_date").fetchall()
    return [r[0] for r in rows]
