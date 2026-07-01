"""Polymarket 本地 Parquet 存储访问（DuckDB）。

统一提供 DuckDB 连接（会话时区置 UTC，确保 ``to_timestamp`` 得到 UTC 瞬时）与各层路径。
``daily_aligned`` 为按日分区的清洗层，用 glob 交给 DuckDB 直接查询，不全量载入内存。
"""

from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
POLY_DIR = ROOT / "data" / "polymarket"
CTF_DIR = POLY_DIR / "CTF"
FEATURES_DIR = POLY_DIR / "features"


def connect(threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """新建 DuckDB 连接（会话时区 UTC）。"""
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    if threads:
        con.execute(f"PRAGMA threads={int(threads)}")
    return con


def daily_aligned_glob() -> str:
    """``daily_aligned`` 层 Parquet glob（供 DuckDB ``read_parquet``）。"""
    return str(POLY_DIR / "daily_aligned" / "*.parquet")
