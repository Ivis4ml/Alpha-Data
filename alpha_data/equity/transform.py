"""Flat Files 分钟 CSV.gz -> AlphaForge 口径分钟 bar。

口径转换（与 CLAUDE.md 第 4 节一致）：

- ``window_start``（纳秒 epoch，UTC，bar 起始）+60s 得 bar 收盘时刻 -> ``America/New_York`` 本地
  ``ts``；夏令时由时区转换自动处理。据 ``ts`` 拆出 ``trade_date``（``'YYYY-MM-DD'``）与 ``minute``
  （``'HH:MM'``，bar 收盘戳；与 AlphaForge ``core/schema.py`` / ``is_rth`` 一致，RTH 为
  09:30 < 收盘戳 <= 16:00，即 09:31..16:00）。
- ``symbol`` = 大写 ``ticker``（保留原始形态，不去后缀；归一由 AlphaForge 下游 ``normalize_symbol``
  负责）。
- 价格为原始未复权。Flat Files 分钟聚合无 vwap，``trade_count`` 取 ``transactions``；下游
  ``amount`` 退化为 ``close×volume``。

用 DuckDB 直接读取 gzip CSV（含表头），SQL 完成时区转换，出核处理不全量载入内存。
"""

from __future__ import annotations

import duckdb

# RTH（bar 收盘口径）：09:30 < 收盘戳 <= 16:00（即 09:31..16:00）。
_RTH_OPEN = "09:30"
_RTH_CLOSE = "16:00"


def canonical_select(glob: str, *, rth_only: bool = False) -> str:
    """构造把分钟 CSV.gz 转为 AlphaForge 口径列的 SELECT。

    输出列：``symbol, trade_date, minute, ts, open, high, low, close, volume, trade_count``。
    """
    g = glob.replace("'", "''")
    where = ""
    if rth_only:
        where = f"WHERE minute > '{_RTH_OPEN}' AND minute <= '{_RTH_CLOSE}'"
    return f"""
        SELECT symbol, trade_date, minute, ts, open, high, low, close, volume, trade_count
        FROM (
            SELECT upper(ticker)                         AS symbol,
                   strftime(et, '%Y-%m-%d')              AS trade_date,
                   strftime(et, '%H:%M')                 AS minute,
                   et                                    AS ts,
                   open::DOUBLE                          AS open,
                   high::DOUBLE                          AS high,
                   low::DOUBLE                           AS low,
                   close::DOUBLE                         AS close,
                   CAST(round(volume) AS BIGINT)         AS volume,
                   CAST(round(transactions) AS BIGINT)   AS trade_count
            FROM (
                SELECT ticker, open, high, low, close, volume, transactions,
                       timezone('America/New_York', to_timestamp(window_start / 1e9 + 60)) AS et
                FROM read_csv_auto('{g}', compression='gzip', header=true, union_by_name=true)
            )
        )
        {where}
    """


def read_canonical(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    *,
    rth_only: bool = False,
):
    """返回 DuckDB 关系（惰性）：Flat Files -> AlphaForge 口径分钟 bar。"""
    return con.sql(canonical_select(glob, rth_only=rth_only))
