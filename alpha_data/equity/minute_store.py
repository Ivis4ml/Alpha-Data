"""构建 AlphaForge ``MinuteDB`` 布局的分钟 / 日线库（DuckDB 出核，直接产出分区 Parquet）。

布局与列（对齐 AlphaForge ``providers/db/minute_db.py``）：

- ``minute/{safe_symbol}/{year}.parquet``：``symbol, ts, open/high/low/close(float32),
  volume/trade_count(int64), source, ingested_utc, raw_hash``。
- ``daily/{safe_symbol}.parquet``：``symbol, trade_date, open/high/low/close(float32),
  volume(int64), amount(float64), source``（由 RTH 分钟聚合：open=首 bar，close=末 bar，
  high=max，low=min，volume/amount=求和）。

``safe_symbol`` 与 ``MinuteDB._safe_symbol`` 一致（非法字符替 ``_``）。Flat Files 分钟聚合无
vwap，故不写 vwap 列；下游 ``amount`` 退化为 ``close×volume``（此处日线 ``amount`` 即按
``Σ close×volume`` 计）。用 DuckDB ``PARTITION_BY`` 出核写分区，再归并到 MinuteDB 命名。
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import duckdb

# 与 MinuteDB._safe_symbol 一致的文件系统安全化。
_SAFE = r"regexp_replace(upper(ticker), '[^A-Za-z0-9_.\-]', '_', 'g')"
# window_start 纳秒 UTC 为 bar 起始；+60s 得 bar 收盘时刻，再转美东本地。
# AlphaForge 契约（core/schema.py、market/us_equity/calendar.is_rth）以 bar 收盘戳标注 minute，
# RTH 为 09:30 < 收盘戳 <= 16:00（即 09:31..16:00），故此处存收盘时刻。
_ET = "timezone('America/New_York', to_timestamp(window_start / 1e9 + 60))"
# RTH（bar 收盘口径）：09:30 < 收盘戳 <= 16:00（用于日线聚合）。
_RTH_PRED = "strftime(_et, '%H:%M') > '09:30' AND strftime(_et, '%H:%M') <= '16:00'"


def _files_by_year(csv_paths: list[Path]) -> dict[int, list[Path]]:
    """按文件名年份（``YYYY-MM-DD.csv.gz``）分组。"""
    groups: dict[int, list[Path]] = defaultdict(list)
    for p in csv_paths:
        groups[int(p.name[:4])].append(p)
    return dict(sorted(groups.items()))


def _duck_list(paths: list[Path]) -> str:
    return "[" + ", ".join("'" + p.as_posix().replace("'", "''") + "'" for p in paths) + "]"


def _reorg_partitions(
    con: duckdb.DuckDBPyConnection, tmp: Path, dst_for, *, skip_existing: bool = False
) -> int:
    """把 ``tmp/_safe=X/*.parquet`` 归并到 ``dst_for(X)``（单文件直接移动，多文件合并）。

    ``skip_existing=True`` 时，目标已存在的分区直接跳过（可续传：中断后重跑只补未完成的）。
    """
    n = 0
    for part in sorted(tmp.glob("_safe=*")):
        safe = part.name.split("=", 1)[1]
        files = sorted(part.glob("*.parquet"))
        if not files:
            continue
        dst = dst_for(safe)
        if skip_existing and dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if len(files) == 1:
            shutil.move(str(files[0]), str(dst))
        else:
            con.execute(
                f"COPY (SELECT * FROM read_parquet({_duck_list(files)})) "
                f"TO '{dst.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
            )
        n += 1
    shutil.rmtree(tmp, ignore_errors=True)
    return n


def build_minute(
    con: duckdb.DuckDBPyConnection,
    csv_paths: list[Path],
    db_root: Path,
    *,
    skip_done: bool = True,
) -> int:
    """把分钟 CSV.gz 写入 ``minute/{safe}/{year}.parquet``。返回写出的 (symbol,year) 分区数。

    每年成功后写 ``minute/.done_{year}`` 标记；``skip_done=True`` 时跳过已标记的年份（可续传）。
    未标记的年份会重建并覆盖其残留分区。
    """
    db_root = Path(db_root)
    minute_dir = db_root / "minute"
    minute_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for year, paths in _files_by_year(csv_paths).items():
        marker = minute_dir / f".done_{year}"
        if skip_done and marker.exists():
            continue
        tmp = db_root / f".tmp_minute_{year}"
        copydone = minute_dir / f".copydone_{year}"
        # 若 COPY 阶段已完成（有 copydone 标记且暂存在），则跳过重读，直接续做归并。
        if not (copydone.exists() and tmp.exists()):
            shutil.rmtree(tmp, ignore_errors=True)
            copydone.unlink(missing_ok=True)
            select = f"""
                SELECT {_SAFE} AS _safe,
                       upper(ticker)                       AS symbol,
                       {_ET}                               AS ts,
                       CAST(open AS FLOAT)                 AS open,
                       CAST(high AS FLOAT)                 AS high,
                       CAST(low AS FLOAT)                  AS low,
                       CAST(close AS FLOAT)                AS close,
                       CAST(round(volume) AS BIGINT)       AS volume,
                       CAST(round(transactions) AS BIGINT) AS trade_count,
                       'massive'                           AS source,
                       ''                                  AS ingested_utc,
                       ''                                  AS raw_hash
                FROM read_csv_auto({_duck_list(paths)}, compression='gzip', header=true,
                                   union_by_name=true)
            """
            con.execute(
                f"COPY ({select}) TO '{tmp.as_posix()}' "
                f"(FORMAT parquet, PARTITION_BY (_safe), COMPRESSION zstd)"
            )
            copydone.write_text("ok")
        total += _reorg_partitions(
            con, tmp, lambda safe, y=year: minute_dir / safe / f"{y}.parquet",
            skip_existing=True,
        )
        copydone.unlink(missing_ok=True)
        marker.write_text("ok")
    return total


def build_daily(con: duckdb.DuckDBPyConnection, csv_paths: list[Path], db_root: Path) -> int:
    """由 RTH 分钟聚合出日线，写入 ``daily/{safe}.parquet``（一文件含该 symbol 全部交易日）。

    可续传：COPY 完成后写 ``.copydone_daily`` 标记；中断后重跑跳过 COPY、只补未归并的分区。
    """
    db_root = Path(db_root)
    db_root.mkdir(parents=True, exist_ok=True)
    daily_dir = db_root / "daily"
    tmp = db_root / ".tmp_daily"
    copydone = db_root / ".copydone_daily"
    if not (copydone.exists() and tmp.exists()):
        shutil.rmtree(tmp, ignore_errors=True)
        copydone.unlink(missing_ok=True)
        inner = f"""
            SELECT {_SAFE} AS _safe, upper(ticker) AS symbol,
                   strftime({_ET}, '%Y-%m-%d') AS trade_date,
                   {_ET} AS _et, open, high, low, close, volume
            FROM read_csv_auto({_duck_list(csv_paths)}, compression='gzip', header=true,
                               union_by_name=true)
        """
        agg = f"""
            SELECT _safe, symbol, trade_date,
                   CAST(arg_min(open, _et) AS FLOAT)  AS open,
                   CAST(max(high) AS FLOAT)           AS high,
                   CAST(min(low) AS FLOAT)            AS low,
                   CAST(arg_max(close, _et) AS FLOAT) AS close,
                   CAST(sum(volume) AS BIGINT)        AS volume,
                   CAST(sum(close * volume) AS DOUBLE) AS amount,
                   'massive' AS source
            FROM ({inner})
            WHERE {_RTH_PRED}
            GROUP BY _safe, symbol, trade_date
        """
        con.execute(
            f"COPY ({agg}) TO '{tmp.as_posix()}' "
            f"(FORMAT parquet, PARTITION_BY (_safe), COMPRESSION zstd)"
        )
        copydone.write_text("ok")
    n = _reorg_partitions(con, tmp, lambda safe: daily_dir / f"{safe}.parquet", skip_existing=True)
    copydone.unlink(missing_ok=True)
    return n
