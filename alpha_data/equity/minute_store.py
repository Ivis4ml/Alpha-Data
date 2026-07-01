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

from alpha_data.common.hashing import content_hash

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


def _input_fingerprint(paths: list[Path]) -> str:
    """输入文件集的指纹（文件名 + 大小），供续传前校验输入未变化。"""
    parts = sorted((p.name, p.stat().st_size if p.exists() else -1) for p in paths)
    return content_hash(parts)


def _reorg_partitions(
    con: duckdb.DuckDBPyConnection, tmp: Path, dst_for, *, skip_existing: bool = False
) -> int:
    """把 ``tmp/_safe=X/*.parquet`` 归并到 ``dst_for(X)``（单文件直接移动，多文件合并）。

    ``skip_existing=True`` 时，目标已存在的分区直接跳过（可续传：中断后重跑只补未完成的）。
    两条路径的目标写入均为原子操作：单文件走同文件系统 rename；多文件先合并到临时
    ``*.parquet.merge`` 再 rename，中断不会留下截断的目标 parquet（残留的 ``.merge``
    下次重跑会被覆盖后原子替换）。
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
            merge_tmp = dst.with_name(dst.name + ".merge")
            con.execute(
                f"COPY (SELECT * FROM read_parquet({_duck_list(files)})) "
                f"TO '{merge_tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
            )
            merge_tmp.replace(dst)
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

    每年两阶段、可续传，状态机如下：

    1. 全新构建（无 ``.copydone_{year}``、暂存目录缺失、或输入指纹与标记不符）：先删除该年
       已有的 ``minute/*/{year}.parquet`` 残留分区（重建即整年替换，删旧保证不会静默保留
       过期数据），再 COPY 出暂存分区并把输入指纹写入 ``.copydone_{year}``，随后归并
       （此时目标必为空，全部移入）。
    2. 归并续传（``.copydone_{year}`` 与暂存目录均存在，且标记内的输入指纹与本次一致）：
       跳过 COPY，仅归并；目标已存在的分区来自同一次 COPY 的已完成移动，跳过即可。
       指纹校验保证"COPY 后中断、随后补充下载新文件再重跑"不会把旧暂存误当续传而丢新数据。

    年份全部归并后写 ``minute/.done_{year}``；``skip_done=True`` 时跳过已标记年份。
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
        fingerprint = _input_fingerprint(paths)
        resumed = (
            copydone.exists() and tmp.exists() and copydone.read_text() == fingerprint
        )
        if not resumed:
            shutil.rmtree(tmp, ignore_errors=True)
            copydone.unlink(missing_ok=True)
            # 整年重建：先清掉旧分区，避免归并阶段把过期数据误当作已完成而保留。
            for stale in minute_dir.glob(f"*/{year}.parquet"):
                stale.unlink()
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
            copydone.write_text(fingerprint)
        total += _reorg_partitions(
            con, tmp, lambda safe, y=year: minute_dir / safe / f"{y}.parquet",
            skip_existing=resumed,
        )
        copydone.unlink(missing_ok=True)
        marker.write_text("ok")
    return total


def build_daily(con: duckdb.DuckDBPyConnection, csv_paths: list[Path], db_root: Path) -> int:
    """由 RTH 分钟聚合出日线，写入 ``daily/{safe}.parquet``（一文件含该 symbol 全部交易日）。

    重要：``csv_paths`` 必须是**完整**的 Flat Files 集合（全部已下载交易日），不能只传
    部分区间。日线一 symbol 一文件、覆盖全部交易日，每次构建整体替换：全新构建会先清空
    旧的 ``daily/*.parquet``，只传部分区间会把区间外的日线数据一并清掉。

    与 :func:`build_minute` 同一状态机：全新构建先清旧、COPY 完成后把输入指纹写入
    ``.copydone_daily``；归并续传（标记与暂存俱在、指纹一致）时跳过 COPY、只补未移动的分区。
    """
    db_root = Path(db_root)
    db_root.mkdir(parents=True, exist_ok=True)
    daily_dir = db_root / "daily"
    tmp = db_root / ".tmp_daily"
    copydone = db_root / ".copydone_daily"
    fingerprint = _input_fingerprint(csv_paths)
    resumed = copydone.exists() and tmp.exists() and copydone.read_text() == fingerprint
    if not resumed:
        shutil.rmtree(tmp, ignore_errors=True)
        copydone.unlink(missing_ok=True)
        for stale in daily_dir.glob("*.parquet"):
            stale.unlink()
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
        copydone.write_text(fingerprint)
    n = _reorg_partitions(
        con, tmp, lambda safe: daily_dir / f"{safe}.parquet", skip_existing=resumed
    )
    copydone.unlink(missing_ok=True)
    return n
