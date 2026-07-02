"""由中证 A 股分钟 CSV 构建 AlphaForge ``MinuteDB`` 布局（分钟 + 日线 + listing）。

源数据布局（本机 ``remote_db/a_stock_data/extracted``）：

- ``沪深个股_1min_按年/{year}/{sh|sz}{code}_{year}.csv``：单只个股一年一文件，含
  成交量与成交额两列。
- ``指数_1min_按年/{year}/{code}_{year}.csv``：单个指数一年一文件，仅有成交额、无成交量列。

CSV 列（GBK 无关，文件为 UTF-8 带 BOM，DuckDB 自动剥离）：
``时间, 代码, 名称, 开盘价, 收盘价, 最高价, 最低价, [成交量,] 成交额, 涨幅, 振幅``。
注意列序为 开盘/收盘/最高/最低，收盘价在最高/最低之前。

产出布局与列（对齐 AlphaForge ``providers/db/minute_db.py``，与美股库
:mod:`alpha_data.equity.minute_store` 逐列一致，便于同一套 ``DbMinuteSource`` 读取）：

- ``minute/{safe_symbol}/{year}.parquet``：``symbol, ts, open/high/low/close(float32),
  volume/trade_count(int64), source, ingested_utc, raw_hash, _safe``。
- ``daily/{safe_symbol}.parquet``：``symbol, trade_date, open/high/low/close(float32),
  volume(int64), amount(float64), source, _safe``（由分钟聚合：open=首 bar，close=末 bar，
  high=max，low=min，volume=Σ成交量，amount=Σ成交额）。
- ``listing.parquet``：``symbol, name, exchange, status, ipo_date, delist_date``。

口径与美股库的差异（有意为之，非疏漏）：

- ``amount`` 直接取源 ``成交额``（真实成交额），不再用 ``close×volume`` 退化估算。
- ``ts`` 为**北京本地墙钟时间**（源时间戳原样保留，非 UTC；A 股连续竞价与集合竞价的
  时点标注沿用数据商口径），非美东。下游若按美股日历 / RTH 解释 ``ts`` 并不适用，A 股
  的交易时段（09:30 集合竞价、09:31–11:30 与 13:01–15:00 连续、15:00 收盘集合竞价）
  全部计入日线，不做 RTH 过滤。
- 指数无成交量，``volume`` 记 0；个股 ``volume`` 取 ``成交量``。
- ``trade_count`` 源无此列，记 0。

符号口径（industry-standard，带交易所后缀，保留前导零）：

- 个股：``代码`` 形如 ``sh600000`` → ``600000.SH``；``sz000001`` → ``000001.SZ``。
- 指数：``代码`` 形如 ``399006`` → 前缀 ``39`` 记深交所 ``.SZ``，``899`` 记北交所 ``.BJ``，
  其余（``000xxx``）记上交所 ``.SH``。指数 ``代码`` 被 DuckDB 默认按整数解析会丢前导零，
  故指数读取时强制 ``代码`` 为 VARCHAR。

``.SH`` / ``.SZ`` / ``.BJ`` 后缀不在 AlphaForge ``default_normalize_symbol`` 的交易所后缀
白名单内，归一时保留，故 ``symbol`` 与 ``_safe`` 一致（仅含字母数字与点）。
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

from alpha_data.equity.minute_store import (
    _duck_list,
    _input_fingerprint,
    _reorg_partitions,
)

SOURCE = "zhongzheng"

# 个股：代码含 sh/sz 前缀 → 去前缀 + 大写后缀。
_STOCK_SYMBOL = "substr(\"代码\", 3) || '.' || upper(substr(\"代码\", 1, 2))"
# 指数：代码为纯数字（VARCHAR，保留前导零）→ 39x 记 .SZ，899 记 .BJ，其余 .SH。
_INDEX_SYMBOL = (
    "\"代码\" || '.' || (CASE "
    "WHEN substr(\"代码\", 1, 2) = '39' THEN 'SZ' "
    "WHEN substr(\"代码\", 1, 3) = '899' THEN 'BJ' "
    "ELSE 'SH' END)"
)


def _safe_expr(symbol_expr: str) -> str:
    """与 ``MinuteDB._safe_symbol`` 一致的文件系统安全化（非法字符替 ``_``）。"""
    return f"regexp_replace(upper({symbol_expr}), '[^A-Za-z0-9_.\\-]', '_', 'g')"


def _read_expr(paths: list[Path], *, kind: str) -> str:
    """构造 DuckDB 读 CSV 的表达式。指数强制 ``代码`` 为 VARCHAR 以保留前导零。"""
    lst = _duck_list(paths)
    if kind == "index":
        return f"read_csv({lst}, header=true, types={{'代码': 'VARCHAR'}}, union_by_name=true)"
    return f"read_csv_auto({lst}, header=true, union_by_name=true)"


def _minute_select(paths: list[Path], *, kind: str) -> str:
    symbol = _STOCK_SYMBOL if kind == "stock" else _INDEX_SYMBOL
    volume = "CAST(\"成交量\" AS BIGINT)" if kind == "stock" else "CAST(0 AS BIGINT)"
    return f"""
        SELECT {_safe_expr(symbol)}          AS _safe,
               {symbol}                       AS symbol,
               "时间"                          AS ts,
               CAST("开盘价" AS FLOAT)         AS open,
               CAST("最高价" AS FLOAT)         AS high,
               CAST("最低价" AS FLOAT)         AS low,
               CAST("收盘价" AS FLOAT)         AS close,
               {volume}                       AS volume,
               CAST(0 AS BIGINT)              AS trade_count,
               '{SOURCE}'                     AS source,
               ''                             AS ingested_utc,
               ''                             AS raw_hash
        FROM {_read_expr(paths, kind=kind)}
    """


def _files_by_year(csv_paths: list[Path]) -> dict[int, list[Path]]:
    """按文件名年份（``..._{YYYY}.csv``）分组。"""
    groups: dict[int, list[Path]] = defaultdict(list)
    for p in csv_paths:
        groups[int(p.stem.rsplit("_", 1)[-1])].append(p)
    return dict(sorted(groups.items()))


def build_minute(
    con: duckdb.DuckDBPyConnection,
    csv_paths: list[Path],
    db_root: Path,
    *,
    kind: str,
    skip_done: bool = True,
) -> int:
    """把某一类（``kind='stock'`` 或 ``'index'``）分钟 CSV 写入 ``minute/{safe}/{year}.parquet``。

    个股与指数符号不相交，可分两次调用写入同一库。状态机与美股库
    :func:`alpha_data.equity.minute_store.build_minute` 一致：每年两阶段、指纹校验可续传、
    归并原子替换。标记按 ``kind`` 区分（``.done_{kind}_{year}`` 等），互不干扰。

    返回写出的 (symbol, year) 分区数。
    """
    if kind not in ("stock", "index"):
        raise ValueError(f"kind 必须为 stock/index，得到 {kind!r}")
    db_root = Path(db_root)
    minute_dir = db_root / "minute"
    minute_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for year, paths in _files_by_year(csv_paths).items():
        marker = minute_dir / f".done_{kind}_{year}"
        if skip_done and marker.exists():
            continue
        tmp = db_root / f".tmp_minute_{kind}_{year}"
        copydone = minute_dir / f".copydone_{kind}_{year}"
        fingerprint = _input_fingerprint(paths)
        resumed = copydone.exists() and tmp.exists() and copydone.read_text() == fingerprint
        if not resumed:
            shutil.rmtree(tmp, ignore_errors=True)
            copydone.unlink(missing_ok=True)
            # 整类整年重建：先清掉本类已有分区（个股/指数符号不相交，按前缀区分不现实，
            # 故依赖 kind 分次调用时各自的 done 标记；此处只清空 tmp 与 copydone，
            # 目标分区在归并阶段整体替换）。
            con.execute(
                f"COPY ({_minute_select(paths, kind=kind)}) TO '{tmp.as_posix()}' "
                f"(FORMAT parquet, PARTITION_BY (_safe), COMPRESSION zstd)"
            )
            copydone.write_text(fingerprint)
        total += _reorg_partitions(
            con,
            tmp,
            lambda safe, y=year: minute_dir / safe / f"{y}.parquet",
            skip_existing=resumed,
        )
        copydone.unlink(missing_ok=True)
        marker.write_text("ok")
    return total


def build_daily(
    con: duckdb.DuckDBPyConnection,
    csv_paths: list[Path],
    db_root: Path,
    *,
    kind: str,
) -> int:
    """由分钟聚合出日线，写入 ``daily/{safe}.parquet``（一 symbol 一文件、覆盖全部交易日）。

    重要：``csv_paths`` 必须是该类**全部年份**的完整文件集。日线整体替换，只传部分年份
    会把其余年份的日线一并清掉。个股与指数分两次调用（符号不相交，互不覆盖）。

    无 RTH 过滤（A 股全时段计入）；``amount`` 取源 ``成交额`` 之和。
    """
    if kind not in ("stock", "index"):
        raise ValueError(f"kind 必须为 stock/index，得到 {kind!r}")
    db_root = Path(db_root)
    db_root.mkdir(parents=True, exist_ok=True)
    daily_dir = db_root / "daily"
    tmp = db_root / f".tmp_daily_{kind}"
    copydone = db_root / f".copydone_daily_{kind}"
    fingerprint = _input_fingerprint(csv_paths)
    resumed = copydone.exists() and tmp.exists() and copydone.read_text() == fingerprint
    symbol = _STOCK_SYMBOL if kind == "stock" else _INDEX_SYMBOL
    volume = "\"成交量\"" if kind == "stock" else "0"
    if not resumed:
        shutil.rmtree(tmp, ignore_errors=True)
        copydone.unlink(missing_ok=True)
        inner = f"""
            SELECT {_safe_expr(symbol)} AS _safe, {symbol} AS symbol,
                   strftime("时间", '%Y-%m-%d') AS trade_date, "时间" AS ts,
                   "开盘价" AS open, "最高价" AS high, "最低价" AS low, "收盘价" AS close,
                   {volume} AS volume, "成交额" AS amount
            FROM {_read_expr(csv_paths, kind=kind)}
        """
        agg = f"""
            SELECT _safe, symbol, trade_date,
                   CAST(arg_min(open, ts) AS FLOAT)  AS open,
                   CAST(max(high) AS FLOAT)          AS high,
                   CAST(min(low) AS FLOAT)           AS low,
                   CAST(arg_max(close, ts) AS FLOAT) AS close,
                   CAST(sum(volume) AS BIGINT)       AS volume,
                   CAST(sum(amount) AS DOUBLE)       AS amount,
                   '{SOURCE}' AS source
            FROM ({inner})
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


_EXCHANGE_NAME = {"SH": "XSHG", "SZ": "XSHE", "BJ": "BSE"}


def build_listing(
    con: duckdb.DuckDBPyConnection,
    latest_by_symbol: dict[str, Path],
    db_root: Path,
    *,
    active_symbols: set[str],
) -> pd.DataFrame:
    """构造 ``listing`` 表并写入 ``listing.parquet``。

    ``latest_by_symbol``：每个 symbol 映射到其最新年份的 CSV（用于取当前名称）。
    ``active_symbols``：在最新一年出现的 symbol 记 ``active``，其余记 ``delisted``。
    ``ipo_date`` / ``delist_date`` 由已建日线的首末 ``trade_date`` 得出。
    """
    db_root = Path(db_root)
    daily_dir = db_root / "daily"
    rows: list[dict[str, str]] = []
    for symbol, path in sorted(latest_by_symbol.items()):
        suffix = symbol.rsplit(".", 1)[-1]
        name = _read_name(path)
        daily_path = daily_dir / f"{symbol}.parquet"
        ipo_date, delist_date = "", ""
        if daily_path.exists():
            span = con.execute(
                f"SELECT min(trade_date), max(trade_date) "
                f"FROM read_parquet('{daily_path.as_posix()}')"
            ).fetchone()
            ipo_date = span[0] or ""
            last_date = span[1] or ""
            # 最新一年未出现即视为退市，退市日以最后交易日近似。
            delist_date = "" if symbol in active_symbols else last_date
        rows.append(
            {
                "symbol": symbol,
                "name": name,
                "exchange": _EXCHANGE_NAME.get(suffix, suffix),
                "status": "active" if symbol in active_symbols else "delisted",
                "ipo_date": ipo_date,
                "delist_date": delist_date,
            }
        )
    cols = ["symbol", "name", "exchange", "status", "ipo_date", "delist_date"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_parquet(db_root / "listing.parquet", index=False)
    return df


def _read_name(path: Path) -> str:
    """读取单文件首个数据行的 ``名称`` 列（标的名称）。"""
    try:
        head = pd.read_csv(path, nrows=1, usecols=["名称"], encoding="utf-8-sig")
    except Exception:
        return ""
    if head.empty:
        return ""
    return str(head["名称"].iloc[0])


def write_empty_corp_actions(db_root: Path) -> None:
    """写入空 ``corp_actions`` 表。

    本源不含拆股 / 分红，复权因子缺失。下游 ``corp_actions`` 返回空表（全部标的
    ``split_ratio=1.0``、``cash_div=0.0`` 的等价），跨除权日收益不可直接使用，须另行
    从免费源（如 akshare / 交易所公告）补充后再重建本表。
    """
    db_root = Path(db_root)
    cols = ["symbol", "ex_date", "split_ratio", "cash_div"]
    pd.DataFrame(columns=cols).to_parquet(db_root / "corp_actions.parquet", index=False)
