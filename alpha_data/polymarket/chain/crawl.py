"""按区块窗口并行抓取、解码、写入 Parquet 分片，并支持断点续爬与按日合并。

产物布局（``data/polymarket_chain/``）::

    shards/{event}/{from_block:09d}-{to_block:09d}.parquet   # 抓取单元，存在即视为已完成
    daily/{event}/{YYYY-MM-DD}.parquet                       # 按 UTC 日合并的分区
    cursor.json                                              # 已完成的连续区块上界

幂等性来自事件主键 ``{chainId}_{blockNumber}_{logIndex}``：同一区块重复抓取只会产出相同的行，
合并阶段按 ``id`` 去重即可。分片以区块区间命名，重跑时跳过已存在的分片，故中断可续。

重组（reorg）：Polygon PoS 的最终性由检查点提供。抓取上界默认取 ``finalized`` 标签
（端点不支持时回退为 ``链头 - 256``），已最终确定的区块不会回滚，故分片可安全视为不可变。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_data.polymarket.chain import decode, venues
from alpha_data.polymarket.chain.rpc import RpcPool

SHARD_SPAN = 10_000  # 单个分片覆盖的区块数（实测端点上限）
REORG_MARGIN = 256  # 端点不支持 finalized 标签时的保守回退深度


def safe_head(pool: RpcPool) -> int:
    """可安全抓取的最高区块（已最终确定）。"""
    try:
        blk = pool.call("eth_getBlockByNumber", ["finalized", False])
        if blk and blk.get("number"):
            return int(blk["number"], 16)
    except Exception:
        pass
    return pool.block_number() - REORG_MARGIN


def iter_windows(start: int, end: int, span: int = SHARD_SPAN) -> Iterator[tuple[int, int]]:
    """把 ``[start, end]`` 切成对齐到 ``span`` 网格的窗口。

    对齐网格可保证不同次运行切出的分片边界一致，从而复用已完成的分片。
    """
    lo = (start // span) * span
    while lo <= end:
        yield max(lo, start), min(lo + span - 1, end)
        lo += span


def _shard_path(out_dir: Path, event: str, lo: int, hi: int) -> Path:
    return out_dir / "shards" / event / f"{lo:09d}-{hi:09d}.parquet"


def _resolve_timestamps(pool: RpcPool, logs: list[dict[str, Any]]) -> dict[int, int] | None:
    """日志若不带 ``blockTimestamp``，批量补齐区块时间戳。"""
    if not logs or logs[0].get("blockTimestamp") is not None:
        return None
    blocks = {int(x["blockNumber"], 16) for x in logs}
    return pool.block_timestamps(blocks)


def crawl_trades_window(pool: RpcPool, lo: int, hi: int) -> pd.DataFrame:
    """抓取并解码一个窗口内的全部成交（新旧两版 ``OrderFilled`` 一次取回）。

    ``eth_getLogs`` 的 ``topics[0]`` 允许传数组表示或运算，故两版事件合并为一次请求。
    不加地址过滤：新协议的交易所合约会增加（已见三个），按 topic 扫描可自动覆盖。
    """
    logs = pool.get_logs(
        from_block=lo,
        to_block=hi,
        topics=[[venues.TOPIC_ORDER_FILLED_V1, venues.TOPIC_ORDER_FILLED_V2]],
    )
    if not logs:
        return pd.DataFrame(columns=list(decode.TRADE_COLUMNS))
    ts_by_block = _resolve_timestamps(pool, logs)
    rows = [r for r in (decode.decode_trade(x, ts_by_block) for x in logs) if r is not None]
    # 2026 年的密度下，单个 10,000 块窗口有逾百万条日志；解码后立即释放原始对象，
    # 避免原始日志与解码结果在内存中同时存在（并发窗口数乘以此开销即为峰值内存）。
    logs.clear()
    df = pd.DataFrame(rows, columns=list(decode.TRADE_COLUMNS))
    rows.clear()
    return df


def crawl_ctf_window(pool: RpcPool, spec: venues.EventSpec, lo: int, hi: int) -> pd.DataFrame:
    """抓取并解码一个窗口内某类 ConditionalTokens 事件。"""
    logs = pool.get_logs(
        from_block=lo, to_block=hi, topics=[spec.topic0], addresses=spec.addresses
    )
    rows = [decode.decode_ctf(spec.name, x) for x in logs]
    return pd.DataFrame(rows, columns=list(decode.CTF_COLUMNS[spec.name]))


def crawl_range(
    pool: RpcPool,
    *,
    start: int,
    end: int,
    out_dir: Path,
    events: tuple[str, ...] = ("trades",),
    workers: int = 8,
    span: int = SHARD_SPAN,
    on_progress: Any = None,
) -> dict[str, int]:
    """并行抓取区块区间，按分片写盘（已存在的分片跳过）。

    Args:
        pool: RPC 池。
        start: 起始区块（含）。
        end: 结束区块（含）。
        out_dir: 输出根目录。
        events: 要抓的事件；``trades`` 表示两版 ``OrderFilled``，其余取 CTF 事件名。
        workers: 并发窗口数。公共端点建议 8；付费端点可上调。
        span: 单分片区块数。
        on_progress: 可选回调 ``(event, lo, hi, rows)``。

    Returns:
        ``{事件名: 新增行数}``。
    """
    ctf_specs = {s.name: s for s in venues.CTF_EVENTS}
    jobs: list[tuple[str, int, int]] = []
    for event in events:
        first = venues.GENESIS_BLOCK if event == "trades" else ctf_specs[event].first_block
        for lo, hi in iter_windows(max(start, first), end, span):
            if not _shard_path(out_dir, event, lo, hi).exists():
                jobs.append((event, lo, hi))

    written: dict[str, int] = dict.fromkeys(events, 0)
    if not jobs:
        return written

    def run(job: tuple[str, int, int]) -> tuple[str, int, int, int]:
        event, lo, hi = job
        if event == "trades":
            df = crawl_trades_window(pool, lo, hi)
        else:
            df = crawl_ctf_window(pool, ctf_specs[event], lo, hi)
        path = _shard_path(out_dir, event, lo, hi)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, compression="zstd", index=False)
        tmp.replace(path)  # 原子落位：中断不会留下半截分片
        return event, lo, hi, len(df)

    with ThreadPoolExecutor(max_workers=workers) as pool_exec:
        futures = [pool_exec.submit(run, j) for j in jobs]
        for fut in as_completed(futures):
            event, lo, hi, n = fut.result()
            written[event] += n
            if on_progress:
                on_progress(event, lo, hi, n)
    return written


def compact_days(out_dir: Path, event: str = "trades") -> int:
    """把分片按 UTC 日合并为日分区（与公开数据集同构），返回写出的天数。

    **流式合并，内存有界**：分片文件名以零填充的起始区块开头，故字典序即区块序；
    区块号与时间戳单调，因此当读到某分片时，早于该分片最小日期的所有日期都已收齐，
    可立即写盘并释放。全量回填有 12 亿行，一次性载入会耗尽内存，故不能用简单的 concat。
    """
    shard_dir = out_dir / "shards" / event
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        return 0
    daily_dir = out_dir / "daily" / event
    daily_dir.mkdir(parents=True, exist_ok=True)

    buffers: dict[str, list[pd.DataFrame]] = {}
    days_written = 0

    def flush(day: str) -> None:
        nonlocal days_written
        part = pd.concat(buffers.pop(day), ignore_index=True)
        part = part.drop_duplicates(subset=["id"]).sort_values(["block_number", "log_index"])
        part.to_parquet(daily_dir / f"{day}.parquet", compression="zstd", index=False)
        days_written += 1

    for path in shards:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        day_col = pd.to_datetime(df["block_timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
        for day, part in df.groupby(day_col, sort=True):
            buffers.setdefault(str(day), []).append(part)
        oldest_open = str(day_col.min())
        for day in sorted(d for d in buffers if d < oldest_open):
            flush(day)
    for day in sorted(buffers):
        flush(day)
    return days_written


def read_cursor(out_dir: Path) -> int | None:
    """读取续爬游标（已完成的连续区块上界）。"""
    path = out_dir / "cursor.json"
    if not path.exists():
        return None
    return int(json.loads(path.read_text())["last_block"])


def write_cursor(out_dir: Path, last_block: int) -> None:
    """写入续爬游标。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cursor.json").write_text(
        json.dumps(
            {
                "last_block": int(last_block),
                "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
        )
    )
