"""Polymarket 链上爬虫 CLI：全量回填、增量续爬、元数据、分析层。

数据来源为 Polygon 主网公开日志与 Polymarket 两个免鉴权只读 API，不依赖任何第三方数据集。

子命令::

    # 1. 增量续爬（默认）：从游标或公开快照末区块 86126998 起，抓到最新已最终确定的区块
    python scripts/crawl_polymarket_chain.py trades

    # 2. 指定区间回填（含 CTF 生命周期事件）
    python scripts/crawl_polymarket_chain.py trades --start 35896869 --end 40000000 \
        --events trades splits merges redemptions preparations resolutions

    # 3. 抓市场元数据（asset_id -> 市场 / 结果 / 胜出结果）
    python scripts/crawl_polymarket_chain.py metadata

    # 4. 由分片构建分析层（剔除中继腿 + 连接元数据 + 计算 p_event / D）
    python scripts/crawl_polymarket_chain.py analysis

并发默认 8（公共端点的稳妥值）。配置 ``POLYGON_RPC_URLS``（逗号分隔）接入付费端点后可上调。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_data.polymarket.chain import crawl, enrich, metadata, positionid, venues  # noqa: E402
from alpha_data.polymarket.chain.rpc import RpcPool  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "polymarket_chain"

# 公开快照 TimeSeventeen/Polymarket-v1 的末区块（旧交易所最后一条 OrderFilled）。
# 未指定 --start 且无游标时，从此处接续，正好补上快照之后的全部数据。
SNAPSHOT_LAST_BLOCK = venues.LEGACY_LAST_BLOCK


def cmd_trades(args: argparse.Namespace) -> int:
    pool = RpcPool()
    out = Path(args.out)
    head = crawl.safe_head(pool)
    start = args.start or crawl.read_cursor(out) or (SNAPSHOT_LAST_BLOCK + 1)
    end = args.end or head
    if start > end:
        print(f"无需抓取：起始 {start} 已超过可安全抓取的上界 {end}")
        return 0

    span_blocks = end - start + 1
    print(f"抓取区块 {start} .. {end}（{span_blocks:,} 块，约 {span_blocks * 2.1 / 86400:.1f} 天）")
    print(f"事件：{', '.join(args.events)}    并发：{args.workers}    输出：{out}")

    t0 = time.time()
    done = [0]

    def progress(event: str, lo: int, hi: int, n: int) -> None:
        done[0] += 1
        if done[0] % 10 == 0:
            print(f"  [{done[0]:>5}] {event} {lo}-{hi} +{n} 行，累计 {time.time() - t0:.0f}s")

    written = crawl.crawl_range(
        pool,
        start=start,
        end=end,
        out_dir=out,
        events=tuple(args.events),
        workers=args.workers,
        on_progress=progress,
    )
    for event, n in written.items():
        print(f"  {event}: 新增 {n:,} 行")
    crawl.write_cursor(out, end)

    if not args.no_compact:
        for event in args.events:
            days = crawl.compact_days(out, event)
            print(f"  {event}: 合并为 {days} 个日分区")
    print(f"完成，耗时 {time.time() - t0:.0f}s")
    return 0


def cmd_metadata(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("抓取 CLOB 市场目录（游标 = base64(offset)，跳页并行）...")
    clob = metadata.fetch_clob_markets(workers=args.workers, max_markets=args.max_markets)
    clob.to_parquet(out / "markets_clob.parquet", compression="zstd", index=False)
    print(f"  市场 {len(clob):,} 个，耗时 {time.time() - t0:.0f}s")

    asset_map = metadata.build_asset_map(clob)
    asset_map.to_parquet(out / "asset_map.parquet", compression="zstd", index=False)
    print(f"  asset_id 映射 {len(asset_map):,} 条")

    if args.gamma:
        print("抓取 Gamma 市场（keyset 分页，closed=true/false 各走一遍）...")
        gamma = metadata.fetch_gamma_markets(max_pages=args.max_pages)
        gamma.to_parquet(out / "markets_gamma.parquet", compression="zstd", index=False)
        print(f"  Gamma 市场 {len(gamma):,} 个")

    if args.verify:
        print("链上抽样校验 asset_id 映射（不信任 HTTP 接口）...")
        report = positionid.verify_asset_map(RpcPool(), asset_map, sample=args.verify)
        print(f"  校验 {report['checked']} 条，链上推导一致 {report['matched']} 条")
        for bad in report["mismatched"][:5]:
            print(f"  不一致：{bad}")
    print(f"完成，耗时 {time.time() - t0:.0f}s")
    return 0


def cmd_analysis(args: argparse.Namespace) -> int:
    out = Path(args.out)
    asset_path = out / "asset_map.parquet"
    if not asset_path.exists():
        print("缺少 asset_map.parquet，请先执行 metadata 子命令")
        return 2
    asset_map = pd.read_parquet(asset_path)
    gamma_path = out / "markets_gamma.parquet"
    gamma = pd.read_parquet(gamma_path) if gamma_path.exists() else None

    resolutions = None
    res_dir = out / "shards" / "resolutions"
    if res_dir.exists():
        res = pd.concat([pd.read_parquet(p) for p in res_dir.glob("*.parquet")], ignore_index=True)
        if not res.empty:
            res["block_number"] = res["id"].str.split("_").str[1].astype("int64")
            pool = RpcPool()
            ts = pool.block_timestamps(res["block_number"].unique().tolist())
            res["block_timestamp"] = res["block_number"].map(ts)
            resolutions = res

    daily_dir = out / "daily" / "trades"
    files = sorted(daily_dir.glob("*.parquet"))
    if not files:
        print("缺少 daily/trades 分区，请先执行 trades 子命令")
        return 2

    analysis_dir = out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    excluded: dict[str, int] = {}
    for path in files:
        trades = pd.read_parquet(path)
        poly, other = enrich.keep_polymarket_venues(trades)
        for addr, n in other["exchange"].value_counts().items():
            excluded[str(addr)] = excluded.get(str(addr), 0) + int(n)
        clean = enrich.drop_relay_legs(poly)
        df = enrich.attach_metadata(clean, asset_map, gamma, resolutions)
        df.to_parquet(analysis_dir / path.name, compression="zstd", index=False)
        total += len(df)
        matched = int(df["condition_id"].notna().sum())
        print(
            f"  {path.stem}: 成交 {len(trades):,} -> Polymarket 场馆 {len(poly):,} "
            f"-> 去中继腿 {len(clean):,} -> 元数据命中 {matched:,}"
        )
    if excluded:
        print("\n未纳入分析层的场馆（非 Polymarket 或待核验，不静默丢弃）：")
        for addr, n in sorted(excluded.items(), key=lambda x: -x[1]):
            print(f"  {addr}  {venues.venue_class(addr):10s} {n:,} 行")
        print("  如需判定某地址，用 venues.classify_venue_onchain(pool, addr) 核验其 getCtf()")
    print(f"\n分析层写出 {total:,} 行 -> {analysis_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Polymarket 链上爬虫")
    parser.add_argument("--out", default=str(OUT), help="输出根目录")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_trades = sub.add_parser("trades", help="抓取成交（两版 OrderFilled）与 CTF 事件")
    p_trades.add_argument("--start", type=int, help="起始区块（默认：游标 / 快照末区块 + 1）")
    p_trades.add_argument("--end", type=int, help="结束区块（默认：最新已最终确定的区块）")
    p_trades.add_argument(
        "--events",
        nargs="+",
        default=["trades"],
        choices=["trades", "splits", "merges", "redemptions", "preparations", "resolutions"],
    )
    p_trades.add_argument("--workers", type=int, default=8)
    p_trades.add_argument("--no-compact", action="store_true", help="只写分片，不合并日分区")
    p_trades.set_defaults(func=cmd_trades)

    p_meta = sub.add_parser("metadata", help="抓取市场元数据")
    p_meta.add_argument("--gamma", action="store_true", help="同时抓 Gamma 类别（较慢，可选）")
    p_meta.add_argument("--workers", type=int, default=8, help="CLOB 目录并发页数")
    p_meta.add_argument("--max-markets", type=int, help="只取前若干市场（调试用）")
    p_meta.add_argument("--max-pages", type=int, help="Gamma 页数上限（调试用）")
    p_meta.add_argument(
        "--verify", type=int, default=0, metavar="N",
        help="抽 N 个市场用链上 getPositionId 校验 asset_id 映射",
    )
    p_meta.set_defaults(func=cmd_metadata)

    p_ana = sub.add_parser("analysis", help="构建分析层")
    p_ana.set_defaults(func=cmd_analysis)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
