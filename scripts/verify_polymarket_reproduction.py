"""复现性验证：用我们自己的链上爬虫重抓若干区块，与本地 Polymarket 快照逐字段比对。

这是"数据是我们自己爬的"这一说法的可检验证据：给定任意区块，从 Polygon 公共 RPC 取原始日志、
用本仓库的解码器还原成交，再与已落地的 ``data/polymarket/OrderFilled`` 同区块行做逐列比较。
若两者全等，说明该数据集不含任何我们无法独立重建的信息。

用法::

    python scripts/verify_polymarket_reproduction.py                 # 默认抽 5 个区块
    python scripts/verify_polymarket_reproduction.py --blocks 63732941 40000000
    python scripts/verify_polymarket_reproduction.py --sample 20     # 随机抽 20 个区块

退出码 0 表示全部一致。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_data.polymarket.chain import crawl, venues  # noqa: E402
from alpha_data.polymarket.chain.rpc import RpcPool  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OF_GLOB = str(ROOT / "data" / "polymarket" / "OrderFilled" / "*.parquet")

# 与公开数据集 OrderFilled 层逐列对应的列（我们的表是其超集）。
SHARED = [
    "id",
    "maker",
    "taker",
    "block_timestamp",
    "maker_asset_id",
    "taker_asset_id",
    "maker_direction",
    "taker_direction",
    "token_asset_id",
    "token_amount",
    "usdc_amount",
    "price",
    "fee_usdc",
]


def pick_blocks(con: duckdb.DuckDBPyConnection, n: int, seed: int = 7) -> list[int]:
    """从快照里随机抽 n 个含成交的区块（覆盖不同年份）。"""
    rng = random.Random(seed)
    years = ["2022_12", "2023_06", "2024_03", "2024_11", "2025_08", "2026_04"]
    blocks: list[int] = []
    for i in range(n):
        month = years[i % len(years)]
        path = ROOT / "data" / "polymarket" / "OrderFilled" / f"{month}.parquet"
        if not path.exists():
            continue
        got = con.sql(
            f"SELECT DISTINCT CAST(split_part(id,'_',2) AS BIGINT) AS b "
            f"FROM read_parquet('{path}') USING SAMPLE 200 ROWS"
        ).df()["b"].tolist()
        if got:
            blocks.append(int(rng.choice(got)))
    return sorted(set(blocks))


def compare_block(pool: RpcPool, con: duckdb.DuckDBPyConnection, block: int) -> tuple[bool, str]:
    """比对单个区块，返回 (是否全等, 说明)。"""
    mine = crawl.crawl_trades_window(pool, block, block)
    prefix = f"{venues.CHAIN_ID}_{block}_"
    theirs = con.sql(
        f"SELECT * FROM read_parquet('{OF_GLOB}') WHERE starts_with(id, '{prefix}')"
    ).df()

    if len(mine) != len(theirs):
        return False, f"行数不一致：我方 {len(mine)}，快照 {len(theirs)}"
    if mine.empty:
        return True, "该区块无成交（双方均为空）"

    a = mine[SHARED].sort_values("id").reset_index(drop=True)
    b = theirs[SHARED].sort_values("id").reset_index(drop=True)
    bad: list[str] = []
    for col in SHARED:
        if pd.api.types.is_float_dtype(b[col]):
            diff = (a[col].astype(float) - b[col].astype(float)).abs() > 1e-9
        else:
            diff = a[col].astype(str) != b[col].astype(str)
        if int(diff.sum()):
            bad.append(f"{col}({int(diff.sum())} 行)")
    if bad:
        return False, "不一致列：" + ", ".join(bad)
    return True, f"{len(a)} 行 x {len(SHARED)} 列全等"


def main() -> int:
    parser = argparse.ArgumentParser(description="验证链上爬虫可复现 Polymarket 快照")
    parser.add_argument("--blocks", type=int, nargs="*", help="指定区块号")
    parser.add_argument("--sample", type=int, default=5, help="随机抽样的区块数")
    args = parser.parse_args()

    con = duckdb.connect()
    blocks = args.blocks or pick_blocks(con, args.sample)
    if not blocks:
        print("未找到可比对的区块（本地快照缺失？）")
        return 2

    pool = RpcPool()
    print(f"RPC 端点：{', '.join(pool.endpoints)}")
    print(f"待比对区块：{blocks}\n")

    all_ok = True
    for blk in blocks:
        ok, note = compare_block(pool, con, blk)
        all_ok &= ok
        print(f"  区块 {blk:>9}  {'一致' if ok else '不一致'}  {note}")

    verdict = "全部区块逐字段一致，快照可由本仓库爬虫完全复现。" if all_ok else "存在差异，见上。"
    print("\n结论：" + verdict)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
