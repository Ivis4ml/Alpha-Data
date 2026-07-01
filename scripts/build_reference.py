"""拉取全市场标的（含退市）+ 拆股 + 分红，写入 MinuteDB 的 ``listing`` / ``corp_actions`` 表。

Developer 档已无限速。产物同时另存 Parquet 便于查看。拆股 / 分红默认限定 ``--since`` 起（默认
2019-01-01，早于数据窗口以保证复权连续性）。

用法::

    python scripts/build_reference.py
    python scripts/build_reference.py --since 2019-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha_data.common.env import load_env
from alpha_data.equity import reference
from alpha_data.equity.massive_client import MassiveClient

ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = ROOT / "data" / "equity" / "minute_db"
REF_DIR = ROOT / "data" / "equity" / "reference"


def main() -> int:
    parser = argparse.ArgumentParser(description="拉取参考数据并写入 MinuteDB")
    parser.add_argument("--db", default=str(DB_ROOT))
    parser.add_argument("--since", default="2019-01-01", help="拆股/分红起始 ex_date")
    parser.add_argument("--until", default="", help="拆股/分红截止 ex_date（空=不限）")
    parser.add_argument("--no-write-db", action="store_true", help="只存 Parquet，不写 MinuteDB")
    args = parser.parse_args()

    load_env()
    client = MassiveClient()
    until = args.until or None

    print("拉取标的全集（active + delisted）……", flush=True)
    listing = reference.fetch_listing(client)
    print(f"  标的 {len(listing)}（active={int((listing['status'] == 'active').sum())}）")

    print("拉取拆股……", flush=True)
    splits = reference.fetch_splits(client, since=args.since, until=until)
    print(f"  拆股 {len(splits)}")

    print("拉取分红……", flush=True)
    dividends = reference.fetch_dividends(client, since=args.since, until=until)
    print(f"  分红 {len(dividends)}")

    # splits/dividends 端点对类别股返回无点 ticker（BFB/MOGA），映射回带点原生形式（BF.B/MOG.A）
    # 以与分钟库、listing 一致；否则这些标的的公司行动在下游永远匹配不上。
    canonical = listing["symbol"]
    splits = reference.remap_to_native_symbols(splits, canonical)
    dividends = reference.remap_to_native_symbols(dividends, canonical)
    n_dotted = int(splits["symbol"].str.contains(".", regex=False).sum()) + int(
        dividends["symbol"].str.contains(".", regex=False).sum()
    )
    print(f"  类别股符号回映射完成（映射后带点行数 {n_dotted}）")

    corp = reference.build_corp_actions(splits, dividends)
    print(f"  corp_actions 合并 {len(corp)} 行")

    REF_DIR.mkdir(parents=True, exist_ok=True)
    listing.to_parquet(REF_DIR / "listing.parquet", index=False)
    splits.to_parquet(REF_DIR / "splits.parquet", index=False)
    dividends.to_parquet(REF_DIR / "dividends.parquet", index=False)
    corp.to_parquet(REF_DIR / "corp_actions.parquet", index=False)
    print(f"  Parquet 已存 -> {REF_DIR}")

    if not args.no_write_db:
        from alphaforge.providers.db.minute_db import MinuteDB

        db = MinuteDB(args.db)
        n_listing = db.write_listing(listing)
        n_ca = db.write_corp_actions(corp)
        print(f"写入 MinuteDB：listing={n_listing}，corp_actions={n_ca} -> {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
