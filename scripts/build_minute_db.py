"""由已下载的 Flat Files 构建 AlphaForge ``MinuteDB``（分钟 + 日线）。

用法::

    python scripts/build_minute_db.py --start 2020-01-02 --end 2020-01-02
    python scripts/build_minute_db.py --start 2020-01-01 --end 2020-12-31 --threads 8

仅处理区间内已下载存在的交易日文件（缺失自动跳过）。分钟按 symbol/year 分区，日线一 symbol 一文件。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha_data.common import calendar
from alpha_data.common.env import load_env
from alpha_data.equity import flatfiles, minute_store
from alpha_data.polymarket import store as duck

ROOT = Path(__file__).resolve().parents[1]
FLAT_ROOT = ROOT / "data" / "equity" / "flatfiles"
DB_ROOT = ROOT / "data" / "equity" / "minute_db"


def main() -> int:
    parser = argparse.ArgumentParser(description="由 Flat Files 构建 MinuteDB")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--db", default=str(DB_ROOT))
    parser.add_argument("--threads", type=int, default=0, help="DuckDB 线程数（0=默认）")
    parser.add_argument("--memory", default="", help="DuckDB 内存上限，如 '12GB'")
    parser.add_argument("--no-daily", action="store_true")
    parser.add_argument("--daily-only", action="store_true", help="只建日线（跳过分钟）")
    parser.add_argument("--no-skip-done", action="store_true", help="不跳过已完成年份（强制重建）")
    args = parser.parse_args()

    load_env()
    days = calendar.trading_days(args.start, args.end)
    paths = [flatfiles.local_path(FLAT_ROOT, d) for d in days]
    existing = [p for p in paths if p.exists() and p.stat().st_size > 0]
    print(f"区间 {args.start}..{args.end}：交易日 {len(days)}，已下载 {len(existing)}", flush=True)
    if not existing:
        print("无可用 Flat Files，先运行 download_equity_flatfiles.py。")
        return 1

    con = duck.connect(threads=args.threads or None)
    con.execute("PRAGMA preserve_insertion_order=false")  # 降低分区写内存
    if args.memory:
        con.execute(f"PRAGMA memory_limit='{args.memory}'")

    if not args.daily_only:
        print("构建分钟分区……", flush=True)
        n_min = minute_store.build_minute(
            con, existing, Path(args.db), skip_done=not args.no_skip_done
        )
        print(f"  分钟：写出 {n_min} 个 (symbol,year) 分区")

    if args.daily_only or not args.no_daily:
        print("构建日线……", flush=True)
        n_day = minute_store.build_daily(con, existing, Path(args.db))
        print(f"  日线：写出 {n_day} 个 symbol 文件")

    con.close()
    print(f"完成 -> {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
