"""由中证 A 股分钟 CSV 构建 AlphaForge ``MinuteDB``（分钟 + 日线 + listing），与美股库同构。

源目录（本机）::

    remote_db/a_stock_data/extracted/沪深个股_1min_按年/{year}/{sh|sz}{code}_{year}.csv
    remote_db/a_stock_data/extracted/指数_1min_按年/{year}/{code}_{year}.csv

用法::

    # 全量（2020-2025，个股 + 指数，分钟 + 日线 + listing）
    python scripts/build_cn_minute_db.py --start-year 2020 --end-year 2025 --memory 12GB

    # 只建某几年的个股分钟（便于分批 / 断点续传）
    python scripts/build_cn_minute_db.py --years 2020 2021 --kind stock --no-daily

日线与 listing 需全部年份的完整文件，故仅在处理到 ``--end-year`` 且未 ``--no-daily`` 时构建。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha_data.cn_equity import minute_store
from alpha_data.polymarket import store as duck

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT.parent / "remote_db" / "a_stock_data" / "extracted"
DB_ROOT = ROOT / "data" / "cn_equity" / "minute_db"

STOCK_DIR = "沪深个股_1min_按年"
INDEX_DIR = "指数_1min_按年"


def _files(src: Path, subdir: str, years: list[int]) -> list[Path]:
    out: list[Path] = []
    for y in years:
        out.extend(sorted((src / subdir / str(y)).glob("*.csv")))
    return out


def _symbol_of(path: Path, *, kind: str) -> str:
    """由文件名推导 symbol（与 SQL 口径一致，供 listing 用）。"""
    code = path.stem.rsplit("_", 1)[0]
    if kind == "stock":
        return code[2:] + "." + code[:2].upper()
    if code[:2] == "39":
        ex = "SZ"
    elif code[:3] == "899":
        ex = "BJ"
    else:
        ex = "SH"
    return code + "." + ex


def main() -> int:
    parser = argparse.ArgumentParser(description="由中证 A 股 CSV 构建 MinuteDB")
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="源根目录")
    parser.add_argument("--db", default=str(DB_ROOT))
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--years", type=int, nargs="*", help="显式指定年份（覆盖 start/end）")
    parser.add_argument(
        "--kind", choices=["stock", "index", "both"], default="both"
    )
    parser.add_argument("--threads", type=int, default=0, help="DuckDB 线程数（0=默认）")
    parser.add_argument("--memory", default="", help="DuckDB 内存上限，如 '12GB'")
    parser.add_argument("--no-minute", action="store_true", help="跳过分钟")
    parser.add_argument("--no-daily", action="store_true", help="跳过日线")
    parser.add_argument("--no-listing", action="store_true", help="跳过 listing")
    parser.add_argument("--no-skip-done", action="store_true", help="强制重建已完成年份")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"源目录不存在：{src}")
        return 1
    years = args.years or list(range(args.start_year, args.end_year + 1))
    kinds = ["stock", "index"] if args.kind == "both" else [args.kind]
    db_root = Path(args.db)
    db_root.mkdir(parents=True, exist_ok=True)
    print(f"源 {src}\n库 {db_root}\n年份 {years}  类别 {kinds}", flush=True)

    con = duck.connect(threads=args.threads or None)
    con.execute("PRAGMA preserve_insertion_order=false")
    if args.memory:
        con.execute(f"PRAGMA memory_limit='{args.memory}'")

    subdir = {"stock": STOCK_DIR, "index": INDEX_DIR}

    if not args.no_minute:
        for kind in kinds:
            paths = _files(src, subdir[kind], years)
            print(f"[{kind}] 分钟：{len(paths)} 个文件", flush=True)
            if paths:
                n = minute_store.build_minute(
                    con, paths, db_root, kind=kind, skip_done=not args.no_skip_done
                )
                print(f"[{kind}] 分钟：写出 {n} 个 (symbol,year) 分区", flush=True)

    all_years = list(range(args.start_year, args.end_year + 1))
    if not args.no_daily:
        for kind in kinds:
            paths = _files(src, subdir[kind], all_years)
            print(
                f"[{kind}] 日线：整体替换，使用全部 {len(paths)} 个文件"
                f"（{args.start_year}-{args.end_year}）",
                flush=True,
            )
            if paths:
                n = minute_store.build_daily(con, paths, db_root, kind=kind)
                print(f"[{kind}] 日线：写出 {n} 个 symbol 文件", flush=True)

    if not args.no_listing:
        latest: dict[str, Path] = {}
        active: set[str] = set()
        last_year = max(all_years)
        for kind in kinds:
            for y in all_years:
                for p in _files(src, subdir[kind], [y]):
                    sym = _symbol_of(p, kind=kind)
                    latest[sym] = p  # 年份升序遍历，最终保留最新一年的文件
                    if y == last_year:
                        active.add(sym)
        print(f"listing：{len(latest)} 标的（active {len(active)}）", flush=True)
        df = minute_store.build_listing(con, latest, db_root, active_symbols=active)
        print(f"listing：写出 {len(df)} 行 -> listing.parquet", flush=True)
        minute_store.write_empty_corp_actions(db_root)
        print("corp_actions：写出空表（本源无拆股 / 分红）", flush=True)

    con.close()
    print(f"完成 -> {db_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
