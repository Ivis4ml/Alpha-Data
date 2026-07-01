"""下载美股分钟 Flat Files（全交易日区间）。可续传（跳过已存在文件）。

用法::

    python scripts/download_equity_flatfiles.py                       # 2020-01-01..2026-06-30
    python scripts/download_equity_flatfiles.py --start 2020-01-01 --end 2020-12-31
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from alpha_data.common import calendar
from alpha_data.common.env import load_env
from alpha_data.equity import flatfiles

ROOT = Path(__file__).resolve().parents[1] / "data" / "equity" / "flatfiles"


def main() -> int:
    parser = argparse.ArgumentParser(description="下载美股分钟 Flat Files")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    load_env()
    days = calendar.trading_days(args.start, args.end)
    print(f"交易日 {len(days)} 天：{args.start}..{args.end}，并行 {args.workers}", flush=True)

    counts: Counter[str] = Counter()

    def progress(res, i: int, total: int) -> None:
        counts[res[1]] += 1
        if i % 50 == 0 or i == total:
            print(
                f"  {i}/{total}  ok={counts['ok']} skip={counts['skip']} "
                f"missing={counts['missing']} error={counts['error']}",
                flush=True,
            )

    results = flatfiles.download_range(days, ROOT, workers=args.workers, progress=progress)
    total_bytes = sum(size for _, _, size in results)
    final = Counter(status for _, status, _ in results)
    print(f"\n完成：{dict(final)}  总体积≈{total_bytes / 1e9:.1f} GB")
    missing = [d for d, s, _ in results if s in ("missing", "error")]
    if missing:
        print(f"缺失/失败 {len(missing)} 天（前 10）：{missing[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
