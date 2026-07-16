"""v1.1 基线信号在扩展样本（2026-01-05 .. 07-13）上的重建。

方法与已提交的 v1.1 管道完全一致（``cn_features.window_signals`` + 主题聚合），
仅两处替换：

1. 逐笔来源从 ``daily_aligned`` 换成 v3 统一 tape（HF 段 + 自爬扩展段）。
2. 登记表用 ``cn_registry_v3.parquet``（同一规则、窗口右端延至 07-13）。

产出作为 v3 框架对比的**基线**：``data/cn_futures/analysis/v3/baseline_theme_signals.parquet``。

用法::

    .venv/bin/python scripts/v3_baseline_extended.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_cn_futures_polymarket import (  # noqa: E402
    aggregate_theme,
    market_signals_by_grid,
)

from alpha_data.cn_futures import store as fut_store  # noqa: E402
from alpha_data.polymarket import store  # noqa: E402
from alpha_data.polymarket.v3 import tape  # noqa: E402

OUT_DIR = ROOT / "data" / "cn_futures" / "analysis" / "v3"
REGISTRY_V3 = store.FEATURES_DIR / "cn_registry_v3.parquet"


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_parquet(REGISTRY_V3)
    con = store.connect()

    ids = registry["condition_id"].unique().tolist()
    trades: dict[str, pd.DataFrame] = {}
    for i, cid in enumerate(ids, 1):
        trades[cid] = tape.fetch_trades_cn(con, cid)
        if i % 50 == 0:
            print(f"  逐笔 {i}/{len(ids)}，耗时 {time.time() - t0:.0f}s")

    specs = fut_store.read_product_specs()
    market_sig = market_signals_by_grid(registry, trades, specs, agg_minutes=15)
    theme_sig = aggregate_theme(market_sig, registry)
    theme_sig["s_pre"] = theme_sig["s_night"] + theme_sig["s_gap"]
    # 无夜盘品种：s_night 恒 NaN，s_pre 即整段闭市信号。
    no_night = theme_sig["s_night"].isna() & theme_sig["s_gap"].notna()
    theme_sig.loc[no_night, "s_pre"] = theme_sig.loc[no_night, "s_gap"]

    out = OUT_DIR / "baseline_theme_signals.parquet"
    theme_sig.to_parquet(out, index=False)
    print(f"基线主题信号已写入 {out}（{len(theme_sig)} 行），耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
