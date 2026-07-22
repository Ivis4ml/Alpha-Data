"""P4 后半：分散度矩族并入面板并按标准引擎评价（clean 标签口径）。

对每个品种，把 ``dispersion_{P}.parquet`` 的 6 列 x 主题连接到信号面板，
用与 40 信号完全相同的引擎（``build_from_frame``）计算 IC / ICIR / 事件
研究。方向先验已预注册为 0（双侧）。

产物：data/cn_futures/analysis/v4/dispersion_grid.parquet

用法::

    .venv/bin/python scripts/v4_dispersion_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_signal_grid import build_from_frame  # noqa: E402
from v4_market_dispersion import DCOLS, PRODUCT_THEMES_V4  # noqa: E402
from v4_signal_grid import classify_locked, clean_labels, find_panel  # noqa: E402

V4 = ROOT / "data" / "cn_futures" / "analysis" / "v4"


def main() -> int:
    rows = []
    for product, themes in PRODUCT_THEMES_V4.items():
        dpath = V4 / f"dispersion_{product}.parquet"
        try:
            panel = pd.read_parquet(find_panel(product))
        except FileNotFoundError:
            print(f"[{product}] 无面板，跳过")
            continue
        if not dpath.exists():
            print(f"[{product}] 无分散度数据，跳过")
            continue
        disp = pd.read_parquet(dpath)
        panel["ts"] = pd.to_datetime(panel["ts"])
        disp["ts"] = pd.to_datetime(disp["ts"])
        df = panel.merge(disp, on="ts", how="left")
        df = clean_labels(classify_locked(df))
        sigs = [f"D_{c}_{th}" for th in themes for c in DCOLS
                if f"D_{c}_{th}" in df.columns]
        got = build_from_frame(df, product, signals=sigs)
        rows.extend(got)
        print(f"[{product}] 评价 {len(got)} 个分散度信号")
    out = pd.DataFrame(rows)
    out.to_parquet(V4 / "dispersion_grid.parquet", index=False)
    print(f"written {V4}/dispersion_grid.parquet ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
