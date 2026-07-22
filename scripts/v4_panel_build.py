"""P5 品种扩展：CF / I / IF 信号面板（与冻结 5 面板同构，零数据成本）。

三品种在 ``cn_registry_v3`` 已有事前映射（CF/I 属 us_china_trade 13 市场、
IF 属 taiwan_risk 8 市场）、分钟数据已在库，属零事后选择的扩展。

纪律（docs/next_phase_design.md §2 P5）：

- 面板构造逻辑**只读 import** 冻结脚本 ``v3_defense_build``（cn_panel /
  pm_minute_theme / add_signals），不复制、不改动；
- **回归门**：先用同一 import 路径重建冻结品种 M 的面板并与存档逐值
  比对，通过后才构建新品种（防 import 路径与当年运行环境有口径漂移）；
- IF 为纯日盘品种：夜盘相关信号（X7 恒 0 等）会退化，网格评价时按
  实际有限值计数、不虚计分母。

产物：data/cn_futures/analysis/v4/panel_{CF,I,IF}.parquet

用法::

    .venv/bin/python scripts/v4_panel_build.py [--skip-regression]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_defense_build import (  # noqa: E402  只读 import 冻结逻辑
    REGISTRY,
    add_signals,
    cn_panel,
    pm_minute_theme,
)
from v4_market_dispersion import PRODUCT_THEMES_V4  # noqa: E402

from alpha_data.polymarket import store  # noqa: E402

DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
V4 = ROOT / "data" / "cn_futures" / "analysis" / "v4"
NEW_PRODUCTS = ("CF", "I", "IF")
START_UTC, END_UTC = "2026-01-02 00:00:00", "2026-07-13 16:00:00"


def build_panel(con, registry: pd.DataFrame, product: str) -> pd.DataFrame:
    """冻结装配方式的逐行复刻：cn_panel -> merge 主题分钟 -> add_signals。"""
    themes = PRODUCT_THEMES_V4[product]
    panel = cn_panel(product)
    for th in themes:
        pm = pm_minute_theme(con, registry, th, product, START_UTC, END_UTC)
        pm = pm.rename(columns={c: f"{c}_{th}" for c in
                                ("dl", "flow", "usdc", "n", "n_mkts")})
        panel = panel.merge(pm, on="ts", how="left")
    return add_signals(panel, themes)


def regression_gate(con, registry: pd.DataFrame) -> None:
    """重建冻结品种 M 并与存档面板逐值比对（数值列，NaN 视为相等）。"""
    fresh = build_panel(con, registry, "M")
    stored = pd.read_parquet(DEF / "panel_M.parquet")
    if list(fresh.columns) != list(stored.columns):
        raise AssertionError(
            f"列集合漂移: {set(fresh.columns) ^ set(stored.columns)}")
    if len(fresh) != len(stored):
        raise AssertionError(f"行数漂移: {len(fresh)} != {len(stored)}")
    for c in stored.columns:
        a, b = stored[c], fresh[c]
        if pd.api.types.is_numeric_dtype(a):
            ok = np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                            rtol=1e-9, atol=1e-12, equal_nan=True)
        else:
            ok = (a.astype(str) == b.astype(str)).to_numpy()
        if not ok.all():
            i = int(np.flatnonzero(~ok)[0])
            raise AssertionError(
                f"回归门失败: 列 {c} 行 {i}: {a.iloc[i]!r} != {b.iloc[i]!r}")
    print("回归门通过：M 面板逐值复现存档")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-regression", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    V4.mkdir(parents=True, exist_ok=True)
    con = store.connect()
    registry = pd.read_parquet(REGISTRY)

    if not args.skip_regression:
        regression_gate(con, registry)

    for product in NEW_PRODUCTS:
        print(f"[{product}] 面板 ...")
        df = build_panel(con, registry, product)
        df.to_parquet(V4 / f"panel_{product}.parquet", index=False)
        print(f"[{product}] {len(df)} 分钟, {df['trade_date'].nunique()} 日, "
              f"夜盘占比 {(df['session'] == 'night').mean():.1%}, "
              f"PM 活跃分钟 {(df['N1'] != 0).mean():.1%}")
    print(f"完成，耗时 {time.time() - t0:.0f}s -> {V4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
