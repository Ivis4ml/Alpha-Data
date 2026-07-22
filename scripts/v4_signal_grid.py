"""P2 评价引擎：涨跌停真封板 / 伪锁分类 + 双口径标签 + clean 信号网格。

设计（docs/next_phase_design.md §2 P2）：

- **真封板** = 该分钟 ``high == low`` 且收盘价落在当日极值上
  （``close >= 当日 high`` 或 ``close <= 当日 low``）：价格被涨跌停约束、
  该处收益不可实现。**伪锁** = 其余 ``high == low``（清淡或最小变动价位），
  不剔除、仅打标。
- **clean 标签**：入场分钟为真封板，该 bar 全部 ``fwd_k`` 置缺失；前向
  窗口 (t, t+k] 内任一分钟真封板，``fwd_k`` 置缺失。raw 口径保留同报。
- **回归门**：raw 口径重算必须与既有 ``signal_grid.parquet`` 逐值一致，
  之后才允许输出 clean 网格（防口径漂移）。

产物（data/cn_futures/analysis/v4/）：
  ``signal_grid_clean.parquet``   clean 口径全信号网格（含扩展品种）
  ``locked_impact.parquet``       每品种真封板 / 伪锁占比与标签剔除率

用法::

    .venv/bin/python scripts/v4_signal_grid.py [--products SC,AU,...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_signal_grid import HORIZONS, build_from_frame  # noqa: E402

DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
V4 = ROOT / "data" / "cn_futures" / "analysis" / "v4"
#: v3 冻结面板所在（5 品种）；v4 扩展面板由 v4_panel_build.py 写入 V4。
PANEL_DIRS = (DEF, V4)


def find_panel(product: str) -> Path:
    for d in PANEL_DIRS:
        p = d / f"panel_{product}.parquet"
        if p.exists():
            return p
    raise FileNotFoundError(f"panel_{product}.parquet 不在 {PANEL_DIRS}")


def classify_locked(df: pd.DataFrame) -> pd.DataFrame:
    """加列 ``true_lock`` / ``pseudo_lock``（按交易日极值判定封板）。"""
    day_hi = df.groupby("trade_date")["high"].transform("max")
    day_lo = df.groupby("trade_date")["low"].transform("min")
    locked = df["high"] == df["low"]
    at_extreme = (df["close"] >= day_hi) | (df["close"] <= day_lo)
    df = df.copy()
    df["true_lock"] = (locked & at_extreme).astype(int)
    df["pseudo_lock"] = (locked & ~at_extreme).astype(int)
    return df


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """把 ``fwd_k`` 换成 clean 口径（原值存 ``fwd_k_raw``）。

    剔除规则：入场分钟真封板，或 (t, t+k] 内任一分钟真封板。
    前向窗口检查按行号进行；跨段的 bar 本就是缺失（段规则先行），
    故这里无需再判段。
    """
    df = df.copy()
    tl = df["true_lock"].to_numpy(dtype=float)
    n = len(df)
    # fut_any[k][i] = (i, i+k] 内是否有真封板
    csum = np.concatenate([[0.0], np.cumsum(tl)])
    for k in HORIZONS:
        end = np.minimum(np.arange(n) + k, n - 1)
        fut_any = (csum[end + 1] - csum[np.arange(n) + 1]) > 0
        bad = (tl > 0) | fut_any
        raw = df[f"fwd_{k}"].to_numpy(dtype=float)
        df[f"fwd_{k}_raw"] = raw
        df[f"fwd_{k}"] = np.where(bad, np.nan, raw)
    return df


def impact_row(product: str, df: pd.DataFrame) -> dict:
    row: dict[str, object] = {
        "product": product,
        "n_rows": int(len(df)),
        "locked_share": float((df["true_lock"] + df["pseudo_lock"]).mean()),
        "true_lock_share": float(df["true_lock"].mean()),
        "pseudo_lock_share": float(df["pseudo_lock"].mean()),
    }
    for k in HORIZONS:
        raw_ok = np.isfinite(df[f"fwd_{k}_raw"].to_numpy(dtype=float))
        cln_ok = np.isfinite(df[f"fwd_{k}"].to_numpy(dtype=float))
        row[f"dropped_{k}"] = int(raw_ok.sum() - cln_ok.sum())
        row[f"dropped_share_{k}"] = float(
            (raw_ok.sum() - cln_ok.sum()) / max(raw_ok.sum(), 1))
    return row


def regression_gate(product: str) -> None:
    """raw 口径重算必须与既有 signal_grid.parquet 逐值一致。"""
    stored = pd.read_parquet(DEF / "signal_grid.parquet")
    stored = stored[stored["product"] == product].reset_index(drop=True)
    if stored.empty:
        return                      # 扩展品种无既有网格，跳过
    df = pd.read_parquet(find_panel(product))
    fresh = pd.DataFrame(build_from_frame(df, product))
    num = [c for c in stored.columns
           if pd.api.types.is_numeric_dtype(stored[c])]
    a = stored.sort_values("signal").reset_index(drop=True)
    b = fresh.sort_values("signal").reset_index(drop=True)
    if list(a["signal"]) != list(b["signal"]):
        raise AssertionError(f"{product}: 信号集合与既有网格不一致")
    for c in num:
        if c not in b.columns:
            raise AssertionError(f"{product}: 缺列 {c}")
        av = a[c].to_numpy(dtype=float)
        bv = b[c].to_numpy(dtype=float)
        ok = np.isclose(av, bv, rtol=1e-9, atol=1e-12, equal_nan=True)
        if not ok.all():
            i = int(np.flatnonzero(~ok)[0])
            raise AssertionError(
                f"{product}: 回归门失败 {c}[{a['signal'].iloc[i]}] "
                f"{av[i]} != {bv[i]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", default="SC,AU,AG,CU,M")
    parser.add_argument("--skip-regression", action="store_true")
    args = parser.parse_args()
    products = args.products.split(",")

    V4.mkdir(parents=True, exist_ok=True)
    rows, impacts = [], []
    for p in products:
        if not args.skip_regression:
            print(f"[{p}] 回归门（raw 口径复现既有网格）...")
            regression_gate(p)
        df = pd.read_parquet(find_panel(p))
        df = classify_locked(df)
        df = clean_labels(df)
        impacts.append(impact_row(p, df))
        print(f"[{p}] clean 网格 ...")
        rows.extend(build_from_frame(df, p))

    grid = pd.DataFrame(rows)
    grid.to_parquet(V4 / "signal_grid_clean.parquet", index=False)
    imp = pd.DataFrame(impacts)
    imp.to_parquet(V4 / "locked_impact.parquet", index=False)
    print(imp[["product", "true_lock_share", "pseudo_lock_share",
               "dropped_share_15"]].round(5).to_string(index=False))
    print(f"written {V4}/signal_grid_clean.parquet ({len(grid)} rows), "
          f"locked_impact.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
