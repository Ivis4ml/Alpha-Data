"""外部审计复核一：C8 双腿分解（PM 腿 vs 期货反转腿）。

审计指控：C8 = z(过去 120 交易分钟 PM 变化) − z(过去 120 交易分钟期货
收益)，第二腿本身是短期价格反转因子，C8 的跨品种稳定性可能主要继承
自期货自身均值回归而非 Polymarket 信息。

复核设计：逐品种把 C8 拆成三个序列各自评估：
  PM 腿   = z(dl120)          只含 Polymarket 信息
  反转腿  = −z(r120)          只含期货自身价格（普通反转因子）
  完整 C8 = PM 腿 + 反转腿    报告原口径
推断单位改为交易日：逐日计算日内 RankIC（signal vs fwd_15），对日 IC
序列做 HAC(5) 均值检验，消除重复分钟的样本量放大。同时验证重建的 C8
与面板存储值一致（防口径漂移）。

产物：analysis/v3/defense/c8_decomposition.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
ZWIN, ZMIN = 4800, 960
PRODUCTS = ["SC", "AU", "AG", "CU", "M"]


def zscore(s: pd.Series) -> pd.Series:
    mu = s.rolling(ZWIN, min_periods=ZMIN).mean()
    sd = s.rolling(ZWIN, min_periods=ZMIN).std()
    return (s - mu) / sd.replace(0, np.nan)


def daily_ic_t(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
    ics = []
    for _, g in df.groupby("trade_date"):
        gg = g.dropna(subset=[col, "fwd_15"])
        if len(gg) < 30 or gg[col].nunique() < 5:
            continue
        ics.append(spearmanr(gg[col], gg["fwd_15"]).statistic)
    s = pd.Series(ics).dropna()
    if len(s) < 20:
        return float("nan"), float("nan"), len(s)
    res = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(res.params[0]), float(res.tvalues[0]), len(s)


def main() -> int:
    rows = []
    for prod in PRODUCTS:
        df = pd.read_parquet(D / f"panel_{prod}.parquet",
                             columns=["trade_date", "N1", "r1", "C8",
                                      "fwd_15"])
        dl120 = df["N1"].rolling(120, min_periods=30).sum()
        r120 = df["r1"].fillna(0.0).rolling(120, min_periods=30).sum()
        df["leg_pm"] = zscore(dl120).fillna(0.0)
        df["leg_rev"] = -zscore(r120).fillna(0.0)
        rebuilt = df["leg_pm"] + df["leg_rev"]
        max_diff = float((rebuilt - df["C8"]).abs().max())
        for col, label in (("leg_pm", "PM腿"), ("leg_rev", "反转腿"),
                           ("C8", "完整C8")):
            m, t, n = daily_ic_t(df, col)
            rows.append({"product": prod, "leg": label, "rank_ic": m,
                         "hac_t": t, "n_days": n, "rebuild_maxdiff": max_diff})
        print(f"{prod}: rebuild max|diff|={max_diff:.2e}")
    out = pd.DataFrame(rows)
    out.to_parquet(D / "c8_decomposition.parquet", index=False)
    print(out.pivot_table(index="product", columns="leg",
                          values=["rank_ic", "hac_t"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
