"""P0-1：C8 的严格增量检验（证伪型，判定标准先于计算注册）。

问题：§12.14 的分解显示 C8 的强度主要由期货反转腿承载。本检验回答
审计的最终问题：<b>在价格基线之上，PM 腿是否有任何可辨别的分钟级
增量</b>。

设计（全部以交易日为推断单位）：
  基线 X_price = [z(r120) 反转腿, mom15_z, rv15_z, volu15_z, 夜盘哑变量]
  PM 腿 x_pm  = z(dl120)（交易分钟窗）；另构造墙钟 120 分钟对照
  目标 y      = fwd_15
  a) partial RankIC：y 与 x_pm 各对基线残差化后，逐日日内 Spearman，
     对日序列 HAC(5)。分全样本 / 日盘 / 夜盘三个口径。
  b) 展开窗 OOS ΔR²：逐日重估 price-only 与 price+PM 两个线性模型
     （严格用 < t 日的分钟），比较逐日 MSE；ΔR² 为总体口径，DM t 为
     逐日 MSE 差序列的 HAC(5) 均值检验。
  c) 安慰剂族（各 199 次，统计量 = partial RankIC 的 HAC t）：
     时间置换 = PM 腿按日循环移位；方向置换 = PM 腿逐日随机翻号；
     映射置换 = SC 与 M 互换 PM 腿（主题不相交）。
  d) 墙钟对照：x_pm 换成过去 120 墙钟分钟的 PM 累积，重复 (a)。

注册判定（缺一不过）：PM 腿具有分钟级增量当且仅当
  |partial HAC t| >= 2 且 时间置换 p < 0.05 且 OOS ΔR² > 0 且 DM t >= 2。

产物：analysis/v3/defense/c8_incremental.parquet
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
N_PLACEBO = 199
RNG = np.random.default_rng(20260718)


def zscore(s: pd.Series) -> pd.Series:
    mu = s.rolling(ZWIN, min_periods=ZMIN).mean()
    sd = s.rolling(ZWIN, min_periods=ZMIN).std()
    return (s - mu) / sd.replace(0, np.nan)


def load_product(prod: str) -> pd.DataFrame:
    df = pd.read_parquet(D / f"panel_{prod}.parquet",
                         columns=["ts", "trade_date", "session", "N1", "r1",
                                  "fwd_15", "mom15_z", "rv15_z", "volu15_z"])
    dl120 = df["N1"].rolling(120, min_periods=30).sum()
    r120 = df["r1"].fillna(0.0).rolling(120, min_periods=30).sum()
    df["x_pm"] = zscore(dl120)
    df["x_rev"] = zscore(r120)
    wall = (df.set_index("ts")["N1"].rolling("120min", min_periods=1)
            .sum().to_numpy())
    df["x_pm_wall"] = zscore(pd.Series(wall, index=df.index))
    df["night"] = (df["session"] == "night").astype(float)
    keep = ["trade_date", "session", "fwd_15", "x_pm", "x_pm_wall",
            "x_rev", "mom15_z", "rv15_z", "volu15_z", "night"]
    return df[keep].dropna(subset=["fwd_15", "x_pm", "x_rev"]).reset_index(
        drop=True)


BASE_COLS = ["x_rev", "mom15_z", "rv15_z", "volu15_z", "night"]


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return y - Xc @ beta


def daily_partial_t(df: pd.DataFrame, x: np.ndarray,
                    mask: np.ndarray | None = None) -> tuple[float, float,
                                                             int]:
    """y 与 x 对基线残差化后的逐日 RankIC 与 HAC t。"""
    sel = np.ones(len(df), bool) if mask is None else mask
    sub = df[sel]
    Xb = sub[BASE_COLS].fillna(0.0).to_numpy()
    ry = residualize(sub["fwd_15"].to_numpy(), Xb)
    rx = residualize(x[sel], Xb)
    tmp = pd.DataFrame({"d": sub["trade_date"].to_numpy(),
                        "rx": rx, "ry": ry})
    ics = []
    for _, g in tmp.groupby("d"):
        if len(g) < 30 or g["rx"].nunique() < 5:
            continue
        ics.append(spearmanr(g["rx"], g["ry"]).statistic)
    s = pd.Series(ics).dropna()
    if len(s) < 20:
        return float("nan"), float("nan"), len(s)
    res = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(res.params[0]), float(res.tvalues[0]), len(s)


def oos_delta_r2(df: pd.DataFrame) -> tuple[float, float, int]:
    """展开窗逐日重估：price-only vs price+PM 的 OOS 比较。"""
    days = df["trade_date"].unique().tolist()
    Xp = np.column_stack([np.ones(len(df)),
                          df[BASE_COLS].fillna(0.0).to_numpy()])
    Xj = np.column_stack([Xp, df["x_pm"].to_numpy()])
    y = df["fwd_15"].to_numpy()
    kp, kj = Xp.shape[1], Xj.shape[1]
    XtXp = np.zeros((kp, kp)); Xtyp = np.zeros(kp)
    XtXj = np.zeros((kj, kj)); Xtyj = np.zeros(kj)
    day_idx = {d: np.flatnonzero(df["trade_date"].to_numpy() == d)
               for d in days}
    sse_p = sse_j = ss0 = 0.0
    ybar_hist: list[float] = []
    d_series = []
    for i, d in enumerate(days):
        idx = day_idx[d]
        if i >= 30:
            try:
                bp = np.linalg.solve(XtXp + 1e-8 * np.eye(kp), Xtyp)
                bj = np.linalg.solve(XtXj + 1e-8 * np.eye(kj), Xtyj)
            except np.linalg.LinAlgError:
                bp = bj = None
            if bp is not None:
                ep = y[idx] - Xp[idx] @ bp
                ej = y[idx] - Xj[idx] @ bj
                sse_p += float(ep @ ep); sse_j += float(ej @ ej)
                mu = np.mean(ybar_hist)
                e0 = y[idx] - mu
                ss0 += float(e0 @ e0)
                d_series.append({"d": d, "diff": float(np.mean(ep**2 - ej**2))})
        XtXp += Xp[idx].T @ Xp[idx]; Xtyp += Xp[idx].T @ y[idx]
        XtXj += Xj[idx].T @ Xj[idx]; Xtyj += Xj[idx].T @ y[idx]
        ybar_hist.append(float(y[idx].mean()))
    ds = pd.DataFrame(d_series)
    if ss0 <= 0 or len(ds) < 20:
        return float("nan"), float("nan"), len(ds)
    dr2 = (sse_p - sse_j) / ss0
    res = sm.OLS(ds["diff"].to_numpy(), np.ones(len(ds))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(dr2), float(res.tvalues[0]), len(ds)


def placebo_p(df: pd.DataFrame, t_real: float, kind: str) -> float:
    """时间循环移位 / 逐日翻号安慰剂的 |t| 经验 p 值。"""
    days = df["trade_date"].unique()
    day_of = df["trade_date"].to_numpy()
    pos_by_day = {d: np.flatnonzero(day_of == d) for d in days}
    x0 = df["x_pm"].to_numpy()
    hits = 0
    for _ in range(N_PLACEBO):
        x = np.empty_like(x0)
        if kind == "shift":
            k = int(RNG.integers(10, len(days) - 10))
            for i, d in enumerate(days):
                src = days[(i + k) % len(days)]
                a, b = pos_by_day[d], pos_by_day[src]
                m = min(len(a), len(b))
                x[a[:m]] = x0[b[:m]]
                if len(a) > m:
                    x[a[m:]] = x0[b[-1]] if len(b) else 0.0
        else:  # sign
            flips = RNG.choice([-1.0, 1.0], size=len(days))
            x = x0.copy()
            for f, d in zip(flips, days):
                x[pos_by_day[d]] *= f
        _, t, _ = daily_partial_t(df, x)
        if np.isfinite(t) and abs(t) >= abs(t_real):
            hits += 1
    return (hits + 1) / (N_PLACEBO + 1)


def main() -> int:
    data = {p: load_product(p) for p in PRODUCTS}
    rows = []
    for prod in PRODUCTS:
        df = data[prod]
        x = df["x_pm"].to_numpy()
        ic, t, nd = daily_partial_t(df, x)
        day_m = (df["session"] == "day").to_numpy()
        ic_d, t_d, _ = daily_partial_t(df, x, day_m)
        ic_n, t_n, _ = daily_partial_t(df, x, ~day_m)
        ic_w, t_w, _ = daily_partial_t(df, df["x_pm_wall"].fillna(0.0)
                                       .to_numpy())
        dr2, dm_t, n_oos = oos_delta_r2(df)
        p_shift = placebo_p(df, t, "shift")
        p_sign = placebo_p(df, t, "sign")
        verdict = (abs(t) >= 2 and p_shift < 0.05 and dr2 > 0
                   and dm_t >= 2)
        rows.append({"product": prod, "partial_ic": ic, "partial_t": t,
                     "n_days": nd, "ic_day": ic_d, "t_day": t_d,
                     "ic_night": ic_n, "t_night": t_n,
                     "ic_wall": ic_w, "t_wall": t_w,
                     "oos_dr2": dr2, "dm_t": dm_t, "n_oos": n_oos,
                     "p_shift": p_shift, "p_sign": p_sign,
                     "verdict_pass": verdict})
        print(f"{prod}: partial {ic:+.4f} (t {t:+.2f}) | day {t_d:+.2f} "
              f"night {t_n:+.2f} wall {t_w:+.2f} | dR2 {dr2:+.5f} "
              f"(DM {dm_t:+.2f}) | p_shift {p_shift:.3f} "
              f"p_sign {p_sign:.3f} | pass={verdict}")

    # 映射安慰剂：SC 与 M 互换 PM 腿（主题不相交）
    for a, b in (("SC", "M"), ("M", "SC")):
        da, db = data[a], data[b]
        xb = (db.groupby("trade_date")["x_pm"].mean()
              .reindex(da["trade_date"]).to_numpy())
        ic, t, nd = daily_partial_t(da, np.nan_to_num(xb))
        rows.append({"product": f"{a}(用{b}的PM腿)", "partial_ic": ic,
                     "partial_t": t, "n_days": nd,
                     "verdict_pass": False})
        print(f"映射安慰剂 {a}<-{b}: {ic:+.4f} (t {t:+.2f})")

    out = pd.DataFrame(rows)
    out.to_parquet(D / "c8_incremental.parquet", index=False)
    print("\nwritten c8_incremental.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
