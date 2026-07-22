"""P3 时序分位检验：decile 多空 + 单调性 + 换手成本净边界。

回应评审意见 5（top-bottom 分位检测）与 6（成本经济门槛）。

口径（docs/next_phase_design.md §2 P3）：

- **分档**：第 d 日的断点取该信号此前 20 个交易日非零取值的十分位
  （严格历史断点，防前视）；当日各 bar 按断点归档，零值与真封板分钟
  不参与。
- **多空价差**：LS_k(d) = 档10当日均值 − 档1当日均值（fwd_k 为 P2 的
  clean 口径），对日序列做 HAC(5) t。
- **单调性**：pooled 档均值对档序的 Spearman；逐日档序回归斜率的
  日间 t。两者同号且斜率 |t| >= 2 记单调。
- **成本净边界**（1 分钟再平衡的真实策略核算，无重叠）：状态
  s_t ∈ {+1(档10), −1(档1), 0}，日毛利 = Σ s_t·fwd1(bp)，
  日成本 = 仓位变动单位数 × (COST_BP/2)（16bp 为完整换手，单边 8bp），
  盈亏平衡毛价差 = 成本 / 毛利单位。
- **准入门**（机械，评价前定死）：连续 / 计数信号；两端极端档非空的
  交易日占比 >= 60%。
- **记账**：k=15、全时段的 LS 检验为注册确认终点，并入扩展主族；
  其余视界为描述性剖面。

产物：data/cn_futures/analysis/v4/quantile_grid.parquet

用法::

    .venv/bin/python scripts/v3_quantile.py [--products SC,AU,...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_signal_grid import HORIZONS, SIGNALS, classify  # noqa: E402
from v4_signal_grid import classify_locked, clean_labels, find_panel  # noqa: E402

V4 = ROOT / "data" / "cn_futures" / "analysis" / "v4"
N_Q = 10
LOOKBACK_DAYS = 20
MIN_COVER = 0.60          # 两端极端档非空的交易日占比下限
COST_BP = 16.0            # 完整换手成本（冻结常数）
HAC_LAG = 5


def hac_t(x: np.ndarray, lag: int = HAC_LAG) -> float:
    """均值为零的 HAC(Newey-West) t。"""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    mu = x.mean()
    e = x - mu
    g0 = float(e @ e) / n
    s = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if s <= 0:
        return float("nan")
    return mu / np.sqrt(s / n)


def assign_deciles(df: pd.DataFrame, sig: str) -> np.ndarray:
    """严格历史断点的档位（1..10；不可分档处为 0）。"""
    days = df["trade_date"].to_numpy()
    uniq = pd.unique(days)
    vals = df[sig].to_numpy(dtype=float)
    ok_base = np.isfinite(vals) & (vals != 0.0) \
        & (df["true_lock"].to_numpy() == 0)
    out = np.zeros(len(df), dtype=int)
    day_index = {d: i for i, d in enumerate(uniq)}
    # 每日历史断点：此前 LOOKBACK_DAYS 日的非零值十分位
    day_rows = {d: np.flatnonzero(days == d) for d in uniq}
    for i, d in enumerate(uniq):
        if i < LOOKBACK_DAYS:
            continue
        hist_days = uniq[i - LOOKBACK_DAYS:i]
        hv = np.concatenate([vals[day_rows[hd]] for hd in hist_days])
        hv = hv[np.isfinite(hv) & (hv != 0.0)]
        if len(hv) < 200 or len(np.unique(hv)) < N_Q * 2:
            continue
        edges = np.quantile(hv, np.linspace(0, 1, N_Q + 1)[1:-1])
        rows = day_rows[d]
        m = ok_base[rows]
        if not m.any():
            continue
        q = np.searchsorted(edges, vals[rows[m]], side="right") + 1
        out[rows[m]] = q
    del day_index
    return out


def evaluate(df: pd.DataFrame, product: str, sig: str) -> dict | None:
    q = assign_deciles(df, sig)
    if (q > 0).sum() == 0:
        return None
    days = df["trade_date"].to_numpy()
    uniq_eval = pd.unique(days[q > 0])
    # 准入门：两端极端档非空的交易日占比
    cover_days = 0
    for d in uniq_eval:
        rows = np.flatnonzero(days == d)
        qq = q[rows]
        if (qq == 1).any() and (qq == N_Q).any():
            cover_days += 1
    cover = cover_days / max(len(uniq_eval), 1)
    rec: dict[str, object] = {
        "product": product, "signal": sig,
        "n_days_eval": int(len(uniq_eval)),
        "extreme_cover": float(cover),
        "eligible": bool(cover >= MIN_COVER),
    }
    if cover < MIN_COVER:
        return rec

    # ---- 各视界：日度 LS 价差 + 单调性 ----
    for k in HORIZONS:
        f = df[f"fwd_{k}"].to_numpy(dtype=float)
        ls_daily, slope_daily = [], []
        dec_sum = np.zeros(N_Q)
        dec_cnt = np.zeros(N_Q)
        for d in uniq_eval:
            rows = np.flatnonzero(days == d)
            qq, ff = q[rows], f[rows]
            m = (qq > 0) & np.isfinite(ff)
            if not m.any():
                continue
            qq, ff = qq[m], ff[m]
            top, bot = ff[qq == N_Q], ff[qq == 1]
            if len(top) and len(bot):
                ls_daily.append(top.mean() - bot.mean())
            if len(np.unique(qq)) >= 3:
                slope_daily.append(
                    float(np.polyfit(qq.astype(float), ff, 1)[0]))
            for j in range(1, N_Q + 1):
                sel = ff[qq == j]
                dec_sum[j - 1] += sel.sum()
                dec_cnt[j - 1] += len(sel)
        ls = np.asarray(ls_daily) * 1e4
        rec[f"ls_mean_bp_{k}"] = float(ls.mean()) if len(ls) else np.nan
        rec[f"ls_t_{k}"] = hac_t(ls)
        dec_mean = np.where(dec_cnt > 0, dec_sum / np.maximum(dec_cnt, 1),
                            np.nan)
        okd = np.isfinite(dec_mean)
        rec[f"mono_rho_{k}"] = (float(stats.spearmanr(
            np.arange(1, N_Q + 1)[okd], dec_mean[okd]).statistic)
            if okd.sum() >= 3 else np.nan)
        sl = np.asarray(slope_daily)
        sl = sl[np.isfinite(sl)]
        rec[f"slope_t_{k}"] = (float(sl.mean() / sl.std(ddof=1)
                                     * np.sqrt(len(sl)))
                               if len(sl) >= 10 and sl.std(ddof=1) > 0
                               else np.nan)
        rho, st_ = rec[f"mono_rho_{k}"], rec[f"slope_t_{k}"]
        rec[f"monotone_{k}"] = bool(
            np.isfinite(rho) and np.isfinite(st_)
            and np.sign(rho) == np.sign(st_) and abs(st_) >= 2.0)

    # ---- 1 分钟再平衡策略的成本核算（真实、无重叠）----
    s_state = np.where(q == N_Q, 1.0, np.where(q == 1, -1.0, 0.0))
    f1 = df["fwd_1"].to_numpy(dtype=float)
    pnl = np.where(np.isfinite(f1), s_state * f1, 0.0) * 1e4
    trades = np.abs(np.diff(s_state, prepend=0.0))
    gross_d, cost_d = [], []
    for d in uniq_eval:
        rows = np.flatnonzero(days == d)
        gross_d.append(pnl[rows].sum())
        cost_d.append(trades[rows].sum() * COST_BP / 2.0)
    gross = np.asarray(gross_d)
    cost = np.asarray(cost_d)
    rec["strat_gross_bp_day"] = float(gross.mean())
    rec["strat_cost_bp_day"] = float(cost.mean())
    rec["strat_net_bp_day"] = float((gross - cost).mean())
    rec["strat_net_t"] = hac_t(gross - cost)
    rec["breakeven_ratio"] = (float(cost.mean() / abs(gross.mean()))
                              if gross.mean() != 0 else np.nan)
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", default="SC,AU,AG,CU,M,CF,I,IF")
    args = parser.parse_args()

    V4.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in args.products.split(","):
        try:
            path = find_panel(p)
        except FileNotFoundError:
            print(f"[{p}] 无面板，跳过")
            continue
        df = pd.read_parquet(path)
        df = clean_labels(classify_locked(df))
        n_run = 0
        for sig in SIGNALS:
            if sig not in df.columns:
                continue
            if classify(df[sig].astype(float)) == "离散":
                continue          # 离散信号走事件研究，不做分位
            r = evaluate(df, p, sig)
            if r is not None:
                rows.append(r)
                n_run += 1
        print(f"[{p}] 分位评价 {n_run} 个信号")
    out = pd.DataFrame(rows)
    out.to_parquet(V4 / "quantile_grid.parquet", index=False)
    el = out[out["eligible"] == True]  # noqa: E712
    print(f"\n合格 (signal, product) 对：{len(el)} / {len(out)}")
    if len(el):
        t15 = el["ls_t_15"].abs()
        print(f"k=15 注册终点：{int(t15.notna().sum())} 个检验，"
              f"max|t| = {t15.max():.3f}")
        print(f"单调(k=15)通过：{int(el['monotone_15'].sum())}")
        print(f"策略净收益为正：{int((el['strat_net_bp_day'] > 0).sum())}"
              f"；盈亏平衡倍数中位 {el['breakeven_ratio'].median():.1f}")
    print(f"written {V4}/quantile_grid.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
