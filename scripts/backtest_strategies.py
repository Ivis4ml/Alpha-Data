"""可交易性检验：四个无前视策略的 Sharpe、成本敏感性与杠杆。

策略定义（信号窗口严格早于持仓窗口，逐条注明）：

S1 日盘方向：position = sign(s_pre(t))，s_pre = s_night + s_gap（两者均在
    09:00 前结束），持有当日日盘 [09:00 开, 15:00 收]，收益 = pos × r_day。
S2 夜盘方向：position = sign(s_gap_pm(t))（傍晚闭市段 [15:00, 21:00) 的信号，
    21:00 夜盘开盘时已知），持有夜盘，收益 = pos × r_night。
S3 反转（升水回归）：position = −sign(s_all(t))（t 日收盘已知），持有其后
    3 个交易日（重叠三档，每日仓位 = 三档均值），收益 = pos × r_cc(t+1)。
    换月日收益缺失记 0（平仓跳过）。
S5 事件（E3 新市场创建）：夜盘内事件桶收盘进场，120 分钟后（或夜盘结束）
    出场，方向 = orientation；逐事件收益按交易日汇总为日收益。

指标：年化收益 / 年化波动 / Sharpe（rf=0，√252 年化）± 95% CI
（SE = sqrt((1+SR²/2)/T)）、最大回撤、胜率、换手。成本：单边 bp 参数
（基线 0 与 4bp——SC 一跳约 1.5bp + 手续费，taker 双边约 4bp/单边 2bp 偏保守）。

杠杆：
- 固定杠杆 L ∈ {1,2,3,5}：无成本时 Sharpe 对 L 不变（收益与波动同乘），
  变化的是年化收益、回撤与保证金占用（SC 保证金约 10-15%，L 上限约 7 倍）；
  成本与仓位同比例，净 Sharpe 亦近似不变。表格如实展示这一点。
- 目标波动动态杠杆（真正可能改善 Sharpe 的杠杆用法）：
  L_t = min(L_cap, σ_target / σ̂_t)，σ̂_t 由 K5 的 HAR-lite 模型
  （|s_gap(t)| 与 RV(t−1)）给出。系数为全样本估计，存在样本内偏差，如实标注。

产物：analysis/deep/backtest_*.parquet 与 docs/figures/deep/l1/l2.png。

用法::

    .venv/bin/python scripts/backtest_strategies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pub_style  # noqa: E402
from analyze_cn_futures_polymarket import (  # noqa: E402
    REGISTRY_PATH,
    SIGNAL_END,
    load_all_trades,
)
from deep_analysis_cn_polymarket import DEEP_DIR, FIG_DIR, futures_fine_returns  # noqa: E402

from alpha_data.cn_futures import store as fut_store  # noqa: E402

P = pub_style.PALETTE
ANN = 252
COST_BP = 4.0  # 单边成本（bp），基线成本情形


def metrics(daily: pd.Series, positions: pd.Series | None = None,
            cost_bp: float = 0.0) -> dict:
    """日收益序列 -> 年化指标。成本按仓位变动（换手）扣除。"""
    r = daily.fillna(0.0).to_numpy(dtype="float64")
    if positions is not None:
        turn = positions.fillna(0.0).diff().abs().fillna(
            positions.fillna(0.0).abs())
        r = r - turn.to_numpy() * cost_bp / 1e4
    T = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    # 日频 SR 的渐近 SE（Lo, 2002），年化时 SR 与 SE 同乘 sqrt(252)：
    # 75 天下年化 Sharpe 的置信区间必然极宽，如实呈现。
    sr_d = mu / sd if sd > 0 else np.nan
    sr = sr_d * np.sqrt(ANN) if np.isfinite(sr_d) else np.nan
    se = (np.sqrt((1 + 0.5 * sr_d**2) / T) * np.sqrt(ANN)
          if np.isfinite(sr_d) else np.nan)
    eq = np.cumsum(r)
    mdd = float((np.maximum.accumulate(eq) - eq).max())
    active = r != 0
    return {
        "T_days": T, "ann_ret_pct": mu * ANN * 100,
        "ann_vol_pct": sd * np.sqrt(ANN) * 100,
        "sharpe": sr, "sharpe_lo": sr - 1.96 * se, "sharpe_hi": sr + 1.96 * se,
        "max_dd_pct": mdd * 100,
        "hit_rate": float((r[active] > 0).mean()) if active.any() else np.nan,
        "n_active": int(active.sum()),
    }


def build_daily_strategies() -> tuple[pd.DataFrame, pd.DataFrame]:
    """S1/S2/S3 的逐日仓位与收益（theme = mideast 与 oil 各一版）。"""
    theme_fine = pd.read_parquet(DEEP_DIR / "theme_signals_fine.parquet")
    rets = futures_fine_returns(["SC"])
    out_r, out_pos = {}, {}
    base = rets[rets["product"] == "SC"].sort_values("trade_date").reset_index(drop=True)
    r_cc_next = base["r_cc"].shift(-1)  # t 收盘进场，赚 t+1 的收对收

    for theme, tag in [("mideast_conflict", "mid"), ("oil_price", "oil")]:
        g = theme_fine[(theme_fine["theme"] == theme)
                       & (theme_fine["product"] == "SC")]
        m = base.merge(g, on="trade_date", how="left").sort_values("trade_date")
        m = m.reset_index(drop=True)
        in_win = m["trade_date"] <= SIGNAL_END

        s_pre = (m["s_night"].fillna(0) + m["s_gap"]).where(in_win)
        pos1 = np.sign(s_pre).fillna(0.0)
        out_pos[f"S1_day_{tag}"] = pos1
        out_r[f"S1_day_{tag}"] = pos1 * m["r_day"].fillna(0)

        pos2 = np.sign(m["s_gap_pm"].where(in_win)).fillna(0.0)
        out_pos[f"S2_night_{tag}"] = pos2
        out_r[f"S2_night_{tag}"] = pos2 * m["r_night"].fillna(0)

        s_all = m[["s_night", "s_gap", "s_day"]].sum(axis=1, skipna=False)
        sig3 = -np.sign(s_all.where(in_win)).fillna(0.0)
        pos3 = sig3.rolling(3, min_periods=1).mean()  # 重叠三档
        out_pos[f"S3_rev_{tag}"] = pos3
        out_r[f"S3_rev_{tag}"] = (pos3 * r_cc_next.fillna(0)).fillna(0)

    # 注意不可用 DataFrame(dict, index=...)：那会按新索引重索引出全 NaN。
    pos_df = pd.DataFrame(out_pos).set_axis(base["trade_date"], axis=0)
    ret_df = pd.DataFrame(out_r).set_axis(base["trade_date"], axis=0)
    return ret_df, pos_df


def build_event_strategy() -> pd.Series:
    """S5：E3 事件后 120 分钟持有的逐日汇总收益。"""
    registry = pd.read_parquet(REGISTRY_PATH)
    sub = registry[(registry["theme"].isin(
        ["oil_price", "mideast_conflict", "russia_ukraine"]))
        & (registry["product"] == "SC")].drop_duplicates("condition_id")
    trades = load_all_trades(sorted(sub["condition_id"].unique()))

    minute = fut_store.read_minute("SC")
    night = minute[minute["session"] == "night"].sort_values("ts")
    px = night.set_index("ts")["close"]
    logret = np.log(px / px.shift(1))
    logret[px.index.to_series().diff() > pd.Timedelta(minutes=2)] = np.nan
    cum = logret.fillna(0).cumsum()
    base_idx = px.index
    td_map = night.set_index("ts")["trade_date"]

    pnl_rows = []
    for rec in sub.itertuples(index=False):
        tr = trades.get(rec.condition_id)
        if tr is None or tr.empty:
            continue
        first_ts = (pd.Timestamp(rec.first_ts, unit="s", tz="UTC")
                    .tz_convert("Asia/Shanghai").tz_localize(None))
        pos = base_idx.searchsorted(first_ts)
        if pos <= 0 or pos >= len(base_idx) - 1:
            continue
        # 进场 = 事件所在 bar 的收盘（事件信息已知后的第一个可成交价，
        # 修正 v1.1 在事件前一分钟定价进场的前视缺陷）；持有 120 根 1 分钟
        # bar（原实现为 25 根却按 120 分钟宣称），夜盘剩余不足则持有至该夜
        # 收盘。
        end = min(pos + 120, len(base_idx) - 1)
        span = base_idx[pos:end + 1].to_series()
        breaks = span.diff() > pd.Timedelta(minutes=2)
        if breaks.any():
            end = pos + int(np.argmax(breaks.to_numpy())) - 1
        if end <= pos:
            continue
        ret = float((cum.iloc[end] - cum.iloc[pos]) * rec.orientation)
        pnl_rows.append({"trade_date": td_map.iloc[pos], "ret": ret,
                         "hold_min": int(end - pos)})
    ev = pd.DataFrame(pnl_rows)
    daily = ev.groupby("trade_date")["ret"].sum()
    print(f"S5 事件数 {len(ev)}，覆盖 {daily.shape[0]} 个交易日")
    all_days = fut_store.read_daily("SC")["trade_date"]
    return daily.reindex(all_days).fillna(0.0).rename("S5_event")


def main() -> int:
    pub_style.setup()
    ret_df, pos_df = build_daily_strategies()
    s5 = build_event_strategy()
    ret_df["S5_event"] = s5.to_numpy()
    pos_df["S5_event"] = (ret_df["S5_event"] != 0).astype(float) * 2  # 逐事件进出

    # 只统计信号窗口内（截至 SIGNAL_END），其后仓位恒为 0
    mask = ret_df.index <= SIGNAL_END
    ret_df, pos_df = ret_df[mask], pos_df[mask]

    # ---- 指标表（成本敏感性 0/2/4/8/12bp 单边）----
    rows = []
    for col in ret_df.columns:
        for cost in (0.0, 2.0, COST_BP, 8.0, 12.0):
            m = metrics(ret_df[col], pos_df[col], cost_bp=cost)
            rows.append({"strategy": col, "cost_bp": cost, **m})
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "backtest_metrics.parquet", index=False)
    print(tab.round(2).to_string(index=False))

    # ---- 交易属性表：换手、进出次数与参与率（容量代理）----
    sc = fut_store.read_daily("SC")
    sc = sc[sc["trade_date"] <= SIGNAL_END]
    adv = float(sc["day_money"].mean())          # 日均成交额（元）
    aoi_val = float((sc["day_oi"] * sc["day_close"]).mean())  # 持仓名义额
    attr_rows = []
    for col in ret_df.columns:
        p = pos_df[col].fillna(0.0)
        turn = p.diff().abs().fillna(p.abs())
        total_turn = float(turn.sum())
        n_entries = int(((p != 0) & (p.shift(1).fillna(0.0) == 0)).sum())
        active = int((p != 0).sum())
        attr_rows.append({
            "strategy": col,
            "total_turnover": total_turn,
            "ann_turnover": total_turn / len(p) * ANN,
            "n_entries": n_entries,
            "avg_holding_days": active / max(n_entries, 1),
            "participation_100m_pct": 1e8 / adv * 100,
            "participation_oi_100m_pct": 1e8 / aoi_val * 100,
        })
    attr = pd.DataFrame(attr_rows)
    attr.to_parquet(DEEP_DIR / "backtest_trade_attrs.parquet", index=False)
    print(attr.round(3).to_string(index=False))
    print(f"SC 日均成交额 {adv / 1e8:.1f} 亿元，持仓名义 {aoi_val / 1e8:.1f} 亿元")

    # ---- 固定杠杆扫描（示例取 Sharpe 最高的策略）----
    best = (tab[tab.cost_bp == COST_BP].sort_values("sharpe", ascending=False)
            .iloc[0]["strategy"])
    lev_rows = []
    for lev in (1, 2, 3, 5):
        m = metrics(ret_df[best] * lev, pos_df[best] * lev, cost_bp=COST_BP)
        lev_rows.append({"strategy": best, "leverage": lev, **m})
    lev_tab = pd.DataFrame(lev_rows)
    lev_tab.to_parquet(DEEP_DIR / "backtest_leverage.parquet", index=False)
    print("\n固定杠杆（含 4bp 成本）：")
    print(lev_tab[["leverage", "ann_ret_pct", "ann_vol_pct", "sharpe",
                   "max_dd_pct"]].round(2).to_string(index=False))

    # ---- 目标波动动态杠杆（K5 模型给 σ̂_t）----
    import statsmodels.api as sm

    theme_fine = pd.read_parquet(DEEP_DIR / "theme_signals_fine.parquet")
    g = theme_fine[(theme_fine["theme"] == "oil_price")
                   & (theme_fine["product"] == "SC")]
    minute = fut_store.read_minute("SC")
    day = minute[minute["session"] == "day"].sort_values("ts")
    px = day.set_index("ts")["close"]
    r5 = (np.log(px / px.shift(1))
          .resample("5min", label="right", closed="right").sum(min_count=1))
    rv = (r5.pow(2).groupby(r5.index.normalize()).sum() ** 0.5)
    rv.index = rv.index.strftime("%Y-%m-%d")
    dv = (g.set_index("trade_date")["s_gap"].abs().rename("abs_s").to_frame()
          .join(rv.rename("rv")).sort_index())
    dv["rv_lag"] = dv["rv"].shift(1)
    d = dv.dropna()
    fit = sm.OLS(d["rv"], sm.add_constant(d[["abs_s", "rv_lag"]])).fit()
    sigma_hat = (fit.params["const"] + fit.params["abs_s"] * dv["abs_s"]
                 + fit.params["rv_lag"] * dv["rv_lag"]).clip(lower=1e-4)
    sigma_target = float(d["rv"].median())
    lev_t = (sigma_target / sigma_hat).clip(upper=3.0)

    # 与 §12.6 正文一致：目标波动杠杆演示应用于净 Sharpe 最高的 S3_rev_oil
    # （样本内择优，报告文字已声明）。
    base_col = "S3_rev_oil"
    lev_series = lev_t.reindex(ret_df.index).fillna(1.0)
    vt_ret = ret_df[base_col] * lev_series
    vt_pos = pos_df[base_col] * lev_series
    m_fix = metrics(ret_df[base_col], pos_df[base_col], cost_bp=COST_BP)
    m_vt = metrics(vt_ret, vt_pos, cost_bp=COST_BP)
    vt_tab = pd.DataFrame([{"variant": "固定 1x", **m_fix},
                           {"variant": "目标波动杠杆（K5 σ̂，cap 3x）", **m_vt}])
    vt_tab.insert(0, "strategy", base_col)
    vt_tab.to_parquet(DEEP_DIR / "backtest_voltarget.parquet", index=False)
    print("\n目标波动杠杆 vs 固定：")
    print(vt_tab[["variant", "ann_ret_pct", "ann_vol_pct", "sharpe",
                  "max_dd_pct"]].round(2).to_string(index=False))

    # ---- 图 l1：净值曲线；l2：杠杆扫描 ----
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    show = ["S2_night_mid", "S1_day_mid", "S3_rev_oil", "S5_event"]
    colors = [P["blue"], P["aqua"], P["orange"], P["red"]]
    x = pd.to_datetime(ret_df.index)
    for col, c in zip(show, colors, strict=True):
        r = ret_df[col].copy()
        turn = pos_df[col].diff().abs().fillna(pos_df[col].abs())
        r = r - turn * COST_BP / 1e4
        ax.plot(x, r.cumsum() * 100, lw=1.1, color=c, label=col)
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_ylabel("累计收益（%，含 4bp 单边成本）")
    ax.legend(fontsize=6)
    pub_style.panel(ax, "a")
    pub_style.soft_grid(ax)

    ax = axes[1]
    ax2 = ax.twinx()  # 杠杆图使用双轴属可读性妥协：左收益/回撤同单位（%），右 Sharpe
    ax.bar(np.arange(4) - 0.15, lev_tab["ann_ret_pct"], width=0.3,
           color=P["aqua"], label="年化收益 %")
    ax.bar(np.arange(4) + 0.15, -lev_tab["max_dd_pct"], width=0.3,
           color=P["red"], label="−最大回撤 %")
    ax2.plot(np.arange(4), lev_tab["sharpe"], "o-", color=P["ink"], lw=1.2,
             label="Sharpe（右轴）")
    ax2.set_ylim(0, max(lev_tab["sharpe"].max() * 1.4, 1))
    ax.set_xticks(np.arange(4), [f"{int(v)}x" for v in lev_tab["leverage"]])
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_ylabel("%")
    ax2.set_ylabel("Sharpe")
    ax.set_title(f"固定杠杆扫描：{best}", fontsize=7.5)
    h1, l1_ = ax.get_legend_handles_labels()
    h2, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1_ + l2_, fontsize=6, loc="upper left")
    pub_style.panel(ax, "b")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "l1_backtest.png")
    plt.close(fig)
    print(f"\n产物：{DEEP_DIR}/backtest_*.parquet 与 {FIG_DIR}/l1_backtest.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
