"""国际基准控制：H1 偏零检验、吸收分解与"升水回归 vs 过度反应"判别。

动机（研究规范 §0 约束 1 / §4 / H1）：国内商品期货与 Polymarket 之间隔着
COMEX / ICE 等国际市场，"Polymarket 领先国内期货"很可能只是"国际期货领先
国内期货"的转述。本脚本用仓库美股分钟库的 ETF 作国际基准：

- USO（WTI 原油 ETF）对 SC；GLD（黄金）对 AU；SLV（白银）对 AG。
- ETF 分钟 bar 为美东本地时间（bar 收盘戳）、含盘前盘后 04:01-20:00 ET，
  换算北京时间恰好覆盖国内闭市窗口（2026-03-08 美国夏令时切换由时区库处理）。

三组检验：

T1（H1 偏零检验）：AU 的窗口收益对 fed_policy / metal_price 信号回归，
    加入同窗口 GLD 收益前后对比信号系数。先验：金价国际套利充分，
    控制 GLD 后 Polymarket 信号应无增量。
T2（吸收分解）：SC 的窗口收益对 mideast_conflict / oil_price 信号回归，
    控制同窗口 USO 收益。问题：事件概率市场是否含有国际油价之外的信息。
T3（反转机制判别）：把 SC 收益拆成 USO 分量 + 价差（升水）分量
    q(t) = r_cc^SC(t) − b_cc^USO(t)，对两者分别做局部投影 IRF。
    若"吸收后反转"（β0>0 且 βh<0）主要出现在价差分量上，则反转是
    SC 对国际锚的升水扩大后收敛（套利摩擦）；若出现在 USO 自身，
    则是全球性过度反应。

基准窗口收益与信号同口径：窗口 [t0,t1) 的基准收益 = log P(t1⁻) − log P(t0⁻)，
P 取该时刻前最后一根 bar 的收盘价（LOCF，严格早于边界），美股不交易的时段
（周末、假日）自然由 LOCF 跨越。

产物：data/cn_futures/analysis/deep/intl_*.parquet 与 docs/figures/deep/i_*.png。

用法::

    .venv/bin/python scripts/intl_benchmark_analysis.py
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

from analyze_cn_futures_polymarket import SIGNAL_END, nw_regression  # noqa: E402
from deep_analysis_cn_polymarket import (  # noqa: E402
    DEEP_DIR,
    FIG_DIR,
    C,
    futures_fine_returns,
    setup_matplotlib,
    style_ax,
)

from alpha_data.cn_futures import sessions  # noqa: E402
from alpha_data.cn_futures import store as fut_store  # noqa: E402

EQUITY_MINUTE = ROOT / "data" / "equity" / "minute_db" / "minute"

#: 品种 -> (基准 ETF, 标的说明)。
BENCHMARKS = {"SC": ("USO", "WTI 原油 ETF"), "AU": ("GLD", "黄金 ETF"),
              "AG": ("SLV", "白银 ETF")}


def load_benchmark_beijing(symbol: str) -> pd.Series:
    """读取 ETF 2026 年分钟收盘价，美东本地 -> 北京时间（tz-naive）。"""
    df = pd.read_parquet(EQUITY_MINUTE / symbol / "2026.parquet",
                         columns=["ts", "close"])
    ts = (pd.to_datetime(df["ts"])
          .dt.tz_localize("America/New_York")
          .dt.tz_convert("Asia/Shanghai")
          .dt.tz_localize(None))
    px = pd.Series(df["close"].to_numpy(dtype="float64"), index=ts).sort_index()
    return px[~px.index.duplicated(keep="last")]


def window_benchmark_returns(px: pd.Series, windows: pd.DataFrame) -> pd.DataFrame:
    """按窗口边界 LOCF 求基准对数收益（与信号端点同口径，严格早于边界）。"""
    logp = np.log(px)
    idx = logp.index

    def at(bound: pd.Series) -> np.ndarray:
        t = pd.to_datetime(bound).to_numpy()
        out = np.full(len(t), np.nan)
        ok = ~pd.isna(t)
        pos = idx.searchsorted(t[ok], side="left") - 1  # 严格早于边界的最后一根
        valid = pos >= 0
        vals = np.full(ok.sum(), np.nan)
        vals[valid] = logp.to_numpy()[pos[valid]]
        out[ok] = vals
        return out

    b = {}
    b["b_gap_pm"] = at(windows["gap1_end"]) - at(windows["gap1_start"])
    b["b_night"] = at(windows["night_end"]) - at(windows["night_start"])
    b["b_gap_am"] = at(windows["gap2_end"]) - at(windows["gap2_start"])
    b["b_day"] = at(windows["day_end"]) - at(windows["day_start"])
    # 整段闭市（前收盘 -> 当日开盘）与收对收
    b["b_gap_total"] = at(windows["day_start"]) - at(windows["gap1_start"])
    b["b_cc"] = at(windows["day_end"]) - at(
        windows["day_end"].shift(1))
    out = pd.DataFrame(b)
    out.insert(0, "trade_date", windows["trade_date"].to_numpy())
    return out


def controlled_regression(
    y: np.ndarray, s: np.ndarray, b: np.ndarray, maxlags: int = 1
) -> dict:
    """r ~ s（无控制）与 r ~ s + b（控制基准）的信号系数对比（HAC）。"""
    import statsmodels.api as sm

    ok = ~(np.isnan(y) | np.isnan(s) | np.isnan(b))
    n = int(ok.sum())
    res = {"n": n}
    if n < 25:
        return {**res, "beta_raw": np.nan, "t_raw": np.nan,
                "beta_ctl": np.nan, "t_ctl": np.nan, "beta_bench": np.nan,
                "t_bench": np.nan, "r2_raw": np.nan, "r2_ctl": np.nan}
    beta_raw, t_raw, _, _ = nw_regression(y[ok], s[ok], maxlags)
    X = sm.add_constant(np.column_stack([s[ok], b[ok]]))
    fit = sm.OLS(y[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    uni = sm.OLS(y[ok], sm.add_constant(s[ok])).fit()
    return {
        **res,
        "beta_raw": beta_raw, "t_raw": t_raw,
        "beta_ctl": float(fit.params[1]), "t_ctl": float(fit.tvalues[1]),
        "beta_bench": float(fit.params[2]), "t_bench": float(fit.tvalues[2]),
        "r2_raw": float(uni.rsquared), "r2_ctl": float(fit.rsquared),
    }


def main() -> int:
    setup_matplotlib()
    DEEP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    theme_fine = pd.read_parquet(DEEP_DIR / "theme_signals_fine.parquet")
    rets = futures_fine_returns(["SC", "AU", "AG"])
    specs = fut_store.read_product_specs().set_index("product")

    # 各品种窗口边界与基准窗口收益
    bench: dict[str, pd.DataFrame] = {}
    for product, (etf, _) in BENCHMARKS.items():
        days = fut_store.read_daily(product)["trade_date"].tolist()
        ne = sessions.parse_night_end(specs.loc[product, "night_end"])
        win = sessions.signal_windows(days, night_end=ne)
        px = load_benchmark_beijing(etf)
        bench[product] = window_benchmark_returns(px, win)

    # ---- T1 / T2：控制前后的信号系数 ----
    tests = [
        # (标签, 主题, 品种, 信号列, 收益列, 基准列)
        ("T1", "fed_policy", "AU", "s_night", "r_night", "b_night"),
        ("T1", "fed_policy", "AU", "s_gap", "r_gap_total", "b_gap_total"),
        ("T1", "metal_price", "AU", "s_night", "r_night", "b_night"),
        ("T1", "metal_price", "AU", "s_gap", "r_gap_total", "b_gap_total"),
        ("T1", "metal_price", "AG", "s_night", "r_night", "b_night"),
        ("T2", "oil_price", "SC", "s_night", "r_night", "b_night"),
        ("T2", "oil_price", "SC", "s_gap", "r_gap_total", "b_gap_total"),
        ("T2", "mideast_conflict", "SC", "s_night", "r_night", "b_night"),
        ("T2", "mideast_conflict", "SC", "s_gap", "r_gap_total", "b_gap_total"),
        ("T2", "russia_ukraine", "SC", "s_gap", "r_gap_total", "b_gap_total"),
    ]
    rows = []
    for test, theme, product, sig_col, ret_col, b_col in tests:
        g = theme_fine[(theme_fine["theme"] == theme)
                       & (theme_fine["product"] == product)]
        r = rets[rets["product"] == product]
        r = r[~r["roll"].fillna(False)]
        if ret_col == "r_gap_total":
            rr = (r["r_gap_pm"].fillna(0) + r["r_night"].fillna(0)
                  + r["r_gap_am"].fillna(0))
            rr[r[["r_gap_pm", "r_night", "r_gap_am"]].isna().all(axis=1)] = np.nan
            r = r.assign(_ret=rr)
        else:
            r = r.assign(_ret=r[ret_col])
        m = (g.merge(r[["trade_date", "_ret"]], on="trade_date")
             .merge(bench[product][["trade_date", b_col]], on="trade_date"))
        m = m[m["trade_date"] <= SIGNAL_END]
        res = controlled_regression(
            m["_ret"].to_numpy(dtype="float64"),
            m[sig_col].to_numpy(dtype="float64"),
            m[b_col].to_numpy(dtype="float64"),
        )
        rows.append({"test": test, "theme": theme, "product": product,
                     "window": sig_col.replace("s_", ""),
                     "bench": BENCHMARKS[product][0], **res})
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "intl_control.parquet", index=False)
    print("=== T1/T2 控制前后信号系数 ===")
    cols = ["test", "theme", "product", "window", "n",
            "t_raw", "t_ctl", "t_bench", "r2_raw", "r2_ctl"]
    print(tab[cols].round(2).to_string(index=False))

    # ---- T3：反转机制判别（SC 收对收拆成 USO 分量 + 价差分量，逐主题）----
    sc = rets[rets["product"] == "SC"].sort_values("trade_date").reset_index(drop=True)
    b_sc = bench["SC"].sort_values("trade_date").reset_index(drop=True)
    irf_rows = []
    for theme in ["oil_price", "mideast_conflict"]:
        g = theme_fine[(theme_fine["theme"] == theme)
                       & (theme_fine["product"] == "SC")]
        s_theme = (g.assign(s_all=lambda d: d[["s_night", "s_gap", "s_day"]]
                            .sum(axis=1, skipna=False))
                   .set_index("trade_date")["s_all"])
        m = (sc.merge(b_sc[["trade_date", "b_cc"]], on="trade_date")
             .merge(s_theme.rename("s_all"), on="trade_date"))
        m["q_cc"] = m["r_cc"] - m["b_cc"]  # 价差（升水变化）分量
        m = m.reset_index(drop=True)
        for comp, col in [("SC 总收益", "r_cc"), ("USO 分量", "b_cc"),
                          ("价差分量", "q_cc")]:
            y_base = m[col].to_numpy(dtype="float64")
            s = m["s_all"].to_numpy(dtype="float64")
            s = np.where(m["trade_date"] <= SIGNAL_END, s, np.nan)
            for h in range(0, 9):
                if h == 0:
                    y = y_base
                else:
                    fwd = pd.Series(y_base).shift(-1).rolling(h, min_periods=h).sum()
                    y = fwd.shift(-(h - 1)).to_numpy()
                beta, t, p, n = nw_regression(y, s, max(h, 1))
                sd = np.nanstd(s)
                irf_rows.append({"theme": theme, "component": comp, "h": h, "n": n,
                                 "beta_bp": beta * 1e4 * sd,
                                 "se_bp": abs(beta / t) * 1e4 * sd
                                 if np.isfinite(t) and t != 0 else np.nan,
                                 "t_hac": t})
    irf = pd.DataFrame(irf_rows)
    irf.to_parquet(DEEP_DIR / "intl_irf_decomp.parquet", index=False)
    print("\n=== T3 IRF 分解（β_h，bp/1σ）===")
    for theme, gg in irf.groupby("theme"):
        print(f"\n[{theme}]")
        print(gg.pivot_table(index="h", columns="component", values="beta_bp")
              .round(0).to_string())

    # ---- 图 ----
    # i1：控制前后 t 值对比
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    lab = tab["theme"] + "×" + tab["product"] + "·" + tab["window"]
    x = np.arange(len(tab))
    ax.bar(x - 0.18, tab["t_raw"], width=0.34, color=C["blue"], label="无控制 t(信号)")
    ax.bar(x + 0.18, tab["t_ctl"], width=0.34, color=C["orange"],
           label="控制国际基准后 t(信号)")
    ax.axhline(2, color=C["ink2"], lw=0.7, ls="--")
    ax.axhline(-2, color=C["ink2"], lw=0.7, ls="--")
    ax.axhline(0, color=C["ink2"], lw=0.7)
    ax.set_xticks(x, lab, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("HAC t（信号系数）")
    ax.set_title("控制同窗口国际基准（USO/GLD/SLV）前后，Polymarket 信号系数的 t 值")
    ax.legend(fontsize=8, frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "i1_intl_control.png", dpi=150)
    plt.close(fig)

    # i2：IRF 分解（oil_price 主题，与正文 IRF 口径一致）
    irf_oil = irf[irf["theme"] == "oil_price"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
    for ax, (comp, gg) in zip(axes, irf_oil.groupby("component", sort=False),
                              strict=False):
        ax.axhline(0, color=C["ink2"], lw=0.7)
        ax.errorbar(gg["h"], gg["beta_bp"], yerr=1.645 * gg["se_bp"], fmt="o-",
                    ms=4, lw=1.5, capsize=3, color=C["blue"], ecolor=C["grid"])
        ax.set_title(comp, fontsize=9.5)
        ax.set_xlabel("视界 h（交易日）")
        style_ax(ax)
    axes[0].set_ylabel("β_h（bp / 1σ 信号，90% CI）")
    fig.suptitle("T3（oil_price 信号）：SC 收益拆成 USO 分量与价差分量的局部投影 IRF",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "i2_irf_decomposition.png", dpi=150)
    plt.close(fig)

    # i3：2026-03-09 前后 SC 对 USO 的累计价差
    win = m[(m["trade_date"] >= "2026-02-16") & (m["trade_date"] <= "2026-03-31")]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = pd.to_datetime(win["trade_date"])
    ax.plot(x, win["r_cc"].fillna(0).cumsum() * 100, lw=1.6, color=C["blue"],
            label="SC 累计收益")
    ax.plot(x, win["b_cc"].fillna(0).cumsum() * 100, lw=1.6, color=C["orange"],
            label="USO 累计收益（北京窗口对齐）")
    ax.plot(x, win["q_cc"].fillna(0).cumsum() * 100, lw=1.8, color=C["red"],
            label="价差（SC-USO）累计")
    ax.axvline(pd.Timestamp("2026-03-09"), color=C["ink2"], lw=0.8, ls=":")
    ax.text(pd.Timestamp("2026-03-09"), ax.get_ylim()[1] * 0.9, " 03-09",
            fontsize=8, color=C["ink2"])
    ax.set_ylabel("累计对数收益（%）")
    ax.set_title("事件高峰期 SC、USO 与两者价差的累计走势（2026-02-16 至 03-31）")
    ax.legend(fontsize=8, frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "i3_premium_timeline.png", dpi=150)
    plt.close(fig)

    print(f"\n产物：{DEEP_DIR}/intl_*.parquet 与 {FIG_DIR}/i*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
