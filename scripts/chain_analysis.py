"""中介链分析：Polymarket -> 美股 -> 国内（股指期货与商品）。

问题：Polymarket 对国内资产的作用，是否（以及多大比例）经由美股 / 国际市场
中介？"引入中间的美股"能否解释直接通道的微弱？

设计（经典中介分解，窗口 = 国内闭市 gap，边界与信号完全同界）：

    总效应   c :  r_CN_gap ~ s_theme_gap
    路径 a   a :  b_US_gap ~ s_theme_gap        （PM 是否被美股在窗口内吸收）
    直接效应 c':  r_CN_gap ~ s_theme_gap + b_US_gap 中 s 的系数
    路径 b   b :  同回归中 b_US 的系数
    中介占比 = 1 − c'/c（c 显著时才有意义）；间接效应 ≈ a×b。

国内股票侧：A 股现货分钟库止于 2025-12-31，与 Polymarket 窗口无重叠，
以中金所股指期货 IF（沪深300）、IM（中证1000）代理；美股中介用 SPY
（含盘前盘后，北京时间对齐）。商品侧把 SC/USO、AU/GLD 的既有结果并入
同一张表，统一回答"中介占比"。

股指的主题方向 m（除 taiwan_risk×IF=−1 为 v1.1 注册外，其余为本节新增的
判断性方向，标注探索性）：mideast −1（风险规避）、us_shutdown −1、
fed_policy +1（σ 已含降息为正）、us_china_trade +1（σ 为升级正，
升级对国内股指为负，故 m=−1；见 RULES 的 σ×m 约定——此处直接给出
等价的 m）。

产物：analysis/deep/chain_mediation.parquet 与 docs/figures/deep/m1_chain.png。

用法::

    .venv/bin/python scripts/chain_analysis.py
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
from deep_analysis_cn_polymarket import DEEP_DIR, FIG_DIR  # noqa: E402
from intl_benchmark_analysis import (  # noqa: E402
    load_benchmark_beijing,
    window_benchmark_returns,
)

from alpha_data.cn_futures import sessions  # noqa: E402
from alpha_data.cn_futures import store as fut_store  # noqa: E402
from alpha_data.polymarket import cn_features  # noqa: E402

P = pub_style.PALETTE

#: 股指侧主题 -> 对股指的方向 m（乘以登记表 σ 后为最终 orientation）。
EQUITY_M = {"taiwan_risk": -1, "mideast_conflict": -1, "us_shutdown": -1,
            "fed_policy": +1, "us_china_trade": -1}


def theme_gap_signals_nonight(registry: pd.DataFrame,
                              trades: dict[str, pd.DataFrame],
                              trade_days: list[str]) -> pd.DataFrame:
    """在无夜盘网格（整段闭市 = gap_pm）上重算五个主题的逐日 gap 信号。"""
    windows = sessions.signal_windows(trade_days, night_end=None)
    rows = []
    for theme, m_eq in EQUITY_M.items():
        sub = registry[registry["theme"] == theme].drop_duplicates("condition_id")
        frames = []
        for rec in sub.itertuples(index=False):
            tr = trades.get(rec.condition_id)
            if tr is None or tr.empty:
                continue
            sig = cn_features.window_signals(tr, windows, agg_minutes=15)
            s = sig[["trade_date", "s_gap"]].copy()
            admit_date = (pd.Timestamp(rec.admit_ts, unit="s", tz="UTC")
                          .tz_convert("Asia/Shanghai").strftime("%Y-%m-%d"))
            s.loc[s["trade_date"] < admit_date, "s_gap"] = np.nan
            # rec.orientation = σ×m_product（登记品种的方向）；此处需要
            # σ×m_eq：先除以登记 m 还原 σ 不可行（m 因品种而异），
            # 直接用 σ = orientation / m_registered。登记表存了 sigma 列。
            s["w"] = np.sqrt(rec.usdc_win) * rec.sigma * m_eq
            frames.append(s)
        allm = pd.concat(frames, ignore_index=True)

        def wavg(g: pd.DataFrame) -> float:
            v = g["s_gap"].to_numpy(dtype="float64")
            w = g["w"].to_numpy(dtype="float64")
            ok = ~np.isnan(v)
            if not ok.any():
                return np.nan
            return float(np.sum(v[ok] * w[ok]) / np.sum(np.abs(w[ok])))

        for day, g in allm.groupby("trade_date"):
            rows.append({"theme": theme, "trade_date": day, "s_gap": wavg(g)})
    return pd.DataFrame(rows)


def mediation(y: np.ndarray, s: np.ndarray, b: np.ndarray) -> dict:
    """c / a / c' / b 四个回归与中介占比（HAC，滞后 1）。"""
    import statsmodels.api as sm

    ok = ~(np.isnan(y) | np.isnan(s) | np.isnan(b))
    n = int(ok.sum())
    if n < 25:
        return {"n": n}
    y, s, b = y[ok], s[ok], b[ok]

    def hac(dep, X):
        return sm.OLS(dep, sm.add_constant(X)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 1})

    f_c = hac(y, s)
    f_a = hac(b, s)
    f_cp = hac(y, np.column_stack([s, b]))
    c, c_t = float(f_c.params[1]), float(f_c.tvalues[1])
    a, a_t = float(f_a.params[1]), float(f_a.tvalues[1])
    cp, cp_t = float(f_cp.params[1]), float(f_cp.tvalues[1])
    bb, b_t = float(f_cp.params[2]), float(f_cp.tvalues[2])
    share = 1 - cp / c if abs(c_t) >= 2 and c != 0 else np.nan
    return {"n": n, "c": c, "c_t": c_t, "a": a, "a_t": a_t,
            "c_prime": cp, "c_prime_t": cp_t, "b": bb, "b_t": b_t,
            "mediation_share": share, "indirect_ab": a * bb}


def main() -> int:
    pub_style.setup()
    registry = pd.read_parquet(REGISTRY_PATH)
    reg = registry[registry["theme"].isin(EQUITY_M)].copy()
    trades = load_all_trades(sorted(reg["condition_id"].unique()))

    results = []
    # ---- 股指链：PM -> SPY -> IF / IM ----
    for cn_prod in ["IF", "IM"]:
        d = fut_store.read_daily(cn_prod)
        days = d["trade_date"].tolist()
        win = sessions.signal_windows(days, night_end=None)
        b_spy = window_benchmark_returns(load_benchmark_beijing("SPY"), win)
        sig = theme_gap_signals_nonight(reg, trades, days)
        base = (d[["trade_date", "roll", "r_gap_full"]]
                .merge(b_spy[["trade_date", "b_gap_total"]], on="trade_date"))
        base = base[(base["trade_date"] <= SIGNAL_END)
                    & ~base["roll"].fillna(False)]
        for theme in EQUITY_M:
            g = base.merge(sig[sig["theme"] == theme], on="trade_date")
            res = mediation(g["r_gap_full"].to_numpy(dtype="float64"),
                            g["s_gap"].to_numpy(dtype="float64"),
                            g["b_gap_total"].to_numpy(dtype="float64"))
            results.append({"chain": f"PM→SPY→{cn_prod}", "theme": theme,
                            "cn": cn_prod, "us": "SPY", **res})

    # ---- 商品链（对照，与 §11 同口径重算便于同表比较）----
    for cn_prod, etf, themes in [("SC", "USO", ["oil_price", "mideast_conflict"]),
                                 ("AU", "GLD", ["metal_price", "fed_policy"])]:
        d = fut_store.read_daily(cn_prod)
        days = d["trade_date"].tolist()
        specs = fut_store.read_product_specs().set_index("product")
        ne = sessions.parse_night_end(specs.loc[cn_prod, "night_end"])
        win = sessions.signal_windows(days, night_end=ne)
        bench = window_benchmark_returns(load_benchmark_beijing(etf), win)
        theme_fine = pd.read_parquet(DEEP_DIR / "theme_signals_fine.parquet")
        rr = d[["trade_date", "roll", "r_gap_pm", "r_night", "r_gap_am"]].copy()
        tot = (rr["r_gap_pm"].fillna(0) + rr["r_night"].fillna(0)
               + rr["r_gap_am"].fillna(0))
        tot[rr[["r_gap_pm", "r_night", "r_gap_am"]].isna().all(axis=1)] = np.nan
        rr["r_gap_total"] = tot
        base = rr.merge(bench[["trade_date", "b_gap_total"]], on="trade_date")
        base = base[(base["trade_date"] <= SIGNAL_END)
                    & ~base["roll"].fillna(False)]
        for theme in themes:
            g = theme_fine[(theme_fine["theme"] == theme)
                           & (theme_fine["product"] == cn_prod)]
            gg = base.merge(g[["trade_date", "s_gap"]], on="trade_date")
            res = mediation(gg["r_gap_total"].to_numpy(dtype="float64"),
                            gg["s_gap"].to_numpy(dtype="float64"),
                            gg["b_gap_total"].to_numpy(dtype="float64"))
            results.append({"chain": f"PM→{etf}→{cn_prod}", "theme": theme,
                            "cn": cn_prod, "us": etf, **res})

    tab = pd.DataFrame(results)
    tab.to_parquet(DEEP_DIR / "chain_mediation.parquet", index=False)
    cols = ["chain", "theme", "n", "c_t", "a_t", "b_t", "c_prime_t",
            "mediation_share"]
    print(tab[cols].round(2).to_string(index=False))

    # ---- 图：总效应 vs 直接效应 t 值 + 中介占比 ----
    plot = tab.dropna(subset=["c_t"]).copy()
    plot["label"] = plot["theme"] + "\n→" + plot["us"] + "→" + plot["cn"]
    plot = plot.sort_values("c_t", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    x = np.arange(len(plot))
    ax.bar(x - 0.18, plot["c_t"], width=0.34, color=P["blue"],
           label="总效应 c（PM→CN）")
    ax.bar(x + 0.18, plot["c_prime_t"], width=0.34, color=P["orange"],
           label="直接效应 c′（控美股后）")
    for yy in (2, -2):
        ax.axhline(yy, color=P["ink2"], lw=0.5, ls="--")
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_xticks(x, plot["label"], fontsize=5.2)
    ax.set_ylabel("HAC t")
    ax.legend(fontsize=6)
    pub_style.panel(ax, "a")

    ax = axes[1]
    ok = plot.dropna(subset=["mediation_share"])
    x = np.arange(len(ok))
    ax.bar(x, ok["mediation_share"].clip(-0.2, 1.2), width=0.6, color=P["aqua"])
    ax.axhline(1.0, color=P["ink2"], lw=0.5, ls="--")
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_xticks(x, ok["label"], fontsize=5.2)
    ax.set_ylabel("中介占比 1 - c'/c")
    ax.set_title("仅对 c 显著的链计算", fontsize=7)
    pub_style.panel(ax, "b")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m1_chain.png")
    plt.close(fig)
    print(f"\n产物：{DEEP_DIR}/chain_mediation.parquet 与 {FIG_DIR}/m1_chain.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
