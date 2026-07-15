"""美股作为代理（surrogate）：充分性检验、最小充分集、影子组合与 LP-IV。

数学框架（潜在因子模型）：设全球事件信息因子 f_t，观测为
PM 信号 s = λf + η、美股向量 B = Λf + u（K 维）、国内资产 y = θf + φ'u + e。

四组实验：

N1 代理充分性（Prentice 1989 替代终点判据的跨市场版）：
    H_S: y ⊥ s | B（美股张成空间是 PM 信息的充分统计量）。
    检验量 = 全谱回归 y ~ s + B 中 s 的系数 c'_full。
    附恒等式自检：c = c' + γ'π（π 为 s 在 B 上的投影系数，γ 为 y 回归中
    B 的系数），数值必须闭合。
N2 最小充分代理集：前向逐步加入美股资产（按对 y 的增量 R² 贪心），
    记录每步后的 |t(c')| 衰减路径——哪个"角度"的美股在承担中介。
N3 影子组合：每个主题对美股横截面的载荷 a_k（s -> 各资产同窗收益的
    单变量 HAC t），主题 × 资产热力图——主题在美股里的"画像"。
N4 LP-IV（外部工具变量）：以 s 为工具识别"PM 驱动的美股变动"对国内的
    结构传导：第一阶段 B_k ~ s，第二阶段 y ~ B̂_k。第一阶段 F < 10 即弱
    工具，如实报告（生成回归元使二阶段 SE 偏小，仅作点估计对比）。

代理集（避开杠杆 ETF）：SPY QQQ IWM SOXX TLT HYG UUP USO XLE GLD FXI ASHR。
窗口 = 各国内品种自身的闭市 gap（与信号完全同界）。

产物：analysis/deep/surrogate_*.parquet 与 docs/figures/deep/n*.png。

用法::

    .venv/bin/python scripts/surrogate_analysis.py
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
from chain_analysis import theme_gap_signals_nonight  # noqa: E402
from deep_analysis_cn_polymarket import DEEP_DIR, FIG_DIR  # noqa: E402
from intl_benchmark_analysis import (  # noqa: E402
    load_benchmark_beijing,
    window_benchmark_returns,
)

from alpha_data.cn_futures import sessions  # noqa: E402
from alpha_data.cn_futures import store as fut_store

P = pub_style.PALETTE
SURROGATES = ["SPY", "QQQ", "IWM", "SOXX", "TLT", "HYG", "UUP",
              "USO", "XLE", "GLD", "FXI", "ASHR"]

#: (theme, CN 品种)；IF/IM 主题信号在无夜盘网格重算（chain_analysis 同源）。
PAIRS = [("mideast_conflict", "SC"), ("oil_price", "SC"),
         ("fed_policy", "AU"),
         ("mideast_conflict", "IF"), ("mideast_conflict", "IM"),
         ("us_china_trade", "IM")]


def hac(dep: np.ndarray, X: np.ndarray):
    import statsmodels.api as sm
    return sm.OLS(dep, sm.add_constant(X)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1})


def build_dataset() -> dict[str, pd.DataFrame]:
    """逐 CN 品种：trade_date、闭市 gap 收益、全部代理同窗收益、各主题信号。"""
    registry = pd.read_parquet(REGISTRY_PATH)
    theme_fine = pd.read_parquet(DEEP_DIR / "theme_signals_fine.parquet")
    specs = fut_store.read_product_specs().set_index("product")

    eq_themes = {t for t, p in PAIRS if p in ("IF", "IM")}
    reg_eq = registry[registry["theme"].isin(eq_themes)]
    trades = load_all_trades(sorted(reg_eq["condition_id"].unique()))

    px_cache = {s: load_benchmark_beijing(s) for s in SURROGATES}
    out: dict[str, pd.DataFrame] = {}
    for product in sorted({p for _, p in PAIRS}):
        d = fut_store.read_daily(product)
        days = d["trade_date"].tolist()
        ne_str = specs.loc[product, "night_end"]
        ne = sessions.parse_night_end(ne_str if isinstance(ne_str, str) else None)
        win = sessions.signal_windows(days, night_end=ne)
        base = d[["trade_date", "roll"]].copy()
        if d["r_night"].isna().all():
            base["r_gap"] = d["r_gap_full"]
        else:
            tot = (d["r_gap_pm"].fillna(0) + d["r_night"].fillna(0)
                   + d["r_gap_am"].fillna(0))
            tot[d[["r_gap_pm", "r_night", "r_gap_am"]].isna().all(axis=1)] = np.nan
            base["r_gap"] = tot
        for sym in SURROGATES:
            b = window_benchmark_returns(px_cache[sym], win)
            base[f"b_{sym}"] = b["b_gap_total"].to_numpy()
        # 主题信号（IF/IM 网格一次算全部权益主题）
        eq_sig = None
        if product in ("IF", "IM"):
            eq_sig = theme_gap_signals_nonight(reg_eq, trades, days)
        for theme, prod in PAIRS:
            if prod != product:
                continue
            if eq_sig is not None:
                sig = eq_sig[eq_sig["theme"] == theme]
                base[f"s_{theme}"] = base["trade_date"].map(
                    sig.set_index("trade_date")["s_gap"])
            else:
                g = theme_fine[(theme_fine["theme"] == theme)
                               & (theme_fine["product"] == product)]
                base[f"s_{theme}"] = base["trade_date"].map(
                    g.set_index("trade_date")["s_gap"])
        base = base[(base["trade_date"] <= SIGNAL_END)
                    & ~base["roll"].fillna(False)]
        out[product] = base.reset_index(drop=True)
        out[product].to_parquet(
            DEEP_DIR / f"surrogate_dataset_{product}.parquet", index=False)
    return out


def n1_sufficiency(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for theme, product in PAIRS:
        df = data[product]
        cols = [f"b_{s}" for s in SURROGATES]
        sub = df[["r_gap", f"s_{theme}", *cols]].dropna()
        n = len(sub)
        if n < 30:
            rows.append({"theme": theme, "cn": product, "n": n})
            continue
        y = sub["r_gap"].to_numpy()
        s = sub[f"s_{theme}"].to_numpy()
        B = sub[cols].to_numpy()
        f_c = hac(y, s)
        c, c_t = float(f_c.params[1]), float(f_c.tvalues[1])
        # 单一最优代理
        best_t, best_sym = np.inf, ""
        for j, sym in enumerate(SURROGATES):
            f1 = hac(y, np.column_stack([s, B[:, j]]))
            if abs(f1.tvalues[1]) < abs(best_t):
                best_t, best_sym = float(f1.tvalues[1]), sym
        # 全谱
        f_full = hac(y, np.column_stack([s, B]))
        cp, cp_t = float(f_full.params[1]), float(f_full.tvalues[1])
        gamma = np.asarray(f_full.params[2:])
        # 遗漏变量恒等式：c = c' + γ'δ，δ_j = B_j 对 s 的一元回归系数
        # （路径 a 向量）。线性代数上精确成立，闭合误差应为数值零。
        s_c = s - s.mean()
        delta = np.array([np.cov(B[:, j], s)[0, 1] / s_c.var(ddof=1)
                          for j in range(B.shape[1])])
        closure = c - cp - float(gamma @ delta)
        rows.append({"theme": theme, "cn": product, "n": n,
                     "c": c, "c_t": c_t,
                     "cp_best1": best_t, "best1_sym": best_sym,
                     "c_prime_full": cp, "c_prime_full_t": cp_t,
                     "identity_gap": closure,
                     "r2_full": float(f_full.rsquared)})
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "surrogate_sufficiency.parquet", index=False)
    return tab


def n2_path(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """前向贪心（按增量 R²）加入代理，跟踪 |t(c')| 衰减。"""
    rows = []
    for theme, product in PAIRS:
        df = data[product]
        cols = [f"b_{s}" for s in SURROGATES]
        sub = df[["r_gap", f"s_{theme}", *cols]].dropna()
        if len(sub) < 30:
            continue
        y = sub["r_gap"].to_numpy()
        s = sub[f"s_{theme}"].to_numpy()
        B = {sym: sub[f"b_{sym}"].to_numpy() for sym in SURROGATES}
        chosen: list[str] = []
        f0 = hac(y, s)
        rows.append({"theme": theme, "cn": product, "k": 0, "added": "",
                     "t_cprime": float(f0.tvalues[1]), "r2": float(f0.rsquared)})
        remaining = list(SURROGATES)
        for k in range(1, 7):
            best_sym, best_r2 = None, -np.inf
            for sym in remaining:
                X = np.column_stack([s] + [B[c] for c in chosen] + [B[sym]])
                r2 = hac(y, X).rsquared
                if r2 > best_r2:
                    best_r2, best_sym = r2, sym
            chosen.append(best_sym)
            remaining.remove(best_sym)
            X = np.column_stack([s] + [B[c] for c in chosen])
            f = hac(y, X)
            rows.append({"theme": theme, "cn": product, "k": k,
                         "added": best_sym,
                         "t_cprime": float(f.tvalues[1]),
                         "r2": float(f.rsquared)})
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "surrogate_path.parquet", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    colors = {("mideast_conflict", "SC"): P["red"],
              ("oil_price", "SC"): P["orange"],
              ("fed_policy", "AU"): P["yellow"],
              ("mideast_conflict", "IF"): P["blue"],
              ("mideast_conflict", "IM"): P["violet"],
              ("us_china_trade", "IM"): P["aqua"]}
    for (theme, cn), g in tab.groupby(["theme", "cn"]):
        g = g.sort_values("k")
        ax.plot(g["k"], g["t_cprime"].abs(), "o-", lw=1.1,
                color=colors.get((theme, cn), P["ink2"]),
                label=f"{theme}×{cn}")
        for _, r in g.iterrows():
            if r["k"] > 0 and r["k"] <= 3:
                ax.annotate(r["added"], (r["k"], abs(r["t_cprime"])),
                            textcoords="offset points", xytext=(0, 5),
                            fontsize=5, ha="center", color=P["ink2"])
    ax.axhline(2, color=P["ink2"], lw=0.6, ls="--")
    ax.set_xlabel("代理集大小 k（前向贪心，标注为该步加入的资产）")
    ax.set_ylabel("|t(c′)|：PM 信号的残余直接效应")
    ax.legend(fontsize=6, ncol=2)
    pub_style.soft_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "n1_sufficiency_path.png")
    plt.close(fig)
    return tab


def n3_shadow(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """主题 -> 各美股资产同窗收益的单变量 t（影子组合画像，SC 闭市网格）。"""
    df = data["SC"]
    themes_sc = [t for t, p in PAIRS if p == "SC"]
    # 补上以 IF 网格计算的股指主题信号（用 IF 表）
    df_if = data["IF"]
    rows = []
    for theme, frame in ([(t, df) for t in themes_sc]
                         + [("mideast_conflict(IF网格)", df_if)]):
        col = (f"s_{theme}" if not theme.endswith("网格)")
               else "s_mideast_conflict")
        for sym in SURROGATES:
            sub = frame[[col, f"b_{sym}"]].dropna()
            if len(sub) < 30:
                continue
            f = hac(sub[f"b_{sym}"].to_numpy(), sub[col].to_numpy())
            rows.append({"theme": theme, "asset": sym,
                         "a_t": float(f.tvalues[1]),
                         "a": float(f.params[1])})
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "surrogate_shadow.parquet", index=False)

    piv = tab.pivot_table(index="theme", columns="asset", values="a_t")
    piv = piv.reindex(columns=SURROGATES)
    from matplotlib.colors import LinearSegmentedColormap
    dv = LinearSegmentedColormap.from_list("dv", [P["blue"], "#f0efe9", P["red"]])
    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(piv) + 1.2))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap=dv, vmin=-5, vmax=5,
                   aspect="auto")
    ax.set_xticks(range(len(SURROGATES)), SURROGATES, fontsize=6.5)
    ax.set_yticks(range(len(piv)), piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, shrink=0.7, label="a 的 HAC t（同窗）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "n2_shadow_portfolio.png")
    plt.close(fig)
    return tab


def n4_lpiv(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """LP-IV：s 为工具，单一代理为内生变量的 2SLS 与 OLS 对比。"""
    designs = [("oil_price", "SC", "USO"), ("mideast_conflict", "SC", "XLE"),
               ("mideast_conflict", "IF", "SPY"),
               ("fed_policy", "AU", "GLD")]
    rows = []
    for theme, product, med in designs:
        df = data[product]
        sub = df[["r_gap", f"s_{theme}", f"b_{med}"]].dropna()
        n = len(sub)
        if n < 30:
            continue
        y = sub["r_gap"].to_numpy()
        s = sub[f"s_{theme}"].to_numpy()
        b = sub[f"b_{med}"].to_numpy()
        f1 = hac(b, s)
        F_first = float(f1.tvalues[1] ** 2)
        b_hat = f1.fittedvalues
        f2 = hac(y, np.asarray(b_hat))
        f_ols = hac(y, b)
        rows.append({"theme": theme, "cn": product, "mediator": med, "n": n,
                     "first_stage_F": F_first,
                     "delta_iv": float(f2.params[1]),
                     "delta_iv_t": float(f2.tvalues[1]),
                     "beta_ols": float(f_ols.params[1]),
                     "beta_ols_t": float(f_ols.tvalues[1]),
                     "weak_iv": F_first < 10})
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "surrogate_iv.parquet", index=False)
    return tab


def main() -> int:
    pub_style.setup()
    data = build_dataset()
    print("N1 充分性")
    t1 = n1_sufficiency(data)
    print(t1.round(3).to_string(index=False))
    print("\nN2 充分集路径")
    t2 = n2_path(data)
    print(t2[t2.k.isin([0, 1, 2, 6])].round(2).to_string(index=False))
    print("\nN3 影子组合")
    t3 = n3_shadow(data)
    _ = t3
    print("\nN4 LP-IV")
    t4 = n4_lpiv(data)
    print(t4.round(3).to_string(index=False))
    print(f"\n产物：{DEEP_DIR}/surrogate_*.parquet 与 {FIG_DIR}/n*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
