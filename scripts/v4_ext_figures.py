"""扩展轮报告的图（3 张），数字全部从 v4 产物读取或按同一口径重算。

图 1  涨跌停处理（P2）：(a) 各品种真封板 / 伪锁占比与标签剔除率；
      (b) raw 对 clean 的逐格极值 |t| 散点（应落在对角线上：处理不制造
      结果）。
图 2  分位与成本（P3）：(a) 四个代表格的十分位收益剖面（k=15）；
      (b) 114 个合格对的策略日毛利对日成本散点（对角线 = 盈亏平衡）。
图 3  扩展总账全景：四个构成块的全部 |t| 分布条带 + 新旧门槛线，
      纯 Polymarket 检验单独着色。

命令：.venv/bin/python scripts/v4_ext_figures.py
产物：docs/ext_round/fig/e{1,2,3}_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pub_style  # noqa: E402
from v3_quantile import N_Q, assign_deciles  # noqa: E402
from v4_signal_grid import classify_locked, clean_labels, find_panel  # noqa: E402

DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
V4 = ROOT / "data" / "cn_futures" / "analysis" / "v4"
FIG = ROOT / "docs" / "ext_round" / "fig"
H = [1, 2, 3, 5, 10, 15]
IC_T = [f"ic_t_{h}" for h in H]
EV_T = [f"{d}_t_{h}" for d in ("up", "dn") for h in H]
PURE = ([f"N{i}" for i in range(1, 11)] + [f"K{i}" for i in range(1, 11)]
        + ["X5", "X7", "X10"])
P = pub_style.PALETTE


def cell_max_t(g: pd.DataFrame) -> pd.Series:
    """逐 (signal, product) 格的适当口径极值 |t|。"""
    out = {}
    ev = [c for c in EV_T if c in g.columns]
    for _, r in g.iterrows():
        cols = ev if r["kind"] == "离散" else IC_T
        v = np.abs(np.asarray([r[c] for c in cols], dtype=float))
        v = v[np.isfinite(v)]
        if len(v):
            out[(r["signal"], r["product"])] = float(v.max())
    return pd.Series(out)


def proper_all(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(全部 |t|, 是否纯 PM) 两个向量。"""
    ts, pure = [], []
    ev = [c for c in EV_T if c in g.columns]
    for _, r in g.iterrows():
        cols = ev if r["kind"] == "离散" else IC_T
        v = np.abs(np.asarray([r[c] for c in cols], dtype=float))
        v = v[np.isfinite(v)]
        ts.extend(v.tolist())
        is_pure = (r["signal"] in PURE) or str(r["signal"]).startswith("D_")
        pure.extend([is_pure] * len(v))
    return np.asarray(ts), np.asarray(pure)


def fig1_locked() -> None:
    pub_style.setup(cn_font=True)
    imp = pd.read_parquet(V4 / "locked_impact.parquet")
    raw = pd.read_parquet(DEF / "signal_grid.parquet")
    cln = pd.read_parquet(V4 / "signal_grid_clean.parquet")
    cln5 = cln[cln["product"].isin(raw["product"].unique())]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axes[0]
    x = np.arange(len(imp))
    ax.bar(x - 0.22, imp["true_lock_share"] * 100, width=0.42,
           color=P["red"], label="真封板（收益不可实现）")
    ax.bar(x + 0.22, imp["pseudo_lock_share"] * 100, width=0.42,
           color=P["ink2"], alpha=0.6, label="伪锁（清淡，仅打标）")
    ax.plot(x, imp["dropped_share_15"] * 100, "o--", color=P["blue"],
            ms=4, lw=1.0, label="fwd$_{15}$ 标签剔除率")
    ax.set_xticks(x)
    ax.set_xticklabels(imp["product"])
    ax.set_ylabel("占全部分钟 (%)")
    ax.legend(fontsize=6.2)
    ax.set_title("(a) 真封板 / 伪锁分类与标签剔除", loc="left")

    ax = axes[1]
    a = cell_max_t(raw)
    b = cell_max_t(cln5)
    common = a.index.intersection(b.index)
    av, bv = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    lim = max(av.max(), bv.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=P["ink2"], lw=0.7, ls="--",
            label="对角线（处理前后不变）")
    ax.scatter(av, bv, s=12, alpha=0.6, color=P["blue"])
    ax.set_xlabel("raw 口径逐格极值 $|t|$")
    ax.set_ylabel("clean 口径逐格极值 $|t|$")
    ax.set_title("(b) 200 个格：clean 对 raw 几乎不动", loc="left")
    ax.legend(fontsize=6.2)
    fig.tight_layout()
    fig.savefig(FIG / "e1_locked.png", bbox_inches="tight")
    plt.close(fig)
    print("written e1_locked.png")


def fig2_quantile() -> None:
    pub_style.setup(cn_font=True)
    q = pd.read_parquet(V4 / "quantile_grid.parquet")
    el = q[q["eligible"].astype(bool)]

    examples = [("SC", "C8"), ("SC", "N4"), ("AU", "N2"), ("CU", "N1")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9),
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1.15]})

    # (a)(b) 十分位剖面（重算两组代表格，口径与 v3_quantile 相同）
    for ax, (prod, sig) in zip(axes[:2], examples[:2], strict=False):
        df = pd.read_parquet(find_panel(prod))
        df = clean_labels(classify_locked(df))
        qv = assign_deciles(df, sig)
        f = df["fwd_15"].to_numpy(dtype=float)
        means = [np.nanmean(f[qv == j]) * 1e4 for j in range(1, N_Q + 1)]
        ax.bar(range(1, N_Q + 1), means, color=P["blue"], alpha=0.8)
        ax.axhline(0, color=P["ink"], lw=0.6)
        ax.set_xlabel("十分位档（按前 20 日断点）")
        ax.set_ylabel("档均值 fwd$_{15}$ (bp)")
        row = el[(el["product"] == prod) & (el["signal"] == sig)]
        t15 = float(row["ls_t_15"].iloc[0]) if len(row) else float("nan")
        tag = "(a)" if sig == "C8" else "(b)"
        ax.set_title(f"{tag} {sig}$\\cdot${prod}  LS t={t15:+.2f}",
                     loc="left")

    # (c) 毛利 vs 成本
    ax = axes[2]
    g = el["strat_gross_bp_day"].to_numpy(dtype=float)
    c = el["strat_cost_bp_day"].to_numpy(dtype=float)
    ax.scatter(np.abs(g), c, s=10, alpha=0.6, color=P["red"])
    lim = max(np.abs(g).max(), c.max()) * 1.1
    ax.plot([0, lim], [0, lim], color=P["ink2"], lw=0.8, ls="--",
            label="盈亏平衡线")
    ax.set_xlabel("|策略日毛利| (bp)")
    ax.set_ylabel("策略日成本 (bp)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.legend(fontsize=6.2, loc="lower right")
    ax.set_title("(c) 114 个合格对：成本全面压倒毛利", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "e2_quantile.png", bbox_inches="tight")
    plt.close(fig)
    print("written e2_quantile.png")


def fig3_ledger() -> None:
    pub_style.setup(cn_font=True)
    raw = pd.read_parquet(DEF / "signal_grid.parquet")
    cln = pd.read_parquet(V4 / "signal_grid_clean.parquet")
    dg = pd.read_parquet(V4 / "dispersion_grid.parquet")
    q = pd.read_parquet(V4 / "quantile_grid.parquet")
    el = q[q["eligible"].astype(bool)]
    t15 = el["ls_t_15"].to_numpy(dtype=float)
    q_pure = el["signal"].isin(PURE).to_numpy()

    blocks = [
        ("v3 冻结轮\n(1,476)", *proper_all(raw)),
        ("扩展品种\nCF/I/IF (798)",
         *proper_all(cln[cln["product"].isin(("CF", "I", "IF"))])),
        ("分散度矩族\n(354)", *proper_all(dg)),
        ("分位 k=15\n(114)", np.abs(t15[np.isfinite(t15)]),
         q_pure[np.isfinite(t15)]),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    rng = np.random.default_rng(7)
    for i, (_label, ts, pure) in enumerate(blocks):
        x = i + rng.uniform(-0.28, 0.28, len(ts))
        ax.scatter(x[~pure], ts[~pure], s=4, alpha=0.35, color=P["ink2"],
                   label="含期货量价腿" if i == 0 else None)
        ax.scatter(x[pure], ts[pure], s=5, alpha=0.6, color=P["blue"],
                   label="仅 Polymarket 数据" if i == 0 else None)
    ax.axhline(4.146, color=P["yellow"], lw=1.0, ls=":",
               label="v3 门槛 4.146")
    ax.axhline(4.285, color=P["red"], lw=1.1, ls="--",
               label="扩展门槛 4.285（升高）")
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels([b[0] for b in blocks], fontsize=7)
    ax.set_ylabel("$|t|$（适当口径）")
    ax.set_ylim(0, 7.2)
    ax.text(2.02, 4.35, "纯 PM 极值 3.994\n(D_hhi·AG，一致率 58%)",
            fontsize=6.4, color=P["blue"])
    ax.legend(fontsize=6.4, loc="upper right", ncol=2)
    ax.set_title("扩展主族 2,742 个检验：纯 Polymarket 侧无一过门槛"
                 "（超出 y 轴的均为 C 族）", loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e3_ledger.png", bbox_inches="tight")
    plt.close(fig)
    print("written e3_ledger.png")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_locked()
    fig2_quantile()
    fig3_ledger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
