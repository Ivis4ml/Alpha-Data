"""v3 结果图（4 幅，出版级样式）。

f_v3_term.png        期限乘子曲线与内部验证摘要
f_v3_power.png       Power Gate：MDE 对经济效应上限
f_v3_oos.png         Layer 5 OOS 增量 R^2（v3 vs v1.1 基线）
f_v3_absorption.png  扩展样本同期吸收（相关系数，基线 / v3 并列）

用法::

    .venv/bin/python scripts/v3_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pub_style  # noqa: E402

V3_DIR = ROOT / "data" / "cn_futures" / "analysis" / "v3"
FIG_DIR = ROOT / "docs" / "figures" / "v3"
P = pub_style.PALETTE


def fig_term() -> None:
    curve = pd.read_parquet(V3_DIR / "v3_term_curve.parquet")
    val = json.loads((V3_DIR / "v3_validation.json").read_text())
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    labels = [
        f"[{int(lo)}, {'inf' if np.isinf(hi) else int(hi)})"
        for lo, hi in zip(curve["tau_lo_days"], curve["tau_hi_days"], strict=True)
    ][::-1]
    vals = curve["var_ratio"].to_numpy()[::-1]
    ax.bar(labels, vals, color=P["blue"], width=0.62)
    ax.axhline(1.0, color=P["ink2"], lw=0.7, ls="--")
    ax.set_ylabel("标准化创新方差比")
    ax.set_xlabel("剩余期限（天，右开区间）")
    ax.text(
        0.02, 0.96,
        (f"H1-lite 创新/差分方差比 {val['h1_lite']['ratio']:.2f}\n"
         f"H2-lite 修复率 {val['h2_lite']['repair_rate']:.3f}"
         f"（违反 n={val['h2_lite']['n_violations']}）"),
        transform=ax.transAxes, va="top", fontsize=7.5, color=P["ink2"],
    )
    pub_style.soft_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_v3_term.png", dpi=300)
    plt.close(fig)


def fig_power() -> None:
    power = pd.read_parquet(V3_DIR / "v3_power_table_expost.parquet")
    power = power.sort_values("mde")
    lab = power["theme"] + "×" + power["product"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    y = np.arange(len(power))
    cap = 0.32
    finite = np.minimum(power["mde"].to_numpy(), cap)
    colors = [P["green"] if d == "category_beta" else P["red"]
              for d in power["decision"]]
    ax.barh(y, finite, color=colors, height=0.62)
    for i, (m, d) in enumerate(zip(power["mde"], power["delta_econ"], strict=True)):
        ax.plot([d], [i], marker="|", ms=10, color=P["ink"])
        if m > cap:
            ax.text(cap + 0.004, i, "inf" if np.isinf(m) else f"{m:.2f}",
                    fontsize=7, va="center", color=P["ink2"])
    ax.set_yticks(y, lab, fontsize=7.5)
    ax.set_xlabel("MDE（单位 surprise 的收益效应，80% 功效）；竖线 = 预注册上限")
    ax.set_xlim(0, cap + 0.05)
    pub_style.soft_grid(ax, axis="x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_v3_power.png", dpi=300)
    plt.close(fig)


def fig_oos() -> None:
    oos = pd.read_parquet(V3_DIR / "v3_oos_results.parquet")
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(oos))
    w = 0.26
    bars = [("M1_base", -w, "v1.1 基线信号", P["ink2"]),
            ("M1_v3slim", 0.0, "v3 精简（同维度）", P["aqua"]),
            ("M1_v3", w, "v3 全测量层", P["blue"])]
    for m, dx, lab, c in bars:
        col = f"dR2_{m}"
        if col not in oos.columns:
            continue
        ax.bar(x + dx, oos[col] * 100, w, label=lab, color=c)
        # Clark-West 是单侧检验：正 t 支持扩展模型（†），负 t 是显著恶化（↓），
        # 二者不能共用同一显著记号（评审第 7 点）。
        for i, row in oos.iterrows():
            t = row.get(f"cw_t_{m}")
            if not np.isfinite(t):
                continue
            if t >= 1.645:
                ax.text(x[i] + dx, max(row[col], 0) * 100 + 0.6, "▲",
                        ha="center", fontsize=6, color=P["green"])
            elif t <= -1.645:
                ax.text(x[i] + dx, max(row[col], 0) * 100 + 0.6, "▼",
                        ha="center", fontsize=6, color=P["red"])
    ax.axhline(0, color=P["ink"], lw=0.7)
    ax.set_xticks(x, oos["product"])
    ax.set_ylabel("OOS 增量 $R^2$ 相对 M0（%）")
    ax.legend(frameon=False, fontsize=7.5)
    pub_style.soft_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_v3_oos.png", dpi=300)
    plt.close(fig)


def fig_absorption() -> None:
    ab = pd.read_parquet(V3_DIR / "v3_absorption.parquet")
    piv = ab.pivot_table(index=["theme", "product"], columns="signal",
                         values="corr")
    piv = piv.dropna()
    lab = [f"{t}×{p}" for t, p in piv.index]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(piv))
    w = 0.36
    ax.bar(x - w / 2, piv["base"], w, label="v1.1 基线", color=P["ink2"])
    ax.bar(x + w / 2, piv["v3"], w, label="v3 coherent", color=P["blue"])
    ax.axhline(0, color=P["ink"], lw=0.7)
    ax.set_xticks(x, lab, fontsize=7, rotation=20)
    ax.set_ylabel("闭市 gap 同期相关")
    ax.legend(frameon=False, fontsize=7.5)
    pub_style.soft_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_v3_absorption.png", dpi=300)
    plt.close(fig)


def fig_episode() -> None:
    """episode 聚类诊断：市场级伪重复与 episode 级支撑域（mideast×SC）。"""
    ep = pd.read_parquet(V3_DIR / "v3_episodes.parquet")
    sub = ep.loc[(ep["theme"] == "mideast_conflict") & (ep["product"] == "SC")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    ax = axes[0]
    ax.bar(range(len(sub)), sub.sort_values("n_markets", ascending=False)
           ["n_markets"], color=P["blue"], width=0.85)
    ax.set_xlabel("episode（按共享市场数降序）")
    ax.set_ylabel("共享同一收益日的市场数")
    pub_style.soft_grid(ax)
    pub_style.panel(ax, "a")
    ax = axes[1]
    ax.hist(sub["surprise"], bins=30, color=P["aqua"])
    ax.set_xlabel("episode 级 surprise（族内均值）")
    ax.set_ylabel("频数")
    pub_style.soft_grid(ax)
    pub_style.panel(ax, "b")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_v3_episode.png", dpi=300)
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pub_style.setup(cn_font=True)
    fig_term()
    fig_power()
    fig_oos()
    fig_absorption()
    fig_episode()
    print(f"图已写入 {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
