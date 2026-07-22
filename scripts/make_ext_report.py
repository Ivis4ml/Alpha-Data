"""扩展轮报告（docs/ext_round/）的表体生成器：数字全部从产物计算。

产物：
  e_tier.tex      P1 档位 x 主题分布
  e_locked.tex    P2 各品种真封板 / 伪锁 / 标签剔除
  e_ext_prod.tex  P5 扩展品种检验数与极值
  e_disp_top.tex  P4 分散度 |t| 前 8 格（带符号 t 与符号一致率）
  e_qtop.tex      P3 分位 |LS t| 前 10 对（单调 / 净收益 / 盈亏平衡）
  e_gate.tex      P6 七关矩阵（6 因子）
  e_ledger.tex    扩展主族总账（同 key_numbers_v2）

命令：.venv/bin/python scripts/make_ext_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "data" / "cn_futures" / "analysis" / "v4"
NEXT = ROOT / "data" / "cn_futures" / "analysis" / "next"
OUT = ROOT / "docs" / "ext_round"
KEY = json.loads((ROOT / "docs" / "concise" /
                  "key_numbers_v2.json").read_text())
H = [1, 2, 3, 5, 10, 15]
IC_T = [f"ic_t_{h}" for h in H]


def esc(s: str) -> str:
    return s.replace("_", r"\_")


def tier_table() -> None:
    tier = pd.read_parquet(
        ROOT / "data" / "polymarket" / "features" / "tier_registry.parquet")
    pv = tier.pivot_table(index="theme", columns="tier", values="product",
                          aggfunc="count", fill_value=0)
    pv = pv.reindex(columns=["primary", "secondary", "exploratory"])
    lines = [f"{esc(t)} & {int(r['primary'])} & {int(r['secondary'])} & "
             f"{int(r['exploratory'])} \\\\" for t, r in pv.iterrows()]
    tot = pv.sum()
    lines.append("\\midrule")
    lines.append(f"\\textbf{{合计}} & \\textbf{{{int(tot['primary'])}}} & "
                 f"\\textbf{{{int(tot['secondary'])}}} & "
                 f"\\textbf{{{int(tot['exploratory'])}}} \\\\")
    (OUT / "e_tier.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "主题 & primary & secondary & exploratory \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def locked_table() -> None:
    imp = pd.read_parquet(V4 / "locked_impact.parquet")
    lines = [
        f"{r['product']} & {r['true_lock_share']:.2%} & "
        f"{r['pseudo_lock_share']:.2%} & {r['dropped_share_1']:.2%} & "
        f"{r['dropped_share_15']:.2%} \\\\".replace("%", "\\%")
        for _, r in imp.iterrows()]
    (OUT / "e_locked.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "品种 & 真封板占比 & 伪锁占比 & fwd$_1$ 剔除 & fwd$_{15}$ 剔除 "
        "\\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def ext_prod_table() -> None:
    lines = []
    for p, v in KEY["ext_products"].items():
        lines.append(f"{p} & {v['tests']} & {v['max_t']:.3f} & "
                     f"{v['pure_pm_tests']} & {v['pure_pm_max_t']:.3f} \\\\")
    (OUT / "e_ext_prod.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "品种 & 检验数 & 极值 $|t|$ & 纯 PM 检验 & 纯 PM 极值 $|t|$ "
        "\\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def disp_table() -> None:
    dg = pd.read_parquet(V4 / "dispersion_grid.parquet")
    dg = dg.assign(m=dg[IC_T].abs().max(axis=1))
    top = dg.nlargest(8, "m")
    lines = []
    for _, r in top.iterrows():
        vals = r[IC_T].to_numpy(dtype=float)
        t_star = float(vals[np.nanargmax(np.abs(vals))])
        a = r[[f"ic_s_{h}" for h in H]].to_numpy(dtype=float)
        a = a[np.isfinite(a) & (a != 0)]
        cons = max((a > 0).sum(), (a < 0).sum()) / len(a) if len(a) else \
            float("nan")
        lines.append(
            f"{esc(r['signal'])}$\\cdot${r['product']} & "
            f"{r['nonzero_share']:.1%} & {t_star:+.3f} & "
            f"{cons:.0%} \\\\".replace("%", "\\%"))
    (OUT / "e_disp_top.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "信号$\\cdot$品种 & 非零占比 & 极值 $t$（带符号） & 六视界内一致率 "
        "\\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def quantile_table() -> None:
    q = pd.read_parquet(V4 / "quantile_grid.parquet")
    el = q[q["eligible"].astype(bool)].copy()
    el["a"] = el["ls_t_15"].abs()
    top = el.nlargest(10, "a")
    lines = []
    for _, r in top.iterrows():
        mono = "是" if r["monotone_15"] else "否"
        lines.append(
            f"{esc(r['signal'])}$\\cdot${r['product']} & "
            f"{r['ls_mean_bp_15']:+.2f} & {r['ls_t_15']:+.2f} & {mono} & "
            f"{r['strat_gross_bp_day']:+.1f} & {r['strat_cost_bp_day']:.1f} "
            f"& {r['strat_net_bp_day']:+.1f} \\\\")
    (OUT / "e_qtop.tex").write_text(
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "信号$\\cdot$品种 & LS$_{15}$(bp) & $t$ & 单调 & 日毛利(bp) & "
        "日成本(bp) & 日净(bp) \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def gate_table() -> None:
    sc = pd.read_parquet(NEXT / "quality_scorecard.parquet")
    gcols = [c for c in sc.columns if c.startswith("g") and
             sc[c].dtype == bool]
    heads = {"g1_prereg": "1 预注册", "g2_coverage": "2 覆盖",
             "g3_orthogonal": "3 正交增量", "g4_monotone": "4 单调",
             "g5_stability": "5 稳定", "g6_cost": "6 成本",
             "g7_no_lookahead": "7 无前视"}
    lines = []
    for _, r in sc.iterrows():
        cells = " & ".join("\\ding{51}" if r[c] else "$\\times$"
                           for c in gcols)
        lines.append(f"{esc(r['factor'])} & {cells} & "
                     f"{int(r['passed'])}/7 \\\\")
    (OUT / "e_gate.tex").write_text(
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l" + "c" * len(gcols) + "r}\n\\toprule\n"
        "因子 & " + " & ".join(heads[c] for c in gcols)
        + " & 通过 \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def ledger_table() -> None:
    fe = KEY["family_ext"]
    lines = [
        f"v3 冻结轮（原样保留） & {fe['n_v3']:,} & "
        f"{fe['threshold_v3']:.3f} \\\\",
        f"$+$ 扩展品种 CF/I/IF（P5） & {fe['n_ext_products']:,} & \\\\",
        f"$+$ 分散度矩族（P4） & {fe['n_dispersion']:,} & \\\\",
        f"$+$ 分位注册终点 $k{{=}}15$（P3） & {fe['n_quantile_k15']:,} & \\\\",
        "\\midrule",
        f"\\textbf{{扩展主族合计}} & \\textbf{{{fe['n_total']:,}}} & "
        f"\\textbf{{{fe['threshold']:.3f}}} \\\\",
    ]
    (OUT / "e_ledger.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "构成 & 检验数 & Bonferroni 门槛 \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tier_table()
    locked_table()
    ext_prod_table()
    disp_table()
    quantile_table()
    gate_table()
    ledger_table()
    print(f"written {OUT}/e_*.tex (7 files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
