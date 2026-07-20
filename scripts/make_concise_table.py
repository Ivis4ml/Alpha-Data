"""精炼版 LaTeX 报告的全部表体生成器（数字一律从产物计算，不手抄）。

产物（docs/concise/）：

  t2_main.tex   规定格式登记表主表（8 条头部记录，源 signal_summary.json）
  t2_stats.tex  规定格式登记表的取值统计量
  t2_base.tex   品种基准收益
  t3_{n,c,k,x}.tex  四族各 10 个信号的中文释义与 LaTeX 公式
  t4_grid.tex   40 信号登记全表（类型 / 月频 / 取值统计 / 跨品种 IC / 最大 |t|）
  t5_ic.tex     连续信号六视界 RankIC 与 ICIR 明细（|t| 前 14 格）
  t6_event.tex  离散信号事件研究（月频 / 六视界收益 / 超 10bp 比例 / 基准 / t）
  t7_arrival.tex 主题事件全历史月度到达率（2023-09..2026-07）

重算：
  .venv/bin/python scripts/v3_signal_grid.py
  .venv/bin/python scripts/export_signal_summary.py
  .venv/bin/python scripts/make_concise_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from concise_signal_defs import DEFS  # noqa: E402

OUT = ROOT / "docs" / "concise"
DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
HORIZONS = [1, 2, 3, 5, 10, 15]


def esc(s: str) -> str:
    return s.replace("_", r"\_")


def fnum(v: object, fmt: str = "+.2f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return format(float(v), fmt)


# --------------------------------------------------------- 规定格式登记表
def tri(r: dict, tpl: str, fmt: str) -> str:
    vs = [r.get(tpl.format(h)) for h in (1, 3, 10)]
    if vs[0] is None:
        return "--"
    return " / ".join(format(v, fmt) for v in vs)


def summary_tables() -> None:
    obj = json.loads((ROOT / "docs" / "signal_summary.json").read_text())
    rows = obj["records"]

    main_lines = []
    for r in rows:
        freq = (f"{r['信号月次数'] / 1000:.1f}k 分钟" if r["信号类型"] == "连续"
                else f"{r['信号月次数']:.0f} 次")
        main_lines.append(
            f"{r['交易品种']} & {esc(r['信号'])} & {r['信号类型']} & {freq} & "
            f"{tri(r, '连续信号IC（发出后{}分钟）', '+.4f')} & "
            f"{tri(r, '连续信号ICIR（发出后{}分钟）', '+.2f')} & "
            f"{tri(r, '离散信号（取1）后{}分钟平均收益', '+.2f')} & "
            f"{tri(r, '离散信号符号RankIC（发出后{}分钟）', '+.4f')} \\\\")
    (OUT / "t2_main.tex").write_text(
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\begin{tabular}{llllllll}\n\\toprule\n"
        "品种 & 信号 & 类型 & 月频 & RankIC & ICIR & "
        "取1收益(bp) & 符号IC \\\\\n\\midrule\n"
        + "\n".join(main_lines) + "\n\\bottomrule\n\\end{tabular}\n")

    stat_lines = []
    for r in rows:
        if r["取值均值"] is not None:
            stats = (f"$\\mu$={r['取值均值']:+.3f}, med={r['取值中位数']:+.3f}, "
                     f"$\\sigma$={r['取值标准差']:.3f}, "
                     f"skew={r['取值偏度']:+.2f}, kurt={r['取值峰度']:+.2f}")
        else:
            stats = "; ".join(f"{k}: {v:,}" for k, v in
                              sorted(r["取值valuecount"].items()))
        stat_lines.append(
            f"{r['交易品种']} & {esc(r['信号'])} & {stats} & "
            f"{r['取值非零频率']:.2%} \\\\".replace("%", "\\%"))
    (OUT / "t2_stats.tex").write_text(
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{llp{8.6cm}l}\n\\toprule\n"
        "品种 & 信号 & 统计量 / valuecount & 非零频率 \\\\\n\\midrule\n"
        + "\n".join(stat_lines) + "\n\\bottomrule\n\\end{tabular}\n")

    base_lines, seen = [], set()
    for r in rows:
        p = r["交易品种"]
        if p in seen:
            continue
        seen.add(p)
        base_lines.append(
            f"{p} & {r['品种全区间收对收累计收益']:+.1%} & "
            f"{r['品种无条件1分钟均值收益_bp']:+.3f} & "
            f"{r['品种无条件3分钟均值收益_bp']:+.3f} & "
            f"{r['品种无条件10分钟均值收益_bp']:+.3f} \\\\".replace("%", "\\%"))
    (OUT / "t2_base.tex").write_text(
        "\\begin{tabular}{lllll}\n\\toprule\n"
        "品种 & 收对收累计 & 无条件 1' (bp) & 3' (bp) & 10' (bp) \\\\\n"
        "\\midrule\n" + "\n".join(base_lines)
        + "\n\\bottomrule\n\\end{tabular}\n")


# --------------------------------------------------------- 四族公式表
def family_tables() -> None:
    for fam in "NCKX":
        lines = []
        for i in range(1, 11):
            sig = f"{fam}{i}"
            name, formula, intent = DEFS[sig]
            lines.append(f"{sig} & {name} & $\\displaystyle {formula}$ & "
                         f"{intent} \\\\")
        (OUT / f"t3_{fam.lower()}.tex").write_text(
            "\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{llp{6.4cm}p{4.5cm}}\n\\toprule\n"
            "信号 & 名称 & 公式 & 构造意图 \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# --------------------------------------------------------- 全信号登记表
def grid_tables() -> None:
    g = pd.read_parquet(DEF / "signal_grid.parquet")

    # ---- 表 4：40 信号 x 跨品种汇总 ----
    lines = []
    for fam in "NCKX":
        for i in range(1, 11):
            sig = f"{fam}{i}"
            sub = g[g["signal"] == sig]
            if sub.empty:
                continue
            kind = sub["kind"].iloc[0]
            pm = sub["per_month"].mean()
            freq = (f"{pm:,.0f} 分钟" if kind != "离散"
                    else f"{pm:,.0f} 次")
            if kind == "离散":
                stat = (f"$\\{{-1,0,1\\}}$，非零 "
                        f"{sub['nonzero_share'].mean():.2%}")
            else:
                stat = (f"$\\mu${sub['mean'].mean():+.2f}, "
                        f"$\\sigma${sub['std'].mean():.2f}, "
                        f"kurt {sub['kurt'].mean():+.1f}")
            ic = " / ".join(fnum(sub[f"ic_s_{h}"].mean() * 100, "+.2f")
                            for h in (1, 5, 15))
            tcol = [c for c in (f"ic_t_{h}" for h in HORIZONS)]
            tmax = np.nanmax(np.abs(sub[tcol].to_numpy()))
            best = sub.loc[sub[tcol].abs().max(axis=1).idxmax(), "product"]
            lines.append(
                f"{sig} & {kind} & {freq} & {stat} & {ic} & "
                f"{tmax:.2f} ({best}) \\\\".replace("%", "\\%"))
        if fam != "X":
            lines.append("\\addlinespace")
    (OUT / "t4_grid.tex").write_text(
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{lllllr}\n\\toprule\n"
        "信号 & 类型 & 月频（均） & 取值统计（跨品种均） & "
        "RankIC$\\times$100 (1'/5'/15') & 最大 $|t|$ \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- 表 5：连续信号六视界明细 ----
    cont = g[g["kind"] != "离散"].copy()
    tcol = [f"ic_t_{h}" for h in HORIZONS]
    cont["score"] = cont[tcol].abs().max(axis=1)
    top = cont.nlargest(14, "score")
    lines = []
    for _, r in top.iterrows():
        ics = " & ".join(fnum(r[f"ic_s_{h}"] * 100, "+.2f") for h in HORIZONS)
        lines.append(
            f"{r['signal']}$\\cdot${r['product']} & {ics} & "
            f"{fnum(r['icir_15'])} & {fnum(r['ic_t_15'])} \\\\")
    (OUT / "t5_ic.tex").write_text(
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l" + "r" * len(HORIZONS) + "rr}\n\\toprule\n"
        "信号$\\cdot$品种 & "
        + " & ".join(f"{h}'" for h in HORIZONS)
        + " & ICIR$_{15}$ & $t_{15}$ \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- 表 6：离散信号事件研究 ----
    disc = g[(g["kind"] == "离散") & (g["n_up"] >= 100)].copy()
    tcols = [f"up_t_{h}" for h in HORIZONS]
    disc["score"] = disc[tcols].abs().max(axis=1)
    top = disc.nlargest(14, "score")
    lines = []
    for _, r in top.iterrows():
        rets = " & ".join(fnum(r[f"up_ret_{h}"]) for h in HORIZONS)
        lines.append(
            f"{r['signal']}$\\cdot${r['product']} & "
            f"{r['up_per_month']:.0f} & {rets} & "
            f"{fnum(r['up_ret_term'])} ({r['up_hold_term']:.0f}') & "
            f"{r['up_pmove_10']:.1%} & {r['base_p_move_10']:.1%} & "
            f"{fnum(r['score'], '.2f')} \\\\".replace("%", "\\%"))
    (OUT / "t6_event.tex").write_text(
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lr" + "r" * len(HORIZONS) + "rrrr}\n\\toprule\n"
        "信号$\\cdot$品种 & 次/月 & "
        + " & ".join(f"{h}'" for h in HORIZONS)
        + " & 至下次 & $P_{10'}$ & 基准 & $\\max|t|$ \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# --------------------------------------------------------- 全历史到达率
def arrival_table() -> None:
    h = pd.read_parquet(DEF / "history_monthly.parquet")
    g = h.groupby("theme").agg(
        months=("month", "nunique"), first=("month", "min"),
        last=("month", "max"), events=("n_events", "sum"),
        usdc=("usdc", "sum"), mkts=("n_markets", "max"),
        tmin=("n_trade_minutes", "sum"))
    g["per_month"] = g["events"] / g["months"]
    g = g.sort_values("events", ascending=False)
    lines = [
        f"{esc(idx)} & {r['first']}..{r['last']} & {int(r['months'])} & "
        f"{int(r['mkts'])} & {int(r['events']):,} & {r['per_month']:,.0f} & "
        f"{r['usdc'] / 1e6:,.1f} \\\\"
        for idx, r in g.iterrows()]
    tot = (f"\\textbf{{合计}} & 2023-09..2026-07 & 35 & -- & "
           f"\\textbf{{{int(g['events'].sum()):,}}} & "
           f"\\textbf{{{g['events'].sum() / 35:,.0f}}} & "
           f"\\textbf{{{g['usdc'].sum() / 1e6:,.1f}}} \\\\")
    (OUT / "t7_arrival.tex").write_text(
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "主题 & 覆盖月份 & 月数 & 市场 & 事件数 & 次/月 & 成交额(M\\$) \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\midrule\n" + tot
        + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_tables()
    family_tables()
    grid_tables()
    arrival_table()
    print(f"written {OUT}/t2..t7 (11 files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
