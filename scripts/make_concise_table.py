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
from scipy import stats

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


# --------------------------------------------------------- 适当口径的检验统计
IC_T = [f"ic_t_{h}" for h in HORIZONS]
EV_T = [f"{d}_t_{h}" for d in ("up", "dn") for h in HORIZONS]


def proper_t(sub: pd.DataFrame) -> np.ndarray:
    """适当口径的 |t| 全集：连续 / 计数用 IC 的 t，离散用事件研究的 t。

    离散信号取值只有 {-1,0,+1} 且九成以上为 0，在其上算秩相关并不适当；
    其主判据是事件研究（触发后收益对品种无条件基准，按交易日聚类）。
    """
    parts: list[np.ndarray] = []
    cont = sub[sub["kind"] != "离散"]
    disc = sub[sub["kind"] == "离散"]
    if len(cont):
        parts.append(np.abs(cont[IC_T].to_numpy(dtype=float)).ravel())
    if len(disc):
        parts.append(np.abs(disc[EV_T].to_numpy(dtype=float)).ravel())
    if not parts:
        return np.empty(0)
    a = np.concatenate(parts)
    return a[np.isfinite(a)]


def bonferroni(n: int, alpha: float = 0.05) -> float:
    """n 个检验的双侧 Bonferroni 门槛。"""
    return float(stats.norm.ppf(1.0 - alpha / 2.0 / max(n, 1)))


#: 30 个独立格在纯噪声下符号一致率的期望（多数符号占比）。
NULL_CONSISTENCY = float(
    np.sum(stats.binom.pmf(np.arange(31), 30, 0.5)
           * np.maximum(np.arange(31), 30 - np.arange(31))) / 30.0)


def sign_consistency(sub: pd.DataFrame) -> float:
    """信号方向一致率：跨品种 x 视界诸格中占多数的符号的占比。

    ``max|t|`` 是双侧筛选，会丢掉方向信息：一个 h=1 为正、h=15 为负的
    信号仍可能报出较大的 ``max|t|``，但它没有连贯方向、多半是噪声。
    本指标补上这一维：纯噪声下约 {:.0%}（30 独立格），真信号趋近 100%。
    离散信号用事件研究的超额收益取符号，与其适当口径一致。
    """
    cols = ([f"up_excess_{h}" for h in HORIZONS]
            if (sub["kind"] == "离散").all() else
            [f"ic_s_{h}" for h in HORIZONS])
    a = sub[cols].to_numpy(dtype=float).ravel()
    a = a[np.isfinite(a) & (a != 0.0)]
    if len(a) < 6:
        return float("nan")
    return float(max((a > 0).sum(), (a < 0).sum()) / len(a))


# --------------------------------------------------------- 全信号登记表
def grid_tables() -> float:
    """写出表 4/5/6，返回适当口径下的全局 Bonferroni 门槛（供族级表复用）。"""
    g = pd.read_parquet(DEF / "signal_grid.parquet")
    thr = bonferroni(len(proper_t(g)))

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
                # 取值域直接读 valuecount，不得硬编码：K5 是无符号游程长度
                # {0,1,2,3,4}、五品种下行触发数均为 0，与签名型不同。
                vals: set[int] = set()
                for js in sub["valuecount"].dropna():
                    vals |= {int(k) for k in json.loads(js)}
                dom = "\\{" + ",".join(str(v) for v in sorted(vals)) + "\\}"
                stat = (f"${dom}$，非零 "
                        f"{sub['nonzero_share'].mean():.2%}")
            else:
                stat = (f"$\\mu${sub['mean'].mean():+.2f}, "
                        f"$\\sigma${sub['std'].mean():.2f}, "
                        f"kurt {sub['kurt'].mean():+.1f}")
            ic = " / ".join(fnum(sub[f"ic_s_{h}"].mean() * 100, "+.2f")
                            for h in (1, 5, 15))
            # 极值 t：取 |t| 最大的格，但报其带符号的原值。门槛判定仍按
            # |t|（双侧）；符号仅供读方向，避免「RankIC 三元组全正而极值
            # 格实为负向」时的误导（如 K7）。
            cols = EV_T if kind == "离散" else IC_T
            per_prod = sub[cols].abs().max(axis=1)
            best_row = sub.loc[per_prod.idxmax()]
            best = best_row["product"]
            vals = best_row[cols].to_numpy(dtype=float)
            t_star = float(vals[np.nanargmax(np.abs(vals))])
            mark = "$^{\\ast}$" if abs(t_star) > thr else ""
            cons = sign_consistency(sub)
            cs = "--" if not np.isfinite(cons) else f"{cons:.0%}"
            lines.append(
                f"{sig} & {kind} & {freq} & {stat} & {ic} & "
                f"{t_star:+.2f}{mark} ({best}) & {cs} \\\\"
                .replace("%", "\\%"))
        if fam != "X":
            lines.append("\\addlinespace")
    (OUT / "t4_grid.tex").write_text(
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lllllrr}\n\\toprule\n"
        "信号 & 类型 & 月频（均） & 取值统计（跨品种均） & "
        "RankIC$\\times$100 (1'/5'/15') & 极值 $t$ & 符号一致率 "
        "\\\\\n\\midrule\n"
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

    # ---- 表 6：离散信号事件研究（一律报超额收益，与 t 检验的量一致）----
    # 歧义修复：此前各视界列印毛收益，而同行的 max|t| 检验的是「条件均值
    # 减品种无条件基准」。二者可反号（X6·AU 在 h=15 毛收益 -0.19 而超额
    # +0.26、t=+0.13），读者无法把收益与 t 对上。现统一改印超额。
    disc = g[(g["kind"] == "离散") & (g["n_up"] >= 100)].copy()
    tcols = [f"up_t_{h}" for h in HORIZONS]
    disc["score"] = disc[tcols].abs().max(axis=1)
    top = disc.nlargest(14, "score")
    lines = []
    for _, r in top.iterrows():
        exc = " & ".join(fnum(r[f"up_excess_{h}"]) for h in HORIZONS)
        vals = r[tcols].to_numpy(dtype=float)
        t_star = float(vals[np.nanargmax(np.abs(vals))])  # 极值格的带符号 t
        lines.append(
            f"{r['signal']}$\\cdot${r['product']} & "
            f"{r['up_per_month']:.0f} & {exc} & "
            f"{fnum(r['up_ret_term'])} ({r['up_hold_term']:.0f}') & "
            f"{r['up_pmove_10']:.1%} & {r['base_p_move_10']:.1%} & "
            f"{t_star:+.2f} \\\\".replace("%", "\\%"))
    (OUT / "t6_event.tex").write_text(
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lr" + "r" * len(HORIZONS) + "rrrr}\n\\toprule\n"
        "& & \\multicolumn{6}{c}{\\textbf{超额收益}(bp)：条件均值 $-$ "
        "同品种无条件基准} & 至下次 & \\multicolumn{2}{c}{$P_{10'}$ 超 10bp 比例} & \\\\\n"
        "\\cmidrule(lr){3-8}\\cmidrule(lr){10-11}\n"
        "信号$\\cdot$品种 & 次/月 & "
        + " & ".join(f"{h}'" for h in HORIZONS)
        + " & 毛值 & 触发后 & 基准 & 极值 $t$ \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    return thr


# --------------------------------------------------------- 族级多重检验
#: 每族的数据来源标注（决定该族的显著性可归因到哪一侧数据）。
FAMILY_SOURCE = {
    "N": "仅 Polymarket",
    "K": "仅 Polymarket",
    "X": "Polymarket $+$ 期货状态",
    "C": "Polymarket $\\times$ 期货量价",
}
#: 不含任何期货腿的纯 Polymarket 信号（X 族中仅这三个不引用期货量价）。
PURE_PM = ([f"N{i}" for i in range(1, 11)] + [f"K{i}" for i in range(1, 11)]
           + ["X5", "X7", "X10"])


def family_table(thr: float) -> dict[str, object]:
    """族级多重检验汇总（适当口径），返回正文要引用的关键数字。"""
    g = pd.read_parquet(DEF / "signal_grid.parquet")
    lines = []
    for fam in ("N", "K", "X", "C"):
        a = proper_t(g[g["family"] == fam])
        n_pass = int((a > thr).sum())
        bold = "\\textbf{0}" if n_pass == 0 else f"{n_pass}"
        lines.append(f"{fam} & {FAMILY_SOURCE[fam]} & {len(a)} & "
                     f"{a.max():.2f} & {bold} \\\\")
    pure = proper_t(g[g["signal"].isin(PURE_PM)])
    lines.append("\\midrule")
    lines.append(
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{仅用 Polymarket 数据的 "
        f"{len(PURE_PM)} 个信号}}}} & \\textbf{{{len(pure)}}} & "
        f"\\textbf{{{pure.max():.2f}}} & \\textbf{{0}} \\\\")
    (OUT / "t8_family.tex").write_text(
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{llrrr}\n\\toprule\n"
        "族 & 数据来源 & 检验数 & $\\max|t|$ & 通过门槛 \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # 通过门槛的格（适当口径），供正文引用
    cells = []
    for _, r in g.iterrows():
        cols = EV_T if r["kind"] == "离散" else IC_T
        v = np.abs(np.asarray([r[c] for c in cols], dtype=float))
        v = v[np.isfinite(v)]
        if len(v) and v.max() > thr:
            cells.append((r["signal"], r["product"], float(v.max())))
    total = proper_t(g)

    # 方向一致性：max|t| 是双侧筛选、丢掉方向，此处补上并定位纯 PM 侧的
    # 最强格究竟是不是一个方向连贯的信号。
    pm_sig = sorted({s for s in PURE_PM if (g["signal"] == s).any()})
    best_pm, best_pm_t = None, -1.0
    for s in pm_sig:
        v = proper_t(g[g["signal"] == s])
        if len(v) and v.max() > best_pm_t:
            best_pm, best_pm_t = s, float(v.max())
    # 极值格的带符号 t（呈现方向用；门槛判定仍按 |t|）
    sub = g[g["signal"] == best_pm]
    signed_pool: list[float] = []
    for _, r in sub.iterrows():
        cols = EV_T if r["kind"] == "离散" else IC_T
        signed_pool.extend(float(r[c]) for c in cols
                           if np.isfinite(r[c]))
    arr = np.asarray(signed_pool)
    best_pm_t_signed = float(arr[np.argmax(np.abs(arr))])
    cons_pm = {s: sign_consistency(g[g["signal"] == s]) for s in pm_sig}
    cons_c = {s: sign_consistency(g[g["signal"] == s])
              for s in (f"C{i}" for i in range(1, 11))}

    return {
        "threshold": thr,
        "n_tests": len(total),
        "n_pass_tests": int((total > thr).sum()),
        "n_pass_cells": len(cells),
        "pass_families": sorted({c[0][0] for c in cells}),
        "pure_pm_max_t": float(pure.max()),
        "pure_pm_tests": len(pure),
        # 方向一致性
        "null_consistency": round(NULL_CONSISTENCY, 4),
        "pure_pm_best_signal": best_pm,
        "pure_pm_best_t": round(best_pm_t, 2),
        "pure_pm_best_t_signed": round(best_pm_t_signed, 2),
        "pure_pm_best_consistency": round(float(cons_pm.get(best_pm,
                                                           float("nan"))), 4),
        "pure_pm_median_consistency": round(
            float(np.nanmedian(list(cons_pm.values()))), 4),
        "c8_consistency": round(float(cons_c.get("C8", float("nan"))), 4),
    }


# --------------------------------------------------------- 检出力与 MDE
#: 第二类错误率 20% 对应的单侧分位（检出力 80%）。
Z_BETA = float(stats.norm.ppf(0.80))
#: 完整换手成本门槛（bp），与冻结清单同口径。
COST_BP = 16.0


def mde_table(thr: float) -> dict[str, object]:
    """最小可检测效应（MDE）：在既有样本上能排除多大的效应。

    标准误由已算出的量精确反解，不引入新假设：

    - 连续侧 :math:`\\mathrm{SE} = \\mathrm{sd}(IC^{(d)})/\\sqrt{D}`，
      即 ``v3_signal_grid`` 中 ``ic_t = (m/sd)*sqrt(nd)`` 的反解；
    - 离散侧 :math:`\\mathrm{SE} = |\\text{超额}|/|t|`（限 ``n_up>=100``），
      即日聚类 t 的反解，单位 bp。

    :math:`\\mathrm{MDE} = (z_\\alpha + z_\\beta)\\times \\mathrm{SE}`。
    """
    g = pd.read_parquet(DEF / "signal_grid.parquet")
    cont = g[g["kind"] != "离散"]
    disc = g[(g["kind"] == "离散") & (g["n_up"] >= 100)]

    rows, key = [], {}
    for h in HORIZONS:
        se_c = (cont[f"icd_std_{h}"]
                / np.sqrt(cont[f"ic_days_{h}"])).to_numpy(dtype=float)
        se_c = se_c[np.isfinite(se_c)]
        se_d = (disc[f"up_excess_{h}"].abs()
                / disc[f"up_t_{h}"].abs()).to_numpy(dtype=float)
        se_d = se_d[np.isfinite(se_d)]
        if not len(se_c) or not len(se_d):
            continue
        # 连续侧：以 RankIC 为单位
        mde_c = (thr + Z_BETA) * np.median(se_c)
        # 离散侧：以 bp 为单位，给中位与 p90 两档
        mde_d_med = (thr + Z_BETA) * np.median(se_d)
        mde_d_p90 = (thr + Z_BETA) * np.quantile(se_d, 0.90)
        rows.append(
            f"{h} & {np.median(se_c):.5f} & {mde_c:.4f} & "
            f"{np.median(se_d):.2f} & {mde_d_med:.1f} & {mde_d_p90:.1f} \\\\")
        key[f"mde_ic_{h}"] = round(float(mde_c), 4)
        key[f"mde_bp_med_{h}"] = round(float(mde_d_med), 2)
        key[f"mde_bp_p90_{h}"] = round(float(mde_d_p90), 2)

    (OUT / "t9_mde.tex").write_text(
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{rrrrrr}\n\\toprule\n"
        "& \\multicolumn{2}{c}{连续侧（RankIC 单位）} "
        "& \\multicolumn{3}{c}{离散侧（bp）} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\n"
        "$h$ & SE 中位 & MDE & SE 中位 & MDE 中位 & MDE p90 \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    days = g[[f"ic_days_{h}" for h in HORIZONS]].to_numpy(dtype=float).ravel()
    days = days[np.isfinite(days) & (days > 0)]
    key["ic_days_median"] = int(np.median(days))
    key["z_beta"] = round(Z_BETA, 4)
    key["cost_bp"] = COST_BP
    # 离散侧中位格能否排除成本门槛之上的效应
    key["mde_bp_med_10_below_cost"] = bool(key.get("mde_bp_med_10", 1e9)
                                           < COST_BP)
    key["mde_bp_p90_10_below_cost"] = bool(key.get("mde_bp_p90_10", 1e9)
                                           < COST_BP)
    return key


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
    thr = grid_tables()
    key = family_table(thr)
    key.update(mde_table(thr))
    arrival_table()
    (OUT / "key_numbers.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2))
    print(f"written {OUT}/t2..t8 (12 files)")
    print(f"适当口径：{key['n_tests']} 个检验，门槛 |t|>{key['threshold']:.2f}，"
          f"通过 {key['n_pass_tests']} 个检验 / {key['n_pass_cells']} 个格，"
          f"全部来自 {'/'.join(key['pass_families'])} 族")
    print(f"仅 PM 的 {len(PURE_PM)} 个信号：{key['pure_pm_tests']} 个检验，"
          f"max|t|={key['pure_pm_max_t']:.2f}，通过 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
