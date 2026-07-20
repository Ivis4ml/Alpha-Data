"""精炼版 LaTeX 报告的表格生成器（从 signal_summary.json 生成）。

产物（docs/concise/）：
  t2_main.tex   表 2 主表体（IC / ICIR / 取1收益 / 符号IC，三元组打包）
  t2_stats.tex  取值统计量表体（均值/中位/σ/偏度/峰度 或 valuecount）
  t2_base.tex   品种基准收益表体
数字与正文表 2 同源，重算命令：
  python scripts/export_signal_summary.py && python scripts/make_concise_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "concise"


def tri(r: dict, tpl: str, fmt: str) -> str:
    vs = [r.get(tpl.format(h)) for h in (1, 3, 10)]
    if vs[0] is None:
        return "--"
    return " / ".join(format(v, fmt) for v in vs)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    obj = json.loads((ROOT / "docs" / "signal_summary.json").read_text())
    rows = obj["records"]

    main_lines = []
    for r in rows:
        freq = (f"{r['信号月次数']:,.0f} 分钟/月" if r["信号类型"] == "连续"
                else f"{r['信号月次数']:.1f} 次/月")
        main_lines.append(
            f"{r['交易品种']} & {r['信号'].replace('_', chr(92) + '_')} & "
            f"{r['信号类型']} & {freq} & "
            f"{tri(r, '连续信号IC（发出后{}分钟）', '+.4f')} & "
            f"{tri(r, '连续信号ICIR（发出后{}分钟）', '+.2f')} & "
            f"{tri(r, '离散信号（取1）后{}分钟平均收益', '+.2f')} & "
            f"{tri(r, '离散信号符号RankIC（发出后{}分钟）', '+.4f')} \\\\")
    main_tab = (
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{llllllll}\n\\toprule\n"
        "品种 & 信号 & 类型 & 月频 & RankIC & ICIR（日度） & "
        "取1收益(bp) & 符号IC \\\\\n\\midrule\n"
        + "\n".join(main_lines)
        + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / "t2_main.tex").write_text(main_tab)

    stat_lines = []
    for r in rows:
        if r["取值均值"] is not None:
            stats = (f"$\\mu$={r['取值均值']:+.3f}, med={r['取值中位数']:+.3f}, "
                     f"$\\sigma$={r['取值标准差']:.3f}, "
                     f"skew={r['取值偏度']:+.2f}, kurt={r['取值峰度']:+.2f}")
        else:
            vc = r["取值valuecount"]
            stats = "; ".join(f"{k}: {v:,}" for k, v in
                              sorted(vc.items()))
        stat_lines.append(
            f"{r['交易品种']} & {r['信号'].replace('_', chr(92) + '_')} & "
            f"{stats} & {r['取值非零频率']:.2%} \\\\"
            .replace("%", "\\%"))
    stats_tab = (
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{llp{8.6cm}l}\n\\toprule\n"
        "品种 & 信号 & 统计量 / valuecount & 非零频率 \\\\\n"
        "\\midrule\n"
        + "\n".join(stat_lines)
        + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / "t2_stats.tex").write_text(stats_tab)

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
            f"{r['品种无条件10分钟均值收益_bp']:+.3f} \\\\"
            .replace("%", "\\%"))
    base_tab = (
        "\\begin{tabular}{lllll}\n\\toprule\n"
        "品种 & 收对收累计 & 无条件 1' (bp) & 3' (bp) & 10' (bp) "
        "\\\\\n\\midrule\n"
        + "\n".join(base_lines)
        + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / "t2_base.tex").write_text(base_tab)
    print(f"written {OUT}/t2_{{main,stats,base}}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
