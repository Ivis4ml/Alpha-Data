"""按导师规定的结构化格式导出已发现信号的汇总 JSON。

字段（每条记录）：交易品种 / 信号定义 / 信号频率 / 信号月次数 /
连续信号 IC（发出后 1 / 3 / 10 分钟）/ 离散信号（取 1）后
1 / 3 / 10 分钟平均收益。

口径说明（全部数字构建时从 parquet 计算，不手抄）：
- IC = 全时段 pooled RankIC（Spearman，ic_table 口径）。注意该口径按
  分钟行计数，推断层面的降格与警示见报告 §12.14-12.15，此表为描述。
- 离散信号"取 1"= 上行触发（E_up = 1 或事件跳 J_evt > 0），平均收益
  为触发后前向收益均值（bp，未扣成本）。
- 信号月次数：连续信号 = 有效信号分钟数 / 月；离散 = 触发次数 / 月。
  月数按样本 125 个交易日 / 21 折算（约 5.95 个月）。

扩展字段（同一 JSON 内，正文统计量表与直方图同源渲染）：公式 /
取值统计量（连续：均值、中位数、标准差、偏度、峰度、直方图面板号；
离散：value count 与触发频率）/ 品种全区间平均收益（无条件基准：
样本期收对收累计与 1/3/10 分钟无条件均值，供离散条件均值对照）。

产物：docs/signal_summary.json 与 docs/figures/f_signal_summary_hist.png
"""
from __future__ import annotations

import json
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
JD = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "jump"
OUT = ROOT / "docs" / "signal_summary.json"
N_MONTHS = 125 / 21
NA = "不适用"

C8_DEF = ("C8 未兑现缺口 = z(过去120交易分钟PM信念累积) − z(同窗期货"
          "收益累积)；复合信号，强度主要由期货反转腿承载，PM 腿增量"
          "未通过确认门槛（报告 §12.14-12.15），本行为描述性登记")
C8_FORMULA = ("C8_t = z_4800(Σ_{u=t-119..t} N1_u) − "
              "z_4800(Σ_{u=t-119..t} r1_u)，z 为 4,800 分钟滚动"
              "（min 960），N1 为主题分钟信念创新、r1 为期货 1 分钟收益")
E1_FORMULA = ("E_up,t = 1{N1_t > κ_t}，κ_t = max(3×1.4826×"
              "MAD_30日(N1≠0), 0.05)")
JUMP_FORMULA = ("15 分钟桶 |Δℓ| > 3×1.4826×MAD_48桶 且桶内成交 ≥ 1 万"
                "美元；J_evt = √usdc 加权的 orientation×Δℓ（(theme,ts) "
                "折叠）；取 1 = J_evt > 0")


def baseline_of(prod: str) -> str:
    """品种全区间平均收益：收对收累计 + 无条件 1/3/10 分钟均值。"""
    daily = pd.read_parquet(ROOT / "data" / "cn_futures" / "daily"
                            / f"{prod}.parquet",
                            columns=["r_cc", "roll"])
    r = daily.loc[~daily["roll"].astype(bool), "r_cc"].dropna()
    cum = float(np.expm1(r.sum()))
    panel = pd.read_parquet(DEF / f"panel_{prod}.parquet",
                            columns=["fwd_1", "fwd_3", "fwd_10"])
    m = {h: float(panel[f"fwd_{h}"].mean()) for h in (1, 3, 10)}
    return (f"样本期收对收累计 {cum:+.1%}（剔换月日）；无条件分钟均值 "
            f"1'={m[1]*1e4:+.3f} / 3'={m[3]*1e4:+.3f} / "
            f"10'={m[10]*1e4:+.3f} bp")


def fmt_ic(x: float) -> str:
    return f"{x:+.4f}"


def fmt_bp(x: float) -> str:
    return f"{x * 1e4:+.2f}bp"


def cont_row(prod: str, ic: pd.DataFrame, panel: pd.DataFrame,
             hist_panel: str) -> dict:
    sub = ic[(ic["product"] == prod) & (ic["signal"] == "C8")
             & (ic["scope"] == "all")].set_index("horizon")
    v = panel["C8"].replace(0.0, np.nan).dropna()
    n_valid = int(len(v))
    return {
        "交易品种": prod,
        "信号定义": C8_DEF,
        "公式": C8_FORMULA,
        "信号频率": "逐分钟连续值（120 分钟滚动窗，跨段按交易分钟序）",
        "信号月次数": f"约 {n_valid / N_MONTHS:,.0f} 个有效信号分钟/月",
        "连续信号IC（发出后1分钟）": fmt_ic(sub.loc[1, "rank_ic"]),
        "连续信号IC（发出后3分钟）": fmt_ic(sub.loc[3, "rank_ic"]),
        "连续信号IC（发出后10分钟）": fmt_ic(sub.loc[10, "rank_ic"]),
        "离散信号（取1）后1分钟平均收益": NA,
        "离散信号（取1）后3分钟平均收益": NA,
        "离散信号（取1）后10分钟平均收益": NA,
        "取值均值": f"{v.mean():+.4f}",
        "取值中位数": f"{v.median():+.4f}",
        "取值标准差": f"{v.std():.4f}",
        "取值偏度": f"{v.skew():+.3f}",
        "取值峰度": f"{v.kurt():+.3f}",
        "取值valuecount": NA,
        "取值频率": f"有效分钟占比 {n_valid / len(panel):.1%}",
        "直方图": f"docs/figures/f_signal_summary_hist.png 面板 {hist_panel}",
        "品种全区间平均收益": baseline_of(prod),
    }


def e1_row(panel: pd.DataFrame) -> dict:
    m = panel["E_up"] == 1
    n = int(m.sum())
    n_dn = int((panel["E_dn"] == 1).sum())
    n_zero = int(len(panel) - n - n_dn)
    vals = {h: float(panel.loc[m, f"fwd_{h}"].mean()) for h in (1, 3, 10)}
    return {
        "交易品种": "SC",
        "信号定义": "E1 上行价格跳：主题 1 分钟 logit 创新 > 3×1.4826×"
                    "MAD48 稳健阈值（E_up = 1；下行对称、此处只登记取 1）",
        "信号频率": "事件型（1 分钟粒度触发）",
        "信号月次数": f"约 {n / N_MONTHS:.0f} 次/月",
        "连续信号IC（发出后1分钟）": NA,
        "连续信号IC（发出后3分钟）": NA,
        "连续信号IC（发出后10分钟）": NA,
        "离散信号（取1）后1分钟平均收益": fmt_bp(vals[1]),
        "离散信号（取1）后3分钟平均收益": fmt_bp(vals[3]),
        "离散信号（取1）后10分钟平均收益": fmt_bp(vals[10]),
        "公式": E1_FORMULA,
        "取值均值": NA, "取值中位数": NA, "取值标准差": NA,
        "取值偏度": NA, "取值峰度": NA,
        "取值valuecount": f"+1（上行跳）: {n:,} 次；−1（下行跳，"
                          f"对称登记）: {n_dn:,} 次；0: {n_zero:,} 分钟",
        "取值频率": f"触发占比 {(n + n_dn) / len(panel):.2%}"
                    f"（取 1 占 {n / len(panel):.2%}）",
        "直方图": "docs/figures/f_signal_summary_hist.png 面板 (f)",
        "品种全区间平均收益": baseline_of("SC"),
    }


def jump_row(prod: str, themes: list[str], isolated: bool, label: str,
             session_day_only: bool, defn: str, formula: str) -> dict:
    jumps = pd.read_parquet(JD / "jumps.parquet")
    panel = pd.read_parquet(DEF / f"panel_{prod}.parquet",
                            columns=["ts", "trade_date", "session",
                                     "fwd_1", "fwd_3", "fwd_10"])
    jp = jumps[jumps["theme"].isin(themes)]
    if isolated:
        jp = jp[jp["n_cojump"] == 0]
    ev = (jp.groupby(["theme", "ts"])
          .apply(lambda x: float(np.average(
              x["J"], weights=np.sqrt(x["usdc"]))), include_groups=False)
          .rename("J_evt").reset_index())
    ts_idx = panel["ts"].to_numpy()
    minute = ev["ts"].dt.ceil("min")
    pos = np.searchsorted(ts_idx, minute.to_numpy())
    ok = pos < len(ts_idx)
    ev = ev[ok].assign(pos=pos[ok], minute=minute[ok].to_numpy())
    ev["delay"] = ((panel["ts"].to_numpy()[ev["pos"]] - ev["minute"])
                   .dt.total_seconds() / 60)
    ev["sess"] = panel["session"].to_numpy()[ev["pos"]]
    if session_day_only:
        ev = ev[(ev["delay"] <= 2) & (ev["sess"] == "day")]
    up = ev[ev["J_evt"] > 0]
    n = int(len(up))
    n_dn = int((ev["J_evt"] < 0).sum())
    vals = {}
    for h in (1, 3, 10):
        f = panel[f"fwd_{h}"].to_numpy()[up["pos"]]
        vals[h] = float(np.nanmean(f)) if np.isfinite(f).any() else np.nan
    return {
        "交易品种": prod,
        "信号定义": defn,
        "信号频率": "事件型（PM 15 分钟桶检出、映射到期货分钟网格）",
        "信号月次数": f"约 {n / N_MONTHS:.1f} 次/月（取 1，即上行跳）",
        "连续信号IC（发出后1分钟）": NA,
        "连续信号IC（发出后3分钟）": NA,
        "连续信号IC（发出后10分钟）": NA,
        "离散信号（取1）后1分钟平均收益": fmt_bp(vals[1]),
        "离散信号（取1）后3分钟平均收益": fmt_bp(vals[3]),
        "离散信号（取1）后10分钟平均收益": fmt_bp(vals[10]),
        "公式": formula,
        "取值均值": NA, "取值中位数": NA, "取值标准差": NA,
        "取值偏度": NA, "取值峰度": NA,
        "取值valuecount": f"+1（上行事件跳）: {n} 次；−1（下行）: "
                          f"{n_dn} 次",
        "取值频率": f"约 {(n + n_dn) / N_MONTHS:.1f} 次/月（双向合计）",
        "直方图": "docs/figures/f_signal_summary_hist.png 面板 (f)",
        "品种全区间平均收益": baseline_of(prod),
    }


def make_histogram(panels: dict[str, pd.Series],
                   discrete_counts: dict[str, tuple[int, int]]) -> None:
    """(a)-(e) 五品种 C8 直方图；(f) 离散信号 value count 条形图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import pub_style
    pub_style.setup(cn_font=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, (prod, v), tag in zip(axes.flat, panels.items(),
                                  "abcde"):
        ax.hist(v.clip(-6, 6), bins=60, color="#3b6db3", alpha=0.85)
        ax.set_title(f"({tag}) C8 x {prod}（均值 {v.mean():+.3f}，"
                     f"σ {v.std():.2f}）", fontsize=9)
        ax.set_yscale("log")
    ax = axes.flat[5]
    labels = list(discrete_counts)
    ups = [discrete_counts[k][0] for k in labels]
    dns = [discrete_counts[k][1] for k in labels]
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, ups, width=0.4, label="+1（上行）", color="#2e7d32")
    ax.bar(xs + 0.2, dns, width=0.4, label="-1（下行）", color="#b03a2e")
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_title("(f) 离散信号 value count（对数轴）", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "f_signal_summary_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"histogram -> {out}")


def main() -> int:
    ic = pd.read_parquet(DEF / "ic_table.parquet")
    rows = []
    hist_data: dict[str, pd.Series] = {}
    for prod, tag in zip(("SC", "AU", "AG", "CU", "M"), "abcde"):
        panel = pd.read_parquet(DEF / f"panel_{prod}.parquet",
                                columns=["C8"])
        rows.append(cont_row(prod, ic, panel, f"({tag})"))
        hist_data[prod] = panel["C8"].replace(0.0, np.nan).dropna()
    sc_panel = pd.read_parquet(DEF / "panel_SC.parquet",
                               columns=["E_up", "E_dn",
                                        "fwd_1", "fwd_3", "fwd_10"])
    rows.append(e1_row(sc_panel))
    rows.append(jump_row(
        "SC", ["mideast_conflict", "oil_price"], False, "盘中跳", True,
        "SC 盘中日盘事件跳（取 1 = 上行）：中东/油价主题 E1 口径跳、"
        "(theme,ts) 事件折叠、映射延迟 ≤2 分钟且落在日盘；置换单方法"
        "支持的探索候选（§12.15），毛收益未过 2× 成本门槛",
        JUMP_FORMULA + "；限映射延迟 ≤2 分钟且落在日盘"))
    rows.append(jump_row(
        "M", ["us_china_trade"], True, "孤立跳", False,
        "M 孤立事件跳（取 1 = 上行）：贸易主题 E1 口径跳且同桶无其他"
        "市场共跳（n_cojump=0）；低功效不显著（§12.15），候选不升级",
        JUMP_FORMULA + "；限 n_cojump = 0（孤立）"))

    def _vc_pair(r: dict) -> tuple[int, int]:
        import re as _re
        nums = _re.findall(r"([\d,]+) 次", r["取值valuecount"])
        return (int(nums[0].replace(",", "")),
                int(nums[1].replace(",", "")))

    discrete_counts = {
        "E1xSC": _vc_pair(rows[5]),
        "SC盘中跳": _vc_pair(rows[6]),
        "M孤立跳": _vc_pair(rows[7]),
    }
    make_histogram(hist_data, discrete_counts)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"written {OUT}（{len(rows)} 条，含扩展字段）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
