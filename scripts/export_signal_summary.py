"""按导师规定的结构化格式导出已发现信号的汇总 JSON（纯数字模式）。

设计原则：**全部量化字段为 JSON number，不适用为 null**，文字只出现
在本质为文字的字段（定义 / 公式 / 频率描述）；单位写进字段名后缀
（_bp）或 meta.units，供下游程序直接处理。正文表格由渲染层
（v3_paper_body）负责把数字格式化为展示文本。

输出结构：{"meta": {...单位与口径...}, "records": [8 条]}。
规定的 10 个字段名保持原样（值为 number / null）；扩展字段：
信号 / 信号类型 / 公式 / 取值均值 / 取值中位数 / 取值标准差 /
取值偏度 / 取值峰度 / 取值非零频率 / 取值valuecount（对象或 null）/
直方图文件 / 直方图面板 / 品种全区间收对收累计收益 /
品种无条件{1,3,10}分钟均值收益_bp。

产物：docs/signal_summary.json 与 docs/figures/f_signal_summary_hist.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
JD = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "jump"
DAILY = ROOT / "data" / "cn_futures" / "daily"
OUT = ROOT / "docs" / "signal_summary.json"
#: 全信号统计网格（IC / ICIR 的唯一来源，与正文表 5、图 2 同源）。
GRID = DEF / "signal_grid.parquet"
HIST = "docs/figures/f_signal_summary_hist.png"
N_MONTHS = 125 / 21

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

META = {
    "generated_by": "scripts/export_signal_summary.py",
    "sample": "2026-01-05 .. 2026-07-13（125 个交易日）",
    "units": {
        "连续信号IC（发出后N分钟）": "pooled RankIC，无量纲（描述性；"
                                     "推断以报告 §12.14-12.15 为准）",
        "离散信号（取1）后N分钟平均收益": "bp（毛值、未扣成本）",
        "信号月次数": "连续 = 有效信号分钟数/月；离散 = 取1触发次数/月"
                      "（125 交易日按 21 折月）",
        "取值非零频率": "非零（或触发）分钟占全部分钟比例，小数",
        "品种全区间收对收累计收益": "小数（0.088 = +8.8%），剔换月日",
        "品种无条件N分钟均值收益_bp": "bp，同品种全样本无条件基准",
        "取值valuecount": "对象 {取值: 次数}，仅离散信号，连续为 null",
        "连续信号ICIR（发出后N分钟）": "日度 IC 均值/日度 IC 标准差，"
                                       "无量纲，仅连续信号",
        "IC统计天数": "日度 IC 序列的天数",
        "离散信号符号RankIC（发出后N分钟）": "signed 指示变量（+1/0/-1）"
                                             "与前向收益的 pooled RankIC"
                                             "（与 X 族同口径），仅离散",
        "离散信号（取1）后N分钟超额收益_bp": "bp，= 取1条件均值 - 品种"
                                             "无条件基准；事件研究 t 检验"
                                             "的即为此量，后续衡量一律以"
                                             "超额为准（毛值字段仅为保持"
                                             "规定格式原样）",
    },
    "na_convention": "不适用的字段一律为 null",
}


def rnd(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def baseline_of(prod: str) -> dict:
    daily = pd.read_parquet(DAILY / f"{prod}.parquet",
                            columns=["r_cc", "roll"])
    r = daily.loc[~daily["roll"].astype(bool), "r_cc"].dropna()
    panel = pd.read_parquet(DEF / f"panel_{prod}.parquet",
                            columns=["fwd_1", "fwd_3", "fwd_10"])
    return {
        "品种全区间收对收累计收益": rnd(np.expm1(r.sum()), 4),
        "品种无条件1分钟均值收益_bp": rnd(panel["fwd_1"].mean() * 1e4, 4),
        "品种无条件3分钟均值收益_bp": rnd(panel["fwd_3"].mean() * 1e4, 4),
        "品种无条件10分钟均值收益_bp": rnd(panel["fwd_10"].mean() * 1e4, 4),
    }


def cont_row(prod: str, ic: pd.DataFrame, panel: pd.DataFrame,
             hist_panel: str) -> dict:
    """连续信号（C8）的登记记录。

    IC / ICIR 一律取自 ``signal_grid.parquet``，与正文 §指标口径的定义、
    表 5（IC 期限结构）与图 2 同源。历史上本函数读 ``ic_table.parquet``，
    该表的 ICIR 走另一条口径（日内最少观测数与相关类型均不同），会使同一个
    C8 的同一指标在同一份报告里印出两个值（AG 曾为 0.39 对 0.89）。
    """
    sub = ic[(ic["product"] == prod) & (ic["signal"] == "C8")
             & (ic["scope"] == "all")].set_index("horizon")
    del sub  # ic_table 仅保留入参兼容，数字一律以 signal_grid 为准
    grid = pd.read_parquet(GRID)
    gr = grid[(grid["product"] == prod) & (grid["signal"] == "C8")].iloc[0]
    v = panel["C8"].replace(0.0, np.nan).dropna()
    return {
        "交易品种": prod,
        "信号": "C8",
        "信号类型": "连续",
        "信号定义": C8_DEF,
        "公式": C8_FORMULA,
        "信号频率": "逐分钟连续值（120 分钟滚动窗，跨段按交易分钟序）",
        "信号月次数": rnd(len(v) / N_MONTHS, 1),
        "连续信号IC（发出后1分钟）": rnd(gr["ic_s_1"]),
        "连续信号IC（发出后3分钟）": rnd(gr["ic_s_3"]),
        "连续信号IC（发出后10分钟）": rnd(gr["ic_s_10"]),
        "连续信号ICIR（发出后1分钟）": rnd(gr["icir_1"], 3),
        "连续信号ICIR（发出后3分钟）": rnd(gr["icir_3"], 3),
        "连续信号ICIR（发出后10分钟）": rnd(gr["icir_10"], 3),
        "IC统计天数": int(gr["ic_days_1"]),
        "离散信号（取1）后1分钟平均收益": None,
        "离散信号（取1）后3分钟平均收益": None,
        "离散信号（取1）后10分钟平均收益": None,
        "离散信号符号RankIC（发出后1分钟）": None,
        "离散信号符号RankIC（发出后3分钟）": None,
        "离散信号符号RankIC（发出后10分钟）": None,
        "取值均值": rnd(v.mean()),
        "取值中位数": rnd(v.median()),
        "取值标准差": rnd(v.std()),
        "取值偏度": rnd(v.skew(), 3),
        "取值峰度": rnd(v.kurt(), 3),
        "取值非零频率": rnd(len(v) / len(panel)),
        "取值valuecount": None,
        "直方图文件": HIST,
        "直方图面板": hist_panel,
        **baseline_of(prod),
    }


def signed_rank_ic(signed: np.ndarray, panel: pd.DataFrame,
                   horizons: tuple[int, ...] = (1, 3, 10)) -> dict[int,
                                                                   float]:
    """signed 指示变量与前向收益的 pooled RankIC（与 X 族同口径）。"""
    from scipy.stats import spearmanr
    out = {}
    for h in horizons:
        f = panel[f"fwd_{h}"].to_numpy()
        ok = np.isfinite(f)
        out[h] = float(spearmanr(signed[ok], f[ok]).statistic)
    return out


def e1_row(panel: pd.DataFrame) -> tuple[dict, np.ndarray, pd.DataFrame]:
    m = panel["E_up"] == 1
    n_up = int(m.sum())
    n_dn = int((panel["E_dn"] == 1).sum())
    n_zero = int(len(panel) - n_up - n_dn)
    signed = (panel["E_up"] - panel["E_dn"]).to_numpy()
    sic = signed_rank_ic(signed, panel)
    row = {
        "交易品种": "SC",
        "信号": "E1_up",
        "信号类型": "离散",
        "信号定义": "E1 上行价格跳：主题 1 分钟 logit 创新超稳健阈值"
                    "（E_up = 1；下行对称、本行登记取 1）",
        "公式": E1_FORMULA,
        "信号频率": "事件型（1 分钟粒度触发）",
        "信号月次数": rnd(n_up / N_MONTHS, 1),
        "连续信号IC（发出后1分钟）": None,
        "连续信号IC（发出后3分钟）": None,
        "连续信号IC（发出后10分钟）": None,
        "离散信号（取1）后1分钟平均收益": rnd(
            panel.loc[m, "fwd_1"].mean() * 1e4, 3),
        "离散信号（取1）后3分钟平均收益": rnd(
            panel.loc[m, "fwd_3"].mean() * 1e4, 3),
        "离散信号（取1）后10分钟平均收益": rnd(
            panel.loc[m, "fwd_10"].mean() * 1e4, 3),
        "连续信号ICIR（发出后1分钟）": None,
        "连续信号ICIR（发出后3分钟）": None,
        "连续信号ICIR（发出后10分钟）": None,
        "IC统计天数": None,
        "离散信号符号RankIC（发出后1分钟）": rnd(sic[1]),
        "离散信号符号RankIC（发出后3分钟）": rnd(sic[3]),
        "离散信号符号RankIC（发出后10分钟）": rnd(sic[10]),
        "取值均值": None, "取值中位数": None, "取值标准差": None,
        "取值偏度": None, "取值峰度": None,
        "取值非零频率": rnd((n_up + n_dn) / len(panel)),
        "取值valuecount": {"-1": n_dn, "0": n_zero, "1": n_up},
        "直方图文件": HIST,
        "直方图面板": "f",
        **baseline_of("SC"),
    }
    return row, signed, panel


def jump_row(prod: str, themes: list[str], isolated: bool,
             session_day_only: bool, name: str, defn: str,
             formula: str) -> tuple[dict, np.ndarray, pd.DataFrame]:
    jumps = pd.read_parquet(JD / "jumps.parquet")
    panel = pd.read_parquet(DEF / f"panel_{prod}.parquet",
                            columns=["ts", "trade_date", "session",
                                     "fwd_1", "fwd_2", "fwd_3", "fwd_5",
                                     "fwd_10", "fwd_15"])
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
    n_up, n_dn = int(len(up)), int((ev["J_evt"] < 0).sum())
    vals: dict[int, float | None] = {}
    for h in (1, 3, 10):
        f = panel[f"fwd_{h}"].to_numpy()[up["pos"]]
        vals[h] = (float(np.nanmean(f)) * 1e4
                   if np.isfinite(f).any() else None)
    signed = np.zeros(len(panel))
    signed[ev["pos"].to_numpy()] = np.sign(ev["J_evt"].to_numpy())
    sic = signed_rank_ic(signed, panel)
    row = {
        "交易品种": prod,
        "信号": name,
        "信号类型": "离散",
        "信号定义": defn,
        "公式": formula,
        "信号频率": "事件型（PM 15 分钟桶检出、映射到期货分钟网格）",
        "信号月次数": rnd(n_up / N_MONTHS, 2),
        "连续信号IC（发出后1分钟）": None,
        "连续信号IC（发出后3分钟）": None,
        "连续信号IC（发出后10分钟）": None,
        "离散信号（取1）后1分钟平均收益": rnd(vals[1], 3),
        "离散信号（取1）后3分钟平均收益": rnd(vals[3], 3),
        "离散信号（取1）后10分钟平均收益": rnd(vals[10], 3),
        "连续信号ICIR（发出后1分钟）": None,
        "连续信号ICIR（发出后3分钟）": None,
        "连续信号ICIR（发出后10分钟）": None,
        "IC统计天数": None,
        "离散信号符号RankIC（发出后1分钟）": rnd(sic[1]),
        "离散信号符号RankIC（发出后3分钟）": rnd(sic[3]),
        "离散信号符号RankIC（发出后10分钟）": rnd(sic[10]),
        "取值均值": None, "取值中位数": None, "取值标准差": None,
        "取值偏度": None, "取值峰度": None,
        "取值非零频率": rnd((n_up + n_dn) / len(panel), 6),
        "取值valuecount": {"-1": n_dn, "1": n_up},
        "直方图文件": HIST,
        "直方图面板": "f",
        **baseline_of(prod),
    }
    return row, signed, panel


def make_histogram(panels: dict[str, pd.Series],
                   discrete_counts: dict[str, tuple[int, int]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import pub_style
    pub_style.setup(cn_font=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, (prod, v), tag in zip(axes.flat, panels.items(), "abcde"):
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


def make_ic_decay(ic: pd.DataFrame,
                  signed_series: dict[str, tuple[np.ndarray,
                                                 pd.DataFrame]]) -> None:
    """IC 衰减三联图：(a) C8 RankIC；(b) C8 日度 ICIR；(c) 离散符号 RankIC。"""
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import pub_style
    pub_style.setup(cn_font=True)
    hs = [1, 2, 3, 5, 10, 15]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    grid = pd.read_parquet(GRID)          # 与表 2、表 5 同源，防止图表分叉
    for prod in ("SC", "AU", "AG", "CU", "M"):
        gr = grid[(grid["product"] == prod) & (grid["signal"] == "C8")].iloc[0]
        axes[0].plot(hs, [float(gr[f"ic_s_{h}"]) for h in hs],
                     marker="o", ms=3, label=prod)
        axes[1].plot(hs, [float(gr[f"icir_{h}"]) for h in hs],
                     marker="o", ms=3, label=prod)
    axes[0].set_title("(a) C8 pooled RankIC 衰减", fontsize=9)
    axes[1].set_title("(b) C8 日度 ICIR 衰减", fontsize=9)
    for name, (signed, panel) in signed_series.items():
        vals = []
        for h in hs:
            f = panel[f"fwd_{h}"].to_numpy()
            okm = np.isfinite(f)
            vals.append(float(spearmanr(signed[okm], f[okm]).statistic))
        axes[2].plot(hs, vals, marker="o", ms=3, label=name)
    axes[2].set_title("(c) 离散信号符号 RankIC 衰减", fontsize=9)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("发出后分钟数")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "f_signal_summary_ic.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"ic decay -> {out}")


def main() -> int:
    ic = pd.read_parquet(DEF / "ic_table.parquet")
    rows = []
    hist_data: dict[str, pd.Series] = {}
    for prod, tag in zip(("SC", "AU", "AG", "CU", "M"), "abcde"):
        panel = pd.read_parquet(DEF / f"panel_{prod}.parquet",
                                columns=["C8"])
        rows.append(cont_row(prod, ic, panel, tag))
        hist_data[prod] = panel["C8"].replace(0.0, np.nan).dropna()
    sc_panel = pd.read_parquet(DEF / "panel_SC.parquet",
                               columns=["E_up", "E_dn", "fwd_1", "fwd_2",
                                        "fwd_3", "fwd_5", "fwd_10",
                                        "fwd_15"])
    e1, e1_signed, e1_panel = e1_row(sc_panel)
    rows.append(e1)
    j_sc, sc_signed, sc_jpanel = jump_row(
        "SC", ["mideast_conflict", "oil_price"], False, True,
        "JUMP_SC_day",
        "SC 盘中日盘事件跳（取 1 = 上行）：中东/油价主题 E1 口径跳、"
        "(theme,ts) 事件折叠、映射延迟 ≤2 分钟且落在日盘；置换单方法"
        "支持的探索候选（§12.15），毛收益未过 2× 成本门槛",
        JUMP_FORMULA + "；限映射延迟 ≤2 分钟且落在日盘")
    rows.append(j_sc)
    j_m, m_signed, m_jpanel = jump_row(
        "M", ["us_china_trade"], True, False,
        "JUMP_M_isolated",
        "M 孤立事件跳（取 1 = 上行）：贸易主题 E1 口径跳且同桶无其他"
        "市场共跳（n_cojump=0）；低功效不显著（§12.15），候选不升级",
        JUMP_FORMULA + "；限 n_cojump = 0（孤立）")
    rows.append(j_m)
    make_ic_decay(ic, {"E1xSC": (e1_signed, e1_panel),
                       "SC盘中跳": (sc_signed, sc_jpanel),
                       "M孤立跳": (m_signed, m_jpanel)})

    discrete_counts = {
        "E1xSC": (rows[5]["取值valuecount"]["1"],
                  rows[5]["取值valuecount"]["-1"]),
        "SC盘中跳": (rows[6]["取值valuecount"]["1"],
                    rows[6]["取值valuecount"]["-1"]),
        "M孤立跳": (rows[7]["取值valuecount"]["1"],
                   rows[7]["取值valuecount"]["-1"]),
    }
    make_histogram(hist_data, discrete_counts)

    # 超额收益字段（离散行）：条件均值 - 品种无条件基准。规定的毛值字段
    # 名与语义保持原样不动，另加 _bp 后缀的超额字段供下游直接按超额衡量
    # （与正文表 13 及事件研究 t 检验的量一致）。
    for r in rows:
        for h in (1, 3, 10):
            raw = r[f"离散信号（取1）后{h}分钟平均收益"]
            base = r[f"品种无条件{h}分钟均值收益_bp"]
            r[f"离散信号（取1）后{h}分钟超额收益_bp"] = (
                None if raw is None or base is None
                else rnd(raw - base, 3))

    OUT.write_text(json.dumps({"meta": META, "records": rows},
                              ensure_ascii=False, indent=2))
    print(f"written {OUT}（{len(rows)} 条，纯数字模式）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
