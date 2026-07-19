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

产物：docs/signal_summary.json
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


def fmt_ic(x: float) -> str:
    return f"{x:+.4f}"


def fmt_bp(x: float) -> str:
    return f"{x * 1e4:+.2f}bp"


def cont_row(prod: str, ic: pd.DataFrame, panel: pd.DataFrame) -> dict:
    sub = ic[(ic["product"] == prod) & (ic["signal"] == "C8")
             & (ic["scope"] == "all")].set_index("horizon")
    n_valid = int(panel["C8"].replace(0.0, np.nan).notna().sum())
    return {
        "交易品种": prod,
        "信号定义": C8_DEF,
        "信号频率": "逐分钟连续值（120 分钟滚动窗，跨段按交易分钟序）",
        "信号月次数": f"约 {n_valid / N_MONTHS:,.0f} 个有效信号分钟/月",
        "连续信号IC（发出后1分钟）": fmt_ic(sub.loc[1, "rank_ic"]),
        "连续信号IC（发出后3分钟）": fmt_ic(sub.loc[3, "rank_ic"]),
        "连续信号IC（发出后10分钟）": fmt_ic(sub.loc[10, "rank_ic"]),
        "离散信号（取1）后1分钟平均收益": NA,
        "离散信号（取1）后3分钟平均收益": NA,
        "离散信号（取1）后10分钟平均收益": NA,
    }


def e1_row(panel: pd.DataFrame) -> dict:
    m = panel["E_up"] == 1
    n = int(m.sum())
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
    }


def jump_row(prod: str, themes: list[str], isolated: bool, label: str,
             session_day_only: bool, defn: str) -> dict:
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
    }


def main() -> int:
    ic = pd.read_parquet(DEF / "ic_table.parquet")
    rows = []
    for prod in ("SC", "AU", "AG", "CU", "M"):
        panel = pd.read_parquet(DEF / f"panel_{prod}.parquet",
                                columns=["C8"])
        rows.append(cont_row(prod, ic, panel))
    sc_panel = pd.read_parquet(DEF / "panel_SC.parquet",
                               columns=["E_up", "fwd_1", "fwd_3", "fwd_10"])
    rows.append(e1_row(sc_panel))
    rows.append(jump_row(
        "SC", ["mideast_conflict", "oil_price"], False, "盘中跳", True,
        "SC 盘中日盘事件跳（取 1 = 上行）：中东/油价主题 E1 口径跳、"
        "(theme,ts) 事件折叠、映射延迟 ≤2 分钟且落在日盘；置换单方法"
        "支持的探索候选（§12.15），毛收益未过 2× 成本门槛"))
    rows.append(jump_row(
        "M", ["us_china_trade"], True, "孤立跳", False,
        "M 孤立事件跳（取 1 = 上行）：贸易主题 E1 口径跳且同桶无其他"
        "市场共跳（n_cojump=0）；低功效不显著（§12.15），候选不升级"))
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\nwritten {OUT}（{len(rows)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
