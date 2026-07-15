"""消融实验（OFAT）：核心吸收结果对每个设计选择的敏感性。

以基线设定为中心，一次只改动一个因子（one-factor-at-a-time），观察两个
头部结论——oil_price×SC 与 mideast_conflict×SC 的闭市 gap / 夜盘同期吸收
（Pearson r 与 HAC t）——的变化。若某结论只在特定设定下成立，则不可信。

基线：桶宽 15min、p_age 上限 120min、概率截断 [0.02, 0.98]、sqrt(usdc) 权重、
时点化准入开启、保留近结算样本。

变体（逐项替换基线的一个因子）：

- 桶宽 5 / 30 分钟（聚合粒度）
- p_age 60 / 240 / 无上限（端点时效）
- 截断 [0.01, 0.99] / [0.05, 0.95]（logit 尾部权重）
- 等权（市场聚合权重）
- 关闭时点化准入（回到 v1.0 的全窗口准入，展示准入偏差的方向）
- 剔除近结算（结算前 3 交易日）

产物：data/cn_futures/analysis/deep/ablation.parquet 与
docs/figures/deep/j_ablation.png。

用法::

    .venv/bin/python scripts/ablation_cn_polymarket.py
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

from analyze_cn_futures_polymarket import (  # noqa: E402
    NEAR_RESOLUTION_DAYS,
    REGISTRY_PATH,
    SIGNAL_END,
    load_all_trades,
    nw_regression,
)
from deep_analysis_cn_polymarket import (  # noqa: E402
    DEEP_DIR,
    FIG_DIR,
    C,
    futures_fine_returns,
    setup_matplotlib,
    style_ax,
)

from alpha_data.cn_futures import sessions  # noqa: E402
from alpha_data.cn_futures import store as fut_store  # noqa: E402
from alpha_data.polymarket import cn_features  # noqa: E402

THEMES = ("oil_price", "mideast_conflict")

#: 变体表：名称 -> 覆盖基线的参数。
VARIANTS: dict[str, dict] = {
    "基线": {},
    "桶宽 5min": {"agg_minutes": 5},
    "桶宽 30min": {"agg_minutes": 30},
    "p_age 60min": {"p_age": 60},
    "p_age 240min": {"p_age": 240},
    "p_age 无上限": {"p_age": None},
    "截断 [0.01,0.99]": {"clip": (0.01, 0.99)},
    "截断 [0.05,0.95]": {"clip": (0.05, 0.95)},
    "等权聚合": {"weight": "equal"},
    "关闭时点化准入": {"admit": False},
    "剔除近结算": {"drop_near_res": True},
}

BASELINE = {"agg_minutes": 15, "p_age": 120, "clip": (0.02, 0.98),
            "weight": "sqrt", "admit": True, "drop_near_res": False}


def theme_signal_variant(
    registry: pd.DataFrame,
    trades: dict[str, pd.DataFrame],
    windows: pd.DataFrame,
    trade_days: list[str],
    params: dict,
) -> pd.DataFrame:
    """在给定参数下重算 SC 两主题的逐日 gap / night 信号。"""
    p = {**BASELINE, **params}
    rows = []
    for theme in THEMES:
        sub = registry[(registry["theme"] == theme) & (registry["product"] == "SC")]
        sub = sub.drop_duplicates("condition_id")
        per_market = []
        for rec in sub.itertuples(index=False):
            tr = trades.get(rec.condition_id)
            if tr is None or tr.empty:
                continue
            sig = cn_features.window_signals(
                tr, windows, agg_minutes=p["agg_minutes"], clip=p["clip"],
                p_age_max_minutes=p["p_age"],
            )
            sig = sig[["trade_date", "s_night", "s_gap"]].copy()
            if p["admit"]:
                admit_date = (pd.Timestamp(rec.admit_ts, unit="s", tz="UTC")
                              .tz_convert("Asia/Shanghai").strftime("%Y-%m-%d"))
                sig.loc[sig["trade_date"] < admit_date,
                        ["s_night", "s_gap"]] = np.nan
            if p["drop_near_res"] and pd.notna(rec.resolved_at):
                res_date = (pd.Timestamp(rec.resolved_at)
                            .tz_convert("Asia/Shanghai").strftime("%Y-%m-%d"))
                idx = np.searchsorted(np.array(trade_days), res_date)
                cutoff = trade_days[max(idx - NEAR_RESOLUTION_DAYS, 0)]
                sig.loc[sig["trade_date"] >= cutoff,
                        ["s_night", "s_gap"]] = np.nan
            w = np.sqrt(rec.usdc_win) if p["weight"] == "sqrt" else 1.0
            sig["w"] = w * rec.orientation
            per_market.append(sig)
        allm = pd.concat(per_market, ignore_index=True)

        def wavg(g: pd.DataFrame, col: str) -> float:
            v = g[col].to_numpy(dtype="float64")
            w = g["w"].to_numpy(dtype="float64")
            ok = ~np.isnan(v)
            if not ok.any():
                return np.nan
            return float(np.sum(v[ok] * w[ok]) / np.sum(np.abs(w[ok])))

        for day, g in allm.groupby("trade_date"):
            rows.append({"theme": theme, "trade_date": day,
                         "s_night": wavg(g, "s_night"), "s_gap": wavg(g, "s_gap")})
    return pd.DataFrame(rows)


def main() -> int:
    setup_matplotlib()
    registry = pd.read_parquet(REGISTRY_PATH)
    reg_sc = registry[(registry["theme"].isin(THEMES))
                      & (registry["product"] == "SC")]
    trades = load_all_trades(sorted(reg_sc["condition_id"].unique()))

    trade_days = fut_store.read_daily("SC")["trade_date"].tolist()
    specs = fut_store.read_product_specs().set_index("product")
    ne = sessions.parse_night_end(specs.loc["SC", "night_end"])
    windows = sessions.signal_windows(trade_days, night_end=ne)

    rets = futures_fine_returns(["SC"])
    rets = rets[~rets["roll"].fillna(False)].copy()
    rr = (rets["r_gap_pm"].fillna(0) + rets["r_night"].fillna(0)
          + rets["r_gap_am"].fillna(0))
    rr[rets[["r_gap_pm", "r_night", "r_gap_am"]].isna().all(axis=1)] = np.nan
    rets["r_gap_total"] = rr

    from scipy import stats as sps

    rows = []
    for name, params in VARIANTS.items():
        sig = theme_signal_variant(reg_sc, trades, windows, trade_days, params)
        for theme in THEMES:
            g = (sig[sig["theme"] == theme]
                 .merge(rets[["trade_date", "r_night", "r_gap_total"]],
                        on="trade_date"))
            g = g[g["trade_date"] <= SIGNAL_END]
            for win_name, sig_col, ret_col in [("gap", "s_gap", "r_gap_total"),
                                               ("night", "s_night", "r_night")]:
                s = g[sig_col].to_numpy(dtype="float64")
                r = g[ret_col].to_numpy(dtype="float64")
                ok = ~(np.isnan(s) | np.isnan(r))
                n = int(ok.sum())
                if n < 15:
                    rows.append({"variant": name, "theme": theme,
                                 "window": win_name, "n": n,
                                 "pearson": np.nan, "t_hac": np.nan})
                    continue
                pear = sps.pearsonr(s[ok], r[ok])
                _, t_hac, _, _ = nw_regression(r, s, 1)
                rows.append({"variant": name, "theme": theme, "window": win_name,
                             "n": n, "pearson": float(pear.statistic),
                             "t_hac": t_hac})
        print(f"  {name} 完成")
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "ablation.parquet", index=False)
    print("\n=== 消融表（Pearson r）===")
    print(tab.pivot_table(index="variant", columns=["theme", "window"],
                          values="pearson").round(3).to_string())

    # 图：r 的点图（按变体）
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    order = list(VARIANTS.keys())
    for ax, theme in zip(axes, THEMES, strict=True):
        for win_name, color in [("gap", C["orange"]), ("night", C["blue"])]:
            g = tab[(tab["theme"] == theme) & (tab["window"] == win_name)]
            g = g.set_index("variant").reindex(order)
            y = np.arange(len(order))
            ax.plot(g["pearson"], y, "o", ms=6, color=color,
                    label=f"{win_name} 窗口")
            base = g.loc["基线", "pearson"]
            ax.axvline(base, color=color, lw=0.8, ls=":", alpha=0.6)
        ax.set_yticks(np.arange(len(order)), order, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("同期吸收 Pearson r")
        ax.set_title(f"{theme} × SC", fontsize=10)
        ax.axvline(0, color=C["ink2"], lw=0.7)
        style_ax(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("消融实验：每次只改一个设计选择（虚线 = 基线值）", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "j_ablation.png", dpi=150)
    plt.close(fig)
    print(f"\n产物：{DEEP_DIR}/ablation.parquet 与 {FIG_DIR}/j_ablation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
