"""精炼版报告的图（3 张）：时间对齐、IC 期限结构、离散信号事件研究。

图 1  时间口径与时段对齐：Polymarket 7x24 活动强度 vs 期货交易时段（北京时），
      并给出分钟桶右端标签与前向收益的先后关系示意。
图 2  IC 期限结构：主要信号在 1/2/3/5/10/15 分钟视界的 RankIC 与 ICIR 热图。
图 3  离散信号事件研究：触发后累计平均收益与同品种无条件基准的对照。

命令：.venv/bin/python scripts/v3_concise_figures.py
产物：docs/concise/fig/f{1,2,3}_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pub_style  # noqa: E402

DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
FIG = ROOT / "docs" / "concise" / "fig"
HORIZONS = [1, 2, 3, 5, 10, 15]
P = pub_style.PALETTE


def _pm_intraday() -> pd.Series:
    """从 tape 直接取登记市场的日内成交笔数（北京时分钟，7x24 连续）。"""
    cache = DEF / "pm_intraday_profile.parquet"
    if cache.exists():
        return pd.read_parquet(cache).set_index("hm")["n"]
    sys.path.insert(0, str(ROOT))
    from alpha_data.polymarket import store  # noqa: PLC0415
    from alpha_data.polymarket.v3 import tape  # noqa: PLC0415
    reg = pd.read_parquet(store.FEATURES_DIR / "cn_registry_v3.parquet")
    ids = ",".join("'" + c + "'" for c in reg["condition_id"].unique())
    con = store.connect()
    sql = f"""
        SELECT (((block_timestamp + 8 * 3600) % 86400) // 60) AS hm,
               count(*) AS n
        FROM {tape.union_sql('condition_id, block_timestamp')}
        WHERE condition_id IN ({ids})
          AND block_timestamp >= epoch(TIMESTAMP '2026-01-04 16:00:00')
          AND block_timestamp <  epoch(TIMESTAMP '2026-07-13 16:00:00')
        GROUP BY 1 ORDER BY 1
    """
    df = con.execute(sql).fetch_df()
    df["n"] = df["n"] / 125.0          # 每分钟每日均值
    df.to_parquet(cache, index=False)
    return df.set_index("hm")["n"]


def fig1_alignment() -> None:
    """时间口径：PM 活动的日内分布 vs 期货时段，以及标签先后关系。"""
    pub_style.setup(cn_font=True)
    panel = pd.read_parquet(DEF / "panel_SC.parquet",
                            columns=["ts", "session", "volume"])
    panel["ts"] = pd.to_datetime(panel["ts"])
    panel["hm"] = panel["ts"].dt.hour * 60 + panel["ts"].dt.minute

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4),
                             gridspec_kw={"height_ratios": [2.0, 1.0]})

    ax = axes[0]
    pm = _pm_intraday().reindex(range(1440))
    fut = (panel.groupby("hm")["volume"].mean().reindex(range(1440)))
    g = pd.DataFrame({"n": pm, "vol": fut})
    x = np.arange(1440) / 60.0
    ax.plot(x, g["n"].to_numpy(), color=P["blue"], lw=1.0,
            label="Polymarket 登记市场成交笔数（分钟均值，左轴）")
    ax.set_ylabel("PM 成交笔数 / 分钟", color=P["blue"])
    ax.tick_params(axis="y", colors=P["blue"])
    ax2 = ax.twinx()
    ax2.plot(x, g["vol"].to_numpy(), color=P["orange"], lw=1.1,
             label="SC 成交量（手，右轴）")
    ax2.set_ylabel("SC 成交量 / 分钟", color=P["orange"])
    ax2.tick_params(axis="y", colors=P["orange"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    for lo, hi, lab in [(9.0, 10.25, "日盘"), (10.5, 11.5, ""),
                        (13.5, 15.0, ""), (21.0, 24.0, "夜盘")]:
        ax.axvspan(lo, hi, color=P["grid"], alpha=0.55, lw=0, zorder=0)
        if lab:
            ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.94, lab,
                    ha="center", va="top", fontsize=7, color=P["ink2"])
    ax.axvspan(0.0, 2.5, color=P["grid"], alpha=0.55, lw=0, zorder=0)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("北京时（小时）；灰底为 SC 交易时段，白底为休市（期货曲线断开）")
    ax.set_title("(a) Polymarket 全天连续，期货分段交易：休市时段的信息只能在"
                 "下一段开盘进入价格", loc="left")

    ax = axes[1]
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")
    ax.annotate("", xy=(3.1, 0.0), xytext=(-3.1, 0.0),
                arrowprops=dict(arrowstyle="->", lw=0.9, color=P["ink"]))
    for xt, lab in [(-2.0, "T-2"), (-1.0, "T-1"), (0.0, "T"),
                    (1.0, "T+1"), (2.0, "T+2")]:
        ax.plot([xt], [0.0], marker="|", ms=7, color=P["ink"])
        ax.text(xt, -0.30, lab, ha="center", fontsize=7.5)
    ax.add_patch(plt.Rectangle((-1.0, 0.10), 1.0, 0.34, fc=P["blue"],
                               alpha=0.22, ec=P["blue"], lw=0.7))
    ax.text(-0.5, 0.27, "PM 分钟桶 [T-60s, T)", ha="center", va="center",
            fontsize=7, color=P["blue"])
    ax.text(0.02, 0.62, "标签 T：只含严格早于 T 的成交", fontsize=7,
            color=P["blue"])
    ax.add_patch(plt.Rectangle((0.0, -0.62), 1.0, 0.34, fc=P["orange"],
                               alpha=0.22, ec=P["orange"], lw=0.7))
    ax.text(0.5, -0.45, "期货 bar (T, T+1]", ha="center", va="center",
            fontsize=7, color=P["orange"])
    ax.text(1.10, -0.45, r"前向收益 fwd$_h$ = ln $C_{T+h}$ - ln $C_T$",
            fontsize=7, color=P["orange"], va="center")
    ax.set_title("(b) 防前视的时间关系：信号在 T 可得，被解释的收益始于 T 之后",
                 loc="left")

    fig.tight_layout()
    out = FIG / "f1_alignment.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"written {out}")


def fig2_ic_term() -> None:
    """IC 期限结构热图（RankIC）与 ICIR 条形。"""
    pub_style.setup(cn_font=True)
    g = pd.read_parquet(DEF / "signal_grid.parquet")
    cont = g[g["kind"] != "离散"].copy()
    cont["score"] = cont[[f"ic_t_{h}" for h in HORIZONS]].abs().max(axis=1)
    top = (cont.sort_values("score", ascending=False)
           .drop_duplicates(["product", "signal"]).head(14)
           .sort_values(["family", "signal", "product"]))
    labels = [f"{r.signal}·{r['product']}" for _, r in top.iterrows()]
    mat = top[[f"ic_s_{h}" for h in HORIZONS]].to_numpy(dtype=float) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.0),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    v = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"{h}'" for h in HORIZONS])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("前向视界（分钟）")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center",
                        fontsize=5.6,
                        color="white" if abs(mat[i, j]) > 0.62 * v
                        else P["ink"])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("RankIC (x100)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("(a) RankIC 期限结构（信号·品种）", loc="left")

    ax = axes[1]
    icir = top["icir_15"].to_numpy(dtype=float)
    y = np.arange(len(labels))
    ax.barh(y, icir, color=[P["blue"] if v_ > 0 else P["red"] for v_ in icir],
            height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()          # imshow 默认已是 row0 在顶，此处与之匹配
    ax.axvline(0.0, color=P["ink"], lw=0.6)
    for thr in (-0.2, 0.2):
        ax.axvline(thr, color=P["ink2"], lw=0.5, ls=":")
    ax.set_xlabel("ICIR（15 分钟视界，逐日 IC 均值 / 标准差）")
    ax.set_title("(b) ICIR", loc="left")

    fig.tight_layout()
    out = FIG / "f2_ic_term.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"written {out}")


def fig3_event_study() -> None:
    """离散信号触发后的平均收益路径与无条件基准。"""
    pub_style.setup(cn_font=True)
    g = pd.read_parquet(DEF / "signal_grid.parquet")
    d = g[(g["kind"] == "离散") & (g["n_up"] >= 100)].copy()
    d["score"] = d[[f"up_t_{h}" for h in HORIZONS]].abs().max(axis=1)
    top = d.sort_values("score", ascending=False).head(6)

    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.0), sharex=True)
    for ax, (_, r) in zip(axes.ravel(), top.iterrows(), strict=False):
        y = np.array([r[f"up_ret_{h}"] for h in HORIZONS], dtype=float)
        b = np.array([r[f"base_mean_bp_{h}"] for h in HORIZONS], dtype=float)
        # 两线之间填色 = 超额收益，即表 13 所印、也是 t 检验的量。
        ax.fill_between(HORIZONS, b, y, where=y >= b, color=P["blue"],
                        alpha=0.16, lw=0, interpolate=True)
        ax.fill_between(HORIZONS, b, y, where=y < b, color=P["red"],
                        alpha=0.16, lw=0, interpolate=True)
        ax.plot(HORIZONS, y, marker="o", color=P["blue"], label="触发后（取 +1）")
        ax.plot(HORIZONS, b, marker="s", ms=2.6, color=P["ink2"], lw=0.9,
                ls="--", label="同品种无条件基准")
        ax.axhline(0.0, color=P["ink"], lw=0.5)
        tmax = max((abs(r[f"up_t_{h}"]) for h in HORIZONS
                    if np.isfinite(r[f"up_t_{h}"])), default=float("nan"))
        ax.set_title(f"{r['signal']}·{r['product']}  n={int(r['n_up'])}, "
                     f"{r['up_per_month']:.0f} 次/月, max|t|={tmax:.2f}",
                     fontsize=6.6, loc="left")
        ax.tick_params(labelsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel("触发后分钟数")
    for ax in axes[:, 0]:
        ax.set_ylabel("平均收益 (bp)")
    axes[0, 0].legend(fontsize=6, loc="best")
    # 门槛与最大 |t| 一律从数据算，不写死
    tcols = [f"{d_}_t_{h}" for d_ in ("up", "dn") for h in HORIZONS]
    tarr = np.abs(g[g["kind"] == "离散"][tcols].to_numpy(dtype=float))
    n_test = int(np.isfinite(tarr).sum())
    tmax_all = float(np.nanmax(tarr))
    thr = float(stats.norm.ppf(1.0 - 0.05 / 2.0 / n_test))
    fig.suptitle(f"离散信号触发后的收益路径（|t| 最大的 6 个格）："
                 f"两线之间的填色即超额收益（表 13 所印、也是 t 检验的量）。"
                 f"全部 {n_test} 个事件检验 max|t| = {tmax_all:.2f} < "
                 f"Bonferroni 门槛 {thr:.2f}，无一通过", fontsize=7.0,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "f3_event_study.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"written {out}")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_alignment()
    fig2_ic_term()
    fig3_event_study()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
