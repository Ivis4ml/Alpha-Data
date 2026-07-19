"""J 族第二阶段：时段剖面、事件级折叠与跨族复合（含配图）。

回应三点评审意见：

1. **同质化归并**：同主题多市场同桶共跳是同一事件的期限结构（如
   "美击伊朗"一族不同截止日市场），原 J1/J2 按市场计数存在重复。
   本脚本把 (theme, bucket) 折叠为**事件级跳**（成交额加权幅度、
   市场广度为属性），重建事件级因子 JE1/JE2 与市场级对照。
2. **时段剖面**：每个跳按品种交易时段分类（盘中日盘 / 盘中夜盘 /
   闭市傍晚 / 闭市凌晨 / 闭市周末），分别测响应路径——盘中跳看
   随后分钟的残余，闭市跳看下一开盘后的残余（开盘跳空吸收假说的
   跳级检验）。
3. **跨族复合**：因子相关结构矩阵（N/C/J 同质性证据）+ 两个不做
   样本内拟合的复合因子——JAGG（事件级带方向跳幅和，纯归并修正）
   与 XF（C8/N4/J2 三族头部信号的等权 z 均值，成分为既有各族头部、
   等权、无拟合，声明为研究内选择）。

产出 ``data/cn_futures/analysis/v3/jump/``：session_response.parquet、
event_factor_ic.parquet、composite_ic.parquet、factor_corr.parquet、
meta_integration.json；图 ``docs/figures/v3/f_jump_*.png``。

用法::

    .venv/bin/python scripts/v3_jump_integration.py
"""

from __future__ import annotations

import json
import sys
from datetime import time as dtime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pub_style  # noqa: E402

from alpha_data.cn_futures import store as fut_store  # noqa: E402

V3_DIR = ROOT / "data" / "cn_futures" / "analysis" / "v3"
DEF_DIR = V3_DIR / "defense"
JUMP_DIR = V3_DIR / "jump"
FIG_DIR = ROOT / "docs" / "figures" / "v3"

PRODUCT_THEMES: dict[str, list[str]] = {
    "SC": ["mideast_conflict", "oil_price"],
    "AU": ["mideast_conflict", "metal_price", "fed_policy"],
    "AG": ["metal_price", "fed_policy"],
    "CU": ["fed_policy", "us_china_trade"],
    "M": ["us_china_trade"],
}
HORIZONS = (1, 2, 3, 5, 10, 15)
ZWIN, ZMIN = 4800, 960
THEME_CN = {
    "mideast_conflict": "中东冲突", "oil_price": "油价阈值",
    "metal_price": "金属价格", "fed_policy": "美联储",
    "us_china_trade": "中美贸易",
}
SESSION_CN = ["盘中·日盘", "盘中·夜盘", "闭市·傍晚", "闭市·凌晨",
              "闭市·周末"]


def zscore(s: pd.Series, win: int = ZWIN, minp: int = ZMIN) -> pd.Series:
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - mu) / sd.replace(0.0, np.nan)


# ------------------------------------------------------ 1. 时段分类
def classify_session(ts: pd.Timestamp, night_end: dtime) -> str:
    """跳时刻（北京）相对某品种交易时段的分类。"""
    t = ts.time()
    wd = ts.weekday()  # 0=周一
    ne = night_end
    in_early_night = t < ne  # 跨零点夜盘的凌晨部分（如 SC 至 02:30）
    if wd == 5:  # 周六
        return "盘中·夜盘" if in_early_night else "闭市·周末"
    if wd == 6:  # 周日
        return "闭市·周末"
    # 周一至周五
    if dtime(9) <= t < dtime(15):
        return "盘中·日盘"
    if t >= dtime(21):
        return "盘中·夜盘"
    if in_early_night and wd != 0:  # 周一凌晨无夜盘（周日夜不开）
        return "盘中·夜盘"
    if dtime(15) <= t < dtime(21):
        return "闭市·傍晚"
    if wd == 0 and t < dtime(9):
        return "闭市·周末"
    return "闭市·凌晨"


def session_response(jumps: pd.DataFrame) -> pd.DataFrame:
    """按时段分类的跳后品种响应（盘中看即时残余，闭市看开盘后残余）。"""
    specs = fut_store.read_product_specs().set_index("product")
    rng = np.random.default_rng(20260718)
    rows: list[dict] = []
    for product, themes in PRODUCT_THEMES.items():
        panel = pd.read_parquet(DEF_DIR / f"panel_{product}.parquet")
        ts_idx = panel["ts"].to_numpy()
        ne_str = specs.loc[product, "night_end"]
        ne = (dtime(*map(int, str(ne_str).split(":")))
              if isinstance(ne_str, str) else dtime(23, 59))
        jp = jumps[jumps["theme"].isin(themes)].copy()
        jp["sess"] = jp["ts"].apply(
            lambda x, ne=ne: classify_session(x, ne))
        jp["minute"] = jp["ts"].dt.ceil("min")
        for sess, sub in jp.groupby("sess"):
            pos = np.searchsorted(ts_idx, sub["minute"].to_numpy())
            ok = pos < len(ts_idx)
            pos = pos[ok]
            sgn = np.sign(sub["J"].to_numpy()[ok])
            # 盘中跳：下一分钟就在盘内；闭市跳：searchsorted 自动落到
            # 下一开盘分钟。响应 = 该分钟起的前向收益（符号化）。
            for h in (5, 15):
                fwd = panel[f"fwd_{h}"].to_numpy()[pos]
                val = np.isfinite(fwd)
                if val.sum() < 15:
                    continue
                signed = fwd[val] * sgn[val]
                boot = np.array([
                    rng.choice(signed, len(signed), replace=True).mean()
                    for _ in range(500)]) * 1e4
                rows.append({
                    "product": product, "session": sess, "h": h,
                    "n": int(val.sum()),
                    "signed_bp": float(signed.mean()) * 1e4,
                    "lo": float(np.quantile(boot, 0.05)),
                    "hi": float(np.quantile(boot, 0.95)),
                    "share_pos": float((signed > 0).mean()),
                    "base_abs_bp": float(
                        panel[f"fwd_{h}"].dropna().abs().mean()) * 1e4,
                })
    return pd.DataFrame(rows)


# --------------------------------------- 2. 事件级折叠与因子对照
def collapse_events(jumps: pd.DataFrame) -> pd.DataFrame:
    """(theme, bucket) 折叠：同一事件的期限结构共跳记为一个事件跳。"""
    g = jumps.groupby(["theme", "ts"])
    ev = pd.DataFrame({
        "J_evt": g.apply(lambda x: float(
            np.average(x["J"], weights=np.sqrt(x["usdc"]))),
            include_groups=False),
        "usdc": g["usdc"].sum(),
        "n_mkts": g["condition_id"].nunique(),
    }).reset_index()
    return ev


def event_factor_ic(jumps: pd.DataFrame) -> pd.DataFrame:
    """事件级 JE1/JE2 与市场级 J1/J2 的 IC 对照（同协议）。"""
    from scipy import stats as sps

    ev = collapse_events(jumps)
    rows: list[dict] = []
    for product, themes in PRODUCT_THEMES.items():
        panel = pd.read_parquet(DEF_DIR / f"panel_{product}.parquet")
        jf = pd.read_parquet(JUMP_DIR / f"factors_{product}.parquet")
        sub = ev[ev["theme"].isin(themes)].copy()
        sub["minute"] = sub["ts"].dt.ceil("min")
        t0 = panel["ts"].min() - pd.Timedelta(days=3)
        grid = pd.date_range(t0, panel["ts"].max(), freq="min")

        def imp(v: pd.Series, idx: pd.Series,
                grid: pd.DatetimeIndex = grid) -> pd.Series:
            s = pd.Series(v.to_numpy(), index=idx.to_numpy())
            return s.groupby(level=0).sum().reindex(grid).fillna(0.0)

        je1 = imp(pd.Series(1.0, index=sub.index), sub["minute"]) \
            .rolling(120, min_periods=1).sum()
        je2 = imp(sub["J_evt"], sub["minute"]) \
            .rolling(120, min_periods=1).sum()
        samp = pd.DataFrame({"JE1": je1, "JE2": je2}) \
            .reindex(panel["ts"].to_numpy())
        samp.index = panel.index
        df = panel[["trade_date"] + [f"fwd_{h}" for h in HORIZONS]].copy()
        df["JE1"] = zscore(samp["JE1"])
        df["JE2"] = zscore(samp["JE2"])
        df["J1"] = jf["J1"].to_numpy()
        df["J2"] = jf["J2"].to_numpy()
        for sig in ("JE1", "JE2", "J1", "J2"):
            for h in HORIZONS:
                x, y = df[sig], df[f"fwd_{h}"]
                ok = x.notna() & y.notna()
                if ok.sum() < 1000 or x[ok].std() == 0:
                    continue
                sp = sps.spearmanr(x[ok], y[ok])
                rows.append({
                    "product": product, "signal": sig, "h": h,
                    "n": int(ok.sum()),
                    "rank_ic": float(sp.statistic),
                    "p_rank": float(sp.pvalue),
                })
    return pd.DataFrame(rows)


# --------------------------------------------- 3. 相关结构与跨族复合
CORR_SET = ["N1", "N4", "C2", "C7", "C8", "J1", "J2", "J5", "J6"]


def factor_corr(products: tuple[str, ...] = ("SC", "M")) -> pd.DataFrame:
    rows = []
    for product in products:
        panel = pd.read_parquet(DEF_DIR / f"panel_{product}.parquet")
        jf = pd.read_parquet(JUMP_DIR / f"factors_{product}.parquet")
        df = pd.concat([panel[[c for c in CORR_SET if c in panel.columns]],
                        jf[[c for c in CORR_SET if c in jf.columns]]], axis=1)
        c = df.corr(min_periods=2000)
        c["product"] = product
        c["row"] = c.index
        rows.append(c.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def composite_ic(jumps: pd.DataFrame) -> pd.DataFrame:
    """跨族复合 XF = mean(z(C8), z(N4), z(J2)) 与成分的 IC 对照。"""
    from scipy import stats as sps

    rows: list[dict] = []
    for product in PRODUCT_THEMES:
        panel = pd.read_parquet(DEF_DIR / f"panel_{product}.parquet")
        jf = pd.read_parquet(JUMP_DIR / f"factors_{product}.parquet")
        comp_in = pd.DataFrame({
            "C8": zscore(panel["C8"]) if "C8" in panel.columns else np.nan,
            "N4": zscore(panel["N4"]) if "N4" in panel.columns else np.nan,
            "J2": jf["J2"],
        })
        xf = comp_in.mean(axis=1, skipna=False)  # 成分齐全才计（声明）
        df = panel[["trade_date"] + [f"fwd_{h}" for h in HORIZONS]].copy()
        df["XF"] = xf
        for c in comp_in.columns:
            df[c] = comp_in[c]
        for sig in ("XF", "C8", "N4", "J2"):
            for h in HORIZONS:
                x, y = df[sig], df[f"fwd_{h}"]
                ok = x.notna() & y.notna()
                if ok.sum() < 1000 or x[ok].std() == 0:
                    continue
                sp = sps.spearmanr(x[ok], y[ok])
                rows.append({
                    "product": product, "signal": sig, "h": h,
                    "n": int(ok.sum()),
                    "rank_ic": float(sp.statistic),
                    "p_rank": float(sp.pvalue),
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 4. 配图
def figures(jumps: pd.DataFrame, sess: pd.DataFrame,
            corr: pd.DataFrame, comp: pd.DataFrame,
            evic: pd.DataFrame) -> None:
    pub_style.setup()
    P = pub_style.PALETTE
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    theme_colors = {"mideast_conflict": P["red"], "oil_price": P["orange"],
                    "metal_price": P["aqua"], "fed_policy": P["blue"],
                    "us_china_trade": P["green"]}

    # 图 1：全景（月度堆叠、|J| 分布、时刻钟面、时段占比）
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    ax = axes[0, 0]
    jm = jumps.assign(month=jumps["ts"].dt.strftime("%y-%m"))
    piv = jm.pivot_table(index="month", columns="theme",
                        values="J", aggfunc="size").fillna(0)
    bottom = np.zeros(len(piv))
    for theme in piv.columns:
        ax.bar(piv.index, piv[theme], bottom=bottom,
               color=theme_colors.get(theme, P["ink2"]),
               label=THEME_CN.get(theme, theme), width=0.8)
        bottom += piv[theme].to_numpy()
    ax.legend(fontsize=5, ncol=2)
    ax.tick_params(axis="x", rotation=60, labelsize=5)
    pub_style.panel(ax, "a")
    ax.set_title("逐月跳数（按主题堆叠）", fontsize=8)

    ax = axes[0, 1]
    ax.hist(jumps["J"].abs(), bins=60, color=P["blue"], log=True)
    ax.set_xlabel("|J|（logit 跳幅）", fontsize=7)
    pub_style.panel(ax, "b")
    ax.set_title("跳幅分布（对数频数）", fontsize=8)

    ax = axes[1, 0]
    hours = jumps["ts"].dt.hour + jumps["ts"].dt.minute / 60
    ax.hist(hours, bins=48, color=P["ink2"])
    ax.axvspan(9, 15, color=P["green"], alpha=0.15, label="SC 日盘")
    ax.axvspan(21, 24, color=P["aqua"], alpha=0.15, label="SC 夜盘")
    ax.axvspan(0, 2.5, color=P["aqua"], alpha=0.15)
    ax.set_xlabel("北京时刻", fontsize=7)
    ax.legend(fontsize=5)
    pub_style.panel(ax, "c")
    ax.set_title("跳的日内时刻分布（对 SC 时段着色）", fontsize=8)

    ax = axes[1, 1]
    sc_sess = sess[(sess["product"] == "SC") & (sess["h"] == 15)]
    order = [s for s in SESSION_CN if s in set(sc_sess["session"])]
    cnt = sc_sess.set_index("session").reindex(order)["n"]
    ax.barh(order, cnt, color=P["orange"])
    pub_style.panel(ax, "d")
    ax.set_title("SC 映射跳的时段分布（n）", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_jump_panorama.png", dpi=150)
    plt.close(fig)

    # 图 2：时段响应（SC 与 M，15 分钟，自助 CI）
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, product in zip(axes, ("SC", "M"), strict=True):
        d = sess[(sess["product"] == product) & (sess["h"] == 15)]
        d = d.set_index("session").reindex(
            [s for s in SESSION_CN if s in set(d["session"])])
        y = np.arange(len(d))
        ax.barh(y, d["signed_bp"], color=[
            P["green"] if v > 0 else P["red"] for v in d["signed_bp"]])
        ax.errorbar(d["signed_bp"], y,
                    xerr=[d["signed_bp"] - d["lo"], d["hi"] - d["signed_bp"]],
                    fmt="none", ecolor=P["ink"], lw=0.8, capsize=2)
        ax.set_yticks(y, d.index, fontsize=7)
        ax.axvline(0, color=P["ink2"], lw=0.6)
        ax.set_title(f"{product}：跳后 15 分钟符号化响应（bp）", fontsize=8)
        for yi, (n, v) in enumerate(zip(d["n"], d["signed_bp"],
                                        strict=True)):
            ax.text(v, yi, f" n={n}", fontsize=5.5, va="center")
    pub_style.panel(axes[0], "a")
    pub_style.panel(axes[1], "b")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_jump_session.png", dpi=150)
    plt.close(fig)

    # 图 3：因子相关结构（同质性证据）
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, product in zip(axes, ("SC", "M"), strict=True):
        c = corr[corr["product"] == product].set_index("row")[CORR_SET]
        im = ax.imshow(c.to_numpy(dtype=float), vmin=-1, vmax=1,
                       cmap="RdBu_r")
        ax.set_xticks(range(len(CORR_SET)), CORR_SET, fontsize=6,
                      rotation=45)
        ax.set_yticks(range(len(CORR_SET)), c.index, fontsize=6)
        for i in range(len(c)):
            for j in range(len(CORR_SET)):
                v = c.iloc[i, j]
                if pd.notna(v) and abs(v) >= 0.25 and i != j:
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=5)
        ax.set_title(f"{product} 面板因子相关", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.savefig(FIG_DIR / "f_jump_corr.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 图 4：复合与事件级折叠的 IC 视界曲线（M）
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    for sig, color in (("XF", P["red"]), ("C8", P["blue"]),
                       ("N4", P["aqua"]), ("J2", P["orange"])):
        d = comp[(comp["product"] == "M") & (comp["signal"] == sig)] \
            .sort_values("h")
        ax.plot(d["h"], d["rank_ic"], marker="o", ms=3, lw=1.1,
                color=color, label=sig)
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_xlabel("视界（分钟）", fontsize=7)
    ax.set_ylabel("RankIC", fontsize=7)
    ax.legend(fontsize=6)
    pub_style.panel(ax, "a")
    ax.set_title("M：跨族复合 XF 与成分", fontsize=8)
    ax = axes[1]
    for sig, color, lab in (("J2", P["ink2"], "J2（市场级）"),
                            ("JE2", P["green"], "JE2（事件级折叠）")):
        d = evic[(evic["product"] == "M") & (evic["signal"] == sig)] \
            .sort_values("h")
        ax.plot(d["h"], d["rank_ic"], marker="o", ms=3, lw=1.1,
                color=color, label=lab)
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_xlabel("视界（分钟）", fontsize=7)
    ax.legend(fontsize=6)
    pub_style.panel(ax, "b")
    ax.set_title("M：同质化归并前后", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_jump_composite.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> int:
    jumps = pd.read_parquet(JUMP_DIR / "jumps.parquet")
    ev = collapse_events(jumps)
    print(f"事件级折叠：{len(jumps)} 跳 -> {len(ev)} 事件"
          f"（平均 {len(jumps) / len(ev):.2f} 市场/事件）")

    sess = session_response(jumps)
    sess.to_parquet(JUMP_DIR / "session_response.parquet", index=False)
    print(f"[1/4] 时段响应 {len(sess)} 行")

    evic = event_factor_ic(jumps)
    evic.to_parquet(JUMP_DIR / "event_factor_ic.parquet", index=False)
    print(f"[2/4] 事件级因子 IC {len(evic)} 行")

    corr = factor_corr()
    corr.to_parquet(JUMP_DIR / "factor_corr.parquet", index=False)
    comp = composite_ic(jumps)
    comp.to_parquet(JUMP_DIR / "composite_ic.parquet", index=False)
    print(f"[3/4] 相关结构 + 复合 IC {len(comp)} 行")

    figures(jumps, sess, corr, comp, evic)
    print("[4/4] 图 4 张已写入 docs/figures/v3/")

    meta = {
        "n_events": int(len(ev)),
        "collapse_ratio": round(len(jumps) / len(ev), 3),
        "multi_market_share": float((ev["n_mkts"] >= 2).mean()),
        "new_cells": int(len(evic) + len(comp)),
    }
    (JUMP_DIR / "meta_integration.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
