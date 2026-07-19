"""P2 横截面检验：主题信号 × 88 品种的逐日截面排序。

动机（与单品种时序检验的差别）：
1. 功效。单品种时序每天贡献 1 个观测；截面每天贡献全宇宙（约 60-75 个
   品种）的相对排序，IC 序列仍是逐日的，但每个 IC 由几十个截面观测支撑。
2. 识别。逐日截面排序天然差分掉当日的全市场共同冲击（宏观共因、人民币、
   风险偏好），恰好回应第一部分安慰剂检验暴露的共因污染：截面检验问的是
   "暴露度高的品种是否比暴露度低的品种动得更多"，而非"大家是否都在动"。

事前登记的检验族（共 6 格，不再扩展）：
    D1 吸收验证：截面得分 vs 同窗开盘前累积收益 r_open
                 （r_gap_pm + r_night + r_gap_am，前收盘到今日日盘开）
    D2 可交易：  截面得分（09:00 开盘前已知）vs 当日日盘收益 r_day
    载荷 E1：    经济先验权重（注册锚点 + 板块外推，无估计、全样本可用）
    载荷 E2：    展开窗单变量 beta（min 40 日、严格用 t 之前数据，PIT）
    组合：       D2 × {E1, E2} 的五分位多空（Q5-Q1，等权、日度再平衡）

信号：主题层 s_pre（前收盘 15:00 至今日 09:00 的闭市累积信念创新，与品种
无关，已核实同主题跨品种逐日相同），过去展开窗 z 归一（min 20 日，仅用
t-1 及以前的均值方差），无市场活跃日按"无新信息"计 0。

产物：data/cn_futures/analysis/v3/xsec/
    theme_signals.parquet   主题参考序列与 PIT z
    panel.parquet           品种 × 日 面板（收益、流动性、得分）
    ic_daily.parquet        逐日截面 Spearman IC（4 格）
    ls_daily.parquet        五分位多空日收益（2 格）
    summary.parquet         汇总（均值 / HAC t / ICIR / n）
    meta.json               口径与登记说明
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "cn_futures" / "daily"
V3 = ROOT / "data" / "cn_futures" / "analysis" / "v3"
OUT = V3 / "xsec"

#: 主题参考品种（信号与品种无关，任取覆盖最全的一行；已核实跨品种一致）
THEME_REF = {
    "mideast_conflict": "SC",
    "oil_price": "SC",
    "metal_price": "AG",
    "fed_policy": "AU",
    "russia_ukraine": "SC",
    "taiwan_risk": "AU",
    "us_china_trade": "M",
    "us_shutdown": "AU",
}

#: 板块定义（88 个品种代码）
SECTORS = {
    "energy": ["SC", "FU", "LU", "BU", "PG", "BZ"],
    "petchem": ["PX", "TA", "EG", "EB", "PP", "L", "V", "PF", "MA"],
    "precious": ["AU", "AG"],
    "base": ["CU", "AL", "ZN", "PB", "NI", "SN", "SS", "AO", "BC", "AD"],
    "ferrous": ["RB", "HC", "I", "J", "JM", "SF", "SM", "WR"],
    "soy": ["M", "RM", "Y", "P", "A", "B"],
    "index": ["IF", "IH", "IC", "IM"],
    "bond": ["T", "TF", "TS", "TL"],
    "shipping": ["EC"],
}

#: E1 经济先验：主题轴 -> {板块或品种: 权重}。
#: 锚点取注册表 sigma×orientation（fed+ -> AG/AU/CU 正、taiwan+ -> IF 负、
#: trade+ -> M/CF 正、CU/I 负），板块外推为登记的探索性延伸。
E1_MAP: dict[str, dict[str, float]] = {
    "mideast_conflict": {"energy": 1.0, "precious": 1.0, "shipping": 1.0,
                         "petchem": 0.5, "bond": 0.5, "index": -0.5},
    "oil_price": {"energy": 1.0, "petchem": 0.5},
    "metal_price": {"precious": 1.0},
    "fed_policy": {"precious": 1.0, "base": 0.5},
    "russia_ukraine": {"energy": 1.0, "precious": 0.5, "NI": 0.5, "AL": 0.5},
    "taiwan_risk": {"index": -1.0, "precious": 0.5, "bond": 0.5},
    "us_china_trade": {"M": 1.0, "CF": 1.0, "CU": -1.0, "I": -1.0,
                       "RM": 0.5, "Y": 0.5, "P": 0.5, "A": 0.5},
    "us_shutdown": {"precious": 0.5},
}

MONEY_FLOOR = 10e8    # 过去 20 日成交额中位数下限（元）
MIN_UNIVERSE = 30     # 单日最小截面规模
Z_MIN, BETA_MIN = 20, 40
HAC_LAGS = 5


def load_returns() -> pd.DataFrame:
    rows = []
    for fp in sorted(DAILY.glob("*.parquet")):
        d = pd.read_parquet(fp, columns=[
            "trade_date", "roll", "day_money", "day_open", "day_close",
            "r_day"])
        d["product"] = fp.stem
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True)
    # 开盘前累积收益 = ln(今日日盘开 / 前日日盘收)。与 gap_pm+night+gap_am
    # 的分量和望远镜式恒等，且对无夜盘品种（股指 / 国债 / 部分农产品，
    # 分量在日度表中无定义）同样有定义。换月日为跨合约伪影，置缺。
    panel = panel.sort_values(["product", "trade_date"])
    prev_close = panel.groupby("product")["day_close"].shift(1)
    panel["r_open"] = np.log(panel["day_open"] / prev_close)
    panel.loc[panel["roll"].astype(bool), "r_open"] = np.nan
    # 流动性门槛只用过去信息：过去 20 日（不含当日）成交额中位数
    med = (panel.groupby("product")["day_money"]
           .transform(lambda s: s.shift(1).rolling(20, min_periods=10)
                      .median()))
    panel["liquid"] = med >= MONEY_FLOOR
    return panel


def load_theme_z() -> pd.DataFrame:
    sig = pd.read_parquet(V3 / "v3_theme_signals.parquet")
    dates = sorted(sig["trade_date"].unique())
    out = pd.DataFrame(index=pd.Index(dates, name="trade_date"))
    for theme, ref in THEME_REF.items():
        sub = sig[(sig.theme == theme) & (sig["product"] == ref)]
        s = sub.set_index("trade_date")["s_pre"].reindex(dates)
        mu = s.expanding(Z_MIN).mean().shift(1)
        sd = s.expanding(Z_MIN).std().shift(1)
        out[theme] = ((s - mu) / sd).where(sd > 0)
    return out.fillna(0.0)


def e1_weights(products: list[str]) -> pd.DataFrame:
    sec_of = {p: s for s, ps in SECTORS.items() for p in ps}
    w = pd.DataFrame(0.0, index=products, columns=list(E1_MAP))
    for theme, mp in E1_MAP.items():
        for key, val in mp.items():
            if key in w.index:                    # 单品种键
                w.loc[key, theme] = val
            else:                                 # 板块键
                for p in products:
                    if sec_of.get(p) == key:
                        w.loc[p, theme] = val
    return w


def e2_scores(panel: pd.DataFrame, z: pd.DataFrame) -> pd.Series:
    """展开窗单变量 beta（r_open ~ z_theta，严格 < t），得分 = sum(b*z)。"""
    dates = z.index.tolist()
    date_pos = {d: i for i, d in enumerate(dates)}
    zv = z.to_numpy()                             # [T, K]
    themes = z.columns.tolist()
    scores = {}
    for prod, g in panel.groupby("product"):
        g = g.set_index("trade_date")["r_open"].reindex(dates)
        y = g.to_numpy()
        valid = ~np.isnan(y)
        sc = np.full(len(dates), np.nan)
        # 逐主题展开协方差（在线累积，只用 < t）
        n = np.zeros(len(themes)); sx = np.zeros(len(themes))
        sy = np.zeros(len(themes)); sxx = np.zeros(len(themes))
        sxy = np.zeros(len(themes))
        for t in range(len(dates)):
            b = np.zeros(len(themes)); ok = n >= BETA_MIN
            if ok.any():
                varx = sxx[ok] / n[ok] - (sx[ok] / n[ok]) ** 2
                cov = sxy[ok] / n[ok] - sx[ok] * sy[ok] / n[ok] ** 2
                bb = np.where(varx > 1e-12, cov / np.maximum(varx, 1e-12), 0.0)
                b[ok] = bb
                sc[t] = float(b @ zv[t])
            if valid[t]:
                x = zv[t]
                n += 1; sx += x; sy += y[t]; sxx += x * x; sxy += x * y[t]
        scores[prod] = pd.Series(sc, index=dates)
    return (pd.DataFrame(scores).stack()
            .rename("score_e2").rename_axis(["trade_date", "product"]))


def make_figure(summary: pd.DataFrame, monthly: pd.DataFrame) -> None:
    """两联图：(a) 6 格汇总（IC 均值与 HAC 95% CI）；(b) 逐月 IC。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pub_style
    pub_style.setup(cn_font=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    icc = summary[summary["metric"] == "RankIC"].reset_index(drop=True)
    labels = {"D1_absorb|score_e1": "吸收\n经济先验",
              "D1_absorb|score_e2": "吸收\n估计beta",
              "D2_predict|score_e1": "日盘预测\n经济先验",
              "D2_predict|score_e2": "日盘预测\n估计beta"}
    xs = np.arange(len(icc))
    se = (icc["mean"] / icc["hac_t"]).abs()
    colors = ["#3b6db3" if c.startswith("D1") else "#c0504d"
              for c in icc["cell"]]
    ax1.bar(xs, icc["mean"], yerr=1.96 * se, capsize=4, color=colors,
            alpha=0.85)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xticks(xs, [labels.get(c, c) for c in icc["cell"]], fontsize=8)
    ax1.set_ylabel("截面 RankIC 均值")
    ax1.set_title("(a) 六格汇总（HAC 95% CI）", fontsize=10)

    for (design, score), g in monthly.groupby(["design", "score"]):
        style = "-" if design == "D1_absorb" else "--"
        lab = ("吸收" if design == "D1_absorb" else "预测") + \
              ("·先验" if score == "score_e1" else "·beta")
        ax2.plot(g["month"], g["mean"], style, marker="o", ms=3, label=lab)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.tick_params(axis="x", rotation=45, labelsize=8)
    ax2.set_ylabel("月均截面 RankIC")
    ax2.set_title("(b) 逐月剖面（吸收各月为正，预测无月份稳定）",
                  fontsize=10)
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "f_supp_xsec.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"figure -> {out}")


def hac_mean(series: pd.Series) -> tuple[float, float, int]:
    s = series.dropna()
    if len(s) < 10:
        return float("nan"), float("nan"), len(s)
    res = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(res.params[0]), float(res.tvalues[0]), len(s)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_returns()
    z = load_theme_z()
    products = sorted(panel["product"].unique())

    w1 = e1_weights(products)
    s1 = z @ w1.T                                 # [日, 品种] E1 得分
    e1 = (s1.stack().rename("score_e1")
          .rename_axis(["trade_date", "product"]))
    e2 = e2_scores(panel, z)
    panel = (panel.set_index(["trade_date", "product"])
             .join(e1).join(e2).reset_index())

    # 检验宇宙：流动性达标、非换月日
    uni = panel[panel["liquid"] & ~panel["roll"]].copy()

    cells = [("D1_absorb", "score_e1", "r_open"),
             ("D1_absorb", "score_e2", "r_open"),
             ("D2_predict", "score_e1", "r_day"),
             ("D2_predict", "score_e2", "r_day")]
    ic_rows, ls_rows = [], []
    for date, g in uni.groupby("trade_date"):
        for design, score, target in cells:
            gg = g.dropna(subset=[score, target])
            gg = gg[gg[score].abs() > 0]          # 无暴露品种不参与排序
            if len(gg) < MIN_UNIVERSE:
                continue
            ic = spearmanr(gg[score], gg[target]).statistic
            ic_rows.append({"trade_date": date, "design": design,
                            "score": score, "ic": ic, "n_cs": len(gg)})
            if design == "D2_predict":            # 五分位多空
                q = pd.qcut(gg[score].rank(method="first"), 5, labels=False)
                ls = (gg.loc[q == 4, target].mean()
                      - gg.loc[q == 0, target].mean())
                ls_rows.append({"trade_date": date, "score": score,
                                "ls_ret": ls, "n_cs": len(gg)})
    ic = pd.DataFrame(ic_rows)
    ls = pd.DataFrame(ls_rows)

    summary = []
    for (design, score), g in ic.groupby(["design", "score"]):
        m, t, n = hac_mean(g.set_index("trade_date")["ic"])
        icir = g["ic"].mean() / g["ic"].std() if g["ic"].std() > 0 else np.nan
        ex = g[~g["trade_date"].str.startswith("2026-06")]
        m_ex, t_ex, n_ex = hac_mean(ex.set_index("trade_date")["ic"])
        summary.append({"cell": f"{design}|{score}", "metric": "RankIC",
                        "mean": m, "hac_t": t, "icir_daily": icir,
                        "n_days": n, "n_cs_med": g["n_cs"].median(),
                        "mean_exjun": m_ex, "t_exjun": t_ex,
                        "n_exjun": n_ex})
    for score, g in ls.groupby("score"):
        m, t, n = hac_mean(g.set_index("trade_date")["ls_ret"])
        sh = (g["ls_ret"].mean() / g["ls_ret"].std() * np.sqrt(244)
              if g["ls_ret"].std() > 0 else np.nan)
        summary.append({"cell": f"D2_LS|{score}", "metric": "Q5-Q1 bp/日",
                        "mean": m * 1e4, "hac_t": t, "icir_daily": sh,
                        "n_days": n, "n_cs_med": g["n_cs"].median()})
    summary = pd.DataFrame(summary)

    # 冲突期（6 月）内外与逐月分解，作描述性输出
    ic["month"] = ic["trade_date"].str[:7]
    monthly = (ic.groupby(["design", "score", "month"])["ic"]
               .agg(["mean", "size"]).reset_index())

    make_figure(summary, monthly)

    z.reset_index().to_parquet(OUT / "theme_signals.parquet", index=False)
    panel.to_parquet(OUT / "panel.parquet", index=False)
    ic.to_parquet(OUT / "ic_daily.parquet", index=False)
    ls.to_parquet(OUT / "ls_daily.parquet", index=False)
    summary.to_parquet(OUT / "summary.parquet", index=False)
    monthly.to_parquet(OUT / "ic_monthly.parquet", index=False)
    meta = {
        "design": "P2 截面检验：6 个登记格（D1/D2 x E1/E2 IC + D2 LS x 2）",
        "signal": "s_pre 主题层（前收盘至 09:00），PIT 展开 z（min 20，"
                  "shift(1)），无活跃计 0",
        "e2": f"展开窗单变量 beta（r_open ~ z），min {BETA_MIN} 日、"
              f"严格 < t",
        "universe": f"过去 20 日成交额中位数 >= {MONEY_FLOOR/1e8:.0f} 亿、"
                    f"非换月日、|暴露|>0、单日 >= {MIN_UNIVERSE} 品种",
        "hac_lags": HAC_LAGS,
    }
    (OUT / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))

    print("=== P2 截面检验（6 登记格）===")
    print(summary.round(4).to_string(index=False))
    print("\n逐月 RankIC（D2 predict）：")
    show = monthly[monthly.design == "D2_predict"].pivot_table(
        index="month", columns="score", values="mean")
    print(show.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
