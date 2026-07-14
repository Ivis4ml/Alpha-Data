"""Polymarket 三分量信号 × 国内期货收益：相关性、HAC 回归与夜盘 lead-lag。

输入：``data/cn_futures``（期货库）、``data/polymarket/features/cn_registry.parquet``
（orientation 登记表）、``data/polymarket/daily_aligned``（逐笔）。

流程：

1. 批量拉取登记表内全部市场的逐笔成交（北京时间）。
2. 逐 (市场, 品种时段网格) 计算 ``window_signals``（15min 聚合、logit innovation、
   p_age<=120、结算截断），按主题 × 品种做方向加权聚合（sqrt(usdc) 权重，
   等权与剔除近结算样本为稳健性变体）。
3. 与期货日线收益对齐做两类检验：
   - 同期吸收：``s_night ~ r_night``、``s_gap ~ r_gap``、``s_day ~ r_day``（信息是否
     被对应时段即时定价，非预测性声称）；
   - 预测性：``s_pre = s_night + s_gap`` 对当日 ``r_day``；全日信号对 ``t+1..t+h``
     前向收益（h ∈ {1, 3, 5}），Newey-West（HAC）标准误，滞后 max(h-1, 1)。
   全部检验做 Benjamini-Hochberg FDR（本样本约 75 个重叠交易日，整体标注探索性）。
4. 夜盘分钟级 lead-lag（研究规范 §5.2 的窗口内版本）：SC × mideast_conflict /
   oil_price，主题 5min innovation 与 SC 夜盘 5min 收益的 ±60min 交叉相关。
5. 图表输出 ``docs/figures/``，数据表输出 ``data/cn_futures/analysis/``。

用法::

    .venv/bin/python scripts/analyze_cn_futures_polymarket.py [--agg-minutes 15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_data.cn_futures import sessions  # noqa: E402
from alpha_data.cn_futures import store as fut_store  # noqa: E402
from alpha_data.polymarket import cn_features  # noqa: E402
from alpha_data.polymarket import store as poly_store  # noqa: E402

ANALYSIS_DIR = fut_store.DB_DIR / "analysis"
FIG_DIR = ROOT / "docs" / "figures"
REGISTRY_PATH = poly_store.FEATURES_DIR / "cn_registry.parquet"

#: 信号有效窗口（Polymarket daily_aligned 覆盖范围内的期货交易日）。
SIGNAL_END = "2026-04-28"
HORIZONS = (1, 3, 5)
NEAR_RESOLUTION_DAYS = 3


def load_all_trades(condition_ids: list[str]) -> dict[str, pd.DataFrame]:
    """一次性拉取全部市场逐笔（北京时间），按 condition_id 分组返回。"""
    con = poly_store.connect()
    glob = poly_store.daily_aligned_glob().replace("'", "''")
    ids = ", ".join(f"'{c}'" for c in condition_ids)
    sql = f"""
        SELECT condition_id,
               timezone('Asia/Shanghai', to_timestamp(block_timestamp)) AS cn_ts,
               p_event, D, usdc_amount,
               timezone('Asia/Shanghai', resolved_at) AS resolved_cn
        FROM read_parquet('{glob}')
        WHERE condition_id IN ({ids})
        ORDER BY condition_id, block_timestamp
    """
    df = con.execute(sql).fetch_df()
    df["cn_ts"] = pd.to_datetime(df["cn_ts"])
    df["resolved_cn"] = pd.to_datetime(df["resolved_cn"])
    print(f"逐笔 {len(df):,} 行，市场 {df['condition_id'].nunique()} 个")
    return {cid: g.reset_index(drop=True) for cid, g in df.groupby("condition_id")}


def market_signals_by_grid(
    registry: pd.DataFrame,
    trades: dict[str, pd.DataFrame],
    specs: pd.DataFrame,
    *,
    agg_minutes: int,
) -> pd.DataFrame:
    """逐 (市场, 夜盘网格) 计算三分量信号，长表返回。

    同一市场对多个品种的信号仅依赖品种的夜盘收盘档位（23:00 / 01:00 / 02:30 / 无），
    按档位缓存避免重复计算。要求所选品种的交易日完全一致（运行时断言）。
    """
    products = sorted(registry["product"].unique())
    spec_map = specs.set_index("product")
    day_sets = {p: tuple(fut_store.read_daily(p)["trade_date"]) for p in products}
    base_days = day_sets[products[0]]
    for p, days in day_sets.items():
        assert days == base_days, f"{p} 交易日与 {products[0]} 不一致，网格缓存不成立"
    trade_days = [d for d in base_days]

    grid_cache: dict[str, pd.DataFrame] = {}

    def grid_windows(product: str) -> tuple[str, pd.DataFrame]:
        ne_str = spec_map.loc[product, "night_end"]
        key = ne_str if isinstance(ne_str, str) else "none"
        if key not in grid_cache:
            ne = sessions.parse_night_end(ne_str if isinstance(ne_str, str) else None)
            grid_cache[key] = sessions.signal_windows(trade_days, night_end=ne)
        return key, grid_cache[key]

    sig_cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows: list[pd.DataFrame] = []
    pairs = registry[["condition_id", "product"]].drop_duplicates()
    for i, rec in enumerate(pairs.itertuples(index=False), 1):
        if rec.condition_id not in trades:
            continue
        key, windows = grid_windows(rec.product)
        cache_key = (rec.condition_id, key)
        if cache_key not in sig_cache:
            sig_cache[cache_key] = cn_features.window_signals(
                trades[rec.condition_id], windows, agg_minutes=agg_minutes
            )
        sig = sig_cache[cache_key].copy()
        sig["condition_id"] = rec.condition_id
        sig["product"] = rec.product
        rows.append(sig)
        if i % 100 == 0:
            print(f"  信号 {i}/{len(pairs)}")
    out = pd.concat(rows, ignore_index=True)
    return out


def aggregate_theme(
    market_sig: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    weight: str = "sqrt_usdc",
    drop_near_resolution: bool = False,
) -> pd.DataFrame:
    """主题 × 品种 × 交易日的方向加权信号与活动量。

    Args:
        market_sig: ``market_signals_by_grid`` 输出。
        registry: orientation 登记表。
        weight: ``sqrt_usdc``（默认）或 ``equal``。
        drop_near_resolution: 剔除市场结算前 ``NEAR_RESOLUTION_DAYS`` 个交易日的
            观测（同义反复稳健性）。

    Returns:
        ``[theme, product, trade_date, s_night, s_gap, s_day, n_active,
        usdc_night, usdc_gap, usdc_day]``。
    """
    meta_cols = ["condition_id", "product", "theme", "orientation", "usdc_win",
                 "admit_ts", "resolved_at"]
    df = market_sig.merge(
        registry[meta_cols].drop_duplicates(["condition_id", "product"]),
        on=["condition_id", "product"],
        how="inner",
    )

    # 时点化准入：早于市场累计流动性达标时刻（admit_ts，北京日期）的信号置 NaN，
    # 使任一交易日的面板构成只依赖当日已知的流动性（避免全窗口筛选的样本内信息）。
    admit_date = (
        pd.to_datetime(df["admit_ts"], unit="s", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.strftime("%Y-%m-%d")
    )
    before_admit = df["trade_date"] < admit_date
    for w in cn_features.WINDOWS:
        df.loc[before_admit, f"s_{w}"] = np.nan
    if drop_near_resolution:
        resolved_date = pd.to_datetime(df["resolved_at"], utc=True).dt.tz_convert(
            "Asia/Shanghai"
        ).dt.strftime("%Y-%m-%d")
        days = np.sort(df["trade_date"].unique())
        idx_of = {d: i for i, d in enumerate(days)}
        td_idx = df["trade_date"].map(idx_of)
        res_idx = resolved_date.map(
            lambda d: np.searchsorted(days, d) if isinstance(d, str) else np.inf
        )
        near = td_idx >= (res_idx - NEAR_RESOLUTION_DAYS)
        for w in cn_features.WINDOWS:
            df.loc[near, f"s_{w}"] = np.nan

    df["w"] = np.sqrt(df["usdc_win"]) if weight == "sqrt_usdc" else 1.0

    def wavg(block: pd.DataFrame, col: str) -> float:
        v = block[col].to_numpy(dtype="float64")
        w = block["w"].to_numpy(dtype="float64") * block["orientation"].to_numpy(dtype="float64")
        ok = ~np.isnan(v)
        if not ok.any():
            return np.nan
        return float(np.sum(v[ok] * w[ok]) / np.sum(np.abs(w[ok])))

    rows: list[dict] = []
    for (theme, product, day), block in df.groupby(["theme", "product", "trade_date"]):
        rows.append(
            {
                "theme": theme, "product": product, "trade_date": day,
                **{f"s_{w}": wavg(block, f"s_{w}") for w in cn_features.WINDOWS},
                "n_active": int(block[[f"s_{w}" for w in cn_features.WINDOWS]]
                                .notna().any(axis=1).sum()),
                **{f"usdc_{w}": float(block[f"usdc_{w}"].sum(skipna=True))
                   for w in cn_features.WINDOWS},
            }
        )
    return pd.DataFrame(rows).sort_values(["theme", "product", "trade_date"]).reset_index(drop=True)


def futures_returns(products: list[str]) -> pd.DataFrame:
    """拼接所选品种的日频收益表（含前向累计收益）。"""
    frames = []
    for p in products:
        d = fut_store.read_daily(p)
        d["product"] = p
        d["r_gap_total"] = d["r_gap_pm"] + d["r_gap_am"]
        # 无夜盘品种（night_* 全 NaN）用整段 gap
        if d["r_night"].isna().all():
            d["r_gap_total"] = d["r_gap_full"]
        for h in HORIZONS:
            fwd = d["r_cc"].shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1))
            d[f"r_fwd_{h}"] = fwd
        frames.append(
            d[["product", "trade_date", "r_night", "r_gap_total", "r_day", "r_cc",
               "roll", *[f"r_fwd_{h}" for h in HORIZONS]]]
        )
    return pd.concat(frames, ignore_index=True)


def nw_regression(
    y: np.ndarray, x: np.ndarray, maxlags: int
) -> tuple[float, float, float, int]:
    """一元 OLS + Newey-West，返回 (beta, t, p, N)。"""
    import statsmodels.api as sm

    ok = ~(np.isnan(y) | np.isnan(x))
    n = int(ok.sum())
    if n < 20 or np.nanstd(x[ok]) == 0:
        return np.nan, np.nan, np.nan, n
    model = sm.OLS(y[ok], sm.add_constant(x[ok]))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": max(maxlags, 1)})
    return float(res.params[1]), float(res.tvalues[1]), float(res.pvalues[1]), n


def corr_tests(theme_sig: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """同期吸收 + 预测性检验总表。"""
    from scipy import stats as sps

    merged = theme_sig.merge(rets, on=["product", "trade_date"], how="inner")
    merged = merged[merged["trade_date"] <= SIGNAL_END]
    rows: list[dict] = []

    def add_row(block: pd.DataFrame, theme: str, product: str, kind: str,
                sig_col: str, ret_col: str, maxlags: int) -> None:
        s = block[sig_col].to_numpy(dtype="float64")
        r = block[ret_col].to_numpy(dtype="float64")
        ok = ~(np.isnan(s) | np.isnan(r))
        n = int(ok.sum())
        if n < 20:
            return
        pear = sps.pearsonr(s[ok], r[ok])
        spear = sps.spearmanr(s[ok], r[ok])
        beta, t_hac, p_hac, _ = nw_regression(r, s, maxlags)
        # 经济量级：信号一个标准差对应的收益（bp）
        bp_per_sd = beta * np.nanstd(s[ok]) * 1e4
        rows.append(
            {
                "theme": theme, "product": product, "kind": kind,
                "signal": sig_col, "ret": ret_col, "n": n,
                "pearson": float(pear.statistic), "p_pearson": float(pear.pvalue),
                "spearman": float(spear.statistic), "p_spearman": float(spear.pvalue),
                "beta": beta, "t_hac": t_hac, "p_hac": p_hac,
                "bp_per_sd": bp_per_sd,
                # n<40 时 HAC（尤其 h=3/5 的 2-4 阶滞后）小样本下过度拒绝，
                # 显著性解读须打折（见报告局限）。
                "small_sample": n < 40,
            }
        )

    for (theme, product), block in merged.groupby(["theme", "product"]):
        block = block.sort_values("trade_date")
        block = block[~block["roll"].fillna(False)]
        # 同期吸收
        add_row(block, theme, product, "contemporaneous", "s_night", "r_night", 1)
        add_row(block, theme, product, "contemporaneous", "s_gap", "r_gap_total", 1)
        add_row(block, theme, product, "contemporaneous", "s_day", "r_day", 1)
        # 预测性：开盘前信息 -> 当日日盘。无夜盘品种（如 IF）的 s_night 为结构性
        # 全 NaN，夜间贡献取 0，使 s_pre 退化为 s_gap（与 futures_returns 对
        # r_gap_total 的特判同理）；有夜盘品种的 NaN（p_age 超时 / 近结算屏蔽）
        # 必须继续传播，故只在全 NaN 时替换。
        s_night_eff = block["s_night"] if block["s_night"].notna().any() else 0.0
        block = block.assign(s_pre=s_night_eff + block["s_gap"],
                             s_all=s_night_eff + block["s_gap"] + block["s_day"])
        add_row(block, theme, product, "predictive", "s_pre", "r_day", 1)
        for h in HORIZONS:
            add_row(block, theme, product, "predictive", "s_all", f"r_fwd_{h}",
                    max(h - 1, 1))

    table = pd.DataFrame(rows)
    # BH-FDR（对预测性检验整体）。用 HAC p 值：h=3/5 的前向收益窗口重叠，
    # iid 假设下的 p_pearson 反保守，与展示用的 t_hac 也不一致。
    pred = table[table["kind"] == "predictive"].copy()
    if not pred.empty:
        p = pred["p_hac"].fillna(1.0).to_numpy()
        order = np.argsort(p)
        m = len(p)
        q = np.full(m, np.nan)
        cummin = 1.0
        for rank_pos in range(m - 1, -1, -1):
            i = order[rank_pos]
            val = p[i] * m / (rank_pos + 1)
            cummin = min(cummin, val)
            q[i] = cummin
        table.loc[pred.index, "q_bh"] = q
    return table


def night_leadlag(
    registry: pd.DataFrame,
    trades: dict[str, pd.DataFrame],
    *,
    product: str = "SC",
    themes: tuple[str, ...] = ("mideast_conflict", "oil_price"),
    step_minutes: int = 5,
    max_lag_steps: int = 12,
) -> pd.DataFrame:
    """夜盘内主题 innovation 与期货分钟收益的交叉相关剖面。

    正滞后 k 表示"信号领先收益 k 步"（信号在前，收益在后）。口径要点：

    - 期货 bar 的 ``ts`` 是收盘戳，重采样必须 ``closed="right"``：标签 ``B`` 的
      收益桶覆盖 ``(B-step, B]``，与信号桶（成交严格早于 ``B``）不重叠，正滞后档
      才是严格的"信号在前"。
    - 信号 innovation 只在相邻桶恰好间隔一个 step 时有效（跨日盘 / 跨夜的
      logit 差不是分钟级 innovation）。
    - 滞后配对要求索引差恰为 ``k*step`` 分钟（不规则索引上位置 shift 会把跨夜 /
      跨假期样本错标为分钟级滞后）。

    注意：信号侧桶内加权中位数使 innovation 相对最新成交价约有半桶滞后，负滞后
    （收益领先信号）档相关偏高属该聚合的固有现象，不应解读为期货预测 Polymarket。
    """
    minute = fut_store.read_minute(product)
    night = minute[minute["session"] == "night"].sort_values("ts")
    px = night.set_index("ts")["close"]
    r_fut = np.log(px / px.shift(1))
    # 跨夜断点置 NaN（相邻 bar 间隔超过 step 即视为断点）
    breaks = px.index.to_series().diff() > pd.Timedelta(minutes=2)
    r_fut[breaks] = np.nan
    r_fut = r_fut.resample(
        f"{step_minutes}min", label="right", closed="right"
    ).sum(min_count=1)

    rows: list[dict] = []
    for theme in themes:
        sub = registry[(registry["theme"] == theme) & (registry["product"] == product)]
        if sub.empty:
            continue
        frames = []
        for rec in sub.drop_duplicates("condition_id").itertuples(index=False):
            tr = trades.get(rec.condition_id)
            if tr is None or tr.empty:
                continue
            agg = cn_features.aggregate_price(tr, agg_minutes=step_minutes)
            lo = pd.Series(
                cn_features.logit(agg["p_agg"].to_numpy()),
                index=pd.to_datetime(agg["bucket_end"]),
            )
            # 只保留相邻桶恰隔一个 step 的 innovation：稀疏索引上的 diff 会把
            # 跨日盘 / 跨夜的累计 logit 变化错当作单桶 innovation。
            consecutive = lo.index.to_series().diff() == pd.Timedelta(
                minutes=step_minutes
            )
            innov = lo.diff().where(consecutive) * rec.orientation * np.sqrt(rec.usdc_win)
            frames.append(innov)
        if not frames:
            continue
        pool = pd.concat(frames, axis=1)
        n_live = (~pool.isna()).sum(axis=1).replace(0, np.nan)
        theme_innov = pool.sum(axis=1, skipna=True) / np.sqrt(n_live)
        aligned = pd.DataFrame({"sig": theme_innov, "ret": r_fut}).dropna()
        gap = aligned.index.to_series()
        for k in range(-max_lag_steps, max_lag_steps + 1):
            # 位置 shift 在不规则索引上会把跨夜 / 跨假期样本错标为 k*step 分钟
            # 滞后；要求配对两端的实际时间差恰为 k*step 分钟。
            spacing_ok = gap.diff(k) == pd.Timedelta(minutes=k * step_minutes)
            paired = pd.DataFrame(
                {"sig": aligned["sig"].shift(k), "ret": aligned["ret"]}
            )[spacing_ok].dropna()
            if len(paired) < 50:
                continue
            rows.append(
                {
                    "theme": theme, "product": product,
                    "lag_minutes": k * step_minutes,
                    "corr": float(paired["sig"].corr(paired["ret"])),
                    "n": len(paired),
                }
            )
    return pd.DataFrame(rows)


def make_figures(
    theme_sig: pd.DataFrame,
    rets: pd.DataFrame,
    table: pd.DataFrame,
    leadlag: pd.DataFrame,
    trades: dict[str, pd.DataFrame],
) -> None:
    """产出全部图表（PNG，中文标签）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Hiragino Sans GB", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # F1: 中东主题累计信号 vs SC 价格
    ms = theme_sig[(theme_sig["theme"] == "mideast_conflict")
                   & (theme_sig["product"] == "SC")].sort_values("trade_date")
    ms = ms[ms["trade_date"] <= SIGNAL_END]
    s_total = (ms[["s_night", "s_gap", "s_day"]].sum(axis=1, skipna=True)).cumsum()
    sc = fut_store.read_daily("SC")
    sc = sc[sc["trade_date"].isin(ms["trade_date"])].sort_values("trade_date")
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    x = pd.to_datetime(ms["trade_date"])
    ax1.plot(x, s_total.to_numpy(), color="#B54708", label="中东冲突累计信号（logit）")
    ax1.set_ylabel("累计方向加权 logit innovation", color="#B54708")
    ax2 = ax1.twinx()
    ax2.plot(pd.to_datetime(sc["trade_date"]), sc["day_close"], color="#175CD3",
             label="SC 主力收盘")
    ax2.set_ylabel("SC 收盘价（元/桶）", color="#175CD3")
    ax1.set_title("中东冲突主题信号 与 INE 原油（SC）主力价格")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f1_mideast_vs_sc.png", dpi=150)
    plt.close(fig)

    # F2: 同期吸收相关性热力图
    cont = table[table["kind"] == "contemporaneous"].copy()
    cont["pair"] = cont["theme"] + " × " + cont["product"]
    piv = cont.pivot_table(index="pair", columns="signal", values="pearson")
    piv = piv.reindex(columns=["s_night", "s_gap", "s_day"])
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(piv))))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                   aspect="auto")
    ax.set_xticks(range(len(piv.columns)), ["夜盘", "闭市 gap", "日盘"])
    ax.set_yticks(range(len(piv)), piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("同期吸收：主题信号与对应时段期货收益的 Pearson 相关")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f2_contemporaneous_heatmap.png", dpi=150)
    plt.close(fig)

    # F3: 预测性检验（h=1 与 s_pre->r_day）
    pred = table[table["kind"] == "predictive"].copy()
    pred["pair"] = (pred["theme"] + "×" + pred["product"] + " "
                    + pred["signal"] + "→" + pred["ret"])
    pred = pred.sort_values("t_hac")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(pred))))
    colors = ["#B42318" if q < 0.1 else "#98A2B3"
              for q in pred["q_bh"].fillna(1.0)]
    hatches = ["///" if small else "" for small in pred["small_sample"]]
    bars = ax.barh(pred["pair"], pred["t_hac"], color=colors)
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    ax.axvline(2.0, color="#475467", lw=0.8, ls="--")
    ax.axvline(-2.0, color="#475467", lw=0.8, ls="--")
    ax.set_xlabel("Newey-West t")
    ax.set_title("预测性检验（红 = BH-FDR q<0.1，FDR 用 HAC p 值；"
                 "斜纹 = n<40 小样本，HAC 过度拒绝，解读打折）")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f3_predictive_tstats.png", dpi=150)
    plt.close(fig)

    # F4: 夜盘 lead-lag 剖面
    if not leadlag.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for theme, block in leadlag.groupby("theme"):
            ax.plot(block["lag_minutes"], block["corr"], marker="o", ms=3, label=theme)
        ax.axvline(0, color="#475467", lw=0.8)
        ax.axhline(0, color="#475467", lw=0.8)
        ax.set_xlabel("滞后（分钟，正值 = 信号领先收益）")
        ax.set_ylabel("交叉相关")
        ax.set_title("SC 夜盘：主题 innovation 与 SC 分钟收益的交叉相关")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "f4_night_leadlag_sc.png", dpi=150)
        plt.close(fig)

    # F5: Polymarket 活动的北京时间时钟（时差结构）
    hours = []
    weights = []
    for tr in trades.values():
        hours.append(tr["cn_ts"].dt.hour.to_numpy())
        weights.append(tr["usdc_amount"].to_numpy(dtype="float64"))
    h = np.concatenate(hours)
    w = np.concatenate(weights)
    prof = pd.Series(w).groupby(pd.Series(h)).sum() / 1e6
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(prof.index, prof.to_numpy(), color="#175CD3")
    ax.axvspan(9, 15, alpha=0.12, color="#12B76A", label="国内日盘 09-15")
    ax.axvspan(21, 24, alpha=0.12, color="#B54708", label="国内夜盘 21-02:30")
    ax.axvspan(0, 2.5, alpha=0.12, color="#B54708")
    ax.set_xlabel("北京时间（小时）")
    ax.set_ylabel("登记市场成交额（百万 USDC）")
    ax.set_title("Polymarket 活动的北京时间分布：美国白天与国内夜盘重叠")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f5_activity_clock.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Polymarket × 国内期货相关性分析")
    parser.add_argument("--agg-minutes", type=int, default=15)
    parser.add_argument("--skip-leadlag", action="store_true")
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_parquet(REGISTRY_PATH)
    specs = fut_store.read_product_specs()
    products = sorted(registry["product"].unique())
    print(f"品种：{products}")

    trades = load_all_trades(sorted(registry["condition_id"].unique()))

    market_sig = market_signals_by_grid(
        registry, trades, specs, agg_minutes=args.agg_minutes
    )
    market_sig.to_parquet(ANALYSIS_DIR / "market_signals.parquet", index=False)

    theme_sig = aggregate_theme(market_sig, registry)
    theme_sig.to_parquet(ANALYSIS_DIR / "theme_signals.parquet", index=False)
    theme_eq = aggregate_theme(market_sig, registry, weight="equal")
    theme_eq.to_parquet(ANALYSIS_DIR / "theme_signals_equal.parquet", index=False)
    theme_nr = aggregate_theme(market_sig, registry, drop_near_resolution=True)
    theme_nr.to_parquet(ANALYSIS_DIR / "theme_signals_nores.parquet", index=False)

    rets = futures_returns(products)
    table = corr_tests(theme_sig, rets)
    table.to_parquet(ANALYSIS_DIR / "corr_table.parquet", index=False)
    table_eq = corr_tests(theme_eq, rets)
    table_eq.to_parquet(ANALYSIS_DIR / "corr_table_equal.parquet", index=False)
    table_nr = corr_tests(theme_nr, rets)
    table_nr.to_parquet(ANALYSIS_DIR / "corr_table_nores.parquet", index=False)

    print("\n=== 同期吸收（|pearson| 前 12）===")
    cont = table[table["kind"] == "contemporaneous"]
    print(cont.reindex(cont["pearson"].abs().sort_values(ascending=False).index)
          .head(12)[["theme", "product", "signal", "ret", "n", "pearson", "t_hac",
                     "bp_per_sd"]].to_string(index=False))
    print("\n=== 预测性（|t_hac| 前 12）===")
    pred = table[table["kind"] == "predictive"]
    print(pred.reindex(pred["t_hac"].abs().sort_values(ascending=False).index)
          .head(12)[["theme", "product", "signal", "ret", "n", "pearson", "t_hac",
                     "bp_per_sd", "q_bh"]].to_string(index=False))

    leadlag = pd.DataFrame()
    if not args.skip_leadlag:
        leadlag = night_leadlag(registry, trades)
        leadlag.to_parquet(ANALYSIS_DIR / "leadlag_sc.parquet", index=False)
        if not leadlag.empty:
            print("\n=== SC 夜盘 lead-lag（信号领先侧，lag>0）===")
            lead = leadlag[leadlag["lag_minutes"] > 0]
            print(lead.reindex(lead["corr"].abs().sort_values(ascending=False).index)
                  .head(8).to_string(index=False))

    make_figures(theme_sig, rets, table, leadlag, trades)
    print(f"\n图表已写入 {FIG_DIR}，数据表已写入 {ANALYSIS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
