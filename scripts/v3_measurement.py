"""v3 测量层驱动（Layer 1-3）：统一 tape -> 主题级测量信号面板。

流程（数据隔离协议：全程不接触期货收益）：

1. 桶级观测（15 分钟，买卖两向中位数均值，DuckDB 聚合）。
2. Layer 1：logit 状态空间滤波，(q, r0) 网格 MLE；第一轮后按剩余期限
   估计期限乘子曲线（框架 §5.3），第二轮带乘子重估。
3. Layer 2：族内加权等张投影 -> coherent 概率与 narrative tension。
4. Layer 3：日期族 hazard innovation、价格阈值族分布特征、平台公共因子
   （全类别高流动市场的 family 级创新截面均值）与主题信号正交化。
5. 主题聚合（orientation × sqrt(usdc) 权重、时点化准入、结算截断——口径与
   v1.1 基线一致，便于对比）。

内部验证（Polymarket 内部目标，写入 v3_validation.json）：

- H1-lite：滤波一步预测创新方差 vs 原始桶间差分方差（应显著缩小）。
- H2-lite：投影修复方向——违反单调约束的市场，其后续价格移动方向与投影
  方向的一致率（> 0.5 支持投影）。

产物（``data/cn_futures/analysis/v3/``）：

    v3_families.parquet          族与门控标注
    v3_filter_params.parquet     逐市场 (q, r0, loglik, n_obs)
    v3_term_curve.parquet        期限乘子曲线
    v3_theme_signals.parquet     主题 × 品种 × 交易日信号（raw / coherent / orth）
    v3_factor.parquet            平台公共因子（逐网格 × 窗口 × 交易日）
    v3_hazard.parquet            主题级 hazard innovation
    v3_price_features.parquet    价格阈值族分布特征 innovation
    v3_zpre.parquet              结算前 24h 滤波状态（Power Gate 输入）
    v3_outcomes.parquet          市场结算结果 Y（Power Gate 输入）
    v3_validation.json           内部验证统计

用法::

    .venv/bin/python scripts/v3_measurement.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_data.cn_futures import sessions  # noqa: E402
from alpha_data.cn_futures import store as fut_store  # noqa: E402
from alpha_data.polymarket import store  # noqa: E402
from alpha_data.polymarket.v3 import (  # noqa: E402
    coherence,
    families,
    geometry,
    latent,
    story,
    tape,
)

OUT_DIR = ROOT / "data" / "cn_futures" / "analysis" / "v3"
REGISTRY_V3 = store.FEATURES_DIR / "cn_registry_v3.parquet"

#: 平台宇宙：窗口内成交额门槛与数量上限（全类别，公共因子用）。
PLATFORM_MIN_USDC = 300_000.0
PLATFORM_TOP_N = 600

#: 窗口类型（atomic；``gap = gap_pm + gap_am``，``pre = [prev 15:00, 09:00)``）。
ATOMIC_WINDOWS = ("night", "gap_pm", "gap_am", "day", "pre")

WINDOW_START_UTC = "2026-01-04 16:00:00"
WINDOW_END_UTC = "2026-07-13 16:00:00"


def beijing_to_epoch(ts: pd.Series) -> np.ndarray:
    """北京 tz-naive 时间戳 -> UTC 秒（NaT -> -1）。

    用 Timedelta 除法而非 ``astype(int64)``：pandas 2 的时间列可能是
    ``datetime64[us]``，直接取整数得到微秒会差三个数量级。
    """
    t = pd.to_datetime(ts)
    loc = t.dt.tz_localize("Asia/Shanghai")
    epoch = (loc - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    return np.where(t.isna(), -1, epoch).astype("int64")


def grid_atomic_windows(trade_days: list[str], night_end_str: str | None) -> pd.DataFrame:
    """交易日列表 + 夜盘收盘档 -> atomic 窗口表 ``[trade_date, window, start_ts, end_ts]``。"""
    ne = sessions.parse_night_end(night_end_str)
    win = sessions.signal_windows(trade_days, night_end=ne)
    cols = {
        "night": ("night_start", "night_end"),
        "gap_pm": ("gap1_start", "gap1_end"),
        "gap_am": ("gap2_start", "gap2_end"),
        "day": ("day_start", "day_end"),
        "pre": ("gap1_start", "day_start"),
    }
    frames = []
    for name, (c0, c1) in cols.items():
        f = pd.DataFrame(
            {
                "trade_date": win["trade_date"],
                "window": name,
                "start_ts": beijing_to_epoch(win[c0]),
                "end_ts": beijing_to_epoch(win[c1]),
            }
        )
        frames.append(f.loc[(f["start_ts"] > 0) & (f["end_ts"] > 0)])
    return pd.concat(frames, ignore_index=True)


def platform_universe(con) -> pd.DataFrame:
    """全类别高流动市场（公共因子宇宙）。"""
    cols = ("condition_id, market_slug, resolved_at, block_timestamp, usdc_amount")
    sql = f"""
        SELECT condition_id,
               any_value(market_slug) AS slug,
               max(resolved_at)       AS resolved_at,
               sum(usdc_amount)       AS usdc_win,
               count(*)               AS n_win
        FROM {tape.union_sql(cols)}
        WHERE block_timestamp >= epoch(TIMESTAMP '{WINDOW_START_UTC}')
          AND block_timestamp <  epoch(TIMESTAMP '{WINDOW_END_UTC}')
        GROUP BY condition_id
        HAVING sum(usdc_amount) >= {PLATFORM_MIN_USDC}
        ORDER BY usdc_win DESC
        LIMIT {PLATFORM_TOP_N}
    """
    return con.execute(sql).fetch_df()


def market_window_signals(
    proj_z: pd.DataFrame,
    pairs: pd.DataFrame,
    grids: dict[str, pd.DataFrame],
    grid_of_product: dict[str, str],
    resolved_ts: pd.Series,
    *,
    z_col: str = "z_proj",
) -> pd.DataFrame:
    """边界状态 -> 逐 (市场, 品种, 交易日, 窗口) 的 logit innovation。

    Args:
        proj_z: ``[condition_id, boundary_ts, {z_col}]``。
        pairs: ``[condition_id, product]``。
        grids: 网格键 -> atomic 窗口表。
        grid_of_product: 品种 -> 网格键。
        resolved_ts: ``condition_id -> 结算 UTC 秒``（NaN 表示未结算）。
        z_col: 使用的状态列（``z_proj`` coherent / ``z`` raw）。

    Returns:
        ``[condition_id, product, trade_date, window, s]``；窗口结束不早于结算
        时刻的行置 NaN（结算后的常数概率不是信号，口径同 v1.1）。
    """
    zmap = proj_z.set_index(["condition_id", "boundary_ts"])[z_col]
    frames: list[pd.DataFrame] = []
    for product, grid_key in grid_of_product.items():
        win = grids[grid_key]
        cids = pairs.loc[pairs["product"] == product, "condition_id"].unique()
        cross = win.assign(key=1).merge(
            pd.DataFrame({"condition_id": cids, "key": 1}), on="key"
        ).drop(columns="key")
        idx0 = pd.MultiIndex.from_frame(cross[["condition_id", "start_ts"]])
        idx1 = pd.MultiIndex.from_frame(cross[["condition_id", "end_ts"]])
        z0 = zmap.reindex(idx0).to_numpy(dtype="float64")
        z1 = zmap.reindex(idx1).to_numpy(dtype="float64")
        s = z1 - z0
        res = cross["condition_id"].map(resolved_ts).to_numpy(dtype="float64")
        s = np.where(
            np.isfinite(res) & (cross["end_ts"].to_numpy() >= res), np.nan, s
        )
        frames.append(
            pd.DataFrame(
                {
                    "condition_id": cross["condition_id"],
                    "product": product,
                    "trade_date": cross["trade_date"],
                    "window": cross["window"],
                    "s": s,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def aggregate_theme_v3(
    market_sig: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """主题聚合（orientation × sqrt(usdc_win)，时点化准入；口径同 v1.1）。

    Returns:
        ``[theme, product, trade_date, window, s, n_active]``（长表）。
    """
    meta = registry[["condition_id", "product", "theme", "orientation",
                     "usdc_win", "admit_ts"]].drop_duplicates(
        ["condition_id", "product"])
    df = market_sig.merge(meta, on=["condition_id", "product"], how="inner")
    admit_date = (
        pd.to_datetime(df["admit_ts"], unit="s", utc=True)
        .dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
    )
    df.loc[df["trade_date"] < admit_date, "s"] = np.nan
    df["w"] = np.sqrt(df["usdc_win"]) * df["orientation"]

    grp = df.groupby(["theme", "product", "trade_date", "window"], sort=False)

    def agg(blk: pd.DataFrame) -> pd.Series:
        v = blk["s"].to_numpy(dtype="float64")
        w = blk["w"].to_numpy(dtype="float64")
        ok = ~np.isnan(v)
        if not ok.any():
            return pd.Series({"s": np.nan, "n_active": 0})
        return pd.Series(
            {"s": float(np.sum(v[ok] * w[ok]) / np.sum(np.abs(w[ok]))),
             "n_active": int(ok.sum())}
        )

    return grp.apply(agg, include_groups=False).reset_index()


def wide_theme_signals(
    theme_long: pd.DataFrame, has_night: dict[str, bool]
) -> pd.DataFrame:
    """长表 -> 宽表（``s_night / s_gap / s_day / s_pre``）。

    有夜盘品种 ``s_gap = s_gap_pm + s_gap_am``（任一缺测则 NaN，口径同 v1.1）；
    无夜盘品种 ``s_gap = s_gap_pm``（该段即整段闭市）。
    """
    piv = theme_long.pivot_table(
        index=["theme", "product", "trade_date"], columns="window", values="s",
        aggfunc="first",
    )
    for c in ATOMIC_WINDOWS:
        if c not in piv.columns:
            piv[c] = np.nan
    out = pd.DataFrame(index=piv.index)
    out["s_night"] = piv["night"]
    night_flag = piv.index.get_level_values("product").map(has_night).to_numpy()
    out["s_gap"] = np.where(night_flag, piv["gap_pm"] + piv["gap_am"], piv["gap_pm"])
    out["s_day"] = piv["day"]
    out["s_pre"] = piv["pre"]
    n = theme_long.pivot_table(
        index=["theme", "product", "trade_date"], columns="window",
        values="n_active", aggfunc="first",
    )
    out["n_active"] = n.max(axis=1)
    return out.reset_index()


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = store.connect()

    registry = pd.read_parquet(REGISTRY_V3)
    fam_cn = families.build_families(registry)
    fam_cn.to_parquet(OUT_DIR / "v3_families.parquet", index=False)
    products = sorted(registry["product"].unique())
    specs = fut_store.read_product_specs().set_index("product")
    trade_days = fut_store.trading_days()
    trade_days = [d for d in trade_days if "2026-01-05" <= d <= "2026-07-13"]

    grids: dict[str, pd.DataFrame] = {}
    grid_of_product: dict[str, str] = {}
    for p in products:
        ne = specs.loc[p, "night_end"]
        key = ne if isinstance(ne, str) else "none"
        if key not in grids:
            grids[key] = grid_atomic_windows(trade_days, ne if isinstance(ne, str) else None)
        grid_of_product[p] = key
    print(f"品种 {len(products)} 个，网格档位：{sorted(grids)}")

    plat = platform_universe(con)
    plat_only = plat.loc[~plat["condition_id"].isin(set(registry["condition_id"]))]
    print(f"平台宇宙：{len(plat)} 个市场（与登记表重叠 {len(plat) - len(plat_only)}）")

    ids_cn = registry["condition_id"].unique().tolist()
    ids_all = sorted(set(ids_cn) | set(plat["condition_id"]))
    print(f"桶级观测：{len(ids_all)} 个市场 ...")
    # 观测窗带三周预热（滤波状态收敛），右端与登记窗口一致。
    warmup_start = int(pd.Timestamp("2025-12-15", tz="UTC").timestamp())
    win_end = int(pd.Timestamp(WINDOW_END_UTC, tz="UTC").timestamp())
    obs = latent.bucket_observations(
        con, ids_all, start_ts=warmup_start, end_ts=win_end
    )
    print(f"  观测 {len(obs):,} 桶，耗时 {time.time() - t0:.0f}s")

    # 期限表：族截止优先，回退结算时刻。
    plat_fam = families.build_families(
        plat.rename(columns={"slug": "slug"}).assign(theme="platform")[
            ["condition_id", "slug", "theme"]]
    )
    fam_all = pd.concat([fam_cn, plat_fam], ignore_index=True).drop_duplicates(
        "condition_id")
    resolved_all = pd.concat(
        [
            registry.drop_duplicates("condition_id")[["condition_id", "resolved_at"]],
            plat[["condition_id", "resolved_at"]],
        ],
        ignore_index=True,
    ).drop_duplicates("condition_id")
    # 链上结算事件补齐（tape 行内的 resolved_at 元数据在 HF 段稀疏）。
    res_chain = tape.resolutions_table(con).rename(
        columns={"resolved_at": "_res_chain"})
    resolved_all = resolved_all.merge(res_chain, on="condition_id", how="left")
    resolved_all["resolved_at"] = pd.to_datetime(
        resolved_all["_res_chain"], utc=True
    ).combine_first(pd.to_datetime(resolved_all["resolved_at"], utc=True))
    resolved_all = resolved_all.drop(columns=["_res_chain"])
    resolved_dt = pd.to_datetime(resolved_all["resolved_at"], utc=True)
    resolved_all["resolved_ts"] = (
        (resolved_dt - pd.Timestamp("1970-01-01", tz="UTC"))
        // pd.Timedelta(seconds=1)
    ).astype("float64")
    resolved_all.loc[resolved_dt.isna(), "resolved_ts"] = np.nan
    deadline = fam_all[["condition_id", "deadline_ts"]].merge(
        resolved_all[["condition_id", "resolved_ts"]], on="condition_id", how="outer"
    )
    deadline["deadline_ts"] = deadline["deadline_ts"].fillna(deadline["resolved_ts"])
    deadlines = deadline.dropna(subset=["deadline_ts"])[
        ["condition_id", "deadline_ts"]].copy()

    # Layer 1：两轮滤波。
    print("Layer 1 第一轮网格 MLE ...")
    fit1 = latent.fit_filter(obs)
    curve = latent.term_multiplier_curve(fit1.states, deadlines)
    print(curve.to_string(index=False))
    curve.to_parquet(OUT_DIR / "v3_term_curve.parquet", index=False)

    # 乘子长表：逐观测桶按剩余天数取曲线值（缺期限 -> 1）。
    dl_map = deadlines.set_index("condition_id")["deadline_ts"]
    tau_days = (
        obs["condition_id"].map(dl_map) - obs["bucket_end"]
    ) / 86400.0
    edges = list(latent.TAU_BUCKETS) + [0.0]
    mult_vals = np.ones(len(obs))
    for (hi, lo), vr in zip(
        zip(edges[:-1], edges[1:], strict=False),
        curve["var_ratio"].to_numpy(), strict=False,
    ):
        if np.isfinite(vr) and vr > 0:
            m = (tau_days < hi) & (tau_days >= lo)
            mult_vals[m.to_numpy()] = vr
    q_mult = pd.DataFrame(
        {"condition_id": obs["condition_id"], "bucket_end": obs["bucket_end"],
         "mult": mult_vals}
    )
    print("Layer 1 第二轮（带期限乘子）...")
    fit = latent.fit_filter(obs, q_mult=q_mult)
    fit.params.to_parquet(OUT_DIR / "v3_filter_params.parquet", index=False)

    # 内部验证 H1-lite：创新方差 vs 桶间差分方差。
    st = fit.states.sort_values(["condition_id", "bucket_end"])
    dy = obs.sort_values(["condition_id", "bucket_end"]).groupby(
        "condition_id")["y"].diff()
    h1 = {
        "innov_var": float(np.nanvar(st["nu"])),
        "raw_diff_var": float(np.nanvar(dy)),
        "ratio": float(np.nanvar(st["nu"]) / np.nanvar(dy)),
    }
    print(f"H1-lite：innov_var / raw_diff_var = {h1['ratio']:.3f}")

    # 边界状态 LOCF。
    all_ts = sorted({int(x) for g in grids.values()
                     for x in np.concatenate([g["start_ts"], g["end_ts"]])})
    boundaries = pd.DataFrame(
        {"condition_id": np.repeat(ids_all, len(all_ts)),
         "boundary_ts": np.tile(all_ts, len(ids_all))}
    )
    print(f"边界状态：{len(boundaries):,} 组合 ...")
    states_at = latent.state_at_boundaries(fit.states, boundaries)

    # Layer 2：投影与 tension。
    usdc_map = pd.concat(
        [registry.drop_duplicates("condition_id").set_index("condition_id")["usdc_win"],
         plat.set_index("condition_id")["usdc_win"]]
    )
    usdc_map = usdc_map[~usdc_map.index.duplicated()]
    weights = np.sqrt(usdc_map.clip(lower=1.0))
    proj = coherence.project_snapshots(states_at, fam_all, weights)
    proj = proj.merge(
        states_at[["condition_id", "boundary_ts", "z"]],
        on=["condition_id", "boundary_ts"], how="left",
    )

    # H2-lite：投影修复方向（违反处，下一边界的价格变动方向 vs 投影方向）。
    v = proj.loc[(proj["p_proj"] - proj["p_filt"]).abs() > 1e-6].copy()
    v = v.sort_values(["condition_id", "boundary_ts"])
    v["p_next"] = v.groupby("condition_id")["p_filt"].shift(-1)
    v = v.dropna(subset=["p_next"])
    repair = np.sign(v["p_next"] - v["p_filt"]) == np.sign(v["p_proj"] - v["p_filt"])
    h2 = {"n_violations": int(len(v)),
          "repair_rate": float(repair.mean()) if len(v) else np.nan}
    print(f"H2-lite：violations={h2['n_violations']}，repair_rate={h2['repair_rate']:.3f}")

    fit.states.to_parquet(OUT_DIR / "v3_filter_states.parquet", index=False)
    proj.to_parquet(OUT_DIR / "v3_proj_states.parquet", index=False)

    resolved_ts_map = resolved_all.set_index("condition_id")["resolved_ts"]
    pairs = registry[["condition_id", "product"]].drop_duplicates()

    # 市场级窗口信号（coherent 与 raw 两套）。
    sig_coh = market_window_signals(
        proj, pairs, grids, grid_of_product, resolved_ts_map, z_col="z_proj")
    sig_raw = market_window_signals(
        proj, pairs, grids, grid_of_product, resolved_ts_map, z_col="z")

    has_night = {p: grid_of_product[p] != "none" for p in products}
    theme_coh = wide_theme_signals(aggregate_theme_v3(sig_coh, registry), has_night)
    theme_raw = wide_theme_signals(aggregate_theme_v3(sig_raw, registry), has_night)
    theme = theme_coh.merge(
        theme_raw, on=["theme", "product", "trade_date"],
        suffixes=("", "_rawz"),
    )

    # 主题 tension（09:00 边界、流动性加权）。
    day_open_ts = {g: grids[g].loc[grids[g]["window"] == "day",
                                   ["trade_date", "start_ts"]]
                   for g in grids}
    tens = proj[["condition_id", "boundary_ts", "tension"]].merge(
        registry[["condition_id", "product", "theme", "usdc_win"]].drop_duplicates(
            ["condition_id", "product"]),
        on="condition_id",
    )
    frames = []
    for p, g in grid_of_product.items():
        dd = day_open_ts[g].rename(columns={"start_ts": "boundary_ts"})
        sub = tens.loc[tens["product"] == p].merge(dd, on="boundary_ts")
        frames.append(sub)
    tens = pd.concat(frames, ignore_index=True)
    tens["w"] = np.sqrt(tens["usdc_win"])
    theme_tension = (
        tens.groupby(["theme", "product", "trade_date"])
        .apply(lambda b: float(np.average(b["tension"], weights=b["w"])),
               include_groups=False)
        .rename("tension").reset_index()
    )
    theme = theme.merge(theme_tension, on=["theme", "product", "trade_date"],
                        how="left")

    # Layer 3：hazard 与价格分布特征（逐品种网格）。
    hz_frames, px_frames = [], []
    for p, g in grid_of_product.items():
        cids = pairs.loc[pairs["product"] == p, "condition_id"]
        sub_proj = proj.loc[proj["condition_id"].isin(set(cids))]
        hz = geometry.hazard_windows(sub_proj, fam_cn, grids[g])
        px = geometry.price_windows(sub_proj, fam_cn, grids[g])
        hz["product"] = p
        px["product"] = p
        hz_frames.append(hz)
        px_frames.append(px)
    hazard = pd.concat(hz_frames, ignore_index=True)
    price_feat = pd.concat(px_frames, ignore_index=True)
    hazard.to_parquet(OUT_DIR / "v3_hazard.parquet", index=False)
    price_feat.to_parquet(OUT_DIR / "v3_price_features.parquet", index=False)

    # hazard 主题聚合：family -> theme（USdc 加权），pre 窗口为主。
    fam_theme = fam_cn.drop_duplicates("family_key")[["family_key", "theme"]]
    fam_usdc = registry.merge(fam_cn[["condition_id", "family_key"]],
                              on="condition_id")
    fam_w = fam_usdc.groupby("family_key")["usdc_win"].sum().pipe(np.sqrt)
    hz = hazard.merge(fam_theme, on="family_key")
    if not hz.empty:
        hz["w"] = hz["family_key"].map(fam_w).fillna(1.0)
        hz_theme = (
            hz.groupby(["theme", "product", "trade_date", "window"])
            .apply(lambda b: float(np.average(b["d_lambda_front"], weights=b["w"])),
                   include_groups=False)
            .rename("hz").reset_index()
        )
        hz_wide = hz_theme.pivot_table(
            index=["theme", "product", "trade_date"], columns="window", values="hz",
            aggfunc="first",
        ).add_prefix("hz_").reset_index()
        theme = theme.merge(hz_wide, on=["theme", "product", "trade_date"],
                            how="left")
    else:
        print("警告：hazard 表为空，跳过主题聚合")

    # 价格分布特征主题聚合（oil_price / metal_price -> 对应品种）。
    px = price_feat.merge(fam_theme, on="family_key")
    if not px.empty:
        px["w"] = px["family_key"].map(fam_w).fillna(1.0)
        px_theme = (
            px.groupby(["theme", "product", "trade_date", "window"])
            .apply(
                lambda b: pd.Series(
                    {
                        "d_med": float(
                            np.average(b["d_implied_med_log"], weights=b["w"])),
                        "d_tail": float(
                            np.average(b["d_tail_mass"], weights=b["w"])),
                        "d_ent": float(np.average(b["d_entropy"], weights=b["w"])),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        px_pre = px_theme.loc[px_theme["window"] == "pre"].drop(columns="window")
        px_pre = px_pre.rename(
            columns={"d_med": "px_med_pre", "d_tail": "px_tail_pre",
                     "d_ent": "px_ent_pre"})
        theme = theme.merge(px_pre, on=["theme", "product", "trade_date"],
                            how="left")
    else:
        print("警告：价格分布特征为空，跳过主题聚合")

    # 平台公共因子（逐网格）与正交化。
    plat_pairs = pd.DataFrame(
        {"condition_id": plat["condition_id"], "product": "PLATFORM"}
    )
    factor_frames = []
    theme_orth_frames = []
    for g in grids:
        plat_sig = market_window_signals(
            proj, plat_pairs, {g: grids[g]}, {"PLATFORM": g},
            resolved_ts_map, z_col="z",
        )
        plat_sig = plat_sig.rename(columns={"s": "s"})
        fam_series = story.family_series(
            plat_sig[["condition_id", "trade_date", "window", "s"]],
            plat_fam.merge(plat[["condition_id", "usdc_win"]], on="condition_id")[
                ["condition_id", "family_key", "usdc_win"]],
        )
        fam_w_plat = (
            plat.merge(plat_fam[["condition_id", "family_key"]], on="condition_id")
            .groupby("family_key")["usdc_win"].sum().pipe(np.sqrt)
        )
        factor = story.common_factor(fam_series, fam_w_plat)
        factor["grid"] = g
        factor_frames.append(factor)

        prods = [p for p, gg in grid_of_product.items() if gg == g]
        sub_theme = theme.loc[theme["product"].isin(prods)]
        # gap 窗口因子 = gap_pm 与 gap_am 因子之和（与信号可加性一致）。
        fpiv = factor.pivot_table(index="trade_date", columns="window",
                                  values="f_pm")
        if "gap_am" in fpiv.columns:
            fpiv["gap"] = fpiv["gap_pm"] + fpiv["gap_am"]
        else:
            fpiv["gap"] = fpiv.get("gap_pm")
        flong = fpiv.reset_index().melt(
            id_vars="trade_date", var_name="window", value_name="f_pm")
        orth = story.orthogonalize(
            sub_theme, flong, cols=("s_night", "s_gap", "s_day", "s_pre"))
        # 附品种网格的因子列（pre 窗口）供 Layer 5 控制。
        orth = orth.merge(
            fpiv[["pre"]].rename(columns={"pre": "f_pm_pre"}).reset_index(),
            on="trade_date", how="left",
        )
        theme_orth_frames.append(orth)
    factor_all = pd.concat(factor_frames, ignore_index=True)
    factor_all.to_parquet(OUT_DIR / "v3_factor.parquet", index=False)
    theme_final = pd.concat(theme_orth_frames, ignore_index=True)
    theme_final.to_parquet(OUT_DIR / "v3_theme_signals.parquet", index=False)

    # Power Gate 输入：结算前 24h 状态与结算结果。
    res_cn = registry.drop_duplicates("condition_id").merge(
        resolved_all[["condition_id", "resolved_ts"]], on="condition_id")
    res_cn = res_cn.dropna(subset=["resolved_ts"])
    zb = pd.DataFrame(
        {"condition_id": res_cn["condition_id"],
         "boundary_ts": (res_cn["resolved_ts"] - 86400).astype("int64")}
    )
    zpre = latent.state_at_boundaries(fit.states, zb, p_age_max_minutes=2880.0)
    zpre = zpre.rename(columns={"z": "z_pre"})[["condition_id", "z_pre"]]
    zpre.to_parquet(OUT_DIR / "v3_zpre.parquet", index=False)

    am = con.execute(
        f"""
        SELECT condition_id,
               max(CASE WHEN outcome_seq = 1
                        AND winning_outcome_label IS NOT NULL
                        AND outcome_label = winning_outcome_label
                        THEN 1 ELSE 0 END) AS y,
               max(CASE WHEN winning_outcome_label IS NOT NULL
                        THEN 1 ELSE 0 END) AS has_winner
        FROM read_parquet('{str(tape.CHAIN_DIR / "asset_map.parquet")}')
        WHERE condition_id IN ({",".join("'" + c + "'" for c in ids_cn)})
        GROUP BY condition_id
        """
    ).fetch_df()
    outcomes = am.loc[am["has_winner"] == 1, ["condition_id", "y"]]
    outcomes.to_parquet(OUT_DIR / "v3_outcomes.parquet", index=False)

    (OUT_DIR / "v3_validation.json").write_text(
        json.dumps({"h1_lite": h1, "h2_lite": h2,
                    "n_markets_cn": len(ids_cn),
                    "n_markets_platform": len(plat),
                    "elapsed_sec": round(time.time() - t0, 1)},
                   ensure_ascii=False, indent=2)
    )
    print(f"完成，耗时 {time.time() - t0:.0f}s；产物在 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
