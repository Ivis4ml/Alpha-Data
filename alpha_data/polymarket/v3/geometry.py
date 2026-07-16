"""Layer 3：日期族 hazard 恢复与价格阈值族分布特征（框架 §7.1 / §7.2）。

日期阶梯族
----------

对同一事件干的截止阶梯 ``F_t(T_k) = P_t(tau_event <= T_k)``（投影后单调），
累计强度 ``Lambda_t(T_k) = -log(1 - F_t(T_k))``。进入模型的是 ``dLambda``
而非原始累计概率变化（框架 H5：hazard innovation 跨期限可比、近 1 处不压缩）。

窗口 innovation 的期限固定规则：以窗口**结束**时刻仍未到期的最近截止（front）
与最远截止（max）为基准，在窗口两端对**同一截止**求 Lambda 之差，避免 front
滚动造成的机械跳变。

价格阈值族
----------

``hit-high`` 阶梯给出期内最大值的生存函数 ``S(K) = P(max >= K)``，特征：

- ``tail_mass``：最高行权价处的生存概率（尾部质量）。
- ``implied_med``：``S(K) = 0.5`` 的线性插值行权价（期内极值的隐含中位数）。
- ``entropy``：离散化质量 ``[1-S(K_1), S(K_1)-S(K_2), ..., S(K_m)]`` 的香农熵。

``hit-low`` 对称（``P(min <= K)`` 为 CDF）。窗口 innovation 为特征在窗口两端之差
（``implied_med`` 取对数差，与期货对数收益同量纲）。价格定义市场只产出这些
分布特征，禁止进入 Layer 4（框架 §7.1）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: F 的截断上界（Lambda 在 F -> 1 时发散；0.995 对应 Lambda 约 5.3）。
F_CAP = 0.995


def cumulative_intensity(f: np.ndarray) -> np.ndarray:
    """``F -> Lambda = -log(1 - min(F, F_CAP))``。"""
    return -np.log1p(-np.minimum(np.asarray(f, dtype="float64"), F_CAP))


def hazard_windows(
    proj: pd.DataFrame,
    fam: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """日期族的逐窗口 hazard innovation。

    Args:
        proj: :func:`coherence.project_snapshots` 输出（含 ``p_proj``）。
        fam: 族表（``date_ladder`` 行使用 ``deadline_ts``）。
        windows: ``[trade_date, window, start_ts, end_ts]``（UTC 秒；窗口左闭右开）。

    Returns:
        ``[family_key, trade_date, window, d_lambda_front, d_lambda_max,
        front_deadline_ts, n_rungs]``。窗口两端任一侧无有效状态则为 NaN 行省略。
    """
    date_fam = fam.loc[fam["family_type"] == "date_ladder",
                       ["condition_id", "family_key", "deadline_ts"]].dropna()
    if date_fam.empty:
        return pd.DataFrame(
            columns=["family_key", "trade_date", "window", "d_lambda_front",
                     "d_lambda_max", "front_deadline_ts", "n_rungs"]
        )
    p = proj.merge(date_fam, on="condition_id", how="inner", suffixes=("", "_f"))
    # (family, deadline, boundary) -> F（同族同截止多市场时取均值；正常情形唯一）。
    key = ["family_key_f", "deadline_ts", "boundary_ts"]
    f_tab = p.groupby(key)["p_proj"].mean()

    rows: list[dict] = []
    for rec in windows.itertuples(index=False):
        start_ts, end_ts = int(rec.start_ts), int(rec.end_ts)
        for fkey in date_fam["family_key"].unique():
            try:
                sub = f_tab.loc[fkey]
            except KeyError:
                continue
            # 窗口结束时未到期的截止阶梯。
            deadlines = np.array(
                sorted({d for d, _ in sub.index if d > end_ts}), dtype="int64"
            )
            if deadlines.size == 0:
                continue
            # front 阶梯（最近未到期截止）必需；max 阶梯远端常稀薄 / 未创建，可选。
            lam: dict[str, float] = {"front": np.nan, "max": np.nan}
            for tag, dl in (("front", deadlines[0]), ("max", deadlines[-1])):
                try:
                    f0 = sub.loc[(dl, start_ts)]
                    f1 = sub.loc[(dl, end_ts)]
                except KeyError:
                    continue
                if np.isnan(f0) or np.isnan(f1):
                    continue
                lam[tag] = float(
                    cumulative_intensity(np.array([f1]))[0]
                    - cumulative_intensity(np.array([f0]))[0]
                )
            if np.isnan(lam["front"]):
                continue
            rows.append(
                {
                    "family_key": fkey,
                    "trade_date": rec.trade_date,
                    "window": rec.window,
                    "d_lambda_front": lam["front"],
                    "d_lambda_max": lam["max"],
                    "front_deadline_ts": int(deadlines[0]),
                    "n_rungs": int(deadlines.size),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=["family_key", "trade_date", "window", "d_lambda_front",
                     "d_lambda_max", "front_deadline_ts", "n_rungs"]
        )
    return out


def price_features_snapshot(
    strikes: np.ndarray, probs: np.ndarray, *, barrier: str
) -> dict[str, float]:
    """价格阈值族单时刻分布特征。

    Args:
        strikes: 升序行权价。
        probs: 对应（投影后）概率；``high`` 为生存函数（单调不增），
            ``low`` 为 CDF（单调不减）。
        barrier: ``high`` / ``low``。
    """
    strikes = np.asarray(strikes, dtype="float64")
    probs = np.asarray(probs, dtype="float64")
    surv = probs if barrier == "high" else 1.0 - probs  # 统一成生存函数
    tail = float(surv[-1])

    # 隐含中位数：S(K) = 0.5 的线性插值；越界取端点。
    if surv[0] <= 0.5:
        med = float(strikes[0])
    elif surv[-1] >= 0.5:
        med = float(strikes[-1])
    else:
        med = float(np.interp(0.5, surv[::-1], strikes[::-1]))

    mass = np.diff(np.concatenate([[1.0], surv])) * -1.0  # 各段质量
    mass = np.concatenate([mass, [surv[-1]]])
    mass = np.clip(mass, 1e-12, None)
    mass = mass / mass.sum()
    entropy = float(-(mass * np.log(mass)).sum())
    return {"tail_mass": tail, "implied_med": med, "entropy": entropy}


def price_windows(
    proj: pd.DataFrame,
    fam: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """价格阈值族的逐窗口分布特征 innovation。

    Returns:
        ``[family_key, trade_date, window, d_implied_med_log, d_tail_mass,
        d_entropy, n_strikes]``。
    """
    price_fam = fam.loc[fam["family_type"] == "price_ladder",
                        ["condition_id", "family_key", "strike", "barrier"]].dropna()
    price_fam = price_fam.rename(
        columns={"family_key": "family_key_f", "strike": "strike_f",
                 "barrier": "barrier_f"}
    )
    out_cols = ["family_key", "trade_date", "window", "d_implied_med_log",
                "d_tail_mass", "d_entropy", "n_strikes"]
    if price_fam.empty:
        return pd.DataFrame(columns=out_cols)
    p = proj.merge(price_fam, on="condition_id", how="inner")

    snap_cache: dict[tuple[str, int], dict[str, float] | None] = {}

    def snapshot(fkey: str, ts: int, blk: pd.DataFrame) -> dict[str, float] | None:
        cached = snap_cache.get((fkey, ts))
        if cached is not None or (fkey, ts) in snap_cache:
            return cached
        sub = blk.loc[blk["boundary_ts"] == ts].dropna(subset=["p_proj"])
        result: dict[str, float] | None = None
        if len(sub) >= 2:
            sub = sub.sort_values("strike_f")
            result = price_features_snapshot(
                sub["strike_f"].to_numpy(),
                sub["p_proj"].to_numpy(),
                barrier=sub["barrier_f"].iloc[0],
            )
        snap_cache[(fkey, ts)] = result
        return result

    rows: list[dict] = []
    for fkey, blk in p.groupby("family_key_f", sort=False):
        for rec in windows.itertuples(index=False):
            s0 = snapshot(fkey, int(rec.start_ts), blk)
            s1 = snapshot(fkey, int(rec.end_ts), blk)
            if s0 is None or s1 is None:
                continue
            rows.append(
                {
                    "family_key": fkey,
                    "trade_date": rec.trade_date,
                    "window": rec.window,
                    "d_implied_med_log": float(
                        np.log(max(s1["implied_med"], 1e-9))
                        - np.log(max(s0["implied_med"], 1e-9))
                    ),
                    "d_tail_mass": s1["tail_mass"] - s0["tail_mass"],
                    "d_entropy": s1["entropy"] - s0["entropy"],
                    "n_strikes": int(
                        blk.loc[blk["boundary_ts"] == int(rec.end_ts),
                                "p_proj"].notna().sum()
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=out_cols)
    return out
