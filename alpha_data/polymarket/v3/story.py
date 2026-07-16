"""Layer 3：story 因子聚合、平台公共因子与正交化（框架 §7.3）。

动机：Polymarket 全平台的资金 / 加密财富冲击会让不相关的事件市场同向波动，
把平台层面的共同创新误认作事件信息。处理：

1. 平台宇宙（全类别高流动市场，不限 CN 登记表）的窗口 innovation 按
   family（slug 事件干）聚合，市场级先做展开窗口标准化（防前视）。
2. 公共因子 ``F_PM`` 取 family 级序列的流动性加权截面均值（框架允许的
   稳健变体；相比 PC1 无载荷估计噪声、无全样本信息）。全样本 PC1 与
   ``F_PM`` 的相关作为诊断输出。
3. CN 主题信号对 ``F_PM`` 做**展开窗口**回归取残差 ``S_orth``（框架 §7.3 的
   ``S_perp``；展开窗口保证 Layer 5 的 OOS 评估无前视）。

数据隔离：全部构造只用 Polymarket 数据；标准化与回归均为展开窗口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 展开窗口统计的最小观测数（此前输出 NaN）。
MIN_OBS = 20


def expanding_zscore(s: pd.Series, *, min_obs: int = MIN_OBS) -> pd.Series:
    """展开窗口 z 分数（均值 / 标准差只用截至当前的观测，含当前）。"""
    mean = s.expanding(min_periods=min_obs).mean()
    std = s.expanding(min_periods=min_obs).std()
    return (s - mean) / std.replace(0.0, np.nan)


def family_series(
    market_windows: pd.DataFrame,
    market_meta: pd.DataFrame,
    *,
    min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    """市场级窗口 innovation -> family 级标准化序列。

    Args:
        market_windows: ``[condition_id, trade_date, window, s]``（滤波 logit 差）。
        market_meta: ``[condition_id, family_key, usdc_win]``。
        min_obs: 市场级展开标准化的最小观测数。

    Returns:
        ``[family_key, trade_date, window, s_family]``（family 内 sqrt(usdc) 加权，
        再对 family 序列做展开标准化）。
    """
    df = market_windows.merge(market_meta, on="condition_id", how="inner")
    df = df.sort_values("trade_date")
    df["s_std"] = (
        df.groupby(["condition_id", "window"], sort=False)["s"]
        .transform(lambda x: expanding_zscore(x, min_obs=min_obs))
    )
    df["w"] = np.sqrt(df["usdc_win"].clip(lower=1.0))
    df = df.dropna(subset=["s_std"])
    if df.empty:
        return pd.DataFrame(columns=["family_key", "trade_date", "window", "s_family"])

    def wmean(blk: pd.DataFrame) -> float:
        w = blk["w"].to_numpy()
        return float(np.sum(w * blk["s_std"].to_numpy()) / np.sum(w))

    fam = (
        df.groupby(["family_key", "trade_date", "window"], sort=False)
        .apply(wmean, include_groups=False)
        .rename("s_family")
        .reset_index()
    )
    fam = fam.sort_values("trade_date")
    fam["s_family"] = (
        fam.groupby(["family_key", "window"], sort=False)["s_family"]
        .transform(lambda x: expanding_zscore(x, min_obs=min_obs))
    )
    return fam.dropna(subset=["s_family"])


def common_factor(
    fam: pd.DataFrame,
    family_weights: pd.Series,
    *,
    min_families: int = 5,
) -> pd.DataFrame:
    """family 级序列 -> 平台公共因子 ``F_PM``（截面加权均值）。

    Args:
        fam: :func:`family_series` 输出。
        family_weights: ``family_key -> 权重``（如 sqrt(族总 usdc)）。
        min_families: 截面最少 family 数（不足时 ``F_PM`` 为 NaN）。

    Returns:
        ``[trade_date, window, f_pm, n_families]``。
    """
    df = fam.copy()
    df["w"] = df["family_key"].map(family_weights).fillna(1.0)

    def agg(blk: pd.DataFrame) -> pd.Series:
        w = blk["w"].to_numpy()
        v = blk["s_family"].to_numpy()
        f = float(np.sum(w * v) / np.sum(w)) if len(blk) >= min_families else np.nan
        return pd.Series({"f_pm": f, "n_families": len(blk)})

    out = (
        df.groupby(["trade_date", "window"], sort=False)
        .apply(agg, include_groups=False)
        .reset_index()
    )
    return out


def orthogonalize(
    theme_sig: pd.DataFrame,
    factor: pd.DataFrame,
    *,
    cols: tuple[str, ...],
    min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    """主题信号对 ``F_PM`` 的展开窗口正交化。

    对每个 ``(theme, product)`` 与每个信号列 ``s_*``：以窗口类型对应的 ``f_pm``
    为回归元，用**截至当前观测**的展开 OLS 系数取残差，输出 ``{col}_orth``。

    Args:
        theme_sig: ``[theme, product, trade_date, s_night, s_gap, s_day, ...]``。
        factor: :func:`common_factor` 输出（window ∈ {night, gap, day, pre}）。
        cols: 要正交化的信号列（列名须形如 ``s_{window}``）。
        min_obs: 展开回归最小观测数（此前残差 = 原值，标注低置信）。
    """
    out = theme_sig.copy().reset_index(drop=True)
    out = out.sort_values(["theme", "product", "trade_date"]).reset_index(drop=True)
    piv = factor.pivot_table(index="trade_date", columns="window", values="f_pm")
    for col in cols:
        window = col.removeprefix("s_")
        if window not in piv.columns:
            out[f"{col}_orth"] = out[col]
            continue
        f_all = out["trade_date"].map(piv[window]).to_numpy(dtype="float64")
        resid = np.full(len(out), np.nan)
        for _, idx in out.groupby(["theme", "product"], sort=False).groups.items():
            idx = np.asarray(idx)
            s = out.loc[idx, col].to_numpy(dtype="float64")
            f = f_all[idx]
            r = np.full(len(idx), np.nan)
            # 系数只用严格早于当前的观测（beta 先取后更，无当期信息）。
            sum_ff = 0.0
            sum_fs = 0.0
            n_eff = 0
            for i in range(len(idx)):
                if np.isnan(s[i]):
                    continue
                if np.isnan(f[i]):
                    r[i] = s[i]  # 因子缺测：不正交化，保留原值
                    continue
                beta = sum_fs / sum_ff if (n_eff >= min_obs and sum_ff > 0) else 0.0
                r[i] = s[i] - beta * f[i]
                sum_ff += f[i] * f[i]
                sum_fs += f[i] * s[i]
                n_eff += 1
            resid[idx] = r
        out[f"{col}_orth"] = resid
    return out
