"""Layer 4：门控后的影响换算（框架 §10）。

只对通过双门控的（品种, 类别）估计 category-level 影响系数：

    R_{j, next(i)} = alpha_j + beta_{j,c} * s_i + eps

- ``s_i``：结算 surprise（:mod:`gates`）。
- ``R_{j, next(i)}``：结算时刻后品种 j 的**下一个可交易日**收盘收益（严格
  next-tradable 对齐，框架 §10.1；北京时间 09:00 前结算的当日即为下一可交易日）。
- 估计只用训练窗口，冻结后用于测试期（框架 §10.2 的滚动冻结在本实现中取
  单次切分——样本仍太小，多次滚动会让每折事件数不足，诚实标注）。

未通过功效门控的类别按框架 §9.4 使用 sign-only：

    Signal_sign = d * S_orth / sigma(S_orth 展开)

方向先验 ``d`` 已折叠在主题聚合的 orientation 权重内（聚合信号本身即
"利多为正"），故 ``d = +1``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def next_trading_return(
    resolved_at: pd.Series,
    returns: pd.DataFrame,
    *,
    product: str,
) -> pd.Series:
    """结算时刻 -> 下一个可交易日的 ``r_cc``。

    Args:
        resolved_at: UTC 时间戳序列。
        returns: ``[product, trade_date, r_cc]``。
        product: 品种。

    Returns:
        与输入同序的收益序列（无对应交易日为 NaN）。
    """
    days = returns.loc[returns["product"] == product, ["trade_date", "r_cc"]]
    days = days.dropna().sort_values("trade_date").reset_index(drop=True)
    dates = days["trade_date"].to_numpy()
    cn = pd.to_datetime(resolved_at, utc=True).dt.tz_convert("Asia/Shanghai")
    # 北京 09:00 前结算：当日日盘尚未开盘，当日即 next-tradable；其后为次一交易日。
    cutoff_date = np.where(
        cn.dt.hour < 9,
        cn.dt.strftime("%Y-%m-%d"),
        (cn + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
    )
    idx = np.searchsorted(dates, cutoff_date, side="left")
    out = np.full(len(resolved_at), np.nan)
    ok = idx < len(dates)
    out[ok] = days["r_cc"].to_numpy()[idx[ok]]
    return pd.Series(out, index=resolved_at.index)


def fit_category_beta(
    surp: pd.DataFrame,
    returns: pd.DataFrame,
    power: pd.DataFrame,
    *,
    train_end: str,
) -> pd.DataFrame:
    """训练窗口内的 category-level 影响系数（仅功效通过的组合）。

    Args:
        surp: :func:`gates.surprises` 输出。
        returns: ``[product, trade_date, r_cc]``。
        power: :func:`gates.power_table` 输出。
        train_end: 训练窗右端（``YYYY-MM-DD``，结算日期严格早于该日的事件入训）。

    Returns:
        ``[theme, product, beta, t_stat, n_train, mode]``；``mode`` 继承功效表
        决策（``category_beta`` 组合给出估计值，其余 beta 为 NaN）。
    """
    rows: list[dict] = []
    for rec in power.itertuples(index=False):
        blk = surp.loc[(surp["theme"] == rec.theme) & (surp["product"] == rec.product)]
        blk = blk.copy()
        blk["r_next"] = next_trading_return(
            blk["resolved_at"], returns, product=rec.product
        )
        res_date = pd.to_datetime(blk["resolved_at"], utc=True).dt.tz_convert(
            "Asia/Shanghai").dt.strftime("%Y-%m-%d")
        train = blk.loc[(res_date < train_end) & blk["r_next"].notna()]
        beta = t_stat = np.nan
        if rec.decision == "category_beta" and len(train) >= 8:
            x = train["surprise"].to_numpy(dtype="float64")
            y = train["r_next"].to_numpy(dtype="float64")
            xc = x - x.mean()
            denom = float(np.sum(xc**2))
            if denom > 0:
                beta = float(np.sum(xc * (y - y.mean())) / denom)
                resid = y - y.mean() - beta * xc
                dof = max(len(train) - 2, 1)
                se = float(np.sqrt(np.sum(resid**2) / dof / denom))
                t_stat = beta / se if se > 0 else np.nan
        rows.append(
            {
                "theme": rec.theme, "product": rec.product,
                "beta": beta, "t_stat": t_stat,
                "n_train": int(len(train)),
                "mode": rec.decision if np.isfinite(beta) else "sign_only",
            }
        )
    return pd.DataFrame(rows)


def impact_signal(
    theme_sig: pd.DataFrame,
    betas: pd.DataFrame,
    *,
    col: str = "s_pre_orth",
    min_obs: int = 20,
) -> pd.DataFrame:
    """门控影响信号：``category_beta`` 组合乘冻结系数，其余 sign-only 标准化。

    Args:
        theme_sig: 含 ``theme / product / trade_date / {col}`` 的主题信号表。
        betas: :func:`fit_category_beta` 输出。
        col: 输入信号列。
        min_obs: sign-only 展开标准差的最小观测数。

    Returns:
        输入表加列 ``impact``（收益单位或标准化符号单位）与 ``impact_mode``。
    """
    out = theme_sig.sort_values(["theme", "product", "trade_date"]).reset_index(drop=True)
    key = betas.set_index(["theme", "product"])
    impact = np.full(len(out), np.nan)
    modes = np.full(len(out), "none", dtype=object)
    for (theme, product), idx in out.groupby(["theme", "product"], sort=False).groups.items():
        idx = np.asarray(idx)
        s = out.loc[idx, col].astype("float64")
        try:
            b = key.loc[(theme, product)]
        except KeyError:
            continue
        if b["mode"] == "category_beta" and np.isfinite(b["beta"]):
            impact[idx] = float(b["beta"]) * s.to_numpy()
            modes[idx] = "category_beta"
        else:
            sd = s.expanding(min_periods=min_obs).std().to_numpy()
            impact[idx] = s.to_numpy() / sd
            modes[idx] = "sign_only"
    out["impact"] = impact
    out["impact_mode"] = modes
    return out
