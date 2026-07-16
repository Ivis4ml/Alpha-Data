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


def _ols_beta(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """一元 OLS：返回 (beta, 常规 t, 残差)。"""
    xc = x - x.mean()
    denom = float(np.sum(xc**2))
    if denom <= 0:
        return np.nan, np.nan, np.full(len(y), np.nan)
    beta = float(np.sum(xc * (y - y.mean())) / denom)
    resid = y - y.mean() - beta * xc
    dof = max(len(y) - 2, 1)
    se = float(np.sqrt(np.sum(resid**2) / dof / denom))
    return beta, (beta / se if se > 0 else np.nan), resid


def wild_bootstrap_p(
    x: np.ndarray, y: np.ndarray, *, n_boot: int = 4999, seed: int = 7
) -> float:
    """Rademacher wild bootstrap 的双侧 p 值（H0: beta = 0，残差重加权）。"""
    beta, _, _ = _ols_beta(x, y)
    if not np.isfinite(beta):
        return np.nan
    # H0 下的残差：y 对常数回归的残差。
    e0 = y - y.mean()
    rng = np.random.default_rng(seed)
    xc = x - x.mean()
    denom = float(np.sum(xc**2))
    count = 0
    for _ in range(n_boot):
        w = rng.choice([-1.0, 1.0], size=len(y))
        yb = y.mean() + e0 * w
        bb = float(np.sum(xc * (yb - yb.mean())) / denom)
        if abs(bb) >= abs(beta):
            count += 1
    return (count + 1) / (n_boot + 1)


def fit_category_beta_episode(
    ep: pd.DataFrame,
    returns: pd.DataFrame,
    power: pd.DataFrame,
    *,
    train_end: str,
) -> pd.DataFrame:
    """episode 级影响系数（主口径）+ 支撑域与稳健性诊断。

    Args:
        ep: :func:`gates.episode_aggregate` 输出（``ev_date`` 即收益日）。
        returns: ``[product, trade_date, r_cc]``。
        power: episode 级功效表（ex-ante）。
        train_end: 训练窗右端（``ev_date`` 严格早于该日的 episode 入训）。

    Returns:
        逐（主题, 品种）一行：``beta / t_ols / p_wild / n_episodes / n_markets /
        sd_s / iqr_s / beta_x_iqr / beta_drop_top1 / beta_drop_top3 /
        loo_beta_min / loo_beta_max / mode``。
    """
    rows: list[dict] = []
    for rec in power.itertuples(index=False):
        blk = ep.loc[(ep["theme"] == rec.theme) & (ep["product"] == rec.product)]
        rmap = returns.loc[returns["product"] == rec.product].set_index(
            "trade_date")["r_cc"]
        blk = blk.assign(r_next=blk["ev_date"].map(rmap))
        train = blk.loc[(blk["ev_date"] < train_end) & blk["r_next"].notna()]
        out = {
            "theme": rec.theme, "product": rec.product,
            "n_episodes": int(len(train)),
            "n_markets": int(train["n_markets"].sum()) if len(train) else 0,
            "beta": np.nan, "t_ols": np.nan, "p_wild": np.nan,
            "sd_s": np.nan, "iqr_s": np.nan, "beta_x_iqr": np.nan,
            "beta_drop_top1": np.nan, "beta_drop_top3": np.nan,
            "loo_beta_min": np.nan, "loo_beta_max": np.nan,
            "mode": rec.decision,
        }
        if rec.decision == "category_beta" and len(train) >= 8:
            x = train["surprise"].to_numpy(dtype="float64")
            y = train["r_next"].to_numpy(dtype="float64")
            beta, t_ols, _ = _ols_beta(x, y)
            out.update(
                beta=beta, t_ols=t_ols,
                p_wild=wild_bootstrap_p(x, y),
                sd_s=float(np.std(x, ddof=1)),
                iqr_s=float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
            )
            out["beta_x_iqr"] = beta * out["iqr_s"]
            order = np.argsort(-np.abs(x))
            for k, key in ((1, "beta_drop_top1"), (3, "beta_drop_top3")):
                if len(x) > k + 4:
                    keep = np.ones(len(x), bool)
                    keep[order[:k]] = False
                    out[key] = _ols_beta(x[keep], y[keep])[0]
            loo = [
                _ols_beta(np.delete(x, i), np.delete(y, i))[0]
                for i in range(len(x))
            ]
            out["loo_beta_min"] = float(np.nanmin(loo))
            out["loo_beta_max"] = float(np.nanmax(loo))
            if not np.isfinite(beta):
                out["mode"] = "sign_only"
        elif rec.decision == "category_beta":
            out["mode"] = "sign_only"
        rows.append(out)
    return pd.DataFrame(rows)


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
