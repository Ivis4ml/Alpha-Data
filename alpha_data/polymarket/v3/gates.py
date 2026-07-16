"""Event Eligibility Gate 与 Power Gate（框架 §8 / §9）。

资格门控：事件类型由 :mod:`families` 标注；价格阈值市场
（``asset_price_threshold``）禁止进入 Layer 4（其结果是标的价格的确定函数，
"影响系数"退化为机械映射）。

功效门控：对每个（品种, 类别）组合核算最小可检测效应（MDE）：

    surprise   s_i  = Y_i - q_pre(i)          （结算结果减事前滤波概率）
    SE(beta)  约等于 sigma_R / sqrt(sum (s_i - s_bar)^2 / DE)
    MDE       = (z_{1-alpha/2} + z_{1-gamma}) * SE

**episode 聚类（v3.1 修订）**：同一现实事件常让整族日期阶梯同时结算——
实测 mideast×SC 的 191 个"市场事件"只对应 97 个收益日，单日最多 25 个市场
共享同一个次日收益。市场级行不是独立样本；真正的样本单位是**下一可交易日**
（episode）。主口径把 surprise 按 (品种, 下一可交易日) 聚合为 episode 级
（同日收益完全相同 = 聚类内相关为 1，聚合等价于最保守的处理），MDE 直接在
episode 设计上核算。family × ISO 周 + rho=0.5 的旧口径保留作对比
（:func:`power_table`，已知偏乐观）。

预注册经济效应上限 ``delta_econ``：单位 surprise（概率 0 -> 1）对应的收益效应
不超过品种日收益率标准差的 2 倍。``MDE > delta_econ`` 时按框架 §9.4 降级：
``category beta -> sign-only -> 关闭``。

**ex-ante / ex-post 分离**：决定 OOS 管道用 ``category_beta`` 还是 ``sign-only``
的门控只能使用训练截止前已结算的事件（ex-ante）；全样本版本（ex-post）仅
回答"以当前数据量哪些参数可研究"，不得反向改变模型选择。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

#: 显著性与功效（框架附录 A：80% 功效）。
ALPHA = 0.05
POWER = 0.80

#: 聚类内相关（预注册，保守）。
RHO_CLUSTER = 0.5

#: 经济效应上限：单位 surprise 的收益效应不超过 K_ECON 倍日波动。
K_ECON = 2.0


def surprises(
    registry: pd.DataFrame,
    fam: pd.DataFrame,
    z_pre: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """已结算合格事件的 surprise 表。

    Args:
        registry: v3 登记表（``condition_id / theme / product / resolved_at``）。
        fam: 族表（取 ``layer4_eligible`` 与 ``family_key``）。
        z_pre: ``[condition_id, z_pre]``（结算前 24h 的滤波 logit，LOCF）。
        outcomes: ``[condition_id, y]``（事件实现 1 / 0）。

    Returns:
        ``[condition_id, theme, product, family_key, resolved_at, q_pre, y,
        surprise, cluster]``；cluster = family × 结算 ISO 周。
    """
    df = registry.merge(
        fam[["condition_id", "family_key", "layer4_eligible"]],
        on="condition_id", how="left",
    )
    df = df.loc[df["layer4_eligible"].fillna(False)]
    df = df.merge(z_pre, on="condition_id", how="inner")
    df = df.merge(outcomes, on="condition_id", how="inner")
    df = df.dropna(subset=["z_pre", "y", "resolved_at"]).copy()
    df["q_pre"] = 1.0 / (1.0 + np.exp(-df["z_pre"]))
    df["surprise"] = df["y"].astype("float64") - df["q_pre"]
    week = pd.to_datetime(df["resolved_at"], utc=True).dt.strftime("%G-%V")
    df["cluster"] = df["family_key"].fillna(df["condition_id"]) + ":" + week
    return df[["condition_id", "theme", "product", "family_key", "resolved_at",
               "q_pre", "y", "surprise", "cluster"]]


def attach_event_date(surp: pd.DataFrame, trading_days: list[str]) -> pd.DataFrame:
    """给 surprise 表附加**下一可交易日** ``ev_date``（episode 键）。

    北京时间 09:00 前结算的事件，其收益窗为当日；其后为次一交易日。周末 /
    假日结算自动归并到下一交易日——这正是"多个市场共享同一收益观测"的
    聚类结构。
    """
    days = np.array(sorted(trading_days))
    cn = pd.to_datetime(surp["resolved_at"], utc=True).dt.tz_convert("Asia/Shanghai")
    cutoff = np.where(
        cn.dt.hour < 9,
        cn.dt.strftime("%Y-%m-%d"),
        (cn + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
    )
    idx = np.searchsorted(days, cutoff, side="left")
    out = surp.copy()
    ev = np.full(len(surp), None, dtype=object)
    ok = idx < len(days)
    ev[ok] = days[idx[ok]]
    out["ev_date"] = ev
    return out.dropna(subset=["ev_date"])


def episode_aggregate(surp: pd.DataFrame, *, agg: str = "mean") -> pd.DataFrame:
    """市场级 surprise -> episode（品种 × 下一可交易日）级。

    Args:
        surp: :func:`attach_event_date` 输出。
        agg: ``mean``（episode 内均值，主口径）或 ``sum``（可加冲击假设，
            稳健性口径）。

    Returns:
        ``[theme, product, ev_date, surprise, n_markets]``。
    """
    fn = "mean" if agg == "mean" else "sum"
    g = surp.groupby(["theme", "product", "ev_date"])
    out = g.agg(surprise=("surprise", fn), n_markets=("surprise", "size"))
    return out.reset_index()


def power_table_episode(
    surp_ev: pd.DataFrame,
    sigma_r: pd.Series,
    *,
    alpha: float = ALPHA,
    power: float = POWER,
    k_econ: float = K_ECON,
    agg: str = "mean",
) -> pd.DataFrame:
    """episode 级功效表（主口径；聚合后各行的收益观测相互独立）。

    Returns:
        ``[theme, product, n_markets, n_episodes, sum_s2, mde, delta_econ,
        decision]``。
    """
    ep = episode_aggregate(surp_ev, agg=agg)
    z_a = stats.norm.ppf(1.0 - alpha / 2.0)
    z_g = stats.norm.ppf(power)
    rows: list[dict] = []
    for (theme, product), blk in ep.groupby(["theme", "product"]):
        s = blk["surprise"].to_numpy(dtype="float64")
        n_ep = len(s)
        sum_s2 = float(np.sum((s - s.mean()) ** 2))
        sig = float(sigma_r.get(product, np.nan))
        if n_ep >= 3 and sum_s2 > 0 and np.isfinite(sig):
            mde = (z_a + z_g) * sig / np.sqrt(sum_s2)
        else:
            mde = np.inf
        delta = k_econ * sig if np.isfinite(sig) else np.nan
        rows.append(
            {
                "theme": theme, "product": product,
                "n_markets": int(blk["n_markets"].sum()),
                "n_episodes": n_ep, "sum_s2": sum_s2,
                "mde": mde, "delta_econ": delta,
                "decision": "category_beta" if mde <= delta else "sign_only",
            }
        )
    return pd.DataFrame(rows).sort_values(["product", "theme"]).reset_index(drop=True)


def power_table(
    surp: pd.DataFrame,
    sigma_r: pd.Series,
    *,
    alpha: float = ALPHA,
    power: float = POWER,
    rho: float = RHO_CLUSTER,
    k_econ: float = K_ECON,
) -> pd.DataFrame:
    """逐（品种, 主题）功效表与降级决策（框架 §9.3 表）。

    Args:
        surp: :func:`surprises` 输出。
        sigma_r: ``product -> 日收益标准差``。
        alpha / power: 双侧显著性与目标功效。
        rho: 聚类内相关。
        k_econ: 经济效应上限倍数。

    Returns:
        ``[theme, product, n_events, n_clusters, sum_s2, design_effect, n_eff,
        mde, delta_econ, decision]``；decision ∈ {category_beta, sign_only}。
    """
    z_a = stats.norm.ppf(1.0 - alpha / 2.0)
    z_g = stats.norm.ppf(power)
    rows: list[dict] = []
    for (theme, product), blk in surp.groupby(["theme", "product"]):
        s = blk["surprise"].to_numpy(dtype="float64")
        n = len(s)
        n_clusters = blk["cluster"].nunique()
        m_bar = n / max(n_clusters, 1)
        de = 1.0 + (m_bar - 1.0) * rho
        sum_s2 = float(np.sum((s - s.mean()) ** 2))
        sig = float(sigma_r.get(product, np.nan))
        if n >= 3 and sum_s2 > 0 and np.isfinite(sig):
            se = sig / np.sqrt(sum_s2 / de)
            mde = (z_a + z_g) * se
        else:
            mde = np.inf
        delta = k_econ * sig if np.isfinite(sig) else np.nan
        decision = "category_beta" if mde <= delta else "sign_only"
        rows.append(
            {
                "theme": theme, "product": product,
                "n_events": n, "n_clusters": n_clusters,
                "sum_s2": sum_s2, "design_effect": de,
                "n_eff": n / de, "mde": mde, "delta_econ": delta,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows).sort_values(["product", "theme"]).reset_index(drop=True)
