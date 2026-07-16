"""Event Eligibility Gate 与 Power Gate（框架 §8 / §9）。

资格门控：事件类型由 :mod:`families` 标注；价格阈值市场
（``asset_price_threshold``）禁止进入 Layer 4（其结果是标的价格的确定函数，
"影响系数"退化为机械映射）。

功效门控：对每个（品种, 类别）组合核算最小可检测效应（MDE）：

    surprise   s_i  = Y_i - q_pre(i)          （结算结果减事前滤波概率）
    SE(beta)  约等于 sigma_R / sqrt(sum (s_i - s_bar)^2 / DE)
    MDE       = (z_{1-alpha/2} + z_{1-gamma}) * SE

设计效应 ``DE = 1 + (m_bar - 1) * rho``：同族同周结算的事件视为一个聚类
（重叠信息窗口），``rho`` 预注册取 0.5（保守）。

预注册经济效应上限 ``delta_econ``：单位 surprise（概率 0 -> 1）对应的收益效应
不超过品种日收益率标准差的 2 倍。``MDE > delta_econ`` 时按框架 §9.4 降级：
``category beta -> sign-only -> 关闭``。
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
