"""Layer 2：族内逻辑一致性软投影与 narrative tension（框架 §6.2 / §6.3）。

只使用**程序可验证的单调约束**（框架的"硬约束仅限程序验证关系"原则在本数据下
可用的子集）：

- 日期阶梯族：``P(事件 <= T_1) <= P(事件 <= T_2)``（T_1 < T_2）。
- 价格 ``hit-high`` 族（同截止）：``P(max >= K)`` 对行权价 K 单调不增。
- 价格 ``hit-low`` 族（同截止）：``P(min <= K)`` 对 K 单调不减。

投影为加权等张回归（PAVA），权重取市场流动性的平方根——流动性高的市场
概率更可信，被移动得更少。这是 KL 投影（框架式 2）在小偏差下的二阶近似，
选择二次距离是因为 PAVA 有精确解、无超参数。

Tension（框架 §6.3 的 raw 轨；本管道不做类别校准，故无 calibrated 轨）：
族内加权均方投影距离。校准分支按框架 §6.1 的 overlap gate 要求 resolved 样本
支撑，本窗口地缘类不满足，按预注册降级为不校准（诚实记录，不是遗漏）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pava(y: np.ndarray, w: np.ndarray, *, increasing: bool = True) -> np.ndarray:
    """加权等张回归（pool adjacent violators），返回单调拟合值。

    Args:
        y: 观测值（按自变量升序排列）。
        w: 正权重。
        increasing: 拟合方向。
    """
    if not increasing:
        return -pava(-y, w, increasing=True)
    y = np.asarray(y, dtype="float64")
    w = np.asarray(w, dtype="float64")
    # 块表示：值、权重、元素个数。
    vals: list[float] = []
    wts: list[float] = []
    cnts: list[int] = []
    for yi, wi in zip(y, w, strict=True):
        vals.append(float(yi))
        wts.append(float(wi))
        cnts.append(1)
        while len(vals) >= 2 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]
            cnts[-2] += cnts[-1]
            vals[-2] = v
            del vals[-1], wts[-1], cnts[-1]
    out = np.empty_like(y)
    i = 0
    for v, c in zip(vals, cnts, strict=True):
        out[i : i + c] = v
        i += c
    return out


def project_family(
    snap: pd.DataFrame,
    *,
    family_type: str,
    barrier: str | None = None,
) -> pd.DataFrame:
    """单个族在单一时刻的快照 -> 投影概率与 tension。

    Args:
        snap: ``[condition_id, order_key, p, w]``；``order_key`` 为族内排序变量
            （日期族为 deadline_ts，价格族为 strike），``p`` 为滤波概率，``w`` 为权重。
        family_type: ``date_ladder`` / ``price_ladder``。
        barrier: 价格族的 ``high`` / ``low``。

    Returns:
        ``snap`` 加列 ``p_proj``；``attrs['tension']`` 为加权均方投影距离。
        少于 2 个成员时投影恒等、tension 为 0。
    """
    snap = snap.sort_values("order_key").reset_index(drop=True)
    p = snap["p"].to_numpy(dtype="float64")
    w = snap["w"].to_numpy(dtype="float64")
    if len(snap) < 2:
        snap["p_proj"] = p
        snap.attrs["tension"] = 0.0
        return snap
    if family_type == "date_ladder":
        increasing = True
    elif family_type == "price_ladder":
        increasing = barrier == "low"
    else:
        snap["p_proj"] = p
        snap.attrs["tension"] = 0.0
        return snap
    q = pava(p, w, increasing=increasing)
    q = np.clip(q, 0.0, 1.0)
    snap["p_proj"] = q
    snap.attrs["tension"] = float(np.sum(w * (p - q) ** 2) / np.sum(w))
    return snap


def project_snapshots(
    states_at: pd.DataFrame,
    fam: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    """全部族 × 全部边界时刻的批量投影。

    Args:
        states_at: ``[condition_id, boundary_ts, z]``（滤波状态 LOCF，z 为 logit）。
        fam: :func:`families.build_families` 输出。
        weights: ``condition_id -> 权重``（如 sqrt(usdc_win)）。

    Returns:
        ``[condition_id, boundary_ts, p_filt, p_proj, z_proj, family_key,
        family_type, tension]``；不属于可投影族（或族内单成员）的市场
        ``p_proj = p_filt``、``tension = 0``。
    """
    df = states_at.merge(
        fam[["condition_id", "family_type", "family_key", "strike", "barrier",
             "deadline_ts"]],
        on="condition_id",
        how="left",
    )
    df["p_filt"] = 1.0 / (1.0 + np.exp(-df["z"]))
    df["w"] = df["condition_id"].map(weights).fillna(1.0)
    df["p_proj"] = df["p_filt"]
    df["tension"] = 0.0

    projectable = df["family_type"].isin(["date_ladder", "price_ladder"]) & df["z"].notna()
    sub = df.loc[projectable].copy()
    sub["order_key"] = np.where(
        sub["family_type"] == "price_ladder",
        sub["strike"],
        sub["deadline_ts"].astype("float64"),
    )
    for _key, blk in sub.groupby(["family_key", "boundary_ts"], sort=False):
        if len(blk) < 2 or blk["order_key"].isna().any():
            continue
        ftype = blk["family_type"].iloc[0]
        barrier = blk["barrier"].iloc[0] if ftype == "price_ladder" else None
        snap = blk[["condition_id", "order_key", "w"]].copy()
        snap["p"] = blk["p_filt"].to_numpy()
        proj = project_family(snap, family_type=ftype, barrier=barrier)
        keyed = proj.set_index("condition_id")
        idx = blk.index
        df.loc[idx, "p_proj"] = keyed["p_proj"].reindex(df.loc[idx, "condition_id"]).to_numpy()
        df.loc[idx, "tension"] = proj.attrs["tension"]

    eps = 1e-6
    p = np.clip(df["p_proj"].to_numpy(dtype="float64"), eps, 1.0 - eps)
    df["z_proj"] = np.log(p / (1.0 - p))
    return df[["condition_id", "boundary_ts", "p_filt", "p_proj", "z_proj",
               "family_key", "family_type", "tension"]]
