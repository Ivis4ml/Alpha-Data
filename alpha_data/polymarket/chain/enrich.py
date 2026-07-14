"""把解码后的成交拼接为分析层（等价于公开数据集的 ``daily_aligned``，并补齐其缺失的市场）。

公开数据集 ``daily_aligned`` 的构造规则已由实证反推得到（见 ``docs/POLYMARKET_CRAWL.md`` 第 4 节），
共两步，且**只有市场级过滤，没有任何行级过滤**：

1. 剔除中继腿（``taker == 交易所合约``）。
2. 与市场元数据做**内连接**，且其元数据只覆盖非 negRisk 的二元市场，
   故 negRisk 市场（含 2024 美国大选那批旗舰市场）被整体丢弃。

实证：2024-11-05 当日，链上非中继腿成交 437,192 行，其中落在 ``daily_aligned`` 市场集合内的
恰为 43,844 行，与 ``daily_aligned`` 当日行数**完全相等**；被丢弃的 2,411 个 asset 全部属于
NegRisk 交易所。

本模块保留全部市场（含 negRisk 与多结果市场），并以 ``is_binary`` 标记二元市场。
``p_event`` 与 ``D`` 的定义只在二元市场上成立，故非二元市场置空，不做外推。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 分析层列序：前 24 列与公开数据集 daily_aligned 对齐，其后为本项目新增列。
ANALYSIS_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "block_timestamp",
    "price",
    "maker",
    "taker",
    "taker_direction",
    "usdc_amount",
    "fee_usdc",
    "condition_id",
    "outcome_seq",
    "neg_risk",
    "category",
    "outcome_label",
    "winning_outcome_label",
    "resolution_status",
    "taker_base_fee",
    "maker_base_fee",
    "opens_at",
    "close_at",
    "resolved_at",
    "market_slug",
    "p_event",
    "D",
    # 本项目新增
    "is_binary",
    "protocol",
    "exchange",
    "block_number",
    "log_index",
)


def drop_relay_legs(trades: pd.DataFrame) -> pd.DataFrame:
    """剔除中继腿。

    交易所以自身地址为对手方，为 taker 的整笔订单额外记一条聚合成交；其份额数恰等于
    对应各 maker 腿之和。保留会使成交量恰好翻倍，故清洗层必须剔除。
    """
    return trades.loc[~trades["is_relay"]].copy()


def keep_polymarket_venues(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按场馆归属拆分：只有经核验的 Polymarket 场馆进入分析层。

    按 topic0 扫描会连带抓到使用同一事件签名的其他协议（已实测到两个）。此处不静默丢弃，
    而是把非 Polymarket 与未判定的行单独返回，交由调用方记录或复核。

    Returns:
        ``(polymarket 行, 其余行)``。
    """
    if "venue_class" not in trades.columns:  # 兼容早期未带该列的分片
        return trades.copy(), trades.iloc[0:0].copy()
    keep = trades["venue_class"] == "polymarket"
    return trades.loc[keep].copy(), trades.loc[~keep].copy()


def attach_metadata(
    trades: pd.DataFrame,
    asset_map: pd.DataFrame,
    gamma: pd.DataFrame | None = None,
    resolutions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按 ``token_asset_id`` 连接市场元数据，并计算 ``p_event`` / ``D``。

    Args:
        trades: :mod:`alpha_data.polymarket.chain.decode` 的成交表（已剔除中继腿）。
        asset_map: :func:`alpha_data.polymarket.chain.metadata.build_asset_map` 的输出。
        gamma: Gamma 市场表（补 ``category`` / ``opens_at`` / ``close_at``），可选。
        resolutions: 链上 ``ConditionResolution`` 表（补 ``resolved_at``），可选。
            需含 ``condition_id`` 与 ``block_timestamp`` 两列。

    Returns:
        分析层 DataFrame。元数据缺失的成交以**左连接**保留（元数据列为空），
        不像公开数据集那样内连接丢弃；缺失以空值标注，不伪造。
    """
    df = trades.merge(
        asset_map, how="left", left_on="token_asset_id", right_on="asset_id", suffixes=("", "_m")
    )

    # 二元市场：该 condition 恰有 2 个结果。
    counts = asset_map.groupby("condition_id")["outcome_seq"].max()
    df["is_binary"] = df["condition_id"].map(counts).eq(2)

    if gamma is not None and not gamma.empty:
        df = df.merge(
            gamma[["condition_id", "category", "opens_at", "close_at"]],
            how="left",
            on="condition_id",
        )
    else:
        for col in ("category", "opens_at", "close_at"):
            df[col] = pd.NA

    if resolutions is not None and not resolutions.empty:
        res = resolutions[["condition_id", "block_timestamp"]].drop_duplicates("condition_id")
        res = res.rename(columns={"block_timestamp": "_resolved_ts"})
        df = df.merge(res, how="left", on="condition_id")
        df["resolved_at"] = pd.to_datetime(df["_resolved_ts"], unit="s", utc=True)
        df = df.drop(columns=["_resolved_ts"])
    else:
        df["resolved_at"] = pd.NaT
    df["resolution_status"] = np.where(df["resolved_at"].notna(), "resolved", None)

    # p_event：事件（第一个结果）发生的概率。仅二元市场有定义。
    # outcome_seq=1 的成交价即为事件概率；outcome_seq=2（互补份额）取 1 - price。
    seq = df["outcome_seq"]
    price = df["price"]
    p_event = np.where(seq == 1, price, np.where(seq == 2, 1.0 - price, np.nan))
    df["p_event"] = np.where(df["is_binary"], p_event, np.nan)

    # D：taker 的成交把事件概率推向哪一边。买入 Yes 或卖出 No 记 +1，反之 -1。
    # 实证真值表（2024-11 全月）：(BUY,1)->+1 (SELL,1)->-1 (BUY,2)->-1 (SELL,2)->+1。
    taker_sign = np.where(df["taker_direction"] == "BUY", 1, -1)
    outcome_sign = np.where(seq == 1, 1, np.where(seq == 2, -1, 0))
    d = taker_sign * outcome_sign
    df["D"] = np.where(df["is_binary"], d, 0).astype("int8")

    for col in ANALYSIS_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[list(ANALYSIS_COLUMNS)]
