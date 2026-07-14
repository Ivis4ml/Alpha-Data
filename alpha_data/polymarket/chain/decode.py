"""链上日志解码：两版 ``OrderFilled`` 与 ConditionalTokens 生命周期事件。

解码口径已与公开数据集 ``TimeSeventeen/Polymarket-v1`` 做过逐字段比对（区块 63732941，
24 行 x 13 列全等，复现脚本 ``scripts/verify_polymarket_reproduction.py``）。

两版 ``OrderFilled`` 的语义差异（均为 **maker 视角**）：

旧版 ``OrderFilled(bytes32 orderHash, address maker, address taker, uint256 makerAssetId,
uint256 takerAssetId, uint256 makerAmountFilled, uint256 takerAmountFilled, uint256 fee)``
方向由 ``makerAssetId == 0`` 推断：为 0 说明 maker 付出抵押品、收到份额，即 maker 买入。

新版 ``OrderFilled(bytes32 orderHash, address maker, address taker, uint8 side,
uint256 assetId, uint256 makerAmountFilled, uint256 takerAmountFilled, uint256 fee,
bytes32, bytes32)`` 只带单个 ``assetId``，方向由 ``side`` 显式给出（0 = maker 买入，
1 = maker 卖出）。因此 ``makerAmountFilled`` / ``takerAmountFilled`` 的币种随 side 互换：
side=0 时前者为抵押品、后者为份额；side=1 时相反。以此规则解码，30,124 条实测样本的
``price = usdc / token`` 全部落在 [0, 1]，无一越界；若忽略 side 直接按固定顺序解码，
则恰好全部 side=1 的行越界。末两个 ``bytes32`` 于成交重建无用，不解码。

中继腿（relay leg）：交易所以自身地址作为对手方，为 taker 的整笔订单再记一条聚合成交。
判定规则为 ``taker == 发出该日志的合约地址``，新旧协议通用。中继腿的份额数恰等于其
对应各 maker 腿之和（已用 Data API 的公开成交口径核对），保留会使成交量恰好翻倍。
"""

from __future__ import annotations

from typing import Any, Final

from eth_abi import decode as abi_decode
from eth_utils import to_checksum_address

from alpha_data.polymarket.chain import venues

_SCALE: Final[float] = float(10**venues.COLLATERAL_DECIMALS)

# 解码后统一的成交表列序（前 13 列与公开数据集 OrderFilled 层逐列一致）。
TRADE_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "maker",
    "taker",
    "block_timestamp",
    "maker_asset_id",
    "taker_asset_id",
    "maker_direction",
    "taker_direction",
    "token_asset_id",
    "token_amount",
    "usdc_amount",
    "price",
    "fee_usdc",
    # 以下为本项目新增（公开数据集没有），用于多协议共存与审计。
    "block_number",
    "log_index",
    "tx_hash",
    "exchange",
    "venue_class",
    "protocol",
    "is_relay",
)

_SPLIT_LIKE = (
    "id",
    "stakeholder",
    "collateral_token",
    "parent_collection_id",
    "condition_id",
    "partition",
    "usdc_amount",
)
CTF_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "splits": _SPLIT_LIKE,
    "merges": _SPLIT_LIKE,
    "redemptions": (
        "id",
        "redeemer",
        "collateral_token",
        "parent_collection_id",
        "condition_id",
        "index_sets",
        "usdc_amount",
    ),
    "preparations": ("id", "condition_id", "oracle", "question_id", "outcome_slot_count"),
    "resolutions": (
        "id",
        "condition_id",
        "oracle",
        "question_id",
        "outcome_slot_count",
        "payout_numerators",
    ),
}


def _addr(topic: str) -> str:
    """32 字节 topic -> 校验和格式地址（与公开数据集的大小写口径一致）。"""
    return to_checksum_address("0x" + topic[-40:])


def _event_id(log: dict[str, Any]) -> str:
    """事件主键 ``{chainId}_{blockNumber}_{logIndex}``（与公开数据集同构，天然幂等去重）。"""
    return f"{venues.CHAIN_ID}_{int(log['blockNumber'], 16)}_{int(log['logIndex'], 16)}"


def _timestamp(log: dict[str, Any], ts_by_block: dict[int, int] | None) -> int:
    """取日志的区块时间戳：优先用端点自带的 ``blockTimestamp``，否则查外部表。"""
    raw = log.get("blockTimestamp")
    if raw is not None:
        return int(raw, 16) if isinstance(raw, str) else int(raw)
    if ts_by_block is None:
        raise KeyError("日志不带 blockTimestamp 且未提供区块时间戳表")
    return ts_by_block[int(log["blockNumber"], 16)]


def _trade_row(
    log: dict[str, Any],
    *,
    maker_buys: bool,
    token_asset_id: int,
    token_raw: int,
    usdc_raw: int,
    fee_raw: int,
    protocol: str,
    ts_by_block: dict[int, int] | None,
) -> dict[str, Any]:
    """把已抽出的字段组装为统一成交行。"""
    exchange = log["address"].lower()
    taker = _addr(log["topics"][3])
    token_amount = token_raw / _SCALE
    usdc_amount = usdc_raw / _SCALE
    return {
        "id": _event_id(log),
        "maker": _addr(log["topics"][2]),
        "taker": taker,
        "block_timestamp": _timestamp(log, ts_by_block),
        # maker 视角的 give / get：买入时付抵押品（资产 id 记 0），卖出时付份额。
        "maker_asset_id": "0" if maker_buys else str(token_asset_id),
        "taker_asset_id": str(token_asset_id) if maker_buys else "0",
        "maker_direction": "BUY" if maker_buys else "SELL",
        "taker_direction": "SELL" if maker_buys else "BUY",
        "token_asset_id": str(token_asset_id),
        "token_amount": token_amount,
        "usdc_amount": usdc_amount,
        "price": (usdc_amount / token_amount) if token_amount else None,
        "fee_usdc": fee_raw / _SCALE,
        "block_number": int(log["blockNumber"], 16),
        "log_index": int(log["logIndex"], 16),
        "tx_hash": log["transactionHash"],
        "exchange": exchange,
        # 同名事件签名亦被其他协议使用，故须标注场馆归属；分析层只取 polymarket。
        "venue_class": venues.venue_class(exchange),
        "protocol": protocol,
        # 中继腿：交易所以自身地址为对手方为 taker 的整笔订单再记一条聚合成交。
        "is_relay": taker.lower() == exchange,
    }


def decode_order_filled_v1(
    log: dict[str, Any], ts_by_block: dict[int, int] | None = None
) -> dict[str, Any]:
    """解码旧版 ``OrderFilled``（2022-11-21 .. 2026-04-28，抵押品 USDC.e）。"""
    maker_asset, taker_asset, maker_amt, taker_amt, fee = abi_decode(
        ["uint256"] * 5, bytes.fromhex(log["data"][2:])
    )
    maker_buys = maker_asset == 0
    if maker_buys:  # maker 付抵押品、收份额
        token_asset, token_raw, usdc_raw = taker_asset, taker_amt, maker_amt
    else:  # maker 付份额、收抵押品
        token_asset, token_raw, usdc_raw = maker_asset, maker_amt, taker_amt
    return _trade_row(
        log,
        maker_buys=maker_buys,
        token_asset_id=token_asset,
        token_raw=token_raw,
        usdc_raw=usdc_raw,
        fee_raw=fee,
        protocol="v1",
        ts_by_block=ts_by_block,
    )


def decode_order_filled_v2(
    log: dict[str, Any], ts_by_block: dict[int, int] | None = None
) -> dict[str, Any]:
    """解码新版 ``OrderFilled``（2026-04-03 起，抵押品 pUSD）。"""
    side, asset_id, maker_amt, taker_amt, fee, _h1, _h2 = abi_decode(
        ["uint8", "uint256", "uint256", "uint256", "uint256", "bytes32", "bytes32"],
        bytes.fromhex(log["data"][2:]),
    )
    maker_buys = side == 0
    if maker_buys:  # side=0：maker 付抵押品、收份额
        usdc_raw, token_raw = maker_amt, taker_amt
    else:  # side=1：maker 付份额、收抵押品
        token_raw, usdc_raw = maker_amt, taker_amt
    return _trade_row(
        log,
        maker_buys=maker_buys,
        token_asset_id=asset_id,
        token_raw=token_raw,
        usdc_raw=usdc_raw,
        fee_raw=fee,
        protocol="v2",
        ts_by_block=ts_by_block,
    )


def decode_trade(
    log: dict[str, Any], ts_by_block: dict[int, int] | None = None
) -> dict[str, Any] | None:
    """按 topic0 分派到对应版本的解码器；非成交事件返回 ``None``。"""
    topic0 = log["topics"][0]
    if topic0 == venues.TOPIC_ORDER_FILLED_V1:
        return decode_order_filled_v1(log, ts_by_block)
    if topic0 == venues.TOPIC_ORDER_FILLED_V2:
        return decode_order_filled_v2(log, ts_by_block)
    return None


# --- ConditionalTokens 生命周期事件 ----------------------------------------


def decode_ctf(
    name: str, log: dict[str, Any], ts_by_block: dict[int, int] | None = None
) -> dict[str, Any]:
    """解码 ConditionalTokens 的五类事件之一。

    Args:
        name: ``splits`` / ``merges`` / ``redemptions`` / ``preparations`` / ``resolutions``。
        log: 原始日志对象。
        ts_by_block: 区块时间戳表（本层输出与公开数据集一致，不含时间戳列，仅用于校验）。

    Returns:
        与公开数据集 ``CTF`` 层同构的一行。
    """
    data = bytes.fromhex(log["data"][2:])
    row: dict[str, Any] = {"id": _event_id(log)}
    if name in ("splits", "merges"):
        # PositionSplit / PositionsMerge(address stakeholder [indexed], IERC20 collateralToken,
        #   bytes32 parentCollectionId [indexed], bytes32 conditionId [indexed],
        #   uint256[] partition, uint256 amount)
        # 注意 collateralToken 在此二者中是非索引字段，与 PayoutRedemption 相反（已实测）。
        collateral, partition, amount = abi_decode(["address", "uint256[]", "uint256"], data)
        row.update(
            stakeholder=_addr(log["topics"][1]),
            collateral_token=to_checksum_address(collateral),
            parent_collection_id=log["topics"][2],
            condition_id=log["topics"][3],
            partition=[str(x) for x in partition],
            usdc_amount=amount / _SCALE,
        )
    elif name == "redemptions":
        # PayoutRedemption(address redeemer [indexed], IERC20 collateralToken [indexed],
        #   bytes32 parentCollectionId [indexed], bytes32 conditionId,
        #   uint256[] indexSets, uint256 payout)
        condition, index_sets, payout = abi_decode(["bytes32", "uint256[]", "uint256"], data)
        row.update(
            redeemer=_addr(log["topics"][1]),
            collateral_token=_addr(log["topics"][2]),
            parent_collection_id=log["topics"][3],
            condition_id="0x" + condition.hex(),
            index_sets=[str(x) for x in index_sets],
            usdc_amount=payout / _SCALE,
        )
    elif name == "preparations":
        # ConditionPreparation(bytes32 conditionId(indexed), address oracle(indexed),
        #   bytes32 questionId(indexed), uint256 outcomeSlotCount)
        (slots,) = abi_decode(["uint256"], data)
        row.update(
            condition_id=log["topics"][1],
            oracle=_addr(log["topics"][2]),
            question_id=log["topics"][3],
            outcome_slot_count=int(slots),
        )
    elif name == "resolutions":
        slots, payouts = abi_decode(["uint256", "uint256[]"], data)
        row.update(
            condition_id=log["topics"][1],
            oracle=_addr(log["topics"][2]),
            question_id=log["topics"][3],
            outcome_slot_count=int(slots),
            payout_numerators=[str(x) for x in payouts],
        )
    else:
        raise ValueError(f"未知 CTF 事件：{name}")
    return row
