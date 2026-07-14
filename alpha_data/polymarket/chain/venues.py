"""Polymarket 链上合约与事件注册表（Polygon 主网，chainId 137）。

本模块是爬虫的唯一事实来源：所有合约地址、事件签名、topic0 与生效区块集中在此，
其余模块不得硬编码地址。

三条经实测确认的事实（核对方法见 ``docs/POLYMARKET_CRAWL.md`` 第 3 节）：

1. 撮合成交只经由交易所合约的 ``OrderFilled`` 事件表达，存在**两套互不兼容的 ABI**：
   旧版（2022-11 .. 2026-04-28）与新版（2026-04-03 起）。两者 topic0 不同，
   字段数与语义也不同，必须分别解码。
2. 2026-04-28 11:00:40 UTC（区块 86126998）旧交易所发出最后一条 ``OrderFilled``，此后归零；
   新交易所自 2026-04-03（区块 85050371）起并行运行并完全接管。公开数据集
   ``TimeSeventeen/Polymarket-v1`` 的末区块正是 86126998，即它恰好覆盖旧合约的完整生命周期。
3. 新版协议同时有**三个**交易所合约在发 ``OrderFilled``（e1111 / e2222 / e3333）。
   按地址白名单抓取会漏数据；正确做法是**按 topic0 全网扫描**，地址仅用于识别中继腿。

抵押品在新旧协议间发生变更：旧版为 USDC.e，新版为 Polymarket 自发的 pUSD（均为 6 位小数），
两者的金额标度一致，故解码后的 ``usdc_amount`` 口径可直接拼接。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from eth_utils import keccak

CHAIN_ID: Final[int] = 137

# --- 交易所合约 -------------------------------------------------------------

# 旧协议（抵押品 USDC.e）。NegRiskCtfExchange 的 getCtf() 指向 NegRiskAdapter 而非裸 CTF。
CTF_EXCHANGE: Final[str] = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
NEG_RISK_CTF_EXCHANGE: Final[str] = "0xc5d563a36ae78145c45a50134d48a1215220f80a"

# 新协议（抵押品 pUSD）。三者并行，覆盖不同市场类型。
EXCHANGE_V2_MAIN: Final[str] = "0xe111180000d2663c0091e4f400237545b87b996b"
EXCHANGE_V2_ALT: Final[str] = "0xe2222d279d744050d28e00520010520000310f59"
EXCHANGE_V2_AUX: Final[str] = "0xe3333700ca9d93003f00f0f71f8515005f6c00aa"

LEGACY_EXCHANGES: Final[frozenset[str]] = frozenset({CTF_EXCHANGE, NEG_RISK_CTF_EXCHANGE})
V2_EXCHANGES: Final[frozenset[str]] = frozenset(
    {EXCHANGE_V2_MAIN, EXCHANGE_V2_ALT, EXCHANGE_V2_AUX}
)

# --- 结算 / 抵押品合约 ------------------------------------------------------

CONDITIONAL_TOKENS: Final[str] = "0x4d97dcd97ec945f40cf65f87097ace5ea0476045"
NEG_RISK_ADAPTER: Final[str] = "0xd91e80cf2e7be2e162c6513ced06f1dd0da35296"
USDC_E: Final[str] = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"  # 旧抵押品
PUSD: Final[str] = "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"  # 新抵押品 Polymarket USD

COLLATERAL_DECIMALS: Final[int] = 6
TOKEN_DECIMALS: Final[int] = 6  # CTF 份额与抵押品同标度

# --- 区块锚点（实测，非估算）-----------------------------------------------

# 链上第一条 OrderFilled（2022-11-21 19:50:09 UTC）。已实测：区块 33,000,000 至此之间无
# 任何 OrderFilled，故此即订单簿协议的起点，亦与公开数据集的最小区块一致。
GENESIS_BLOCK: Final[int] = 35_896_869
# 旧交易所最后一条 OrderFilled（2026-04-28 11:00:40 UTC）。
LEGACY_LAST_BLOCK: Final[int] = 86_126_998
# 新交易所第一条 OrderFilled（2026-04-03 12:52:59 UTC）。
V2_FIRST_BLOCK: Final[int] = 85_050_371


def _topic0(signature: str) -> str:
    """事件签名 -> topic0（keccak256）。"""
    return "0x" + keccak(text=signature).hex()


# --- 事件签名 ---------------------------------------------------------------

# 旧版：方向由 makerAssetId 是否为 0 推断（0 表示 maker 付出抵押品，即 maker 买入）。
SIG_ORDER_FILLED_V1: Final[str] = (
    "OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)"
)
# 新版：显式 side(uint8, 0=maker BUY / 1=maker SELL)，且只带单个 assetId。
SIG_ORDER_FILLED_V2: Final[str] = (
    "OrderFilled(bytes32,address,address,uint8,uint256,uint256,uint256,uint256,bytes32,bytes32)"
)
SIG_POSITION_SPLIT: Final[str] = "PositionSplit(address,address,bytes32,bytes32,uint256[],uint256)"
SIG_POSITIONS_MERGE: Final[str] = (
    "PositionsMerge(address,address,bytes32,bytes32,uint256[],uint256)"
)
SIG_PAYOUT_REDEMPTION: Final[str] = (
    "PayoutRedemption(address,address,bytes32,bytes32,uint256[],uint256)"
)
SIG_CONDITION_PREPARATION: Final[str] = "ConditionPreparation(bytes32,address,bytes32,uint256)"
SIG_CONDITION_RESOLUTION: Final[str] = (
    "ConditionResolution(bytes32,address,bytes32,uint256,uint256[])"
)

TOPIC_ORDER_FILLED_V1: Final[str] = _topic0(SIG_ORDER_FILLED_V1)
TOPIC_ORDER_FILLED_V2: Final[str] = _topic0(SIG_ORDER_FILLED_V2)
TOPIC_POSITION_SPLIT: Final[str] = _topic0(SIG_POSITION_SPLIT)
TOPIC_POSITIONS_MERGE: Final[str] = _topic0(SIG_POSITIONS_MERGE)
TOPIC_PAYOUT_REDEMPTION: Final[str] = _topic0(SIG_PAYOUT_REDEMPTION)
TOPIC_CONDITION_PREPARATION: Final[str] = _topic0(SIG_CONDITION_PREPARATION)
TOPIC_CONDITION_RESOLUTION: Final[str] = _topic0(SIG_CONDITION_RESOLUTION)


@dataclass(frozen=True)
class EventSpec:
    """一类待抓取事件的抓取规格。

    Attributes:
        name: 逻辑名（同时是输出目录名）。
        topic0: 事件 topic0。
        addresses: 地址过滤；``None`` 表示不按地址过滤（按 topic 全网扫描），
            用于交易所合约可能新增地址的场景。
        first_block: 该事件最早可能出现的区块（用于裁剪回填区间）。
    """

    name: str
    topic0: str
    addresses: tuple[str, ...] | None
    first_block: int


# OrderFilled 两版均不按地址过滤：新版合约地址会增加（已见 e1111/e2222/e3333），
# 按 topic 扫描可自动覆盖未来新增的交易所合约。
ORDER_FILLED_V1 = EventSpec("order_filled_v1", TOPIC_ORDER_FILLED_V1, None, GENESIS_BLOCK)
ORDER_FILLED_V2 = EventSpec("order_filled_v2", TOPIC_ORDER_FILLED_V2, None, V2_FIRST_BLOCK)

# CTF 生命周期事件只可能来自 ConditionalTokens（NegRiskAdapter 亦最终调用它），按地址过滤更省。
CTF_EVENTS: Final[tuple[EventSpec, ...]] = (
    EventSpec("splits", TOPIC_POSITION_SPLIT, (CONDITIONAL_TOKENS,), GENESIS_BLOCK),
    EventSpec("merges", TOPIC_POSITIONS_MERGE, (CONDITIONAL_TOKENS,), GENESIS_BLOCK),
    EventSpec("redemptions", TOPIC_PAYOUT_REDEMPTION, (CONDITIONAL_TOKENS,), GENESIS_BLOCK),
    EventSpec("preparations", TOPIC_CONDITION_PREPARATION, (CONDITIONAL_TOKENS,), GENESIS_BLOCK),
    EventSpec("resolutions", TOPIC_CONDITION_RESOLUTION, (CONDITIONAL_TOKENS,), GENESIS_BLOCK),
)

ALL_EVENTS: Final[tuple[EventSpec, ...]] = (ORDER_FILLED_V1, ORDER_FILLED_V2, *CTF_EVENTS)


def is_exchange(address: str) -> bool:
    """地址是否为已知交易所合约（用于识别中继腿）。"""
    a = address.lower()
    return a in LEGACY_EXCHANGES or a in V2_EXCHANGES


# --- 场馆分类 ---------------------------------------------------------------
#
# 按 topic0 全网扫描会连带抓到**其他协议**：``OrderFilled(bytes32,address,address,uint256,
# uint256,uint256,uint256,uint256)`` 并非 Polymarket 独有，链上存在使用同一签名的仿制品
# （已实测到两个，其 getCtf() 指向不同的结算合约）。若不加甄别，这些外部协议的成交会混入数据。
#
# 判定规则（可复核，不依赖名单）：Polymarket 场馆的 ``getCtf()`` 必然返回
# ConditionalTokens 或 NegRiskAdapter。以下名单即由该规则逐一核验得出。

VERIFIED_VENUES: Final[dict[str, str]] = {
    CTF_EXCHANGE: "CTFExchange（结算 CTF，抵押 USDC.e）",
    NEG_RISK_CTF_EXCHANGE: "NegRiskCtfExchange（结算 NegRiskAdapter，抵押 USDC.e）",
    "0x87b81fd5b3845f68854f7c407477bad1bc1108d1": "CTF 变体（抵押 USDC.e）",
    "0x403c94538713f98fd63bde5a0e0810c2ed7f20cd": "NegRisk 变体（抵押 USDC.e）",
    "0x2e0277c26470ef1e138d704666a5baa53911dec9": "NegRisk 变体（抵押 USDC.e）",
    "0xdbf75b4057ced0b6fc9b521acfabfe817613af04": "CTF 变体（抵押 USDT）",
    EXCHANGE_V2_MAIN: "Exchange v2 主（抵押 pUSD）",
    EXCHANGE_V2_ALT: "Exchange v2 次（抵押 pUSD）",
}

# 已核验为**非** Polymarket：事件签名相同但结算合约不同的其他协议，必须排除。
FOREIGN_VENUES: Final[dict[str, str]] = {
    "0x5afa51599b8b91cb025a7863d0cd77b7000293c6": "其他协议（结算 0x82a08b18..）",
    "0xcae049a1184cf71be6b88a3fae49b80a283e4ca8": "其他协议（结算 0x429eba38..）",
}

# 无 getCtf() 接口、无法按上述规则判定的地址（如 e3333）归为 unknown：
# 既不混入分析层，也不静默丢弃，而是记入报告交由人工判定。
UNCLASSIFIED_NOTE: Final[str] = (
    f"{EXCHANGE_V2_AUX} 不暴露 getCtf()，其成交结算于 0x006f54f7.. 而非 ConditionalTokens，"
    "暂列 unknown。"
)


def venue_class(address: str) -> str:
    """场馆分类：``polymarket`` / ``foreign`` / ``unknown``。

    ``unknown`` 表示该地址发出了 Polymarket 格式的成交事件但未经核验，调用方应将其
    隔离并复核（用 :func:`classify_venue_onchain` 核验后补入 :data:`VERIFIED_VENUES`），
    不得默认当作 Polymarket 数据使用。
    """
    a = address.lower()
    if a in VERIFIED_VENUES:
        return "polymarket"
    if a in FOREIGN_VENUES:
        return "foreign"
    return "unknown"


def classify_venue_onchain(pool: object, address: str) -> str:
    """用链上 ``getCtf()`` 判定某地址是否为 Polymarket 场馆。

    Args:
        pool: 具备 ``call(method, params)`` 的 RPC 客户端（:class:`~.rpc.RpcPool`）。
        address: 待判定的合约地址。

    Returns:
        ``polymarket`` / ``foreign`` / ``unknown``（无 ``getCtf()`` 接口时）。
    """
    selector = "0x" + keccak(text="getCtf()").hex()[:8]
    try:
        raw = pool.call("eth_call", [{"to": address, "data": selector}, "latest"])  # type: ignore[attr-defined]
    except Exception:
        return "unknown"
    if not raw or raw == "0x" or len(raw) != 66:
        return "unknown"
    settled = "0x" + raw[-40:]
    if settled in (CONDITIONAL_TOKENS, NEG_RISK_ADAPTER):
        return "polymarket"
    return "foreign"
