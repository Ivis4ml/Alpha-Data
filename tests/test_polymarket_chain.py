"""Polymarket 链上爬虫的离线单元测试。

固定装置是从 Polygon 主网取回的**真实日志**（区块 63732941 与 90198883），
其期望解码值来自本地快照 ``data/polymarket/OrderFilled``（旧版）与 Polymarket
公开 Data API（新版），故本测试同时锁定"解码正确"与"与既有数据口径一致"两件事。
测试不发起任何网络请求。
"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_data.polymarket.chain import crawl, decode, enrich, venues

# --- 固定装置：真实链上日志 ---------------------------------------------------

# 区块 63732941 log 137：旧版 OrderFilled。快照 id=137_63732941_137 的期望值为
# maker=0x463e797d.. taker=0x31eB2055.. maker_asset_id=0 token=38.59 usdc=0.27013 price=0.007
LOG_V1 = {
    "address": venues.NEG_RISK_CTF_EXCHANGE,
    "blockNumber": hex(63732941),
    "logIndex": hex(137),
    "transactionHash": "0x" + "11" * 32,
    "topics": [
        venues.TOPIC_ORDER_FILLED_V1,
        "0x" + "22" * 32,
        "0x000000000000000000000000463e797dccfe0fd56839bc855f1e94b3f6ef664e",
        "0x00000000000000000000000031eb2055c8e164536dda7fdf48749cbd8ac3d1f6",
    ],
    # makerAssetId=0, takerAssetId=1020053931...（快照实值）,
    # makerAmountFilled=270130（0.27013 抵押品）, takerAmountFilled=38590000（38.59 份额）, fee=0
    "data": "0x"
    + f"{0:064x}"
    + f"{102005393153954001616156555364526298244258903553193640605945603006929523666032:064x}"
    + f"{270130:064x}"
    + f"{38590000:064x}"
    + f"{0:064x}",
    "blockTimestamp": hex(1730419200),
}

# 区块 90198883 log 673：新版 OrderFilled（side=0，maker 买入）。
# Data API 对这笔成交的口径：size=10.20408 price=0.4899999804
LOG_V2 = {
    "address": venues.EXCHANGE_V2_MAIN,
    "blockNumber": hex(90198883),
    "logIndex": hex(673),
    "transactionHash": "0x" + "33" * 32,
    "topics": [
        venues.TOPIC_ORDER_FILLED_V2,
        "0x" + "44" * 32,
        "0x00000000000000000000000025a7a9a34a2a709c9f712e23c32d406ed40b4537",
        # taker 就是交易所自身 -> 中继腿
        "0x000000000000000000000000e111180000d2663c0091e4f400237545b87b996b",
    ],
    "data": "0x"
    + f"{0:064x}"  # side = 0（maker 买入）
    + f"{44673194269862996949822103727031113002480017684404327749201539777332965698898:064x}"
    + f"{4999999:064x}"  # makerAmountFilled = 4.999999 抵押品
    + f"{10204080:064x}"  # takerAmountFilled = 10.20408 份额
    + f"{378480:064x}"  # fee = 0.37848
    + f"{0:064x}"
    + f"{0:064x}",
    "blockTimestamp": hex(1784003320),
}


def test_topic0_与链上实测一致() -> None:
    """七个事件的 topic0 必须与链上观测值逐一相等（防止签名写错）。"""
    assert venues.TOPIC_ORDER_FILLED_V1 == (
        "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
    )
    assert venues.TOPIC_ORDER_FILLED_V2 == (
        "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
    )
    assert venues.TOPIC_POSITION_SPLIT == (
        "0x2e6bb91f8cbcda0c93623c54d0403a43514fabc40084ec96b6d5379a74786298"
    )
    assert venues.TOPIC_PAYOUT_REDEMPTION == (
        "0x2682012a4a4f1973119f1c9b90745d1bd91fa2bab387344f044cb3586864d18d"
    )
    assert venues.TOPIC_CONDITION_RESOLUTION == (
        "0xb44d84d3289691f71497564b85d4233648d9dbae8cbdbb4329f301c3a0185894"
    )


def test_旧版解码与快照口径一致() -> None:
    row = decode.decode_trade(LOG_V1)
    assert row is not None
    assert row["id"] == "137_63732941_137"
    assert row["maker"] == "0x463e797dCCfe0fD56839BC855F1E94b3f6Ef664E"
    assert row["taker"] == "0x31eB2055C8e164536dDa7fdF48749CBd8AC3d1F6"
    assert row["maker_asset_id"] == "0"
    assert row["token_asset_id"] == (
        "102005393153954001616156555364526298244258903553193640605945603006929523666032"
    )
    assert row["maker_direction"] == "BUY"  # makerAssetId == 0 -> maker 付抵押品
    assert row["taker_direction"] == "SELL"
    assert row["token_amount"] == pytest.approx(38.59)
    assert row["usdc_amount"] == pytest.approx(0.27013)
    assert row["price"] == pytest.approx(0.007)
    assert row["block_timestamp"] == 1730419200
    assert row["protocol"] == "v1"
    assert row["venue_class"] == "polymarket"


def test_新版解码遵循_side_决定的币种顺序() -> None:
    """side=0 时 makerAmountFilled 是抵押品；价格须落在 [0, 1] 且与 Data API 一致。"""
    row = decode.decode_trade(LOG_V2)
    assert row is not None
    assert row["protocol"] == "v2"
    assert row["maker_direction"] == "BUY"
    assert row["token_amount"] == pytest.approx(10.20408)
    assert row["usdc_amount"] == pytest.approx(4.999999)
    assert row["price"] == pytest.approx(0.4899999804, abs=1e-9)
    assert 0.0 <= row["price"] <= 1.0
    assert row["fee_usdc"] == pytest.approx(0.37848)


def test_中继腿判定为_taker_等于交易所地址() -> None:
    """LOG_V2 的 taker 就是交易所自身，应判为中继腿；LOG_V1 的 taker 是用户，不是。"""
    assert decode.decode_trade(LOG_V2)["is_relay"] is True
    assert decode.decode_trade(LOG_V1)["is_relay"] is False


def test_场馆分类拒绝同名签名的其他协议() -> None:
    assert venues.venue_class(venues.CTF_EXCHANGE) == "polymarket"
    assert venues.venue_class(venues.EXCHANGE_V2_MAIN) == "polymarket"
    # 使用相同事件签名但结算于其他合约的协议，必须排除
    assert venues.venue_class("0x5afa51599b8b91cb025a7863d0cd77b7000293c6") == "foreign"
    # 未核验的地址归为 unknown，交由人工复核，不得默认纳入
    assert venues.venue_class(venues.EXCHANGE_V2_AUX) == "unknown"
    assert venues.venue_class("0x" + "ab" * 20) == "unknown"


def test_分析层只保留_polymarket_场馆() -> None:
    trades = pd.DataFrame(
        {
            "exchange": [venues.CTF_EXCHANGE, "0x5afa51599b8b91cb025a7863d0cd77b7000293c6"],
            "venue_class": ["polymarket", "foreign"],
            "is_relay": [False, False],
        }
    )
    poly, other = enrich.keep_polymarket_venues(trades)
    assert len(poly) == 1
    assert len(other) == 1
    assert other.iloc[0]["venue_class"] == "foreign"


def test_p_event_与_D_的真值表() -> None:
    """真值表来自 2024-11 全月实测：(BUY,1)->+1 (SELL,1)->-1 (BUY,2)->-1 (SELL,2)->+1。"""
    trades = pd.DataFrame(
        {
            "token_asset_id": ["a1", "a1", "a2", "a2"],
            "taker_direction": ["BUY", "SELL", "BUY", "SELL"],
            "price": [0.30, 0.30, 0.70, 0.70],
            "is_relay": [False] * 4,
        }
    )
    asset_map = pd.DataFrame(
        {
            "asset_id": ["a1", "a2"],
            "condition_id": ["c1", "c1"],
            "outcome_seq": [1, 2],
            "outcome_label": ["Yes", "No"],
            "winning_outcome_label": [None, None],
            "neg_risk": [False, False],
            "market_slug": ["m", "m"],
            "maker_base_fee": [0.0, 0.0],
            "taker_base_fee": [0.0, 0.0],
            "closed": [False, False],
        }
    )
    out = enrich.attach_metadata(trades, asset_map)
    assert out["is_binary"].all()
    # outcome_seq=1 的成交价即事件概率；outcome_seq=2 取 1 - price
    assert list(out["p_event"].round(4)) == [0.30, 0.30, 0.30, 0.30]
    assert list(out["D"]) == [1, -1, -1, 1]


def test_多结果市场不外推_p_event() -> None:
    trades = pd.DataFrame(
        {
            "token_asset_id": ["x1"],
            "taker_direction": ["BUY"],
            "price": [0.25],
            "is_relay": [False],
        }
    )
    asset_map = pd.DataFrame(
        {
            "asset_id": ["x1", "x2", "x3"],
            "condition_id": ["c9"] * 3,
            "outcome_seq": [1, 2, 3],
            "outcome_label": ["A", "B", "C"],
            "winning_outcome_label": [None] * 3,
            "neg_risk": [True] * 3,
            "market_slug": ["m"] * 3,
            "maker_base_fee": [0.0] * 3,
            "taker_base_fee": [0.0] * 3,
            "closed": [False] * 3,
        }
    )
    out = enrich.attach_metadata(trades, asset_map)
    assert not out["is_binary"].iloc[0]
    assert pd.isna(out["p_event"].iloc[0])  # 无互补关系，不外推
    assert out["D"].iloc[0] == 0


def test_区块窗口对齐到网格() -> None:
    """窗口边界必须对齐网格，否则不同次运行切出的分片无法复用。"""
    wins = list(crawl.iter_windows(10_500, 30_400, span=10_000))
    assert wins == [(10_500, 19_999), (20_000, 29_999), (30_000, 30_400)]


def test_CTF_事件解码_条件解析() -> None:
    log = {
        "address": venues.CONDITIONAL_TOKENS,
        "blockNumber": hex(35901095),
        "logIndex": hex(0),
        "topics": [
            venues.TOPIC_CONDITION_RESOLUTION,
            "0xdb1f84d8cb40e9c13e9d2ee4a1809bc0519c2d94723a7d55dcf2bce3151527f7",
            "0x000000000000000000000000cb1822859cef82cd2eb4e6276c7916e692995130",
            "0xf9f94da0ee227fbf01add6598be54cc2513995d1ab6a0458263e87cbc180fa53",
        ],
        "data": "0x" + f"{2:064x}" + f"{64:064x}" + f"{2:064x}" + f"{1:064x}" + f"{0:064x}",
    }
    row = decode.decode_ctf("resolutions", log)
    assert row["id"] == "137_35901095_0"
    assert row["condition_id"].startswith("0xdb1f84d8")
    assert row["outcome_slot_count"] == 2
    assert row["payout_numerators"] == ["1", "0"]  # 第一个结果获胜
