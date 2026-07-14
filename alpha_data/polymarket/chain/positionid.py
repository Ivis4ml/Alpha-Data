"""从链上推导 ``asset_id``：不依赖任何 HTTP 接口的市场映射校验。

``asset_id``（ERC1155 份额 id）并非随机编号，而是由 ConditionalTokens 合约按
``(抵押品, conditionId, indexSet)`` 确定性算出的：

```text
collectionId = CTF.getCollectionId(parentCollectionId=0x0, conditionId, indexSet)
asset_id     = CTF.getPositionId(collateralToken, collectionId)
```

其中 ``indexSet`` 是结果的位掩码：``1`` = 第一个结果，``2`` = 第二个结果。
这恰好等于公开数据集里 1 起编号的 ``outcome_seq``。

因此我们**不必信任** Polymarket 的 HTTP 接口就能建立 ``asset_id -> (市场, 结果)`` 的映射：
用 CLOB 目录做批量映射（快），再用本模块对抽样市场做链上推导校验（可信）。
两者若一致，即证明 CLOB 的 ``tokens`` 数组顺序确实是 indexSet 顺序。

**抵押品不能取自 Gamma 的 ``negRisk`` 标记**：实测存在 ``neg_risk`` 为假、却以 WCOL
（NegRiskAdapter 的包装抵押品）计算份额 id 的市场。正确做法是两种抵押品都试：
标准市场用 USDC.e，negRisk 市场用 WCOL，取能对上的那个。
"""

from __future__ import annotations

from typing import Any

from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from eth_utils import keccak

from alpha_data.polymarket.chain import venues

# NegRiskAdapter 构造时新建的包装抵押品（WrappedCollateral, 6 位小数）。
# negRisk 市场的份额 id 以它为抵押品计算，而非 USDC.e。
WRAPPED_COLLATERAL: str = "0x3a3bd7bb9528e159577f7c2e685cc81a765002e2"

_SEL_COLLECTION = "0x" + keccak(text="getCollectionId(bytes32,bytes32,uint256)").hex()[:8]
_SEL_POSITION = "0x" + keccak(text="getPositionId(address,bytes32)").hex()[:8]


def _call(pool: Any, to: str, data: str) -> str:
    return pool.call("eth_call", [{"to": to, "data": data}, "latest"])


def collection_id(pool: Any, condition_id: str, index_set: int) -> bytes:
    """``CTF.getCollectionId(0x0, conditionId, indexSet)``。"""
    payload = _SEL_COLLECTION + abi_encode(
        ["bytes32", "bytes32", "uint256"],
        [bytes(32), bytes.fromhex(condition_id[2:]), int(index_set)],
    ).hex()
    raw = _call(pool, venues.CONDITIONAL_TOKENS, payload)
    return abi_decode(["bytes32"], bytes.fromhex(raw[2:]))[0]


def position_id(pool: Any, collateral: str, collection: bytes) -> int:
    """``CTF.getPositionId(collateralToken, collectionId)`` -> ERC1155 份额 id。"""
    payload = _SEL_POSITION + abi_encode(["address", "bytes32"], [collateral, collection]).hex()
    raw = _call(pool, venues.CONDITIONAL_TOKENS, payload)
    return int(abi_decode(["uint256"], bytes.fromhex(raw[2:]))[0])


def derive_asset_id(pool: Any, condition_id: str, outcome_seq: int, *, neg_risk: bool) -> int:
    """由 ``(condition_id, outcome_seq)`` 推导 ``asset_id``。

    Args:
        pool: :class:`~.rpc.RpcPool`。
        condition_id: ``0x`` 前缀的 32 字节条件 id。
        outcome_seq: 1 起编号的结果序号（等于 CTF 的 indexSet）。
        neg_risk: 是否为 negRisk 市场（决定抵押品取 WCOL 还是 USDC.e）。

    Returns:
        ERC1155 份额 id。
    """
    collateral = WRAPPED_COLLATERAL if neg_risk else venues.USDC_E
    return position_id(pool, collateral, collection_id(pool, condition_id, outcome_seq))


def verify_asset_map(
    pool: Any, asset_map: Any, *, sample: int = 20, seed: int = 7
) -> dict[str, Any]:
    """抽样校验 CLOB 给出的 ``asset_id`` 映射是否与链上推导一致。

    对每个抽样市场，两种抵押品都试一遍（不信任 ``neg_risk`` 标记），只要有一种对上即算通过。

    Args:
        pool: :class:`~.rpc.RpcPool`。
        asset_map: :func:`~.metadata.build_asset_map` 的输出。
        sample: 抽样行数。
        seed: 随机种子。

    Returns:
        ``{"checked": n, "matched": m, "mismatched": [...]}``。
    """
    rows = asset_map.sample(n=min(sample, len(asset_map)), random_state=seed)
    matched = 0
    mismatched: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        expected = str(row.asset_id)
        got: list[str] = []
        for neg in (False, True):
            try:
                got.append(
                    str(derive_asset_id(pool, row.condition_id, int(row.outcome_seq), neg_risk=neg))
                )
            except Exception:  # 条件未在链上准备好时 eth_call 会失败
                continue
        if expected in got:
            matched += 1
        else:
            mismatched.append(
                {
                    "asset_id": expected,
                    "condition_id": row.condition_id,
                    "outcome_seq": int(row.outcome_seq),
                    "derived": got,
                }
            )
    return {"checked": len(rows), "matched": matched, "mismatched": mismatched}
