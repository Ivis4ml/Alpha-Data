"""市场元数据抓取：CLOB ``/markets``（全量目录）+ Gamma keyset（类别与生命周期）。

链上日志只有 ``asset_id``（ERC1155 份额 id）这一个市场标识。要把成交映射到「哪个市场、
哪个结果、事件是否发生」，必须补一层链下元数据。两个公开免鉴权端点各有分工：

- ``clob.polymarket.com/markets``：游标分页，每页 1000 条，**默认即含已关闭市场**。
  给出 ``condition_id``、``tokens[{token_id, outcome, winner}]``、``neg_risk``、
  ``market_slug`` 与手续费档，是 ``asset_id -> (市场, 结果)`` 映射的权威来源。
  其游标为 ``base64(十进制 offset)``，故可跳页并行抓取（实测全量约 179 万个市场）。
- ``gamma-api.polymarket.com``：补 ``category`` / ``startDate`` / ``endDate`` / ``closedTime``。

**Gamma 的四个坑**（均经实测确认；踩中会静默丢数据而不报错）：

1. ``limit`` 被**静默钳制**为 100：传 500 / 1000 仍只返回 100 条，HTTP 200 无任何提示。
2. ``/markets`` 的 ``offset`` **硬上限 2000**，越界返回 HTTP 422。offset 分页最多枚举约
   2,100 个市场，**不能用于全量抓取**。全量必须走 ``/markets/keyset``。
3. keyset 的响应字段名为 ``next_cursor``，但回传时的参数名是 ``after_cursor``。
   若原样以 ``next_cursor`` 回传，服务端**静默忽略并永远返回第一页**，形成死循环。
4. Gamma 默认 ``closed=false``，即默认只返回未关闭的市场。要取全量历史，
   必须对 ``closed=true`` 与 ``closed=false`` 各走一遍。

若只做成交重建，CLOB 一家即已足够；Gamma 仅用于补类别，属可选项。

市场解析时刻（``resolved_at``）以链上 ``ConditionResolution`` 事件的区块时间戳为准，
比 Gamma 的 ``closedTime``（运营侧写入时刻）更严格。
"""

from __future__ import annotations

import base64
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import requests

CLOB_URL = "https://clob.polymarket.com/markets"
GAMMA_KEYSET_URL = "https://gamma-api.polymarket.com/markets/keyset"
GAMMA_PAGE = 100  # 服务端硬上限，传更大值会被静默钳制
CLOB_PAGE = 1000


def _clob_cursor(offset: int) -> str:
    """CLOB 游标 = ``base64(十进制 offset)``；据此可跳页并行抓取。"""
    return base64.b64encode(str(offset).encode()).decode()


def _get(url: str, params: dict[str, Any], *, retries: int = 5, timeout: float = 40.0) -> Any:
    """带退避重试的 GET。"""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last = exc
            time.sleep(min(2**attempt * 0.5, 10.0))
    raise RuntimeError(f"请求失败 {url}：{last}")


def _clob_page(offset: int) -> list[dict[str, Any]]:
    """取 CLOB 目录的一页（offset 为行偏移，非页码）。"""
    payload = _get(CLOB_URL, {"next_cursor": _clob_cursor(offset)})
    rows: list[dict[str, Any]] = []
    for m in payload.get("data") or []:
        tokens = m.get("tokens") or []
        rows.append(
            {
                "condition_id": m.get("condition_id"),
                "question_id": m.get("question_id"),
                "question": m.get("question"),
                "market_slug": m.get("market_slug"),
                "neg_risk": bool(m.get("neg_risk")),
                "closed": bool(m.get("closed")),
                "active": bool(m.get("active")),
                "end_date_iso": m.get("end_date_iso"),
                "game_start_time": m.get("game_start_time"),
                "maker_base_fee": float(m.get("maker_base_fee") or 0.0),
                "taker_base_fee": float(m.get("taker_base_fee") or 0.0),
                "tags": list(m.get("tags") or []),
                "asset_ids": [str(t.get("token_id")) for t in tokens],
                "outcomes": [t.get("outcome") for t in tokens],
                "winners": [bool(t.get("winner")) for t in tokens],
            }
        )
    return rows


def fetch_clob_markets(*, workers: int = 8, max_markets: int | None = None) -> pd.DataFrame:
    """全量抓取 CLOB 市场目录。

    先顺序探到末尾（返回空页即止），再对已知的 offset 网格并行取回。
    由于游标是 base64(offset)，可安全跳页；实测末尾在 offset 约 179 万处。

    Args:
        workers: 并发页数。
        max_markets: 只取前若干个市场（调试用）。

    Returns:
        每行一个市场，``asset_ids`` / ``outcomes`` / ``winners`` 为等长列表列。
    """
    # 指数探测上界：找到第一个返回空页的 offset。
    hi = CLOB_PAGE
    while True:
        if max_markets and hi >= max_markets:
            hi = max_markets
            break
        if not _clob_page(hi):
            break
        hi *= 2
        if hi > 20_000_000:  # 防御：目录不可能这么大
            break

    offsets = list(range(0, hi + CLOB_PAGE, CLOB_PAGE))
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for page in pool.map(_clob_page, offsets):
            rows.extend(page)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["condition_id"])
    return df


def fetch_gamma_markets(*, max_pages: int | None = None) -> pd.DataFrame:
    """抓取 Gamma 市场（补类别与生命周期），走 keyset 分页。

    对 ``closed=true`` 与 ``closed=false`` 各走一遍：Gamma 默认只返回未关闭的市场，
    只走默认值会丢掉全部历史市场。
    """
    rows: list[dict[str, Any]] = []
    for closed in ("true", "false"):
        cursor: str | None = None
        pages = 0
        seen: set[str] = set()
        while True:
            params: dict[str, Any] = {"limit": GAMMA_PAGE, "closed": closed}
            if cursor:
                # 回传参数名是 after_cursor，不是响应里的 next_cursor（用错会死循环）。
                params["after_cursor"] = cursor
            payload = _get(GAMMA_KEYSET_URL, params)
            batch = payload.get("data") if isinstance(payload, dict) else payload
            if not batch:
                break
            for m in batch:
                rows.append(
                    {
                        "condition_id": m.get("conditionId"),
                        "category": m.get("category"),
                        "opens_at": m.get("startDate"),
                        "close_at": m.get("endDate"),
                        "closed_time": m.get("closedTime"),
                        "volume": float(m.get("volumeNum") or 0.0),
                    }
                )
            pages += 1
            nxt = payload.get("next_cursor") if isinstance(payload, dict) else None
            if not nxt or nxt in seen or (max_pages and pages >= max_pages):
                break
            seen.add(nxt)
            cursor = nxt
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["condition_id"]).drop_duplicates(subset=["condition_id"])
    return df


def build_asset_map(clob: pd.DataFrame) -> pd.DataFrame:
    """把市场目录展开为 ``asset_id -> (condition_id, outcome_seq, outcome_label, ...)``。

    ``outcome_seq`` 采用 **1 起编号**，与公开数据集一致，且恰好等于 CTF 的 indexSet
    （1 = 第一个结果，2 = 第二个）。CLOB 的 ``tokens`` 数组顺序即 indexSet 顺序，
    这一点已由链上 ``getPositionId`` 推导逐一核验（见 ``chain/positionid.py``）。

    ``outcome_label`` 并不总是 Yes / No（多结果市场为候选人名等）。
    ``winning_outcome_label`` 取该市场标记为 winner 的结果；未解析市场为 ``None``。
    """
    records: list[dict[str, Any]] = []
    for row in clob.itertuples(index=False):
        assets = list(row.asset_ids)
        outcomes = list(row.outcomes)
        winners = list(row.winners)
        win_label = next((o for o, w in zip(outcomes, winners, strict=False) if w), None)
        for seq, (asset, label) in enumerate(zip(assets, outcomes, strict=False), start=1):
            if not asset or asset == "None":
                continue
            records.append(
                {
                    "asset_id": asset,
                    "condition_id": row.condition_id,
                    "outcome_seq": seq,
                    "outcome_label": label,
                    "winning_outcome_label": win_label,
                    "neg_risk": row.neg_risk,
                    "market_slug": row.market_slug,
                    "maker_base_fee": row.maker_base_fee,
                    "taker_base_fee": row.taker_base_fee,
                    "closed": row.closed,
                }
            )
    return pd.DataFrame(records)
