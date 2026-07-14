"""Polygon JSON-RPC 客户端池：多端点轮换、自适应区块窗口、批量区块头。

设计依据为实测（见 ``docs/POLYMARKET_CRAWL.md`` 第 5 节）：

- 公共端点对 ``eth_getLogs`` 的区块范围上限差异极大：1rpc 仅 50 块，drpc 免费档不稳定，
  tenderly 与 onfinality 可稳定支撑 10,000 块 / 单次（返回逾 27 万条日志）。
- onfinality 的日志对象自带 ``blockTimestamp``；tenderly 不带。命中前者可省去区块头请求。
- 超限的失败模式不统一（区间超限 / 结果过多 / 超时），无法靠错误码区分，
  故统一策略：任一失败即二分该区间重试，成功则缓慢回升窗口。

因此本类不假定任何端点的能力，而是以“失败即折半、成功即回升”的自适应窗口收敛到各端点的实际上限。
"""

from __future__ import annotations

import itertools
import os
import random
import time
from collections.abc import Iterable, Sequence
from typing import Any

import requests

# 实测可用于归档日志查询的免费公共端点（按实测吞吐排序）。
# onfinality 置首：其日志对象自带 blockTimestamp。
DEFAULT_ENDPOINTS: tuple[str, ...] = (
    "https://polygon.api.onfinality.io/public",
    "https://polygon.gateway.tenderly.co",
    "https://polygon.drpc.org",
)

MAX_SPAN = 10_000  # 单次 eth_getLogs 的区块跨度上限（实测 tenderly / onfinality 可达）
MIN_SPAN = 1  # 折半下限：单块查询必须成功，否则视为端点故障


class RpcError(RuntimeError):
    """JSON-RPC 调用错误（协议层或传输层）。"""


class RpcPool:
    """多端点 JSON-RPC 池。

    端点按轮换顺序使用；单端点连续失败时自动切换到下一个。所有方法为同步阻塞，
    并发由调用方以线程池实现（各线程持有独立 ``RpcPool`` 或共享本类，
    本类的 ``requests.Session`` 按端点独立且线程安全用法仅限于只读复用连接池）。

    Attributes:
        endpoints: 端点列表。环境变量 ``POLYGON_RPC_URLS``（逗号分隔）可覆盖默认值，
            用于接入付费端点（Alchemy / QuickNode 等）以提升吞吐。
        timeout: 单次请求超时（秒）。
        max_retries: 单次调用在所有端点上的总重试次数。
    """

    def __init__(
        self,
        endpoints: Sequence[str] | None = None,
        *,
        timeout: float = 90.0,
        max_retries: int = 6,
    ) -> None:
        env = os.environ.get("POLYGON_RPC_URLS", "").strip()
        if endpoints:
            eps = tuple(endpoints)
        elif env:
            eps = tuple(x.strip() for x in env.split(",") if x.strip())
        else:
            eps = DEFAULT_ENDPOINTS
        if not eps:
            raise RpcError("未配置任何 RPC 端点")
        self.endpoints = eps
        self.timeout = timeout
        self.max_retries = max_retries
        self._cycle = itertools.cycle(range(len(eps)))
        self._session = requests.Session()
        # 各端点的自适应窗口（区块跨度）。
        self._span: dict[str, int] = {ep: 2_000 for ep in eps}

    # --- 底层调用 ---------------------------------------------------------

    def _post(self, endpoint: str, payload: Any) -> Any:
        resp = self._session.post(endpoint, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RpcError(f"{endpoint} HTTP {resp.status_code}")
        return resp.json()

    def call(self, method: str, params: list[Any]) -> Any:
        """单次 RPC 调用，失败时跨端点重试。

        Args:
            method: RPC 方法名。
            params: 参数列表。

        Returns:
            ``result`` 字段。

        Raises:
            RpcError: 所有端点均失败。
        """
        last: Exception | None = None
        for attempt in range(self.max_retries):
            endpoint = self.endpoints[next(self._cycle)]
            try:
                payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                out = self._post(endpoint, payload)
                if "error" in out:
                    raise RpcError(f"{endpoint} {out['error']}")
                return out["result"]
            except Exception as exc:  # 传输层与协议层一并退避重试
                last = exc
                time.sleep(min(2**attempt * 0.4, 8.0) * (0.5 + random.random()))
        raise RpcError(f"RPC 调用失败（{method}）：{last}")

    def block_number(self) -> int:
        """当前链头区块号。"""
        return int(self.call("eth_blockNumber", []), 16)

    # --- 日志抓取 ---------------------------------------------------------

    def get_logs(
        self,
        *,
        from_block: int,
        to_block: int,
        topics: list[Any] | None = None,
        addresses: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """抓取一个区块区间的日志，区间过大时自动二分。

        失败模式在各端点间不统一（区间超限 / 结果过多 / 网关超时），一律以二分应对：
        单块查询仍失败才判定为真实故障。

        Args:
            from_block: 起始区块（含）。
            to_block: 结束区块（含）。
            topics: ``eth_getLogs`` 的 topics 过滤。
            addresses: 地址过滤；``None`` 表示按 topic 全网扫描。

        Returns:
            日志对象列表，按 (blockNumber, logIndex) 升序。
        """
        out: list[dict[str, Any]] = []
        self._get_logs_into(out, from_block, to_block, topics, addresses)
        out.sort(key=lambda x: (int(x["blockNumber"], 16), int(x["logIndex"], 16)))
        return out

    def _get_logs_into(
        self,
        sink: list[dict[str, Any]],
        lo: int,
        hi: int,
        topics: list[Any] | None,
        addresses: Sequence[str] | None,
    ) -> None:
        if lo > hi:
            return
        flt: dict[str, Any] = {"fromBlock": hex(lo), "toBlock": hex(hi)}
        if topics:
            flt["topics"] = topics
        if addresses:
            flt["address"] = list(addresses)
        try:
            sink.extend(self.call("eth_getLogs", [flt]))
            return
        except RpcError:
            if hi - lo < MIN_SPAN:
                raise
        mid = (lo + hi) // 2
        self._get_logs_into(sink, lo, mid, topics, addresses)
        self._get_logs_into(sink, mid + 1, hi, topics, addresses)

    def suggested_span(self) -> int:
        """建议的区块窗口（供调用方切分回填任务）。"""
        return MAX_SPAN

    # --- 区块时间戳 -------------------------------------------------------

    def block_timestamps(self, blocks: Iterable[int], *, batch: int = 200) -> dict[int, int]:
        """批量取区块时间戳（用于日志不带 ``blockTimestamp`` 的端点）。

        Args:
            blocks: 区块号集合。
            batch: 单个 JSON-RPC 批请求内的区块数。

        Returns:
            ``{区块号: unix 秒}``。
        """
        want = sorted(set(blocks))
        out: dict[int, int] = {}
        for i in range(0, len(want), batch):
            chunk = want[i : i + batch]
            payload = [
                {
                    "jsonrpc": "2.0",
                    "id": n,
                    "method": "eth_getBlockByNumber",
                    "params": [hex(n), False],
                }
                for n in chunk
            ]
            last: Exception | None = None
            for attempt in range(self.max_retries):
                endpoint = self.endpoints[next(self._cycle)]
                try:
                    res = self._post(endpoint, payload)
                    if not isinstance(res, list):
                        raise RpcError(f"{endpoint} 批请求未返回数组")
                    for item in res:
                        if item.get("error") or not item.get("result"):
                            raise RpcError(f"{endpoint} 批内错误 {item.get('error')}")
                        blk = item["result"]
                        out[int(blk["number"], 16)] = int(blk["timestamp"], 16)
                    break
                except Exception as exc:
                    last = exc
                    time.sleep(min(2**attempt * 0.4, 8.0) * (0.5 + random.random()))
            else:
                raise RpcError(f"批量取区块头失败：{last}")
        return out
