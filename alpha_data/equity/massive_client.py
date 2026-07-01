"""massive.com（Polygon.io 更名）REST 客户端。

职责：注入 API key、按 ``next_url`` 分页、限速（429）退避、瞬时错误重试。massive.com 与
Polygon.io 在迁移期内 API 完全一致，``base_url`` 默认 ``api.massive.com``，可回退
``api.polygon.io``。

环境变量：``MASSIVE_API_KEY``（兼容 ``POLYGON_API_KEY``）。
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import requests

# 视为成功的响应 status（DELAYED 为延迟档数据，仍是有效返回）。
_OK_STATUS = frozenset({"OK", "DELAYED"})


class MassiveError(RuntimeError):
    """massive.com REST 调用错误（鉴权 / 授权 / 协议层）。"""


class MassiveClient:
    """轻量 REST 客户端（仅依赖 requests）。

    Attributes:
        api_key: REST 密钥。
        base_url: API 根（默认 https://api.massive.com）。
        timeout: 单次请求超时（秒）。
        max_retries: 瞬时错误 / 限速的最大重试次数。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.massive.com",
        *,
        timeout: float = 30.0,
        max_retries: int = 6,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("MASSIVE_API_KEY")
            or os.environ.get("POLYGON_API_KEY")
        )
        if not self.api_key:
            raise MassiveError("缺少 API key：请设置 MASSIVE_API_KEY（或 POLYGON_API_KEY）")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._session = requests.Session()

    def get(self, path_or_url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET 一个端点或绝对 URL（用于 next_url），返回解析后的 JSON。

        对 429 / 5xx / 网络错误做指数退避重试；其他非成功 status 抛 :class:`MassiveError`。
        """
        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url}{path_or_url}"
        query = dict(params or {})
        query["apiKey"] = self.api_key
        backoff = 1.0
        last_err = "未知错误"
        for _ in range(self.max_retries):
            try:
                resp = self._session.get(url, params=query, timeout=self.timeout)
            except requests.RequestException as exc:
                last_err = f"网络错误: {exc}"
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            if resp.status_code == 429:
                last_err = "限速 429"
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            if resp.status_code >= 500:
                last_err = f"服务端错误 {resp.status_code}"
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            try:
                data = resp.json()
            except ValueError:
                raise MassiveError(
                    f"非 JSON 响应（HTTP {resp.status_code}）：{resp.text[:200]}"
                ) from None
            status = data.get("status")
            if resp.status_code == 200 and (status is None or status in _OK_STATUS):
                return data
            message = data.get("message") or data.get("error") or ""
            raise MassiveError(f"API status={status} (HTTP {resp.status_code})：{message}")
        raise MassiveError(f"重试 {self.max_retries} 次仍失败：{last_err}")

    def paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        results_key: str = "results",
    ) -> Iterator[dict]:
        """逐页迭代结果（沿 ``next_url`` 翻页，apiKey 自动重注）。"""
        data = self.get(path, params)
        while True:
            yield from data.get(results_key) or []
            next_url = data.get("next_url")
            if not next_url:
                break
            data = self.get(next_url)
