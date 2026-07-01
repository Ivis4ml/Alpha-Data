"""``MassiveProvider``：实现 AlphaForge ``MinuteProvider`` 协议，用 REST 聚合拉取 1min bar。

用途：增量 / 补缺（``Ingester`` 依赖）。批量历史走 Flat Files（见 ``flatfiles`` /
``minute_store``），本 provider 面向单标的、指定区间的按需拉取。

口径（与 Flat Files 一致）：REST ``t`` 为 bar 起始（毫秒 epoch，UTC），``+60s`` 得 bar 收盘
时刻，转 ``America/New_York`` 得 ``ts``（``'YYYY-MM-DD HH:MM:SS'``）。价格 ``adjusted=false``
（原始未复权）。``extended_hours=False`` 时仅保留 RTH（``09:30 < 收盘戳 <= 16:00``）。

依赖 AlphaForge 的 ``FetchResult`` 类型（本 provider 的产物直接交给 AlphaForge ``Ingester``）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from alphaforge.providers.ingest.provider import FetchResult

from alpha_data.common.hashing import content_hash
from alpha_data.equity.massive_client import MassiveClient, MassiveError

_ET = ZoneInfo("America/New_York")


def _close_ts_et(start_ms: int) -> str:
    """bar 起始毫秒 UTC -> bar 收盘时刻（+60s）-> 美东本地 ``'YYYY-MM-DD HH:MM:SS'``。"""
    dt = datetime.fromtimestamp((start_ms + 60_000) / 1000.0, tz=UTC).astimezone(_ET)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class MassiveProvider:
    """massive.com REST 1min 供应商（满足 ``MinuteProvider`` 结构化协议）。"""

    name: str = "massive"

    def __init__(self, client: MassiveClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> MassiveClient:
        if self._client is None:
            self._client = MassiveClient()
        return self._client

    def available(self) -> bool:
        """密钥就绪即可用。"""
        try:
            _ = self.client
            return True
        except MassiveError:
            return False

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        interval: str = "1min",
        extended_hours: bool = False,
    ) -> FetchResult:
        """拉取 ``[start, end]`` 闭区间（``YYYY-MM-DD``）的 1min bar（原始未复权）。"""
        if interval != "1min":
            return FetchResult(
                bars=[], status="error", note=f"massive: 仅支持 1min，收到 {interval}"
            )
        sym = symbol.strip().upper()
        path = f"/v2/aggs/ticker/{sym}/range/1/minute/{start}/{end}"
        params = {"adjusted": "false", "sort": "asc", "limit": 50000}
        try:
            rows = list(self.client.paginate(path, params))
        except MassiveError as exc:
            note = str(exc)
            status = "rate_limited" if ("429" in note or "限速" in note) else "error"
            return FetchResult(bars=[], status=status, note=f"massive: {note}")

        bars: list[dict] = []
        for r in rows:
            ts = _close_ts_et(int(r["t"]))
            hhmm = ts[11:16]
            if not extended_hours and not ("09:30" < hhmm <= "16:00"):
                continue
            bar = {
                "ts": ts,
                "open": float(r["o"]),
                "high": float(r["h"]),
                "low": float(r["l"]),
                "close": float(r["c"]),
                "volume": float(r["v"]),
            }
            if r.get("vw") is not None:
                bar["vwap"] = float(r["vw"])
            if r.get("n") is not None:
                bar["trade_count"] = int(r["n"])
            bars.append(bar)

        if not bars:
            return FetchResult(
                bars=[], status="empty", note=f"massive: {sym} {start}..{end} 无 bar"
            )
        raw_hash = content_hash(
            sym, start, end, extended_hours, len(bars), bars[0]["ts"], bars[-1]["ts"]
        )
        return FetchResult(
            bars=bars, status="ok", note=f"massive: {len(bars)} bars", raw_hash=raw_hash
        )
