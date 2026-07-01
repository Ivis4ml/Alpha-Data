"""确定性内容寻址哈希（供 ``raw_hash`` / 缓存键，用于口径变更检测）。"""

from __future__ import annotations

import hashlib


def content_hash(*parts: object) -> str:
    """把参数序列化为稳定字符串后取 sha1 十六进制摘要。"""
    payload = "|".join(repr(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
