"""读取仓库根的 ``.env`` 到环境变量。

优先使用 python-dotenv；不可用时回退极简解析。``setdefault`` 语义：已存在的环境变量不被覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | None = None) -> Path:
    """加载 ``.env``。返回实际读取的路径（即使文件不存在也返回预期路径）。"""
    target = path or (REPO_ROOT / ".env")
    try:
        from dotenv import load_dotenv

        load_dotenv(target)
        return target
    except Exception:
        pass
    if target.exists():
        for line in target.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    return target
