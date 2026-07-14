"""下载 HuggingFace 数据集 ``TimeSeventeen/Polymarket-v1`` 到本地。

默认下载 ``daily_aligned`` + ``ctf``（约 22 GB）；``--all`` 追加 ``orderfilled``（共约 49 GB）。
``--complete`` 下载整个仓库（含 ``README.md``、``.gitattributes`` 等根目录文件，覆盖 ``--all``
未覆盖的边角文件），忽略分层选择，同样约 49 GB。
下载可断点续传（``snapshot_download`` 跳过已完整文件）。``--list-only`` 仅打印仓库结构不下载。

环境变量 ``HF_TOKEN``（可选，提升吞吐）从仓库根的 ``.env`` 读取。

用法::

    python scripts/download_polymarket.py --list-only
    python scripts/download_polymarket.py
    python scripts/download_polymarket.py --all
    python scripts/download_polymarket.py --complete
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "TimeSeventeen/Polymarket-v1"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "polymarket"
ENV_FILE = ROOT / ".env"

# 默认下载层（大小写不敏感匹配仓库实际顶层目录名）。
DEFAULT_LAYERS: tuple[str, ...] = ("daily_aligned", "ctf")
ALL_EXTRA_LAYERS: tuple[str, ...] = ("orderfilled",)


def _load_env(path: Path) -> None:
    """读取 ``.env`` 到环境变量（优先 python-dotenv，回退极简解析）。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
        return
    except Exception:
        pass
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _list_siblings(token: str | None) -> list:
    """获取仓库全部文件条目（含根目录松散文件，如 README.md）。"""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
    return list(info.siblings)


def _summarize_top_dirs(siblings: list) -> dict[str, tuple[int, int]]:
    """返回仓库顶层目录 -> (文件数, 总字节)；根目录松散文件不计入。"""
    top: dict[str, list[int]] = {}
    for sibling in siblings:
        parts = sibling.rfilename.split("/")
        if len(parts) < 2:
            continue
        directory = parts[0]
        size = int(sibling.size or 0)
        agg = top.setdefault(directory, [0, 0])
        agg[0] += 1
        agg[1] += size
    return {d: (n, b) for d, (n, b) in top.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 Polymarket-v1 指定层或整个仓库")
    parser.add_argument("--all", action="store_true", help="追加 orderfilled 原始层（约 49 GB）")
    parser.add_argument(
        "--complete",
        action="store_true",
        help="下载整个仓库（含 README.md 等根目录文件），忽略 --all 与默认分层选择，约 49 GB",
    )
    parser.add_argument("--list-only", action="store_true", help="仅打印仓库结构，不下载")
    parser.add_argument("--out", default=str(OUT), help="输出目录")
    args = parser.parse_args()

    _load_env(ENV_FILE)
    token = os.environ.get("HF_TOKEN") or None

    # huggingface_hub>=1.x 用 Xet 后端做高性能传输（hf_transfer 已弃用）。
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    siblings = _list_siblings(token)
    top = _summarize_top_dirs(siblings)
    print("仓库顶层目录：")
    for directory, (n, b) in sorted(top.items()):
        print(f"  {directory:16s} files={n:5d}  size={b / 1e9:7.2f} GB")

    patterns: list[str] | None
    if args.complete:
        total_files = len(siblings)
        total_bytes = sum(int(s.size or 0) for s in siblings)
        print(
            f"\n--complete：将下载整个仓库（{total_files} 个文件，含根目录文件）"
            f"约 {total_bytes / 1e9:.1f} GB -> {args.out}"
        )
        patterns = None
    else:
        want = list(DEFAULT_LAYERS) + (list(ALL_EXTRA_LAYERS) if args.all else [])
        lower_to_actual = {d.lower(): d for d in top}
        selected: list[str] = []
        patterns = []
        for layer in want:
            actual = lower_to_actual.get(layer.lower())
            if actual is None:
                print(f"  警告：未在仓库找到层 {layer!r}，跳过")
                continue
            patterns.append(f"{actual}/*")
            selected.append(actual)

        if not patterns:
            print("没有可下载的层，退出。")
            return 1

        selected_bytes = sum(top[d][1] for d in selected)
        print(f"\n将下载：{selected}（约 {selected_bytes / 1e9:.1f} GB）-> {args.out}")

    if args.list_only:
        print("（--list-only：不执行下载）")
        return 0

    print("snapshot_download 断点续传中……\n", flush=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=patterns,
        token=token,
        max_workers=8,
    )
    print("\n下载完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
