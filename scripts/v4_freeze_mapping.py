"""P1 冻结账本：把映射产物哈希写入 freeze/manifest_v2.json。

与旧账本的关系：``freeze/manifest.json``（v3）**绝不触碰**；本脚本产生
独立的 ``manifest_v2.json``。绝不运行 ``make_freeze_manifest.py``（它会用
当前 HEAD 覆盖 v3 冻结记录）。

行为：

- 计算五个对象的 SHA256：分类表 CSV、tier 表 parquet、预注册符号 JSON、
  生成脚本本身、以及只读依赖 ``cn_registry_v3.parquet``（记录其哈希以
  证明本阶段未改动 v3 登记表）。
- 已存在 ``manifest_v2.json`` 时拒绝覆盖（除非 ``--force``），防止一次
  误运行销毁冻结证据。
- 诚实声明：本冻结无第三方时间戳，判据与计算同仓库提交，仅可证明
  自冻结提交起未被篡改。

用法::

    .venv/bin/python scripts/v4_freeze_mapping.py [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_V2 = ROOT / "freeze" / "manifest_v2.json"

#: 冻结对象（相对仓库根）。
FROZEN_FILES: tuple[str, ...] = (
    "docs/product_taxonomy.csv",
    "data/polymarket/features/tier_registry.parquet",
    "freeze/prereg_signs.json",
    "scripts/v4_mapping_strength.py",
)
#: 只读依赖（记录哈希以证明未改动，不属于本阶段产物）。
READONLY_DEPS: tuple[str, ...] = (
    "data/polymarket/features/cn_registry_v3.parquet",
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="写 manifest_v2 冻结账本")
    parser.add_argument("--force", action="store_true",
                        help="覆盖已存在的 manifest_v2（危险，须说明理由）")
    args = parser.parse_args()

    if MANIFEST_V2.exists() and not args.force:
        raise SystemExit(
            f"{MANIFEST_V2} 已存在。冻结账本为一次写入；确需重建请加 "
            f"--force 并在提交说明中写明理由。")

    entry = {
        "frozen_at": date.today().isoformat(),
        "git_commit_at_freeze": git_head(),
        "purpose": ("P1 事前映射强度表冻结：主族 Bonferroni 分母固定为本表"
                    "预注册的最大 (主题, 品种) 对数；后续 gate 剪除映射"
                    "不得回缩分母、不得下调门槛。"),
        "frozen": {p: sha256_of(ROOT / p) for p in FROZEN_FILES},
        "readonly_deps": {p: sha256_of(ROOT / p) for p in READONLY_DEPS},
        "disclaimer": ("本冻结无第三方时间戳；判据与计算在同一仓库提交，"
                       "仅可证明自冻结提交起未被篡改，不构成外部公证。"),
    }
    MANIFEST_V2.write_text(json.dumps(entry, ensure_ascii=False, indent=2))
    print(f"written {MANIFEST_V2}")
    for p, h in {**entry["frozen"], **entry["readonly_deps"]}.items():
        print(f"  {h[:16]}  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
