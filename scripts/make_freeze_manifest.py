"""生成可审计的研究冻结清单 freeze/manifest.json（受 Git 跟踪）。

回应第二轮复审：§12.15 的冻结声明只有文字层，数据产物在 data/ 下不入
库，无法事后证明注册表、权重与候选定义未被修改。本脚本对全部冻结对象
计算 SHA256 并写入受 Git 跟踪的清单；终检日期与判定终点显式固定。

诚实声明：判据与计算在同一提交中出现，Git 历史不能证明判据先于计算，
本清单的性质是"计算前固化于脚本、事后可验未篡改"，不等同于第三方
时间戳的预注册。

用法：.venv/bin/python scripts/make_freeze_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "freeze" / "manifest.json"

#: 冻结对象（相对 ROOT）。数据产物不入库但哈希入库，事后可验。
FROZEN_FILES = [
    # 定义层（代码，同时受 Git 跟踪）
    "scripts/v3_defense_build.py",        # C8 及 40 信号公式
    "scripts/v3_jump_factors.py",         # E1 跳检测口径与 J1-J7
    "scripts/v3_jump_inference.py",       # J 推断口径（第二版）
    "scripts/v3_c8_incremental.py",       # C8 确认门槛（第二版）
    "scripts/v3_cross_section.py",        # E1 权重图谱与截面口径
    "scripts/v3_prereg_tests.py",         # R1-R3 定义与判定
    "scripts/v3_curve_family.py",         # R4 定义与判定
    # 数据层（不入库，哈希入库）
    "data/polymarket/features/cn_registry_v3.parquet",
    "data/cn_futures/analysis/v3/xsec/theme_signals.parquet",
    "data/cn_futures/analysis/v3/prereg/r1_d3.parquet",
    "data/cn_futures/analysis/v3/prereg/r2_resid.parquet",
    "data/cn_futures/analysis/v3/prereg/r3_flag.parquet",
    "data/cn_futures/analysis/v3/prereg/r4_curve.parquet",
    "data/intl/brent_daily.csv",
    "data/intl/usdcnh_daily.csv",
    "data/intl/curve_daily.parquet",
]

MANIFEST_META = {
    "freeze_date": "2026-07-18",
    "sample_cutoff": "2026-07-13",
    "untouched_sample_start": "2026-07-18",
    "evaluation": {
        "jump_sc_daysession": "样本积累至 2026-12-31 或新增 95 个交易日"
                              "（先到者）后一次性裁决",
        "r1_r2_r4": "2027-01-31 一次性裁决",
        "r3_flag": "2027-01-31 一次性裁决",
        "rule": "各候选仅一次预约定检验，不因结果调整定义或重试",
    },
    "primary_endpoints": {
        "jump_sc": "15' 符号化响应，wild(HAC) 与日期块置换均 p<0.05，"
                   "且净响应（毛值减 8bp）在两倍成本（16bp）后为正",
        "r1": "全部日截面 RankIC<0 且 |HAC t|>=2",
        "r2": "事件日残差 3 日自回归系数<0 且 |t|>=2",
        "r3": "lift>=2 且 Fisher p<0.05",
        "r4": "同期响应与事件后回归按注册方向且 |t|>=2",
    },
    "cost_model": "单边 4bp 基准；投产门槛按两倍成本（完整换手 16bp）",
    "multiple_testing_family": "R1-R4 与 SC 跳格共 5 个主终点，"
                               "BH-FDR q<0.10",
    "candidate_ids": ["jump_sc_daysession_15m", "r1_d3", "r2_sc_resid",
                      "r3_tail_flag", "r4_curve_family",
                      "w8_w10_iron_ore", "m_oos_increment"],
}


def sha256_of(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    hashes = {}
    for rel in FROZEN_FILES:
        digest = sha256_of(ROOT / rel)
        hashes[rel] = digest
        status = digest[:16] if digest else "MISSING"
        print(f"  {status}  {rel}")
    manifest = {**MANIFEST_META, "git_commit_at_freeze": commit,
                "sha256": hashes}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
