"""P1 映射强度表的回归测试（覆盖、确定性、v3 一致性、冻结完整性）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v4_mapping_strength as vms  # noqa: E402

MINUTE_DIR = ROOT / "data" / "cn_futures" / "minute"
EQUITY_MIN = ROOT / "data" / "equity" / "minute_db" / "minute"
MANIFEST_V2 = ROOT / "freeze" / "manifest_v2.json"


def test_taxonomy_covers_minute_db_exactly() -> None:
    """分类表品种集合 = 分钟库品种集合（88 个，一一对应）。"""
    if not MINUTE_DIR.exists():
        pytest.skip("分钟库不在本机")
    on_disk = {p.name for p in MINUTE_DIR.iterdir() if p.is_dir()}
    assert set(vms.TAXONOMY) == on_disk


def test_taxonomy_sectors_frozen() -> None:
    tax = vms.build_taxonomy()
    assert set(tax["sector"]) <= set(vms.SECTORS)
    assert tax["product"].is_unique


def test_tier_values_and_signs() -> None:
    tier = vms.build_tier_table()
    assert set(tier["tier"]) <= {"primary", "secondary", "exploratory"}
    assert set(tier["m_prior"]) <= {-1, 1}
    # 每个 (theme, product) 至多一行（首个命中规则生效）
    assert not tier.duplicated(["theme", "product"]).any()


def test_v3_pairs_inherited_exactly() -> None:
    """v3 登记表的 17 对 (theme, product) 方向先验被精确继承。

    build_tier_table 内部已断言；此处独立复核，防止内部断言被削弱。
    """
    if not vms.REGISTRY_V3.exists():
        pytest.skip("v3 登记表不在本机")
    tier = vms.build_tier_table().set_index(["theme", "product"])["m_prior"]
    v3 = vms.registry_m_priors()
    for _, r in v3.iterrows():
        assert tier.loc[(r["theme"], r["product"])] == r["m_v3"], \
            f"{r['theme']}/{r['product']} 方向先验与 v3 不一致"


def test_deterministic_rebuild() -> None:
    """重复构建逐值相同（tier 表与分类表都是纯函数）。"""
    a, b = vms.build_tier_table(), vms.build_tier_table()
    pd.testing.assert_frame_equal(a, b)
    ta, tb = vms.build_taxonomy(), vms.build_taxonomy()
    pd.testing.assert_frame_equal(ta, tb)


def test_anchor_etfs_exist_in_equity_db() -> None:
    """主题锚与品种基准 ETF 必须真实存在于美股分钟库。"""
    if not EQUITY_MIN.exists():
        pytest.skip("美股分钟库不在本机")
    for theme, (anchors, _q) in vms.THEME_ANCHORS.items():
        for etf in anchors:
            assert (EQUITY_MIN / etf).is_dir(), f"{theme} 锚 {etf} 不在库"
    for prod, (_n, _s, _ip, etfs) in vms.TAXONOMY.items():
        for etf in etfs:
            assert (EQUITY_MIN / etf).is_dir(), f"{prod} 基准 {etf} 不在库"


def test_prereg_signs_wellformed() -> None:
    for name, e in vms.PREREG_SIGNS.items():
        assert e["direction_prior"] in (-1, 0, 1), name
        assert e["registered_at"], name
        assert e["definition"] and e["rationale"], name


@pytest.mark.skipif(not MANIFEST_V2.exists(), reason="manifest_v2 未生成")
def test_manifest_v2_hashes_current() -> None:
    """冻结哈希与当前文件一致；v3 登记表未被改动；v3 账本未被触碰。"""
    import v4_freeze_mapping as vfm
    m = json.loads(MANIFEST_V2.read_text())
    for rel, recorded in {**m["frozen"], **m["readonly_deps"]}.items():
        assert vfm.sha256_of(ROOT / rel) == recorded, f"{rel} 与冻结哈希不符"
    # v3 账本仍在且未被本阶段修改（git 层面另有保证，这里做存在性检查）
    assert (ROOT / "freeze" / "manifest.json").exists()
