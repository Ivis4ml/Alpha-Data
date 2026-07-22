"""P1 事前映射强度表：88 品种参照表 + (主题, 品种) 分档 + 预注册符号。

产物（全部由本文件内的冻结字典确定性生成，重跑结果逐值相同）：

1. ``docs/product_taxonomy.csv``：88 品种 -> (中文名, 板块, 国际定价程度,
   国际基准 ETF 清单)。全部为合约规格与公开常识，属事前信息，不含任何
   样本内统计量。
2. ``data/polymarket/features/tier_registry.parquet``：每个 (主题, 品种)
   对的映射强度（primary / secondary / exploratory）、方向先验 m_prior
   与机制理由。v3 已注册的 17 对 (theme, product) 的 m 必须被精确继承
   （m = sigma * orientation，在对内恒定，构建时断言一致）。
3. ``freeze/prereg_signs.json``：下一阶段候选因子的预期符号。凡未在计算
   前登记于此的因子，一律不得进入确认层。

纪律（见 docs/next_phase_design.md §2 P1）：

- 主族 Bonferroni 分母固定为预注册最大对数，后续 gate 剪除映射不得回缩；
- ``cn_registry_v3.parquet`` 保持冻结不动，本脚本只读；
- 分档只用经济先验与外部锚可得性，不读任何期货收益。

用法::

    .venv/bin/python scripts/v4_mapping_strength.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_CSV = ROOT / "docs" / "product_taxonomy.csv"
TIER_PARQUET = ROOT / "data" / "polymarket" / "features" / "tier_registry.parquet"
PREREG_JSON = ROOT / "freeze" / "prereg_signs.json"
REGISTRY_V3 = ROOT / "data" / "polymarket" / "features" / "cn_registry_v3.parquet"

#: 冻结板块集合（分类表中的板块必须取自此集合）。
SECTORS: tuple[str, ...] = (
    "能源", "贵金属", "有色金属", "新能源金属", "黑色金属", "化工",
    "农产品-油脂油料", "农产品-谷物", "农产品-软商品", "农产品-畜牧",
    "林木与纸", "股指", "国债", "航运",
)

#: 国际定价程度：international = 全球套利充分（LME/COMEX/CBOT 直连）；
#: partial = 有国际联动但受配额 / 运费 / 品质差隔离；domestic = 国内定价。
#: 88 品种 -> (中文名, 板块, 国际定价, 基准 ETF 元组)。
#: ETF 全部经核实存在于美股分钟库（data/equity/minute_db/minute/）。
TAXONOMY: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    # ---- 能源 ----
    "SC": ("原油", "能源", "international", ("USO", "BNO", "DBE")),
    "FU": ("燃料油", "能源", "international", ("USO", "BNO", "DBE")),
    "LU": ("低硫燃料油", "能源", "international", ("USO", "BNO", "DBE")),
    "PG": ("液化石油气", "能源", "partial", ("USO", "DBE")),
    "ZC": ("动力煤", "能源", "domestic", ()),
    # ---- 贵金属 ----
    "AU": ("黄金", "贵金属", "international", ("GLD",)),
    "AG": ("白银", "贵金属", "international", ("SLV",)),
    "PT": ("铂", "贵金属", "international", ("PPLT",)),
    "PD": ("钯", "贵金属", "international", ("PALL",)),
    # ---- 有色金属 ----
    "CU": ("铜", "有色金属", "international", ("CPER", "DBB")),
    "BC": ("国际铜", "有色金属", "international", ("CPER", "DBB")),
    "AL": ("铝", "有色金属", "international", ("DBB",)),
    "AO": ("氧化铝", "有色金属", "partial", ("DBB",)),
    "AD": ("铸造铝合金", "有色金属", "partial", ("DBB",)),
    "ZN": ("锌", "有色金属", "international", ("DBB",)),
    "PB": ("铅", "有色金属", "international", ("DBB",)),
    "NI": ("镍", "有色金属", "international", ("JJN", "DBB")),
    "SN": ("锡", "有色金属", "international", ("DBB",)),
    "SS": ("不锈钢", "有色金属", "partial", ("JJN",)),
    # ---- 新能源金属 ----
    "LC": ("碳酸锂", "新能源金属", "partial", ("LIT",)),
    "SI": ("工业硅", "新能源金属", "domestic", ()),
    "PS": ("多晶硅", "新能源金属", "domestic", ()),
    # ---- 黑色金属 ----
    "I": ("铁矿石", "黑色金属", "partial", ("SLX",)),
    "RB": ("螺纹钢", "黑色金属", "domestic", ("SLX",)),
    "HC": ("热轧卷板", "黑色金属", "domestic", ("SLX",)),
    "WR": ("线材", "黑色金属", "domestic", ()),
    "J": ("焦炭", "黑色金属", "domestic", ()),
    "JM": ("焦煤", "黑色金属", "domestic", ()),
    "SF": ("硅铁", "黑色金属", "domestic", ()),
    "SM": ("锰硅", "黑色金属", "domestic", ()),
    # ---- 化工 ----
    "TA": ("PTA", "化工", "partial", ("USO",)),
    "PX": ("对二甲苯", "化工", "partial", ("USO",)),
    "PF": ("涤纶短纤", "化工", "partial", ("USO",)),
    "PR": ("瓶片", "化工", "partial", ("USO",)),
    "MA": ("甲醇", "化工", "partial", ("USO", "UNG")),
    "EG": ("乙二醇", "化工", "partial", ("USO",)),
    "EB": ("苯乙烯", "化工", "partial", ("USO",)),
    "BZ": ("纯苯", "化工", "partial", ("USO",)),
    "PL": ("丙烯", "化工", "partial", ("USO",)),
    "PP": ("聚丙烯", "化工", "partial", ("USO",)),
    "L": ("聚乙烯", "化工", "partial", ("USO",)),
    "V": ("聚氯乙烯", "化工", "domestic", ()),
    "BU": ("石油沥青", "化工", "partial", ("USO", "BNO")),
    "BR": ("丁二烯橡胶", "化工", "partial", ("USO",)),
    "RU": ("天然橡胶", "化工", "partial", ()),
    "NR": ("20号胶", "化工", "partial", ()),
    "SA": ("纯碱", "化工", "domestic", ()),
    "FG": ("玻璃", "化工", "domestic", ()),
    "SH": ("烧碱", "化工", "domestic", ()),
    "UR": ("尿素", "化工", "partial", ()),
    # ---- 农产品：油脂油料 ----
    "M": ("豆粕", "农产品-油脂油料", "partial", ("SOYB",)),
    "A": ("豆一", "农产品-油脂油料", "partial", ("SOYB",)),
    "B": ("豆二", "农产品-油脂油料", "international", ("SOYB",)),
    "Y": ("豆油", "农产品-油脂油料", "partial", ("SOYB", "DBA")),
    "P": ("棕榈油", "农产品-油脂油料", "international", ("DBA",)),
    "OI": ("菜籽油", "农产品-油脂油料", "partial", ("DBA",)),
    "RM": ("菜籽粕", "农产品-油脂油料", "partial", ("DBA",)),
    "RS": ("油菜籽", "农产品-油脂油料", "domestic", ()),
    "PK": ("花生", "农产品-油脂油料", "domestic", ()),
    # ---- 农产品：谷物 ----
    "C": ("玉米", "农产品-谷物", "domestic", ("CORN",)),
    "CS": ("玉米淀粉", "农产品-谷物", "domestic", ("CORN",)),
    "WH": ("强麦", "农产品-谷物", "domestic", ("WEAT",)),
    "PM": ("普麦", "农产品-谷物", "domestic", ("WEAT",)),
    "RI": ("早籼稻", "农产品-谷物", "domestic", ()),
    "LR": ("晚籼稻", "农产品-谷物", "domestic", ()),
    "JR": ("粳稻", "农产品-谷物", "domestic", ()),
    "RR": ("粳米", "农产品-谷物", "domestic", ()),
    # ---- 农产品：软商品 ----
    "CF": ("棉花", "农产品-软商品", "partial", ("BAL",)),
    "CY": ("棉纱", "农产品-软商品", "domestic", ("BAL",)),
    "SR": ("白糖", "农产品-软商品", "partial", ("CANE",)),
    "AP": ("苹果", "农产品-软商品", "domestic", ()),
    "CJ": ("红枣", "农产品-软商品", "domestic", ()),
    # ---- 农产品：畜牧 ----
    "JD": ("鸡蛋", "农产品-畜牧", "domestic", ()),
    "LH": ("生猪", "农产品-畜牧", "domestic", ()),
    # ---- 林木与纸 ----
    "SP": ("纸浆", "林木与纸", "partial", ()),
    "LG": ("原木", "林木与纸", "partial", ()),
    "FB": ("纤维板", "林木与纸", "domestic", ()),
    "BB": ("胶合板", "林木与纸", "domestic", ()),
    "OP": ("胶版印刷纸", "林木与纸", "domestic", ()),
    # ---- 股指 ----
    "IF": ("沪深300", "股指", "partial", ("FXI",)),
    "IH": ("上证50", "股指", "partial", ("FXI",)),
    "IC": ("中证500", "股指", "partial", ("FXI",)),
    "IM": ("中证1000", "股指", "partial", ("FXI", "KWEB")),
    # ---- 国债 ----
    "T": ("10年期国债", "国债", "domestic", ("TLT",)),
    "TF": ("5年期国债", "国债", "domestic", ("TLT",)),
    "TS": ("2年期国债", "国债", "domestic", ("TLT",)),
    "TL": ("30年期国债", "国债", "domestic", ("TLT",)),
    # ---- 航运 ----
    "EC": ("集运指数欧线", "航运", "partial", ("BDRY",)),
}

#: 主题轴（与 select_polymarket_markets.RULES 的 sigma 语义一致，不得改）。
THEME_AXES: dict[str, str] = {
    "mideast_conflict": "+1 = 冲突升级",
    "oil_price": "+1 = 油价上行",
    "metal_price": "+1 = 金银价格上行",
    "fed_policy": "+1 = 宽松（降息）",
    "russia_ukraine": "+1 = 冲突升级",
    "us_china_trade": "+1 = 贸易紧张升级",
    "taiwan_risk": "+1 = 台海风险升级",
    "us_shutdown": "+1 = 政府停摆发生",
}

#: 主题 -> 外部锚 ETF（第二阶段分级器用；strong = 直接商品锚，
#: medium = 宏观锚，weak = 无直接锚只有弱代理）。全部核实在美股分钟库。
THEME_ANCHORS: dict[str, tuple[tuple[str, ...], str]] = {
    "mideast_conflict": (("USO", "BNO", "GLD"), "strong"),
    "oil_price": (("USO", "BNO"), "strong"),
    "metal_price": (("GLD", "SLV"), "strong"),
    "fed_policy": (("TLT", "UUP", "GLD"), "medium"),
    "russia_ukraine": (("WEAT", "USO", "UNG"), "medium"),
    "us_china_trade": (("FXI", "SOYB", "KWEB"), "medium"),
    "taiwan_risk": (("EWT", "FXI"), "medium"),
    "us_shutdown": (("SPY", "TLT", "GLD"), "weak"),
}

#: v3 已按探索性登记的主题（继承，不重判）。
THEMES_EXPLORATORY_V3 = {"us_china_trade", "taiwan_risk", "us_shutdown"}

#: 原油成本链化工子集（原油 -> 芳烃 / 烯烃 / 沥青路线，成本传导直接）。
OIL_CHAIN: tuple[str, ...] = ("TA", "PX", "PF", "PR", "EG", "EB", "BZ",
                              "PL", "PP", "L", "V", "BU", "MA", "BR")

#: 主题 -> [(板块或品种子集, 档位, m_prior, 机制理由)]。
#: 板块规则展开到该板块全部品种；OIL_CHAIN 为显式品种子集。
#: 档位判据（冻结）：primary = 教科书一阶传导（直接标的 / 供给冲击）；
#: secondary = 真实但二阶（避险、成本、风险偏好）；exploratory = 弱先验。
Rule = tuple[str, str, int, str]
THEME_SECTOR_MAP: dict[str, list[Rule]] = {
    "mideast_conflict": [
        ("能源", "primary", +1, "中东供给风险溢价直接推升油价"),
        ("航运", "primary", +1, "红海航线中断迫使绕行好望角推升运价"),
        ("贵金属", "secondary", +1, "地缘避险需求"),
        ("OIL_CHAIN", "secondary", +1, "原油成本沿化工链传导"),
    ],
    "oil_price": [
        ("能源", "primary", +1, "同一标的的价格阶梯市场"),
        ("OIL_CHAIN", "secondary", +1, "原油成本沿化工链传导"),
    ],
    "metal_price": [
        ("贵金属", "primary", +1, "同一标的（金银价格阶梯）"),
        ("有色金属", "exploratory", +1, "贵金属与工业金属的共同宏观因子"),
    ],
    "fed_policy": [
        ("贵金属", "primary", +1, "宽松降低零息资产持有成本"),
        ("有色金属", "secondary", +1, "宽松抬升全球工业需求预期与美元计价"),
        ("股指", "secondary", +1, "宽松改善风险偏好与流动性"),
        ("新能源金属", "exploratory", +1, "利率敏感的资本开支链"),
        ("国债", "exploratory", +1, "全球利率下行的外溢（中国货币政策独立性强）"),
    ],
    "russia_ukraine": [
        ("能源", "primary", +1, "俄油气供给与制裁风险"),
        ("贵金属", "secondary", +1, "地缘避险需求"),
        ("农产品-谷物", "secondary", +1, "黑海谷物出口中断"),
        ("农产品-油脂油料", "exploratory", +1, "葵油供给中断的植物油替代"),
        ("UR", "secondary", +1, "俄罗斯化肥出口受限"),
    ],
    "us_china_trade": [
        ("农产品-油脂油料", "primary", +1, "对美农产品反制导致进口替代、国内蛋白粕上行"),
        ("CF", "secondary", +1, "棉花进口替代"),
        ("有色金属", "secondary", -1, "贸易摩擦压制全球需求与风险偏好"),
        ("黑色金属", "secondary", -1, "出口与制造业需求预期下行"),
        ("股指", "secondary", -1, "风险偏好与盈利预期下行"),
        ("农产品-谷物", "exploratory", +1, "玉米高粱进口替代"),
    ],
    "taiwan_risk": [
        ("股指", "secondary", -1, "地缘风险压制权益估值"),
        ("贵金属", "secondary", +1, "地缘避险需求"),
        ("航运", "exploratory", +1, "台海航线风险溢价"),
    ],
    "us_shutdown": [
        ("贵金属", "secondary", +1, "美国财政不确定性避险"),
        ("股指", "exploratory", -1, "风险偏好小幅下行"),
    ],
}

#: 下一阶段候选因子的预期符号预注册（P4 分散度矩族先行）。
#: direction_prior: +1 / -1 = 单向先验；0 = 无方向先验（双侧检验）。
#: 纪律：direction_prior 必须在任何相关统计量计算之前写入本表；
#: 事后新增条目必须带 registered_at 且不得修改既有条目。
PREREG_SIGNS: dict[str, dict] = {
    "D_disp": {
        "family": "dispersion",
        "definition": "主题内定向创新的成交额加权标准差（分歧度）",
        "direction_prior": 0,
        "vol_channel": "+1",
        "rationale": "分歧度对收益方向无先验；对已实现波动的先验为正",
        "registered_at": "2026-07-22",
    },
    "D_cancel": {
        "family": "dispersion",
        "definition": "|Σ w·Δℓ| / Σ|w·Δℓ|（合并的抵消存活率，1=全同向）",
        "direction_prior": 0,
        "vol_channel": "0",
        "rationale": "作为 N1 方向的调节项使用，本身无方向先验",
        "registered_at": "2026-07-22",
    },
    "D_hhi": {
        "family": "dispersion",
        "definition": "主题内市场成交额份额的 Herfindahl 集中度",
        "direction_prior": 0,
        "vol_channel": "0",
        "rationale": "衡量信息集中于单一市场还是普遍，本身无方向先验",
        "registered_at": "2026-07-22",
    },
    "D_sign_ratio": {
        "family": "dispersion",
        "definition": "主题内创新多数符号占比",
        "direction_prior": 0,
        "vol_channel": "0",
        "rationale": "方向一致性调节项，无独立方向先验",
        "registered_at": "2026-07-22",
    },
    "D_head_share": {
        "family": "dispersion",
        "definition": "头部市场对主题创新的贡献占比",
        "direction_prior": 0,
        "vol_channel": "0",
        "rationale": "信息来源结构，无方向先验",
        "registered_at": "2026-07-22",
    },
    "D_skew": {
        "family": "dispersion",
        "definition": "主题内定向创新的偏度",
        "direction_prior": 0,
        "vol_channel": "0",
        "rationale": "极端市场拉动 vs 普遍位移的形状量，无方向先验",
        "registered_at": "2026-07-22",
    },
}


def registry_m_priors() -> pd.DataFrame:
    """从冻结的 v3 登记表提取 (theme, product) 的方向先验 m（只读）。"""
    reg = pd.read_parquet(REGISTRY_V3)
    reg = reg.assign(m=reg["sigma"] * reg["orientation"])
    grp = reg.groupby(["theme", "product"])["m"]
    if int(grp.nunique().max()) != 1:
        raise AssertionError("v3 登记表中 m 在 (theme, product) 内不恒定")
    return grp.first().astype(int).rename("m_v3").reset_index()


def build_taxonomy() -> pd.DataFrame:
    rows = [
        {"product": p, "name_cn": n, "sector": s, "intl_pricing": ip,
         "benchmark_etfs": ";".join(etfs)}
        for p, (n, s, ip, etfs) in sorted(TAXONOMY.items())
    ]
    df = pd.DataFrame(rows)
    bad = set(df["sector"]) - set(SECTORS)
    if bad:
        raise AssertionError(f"未冻结的板块: {bad}")
    return df


def build_tier_table() -> pd.DataFrame:
    """按冻结规则展开 (主题, 品种) 分档；品种级规则优先于板块级。"""
    tax = {p: s for p, (_, s, _, _) in TAXONOMY.items()}
    rows: list[dict] = []
    for theme, rules in THEME_SECTOR_MAP.items():
        anchors, anchor_q = THEME_ANCHORS[theme]
        seen: set[str] = set()
        # 品种级规则先行（优先级高）
        for target, tier, m, why in rules:
            if target in TAXONOMY:
                products = [target]
            elif target == "OIL_CHAIN":
                products = list(OIL_CHAIN)
            elif target in SECTORS:
                products = [p for p, s in tax.items() if s == target]
            else:
                raise AssertionError(f"未知规则目标: {target}")
            for p in products:
                if p in seen:      # 首个命中的规则生效（与 v3 RULES 同约定）
                    continue
                seen.add(p)
                rows.append({
                    "theme": theme,
                    "product": p,
                    "sector": tax[p],
                    "tier": tier,
                    "m_prior": m,
                    "rationale": why,
                    "theme_axis": THEME_AXES[theme],
                    "theme_anchors": ";".join(anchors),
                    "anchor_quality": anchor_q,
                    "theme_exploratory_v3": theme in THEMES_EXPLORATORY_V3,
                })
    df = pd.DataFrame(rows).sort_values(["theme", "tier", "product"]) \
        .reset_index(drop=True)

    # 与 v3 登记表的 17 对既有 (theme, product) 方向先验必须精确一致
    v3 = registry_m_priors()
    merged = v3.merge(df[["theme", "product", "m_prior"]],
                      on=["theme", "product"], how="left")
    missing = merged[merged["m_prior"].isna()]
    if len(missing):
        raise AssertionError(
            "v3 已注册对未被 v4 覆盖:\n" + missing.to_string(index=False))
    clash = merged[merged["m_v3"] != merged["m_prior"].astype(int)]
    if len(clash):
        raise AssertionError(
            "v4 方向先验与 v3 登记表冲突:\n" + clash.to_string(index=False))
    return df


def main() -> int:
    tax = build_taxonomy()
    tier = build_tier_table()

    TAXONOMY_CSV.parent.mkdir(parents=True, exist_ok=True)
    tax.to_csv(TAXONOMY_CSV, index=False)
    TIER_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tier.to_parquet(TIER_PARQUET, index=False)
    PREREG_JSON.parent.mkdir(parents=True, exist_ok=True)
    PREREG_JSON.write_text(json.dumps(
        {"meta": {
            "purpose": "候选因子预期符号预注册；计算前未登记者不得进确认层",
            "discipline": "既有条目不得修改；新增须带 registered_at",
            "created_at": "2026-07-22",
        }, "signs": PREREG_SIGNS},
        ensure_ascii=False, indent=2))

    print(f"taxonomy -> {TAXONOMY_CSV}  ({len(tax)} 品种)")
    print(f"tier     -> {TIER_PARQUET}  ({len(tier)} 个 (主题,品种) 对)")
    print(f"prereg   -> {PREREG_JSON}  ({len(PREREG_SIGNS)} 个因子符号)")
    print()
    print("档位 x 主题分布：")
    print(tier.pivot_table(index="theme", columns="tier", values="product",
                           aggfunc="count", fill_value=0).to_string())
    n_mapped = tier["product"].nunique()
    unmapped = sorted(set(TAXONOMY) - set(tier["product"]))
    print(f"\n被映射品种 {n_mapped}/88；未映射（归 exploratory 池，不进主族）"
          f"{len(unmapped)} 个：")
    print("  " + " ".join(unmapped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
