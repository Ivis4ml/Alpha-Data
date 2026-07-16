"""市场族识别与事件类型标注（框架 §7 / §8 的输入）。

从 ``market_slug`` 解析两类结构化族：

- **价格阈值族**（``price_ladder``）：``will-{asset}-hit-{high|low}-{strike}-{deadline}``，
  同 (资产, 方向, 截止) 内按行权价构成梯子。这类市场的结果是标的价格的确定函数，
  按框架 §8.2 标注 ``asset_price_threshold``、``layer4_eligible = False``，只进入
  Layer 3 分布恢复。
- **日期阶梯族**（``date_ladder``）：``{stem}-by-{deadline}``，同一事件干（stem）不同
  截止日期构成累计分布 ``F_t(T_k)``，进入 Layer 3 hazard 恢复。

slug 尾部的去重噪音后缀固定为 3 位数字组（实测），日期天数为 1-2 位、年份 4 位、
行权价非尾随，故可无歧义剥离。

截止时刻精度为天（用于族内排序、剩余期限 tau 与 hazard 区间；解析为该日期次日
0 点 UTC，Polymarket 实际结算条款多为美东日终，误差数小时，对天级用途无影响）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december"]
    )
}
_MONTH_RE = "|".join(_MONTHS)

#: 尾部去重噪音：一个或多个 3 位数字组。
_NOISE_RE = re.compile(r"(?:-\d{3})+$")

_PRICE_RE = re.compile(
    rf"^(?:will-)?(?P<asset>[a-z0-9-]+?)-hit-(?P<bar>high|low)-(?P<strike>\d+(?:pt\d+)?)"
    rf"-(?P<dl>(?:by-end-of|in)-(?:{_MONTH_RE})|by-(?:{_MONTH_RE})-\d{{1,2}}(?:-\d{{4}})?)$"
)

_DATE_DL_RE = re.compile(
    rf"^(?P<stem>.+?)-(?P<form>by|before|in|on)-(?P<dl>"
    rf"(?:the-)?(?:end-of-)?"
    rf"(?:(?:{_MONTH_RE})(?:-\d{{1,2}})?(?:-\d{{4}})?(?:-meeting)?"
    rf"|\d{{4}}))$"
)


@dataclass
class ParsedMarket:
    """单个市场的族解析结果。

    Attributes:
        family_type: ``price_ladder`` / ``date_ladder`` / ``single``。
        family_key: 族标识（价格族为 ``{asset}:{bar}:{deadline}``，日期族为 stem）。
        strike: 行权价（价格族），其余 None。
        barrier: ``high`` / ``low``（价格族），其余 None。
        deadline: 截止日（date），无法解析为 None。
    """

    family_type: str
    family_key: str
    strike: float | None
    barrier: str | None
    deadline: object


def _month_end(year: int, month: int) -> datetime:
    nxt = datetime(year + (month == 12), month % 12 + 1, 1)
    return nxt - timedelta(days=1)


def _parse_deadline(dl: str, *, default_year: int = 2026) -> datetime | None:
    """截止串 -> 日期（``end-of-{month}`` / ``{month}[-day][-year][-meeting]`` / ``{year}``）。"""
    dl = dl.removeprefix("the-")
    dl = dl.removeprefix("end-of-")
    dl = dl.removesuffix("-meeting")
    if re.fullmatch(r"\d{4}", dl):
        return datetime(int(dl), 12, 31)
    parts = dl.split("-")
    month = _MONTHS.get(parts[0])
    if month is None:
        return None
    day: int | None = None
    year = default_year
    for tok in parts[1:]:
        if re.fullmatch(r"\d{4}", tok):
            year = int(tok)
        elif re.fullmatch(r"\d{1,2}", tok):
            day = int(tok)
    if day is None:
        return _month_end(year, month)
    try:
        return datetime(year, month, day)
    except ValueError:
        return _month_end(year, month)


def parse_slug(slug: str) -> ParsedMarket:
    """解析单个 slug；不匹配任何族模式时返回 ``single``（family_key 为去噪 slug）。"""
    s = _NOISE_RE.sub("", slug.lower())

    m = _PRICE_RE.match(s)
    if m:
        deadline = _parse_deadline(
            m.group("dl").removeprefix("by-").removeprefix("in-")
        )
        strike = float(m.group("strike").replace("pt", "."))
        dl_key = deadline.strftime("%Y-%m-%d") if deadline else "na"
        return ParsedMarket(
            family_type="price_ladder",
            family_key=f"{m.group('asset')}:{m.group('bar')}:{dl_key}",
            strike=strike,
            barrier=m.group("bar"),
            deadline=deadline.date() if deadline else None,
        )

    m = _DATE_DL_RE.match(s)
    if m:
        deadline = _parse_deadline(m.group("dl"))
        # ``on-{date}`` 为单日区间事件（非累计阶梯），hazard 恢复不适用累计式。
        ftype = "date_point" if m.group("form") == "on" else "date_ladder"
        return ParsedMarket(
            family_type=ftype,
            family_key=m.group("stem"),
            strike=None,
            barrier=None,
            deadline=deadline.date() if deadline else None,
        )

    return ParsedMarket(
        family_type="single", family_key=s, strike=None, barrier=None, deadline=None
    )


#: 主题 -> 事件类型（框架 §8.1；价格阈值另由 family_type 强制覆盖）。
THEME_EVENT_TYPE: dict[str, str] = {
    "mideast_conflict": "exogenous_occurrence",
    "russia_ukraine": "exogenous_occurrence",
    "taiwan_risk": "exogenous_occurrence",
    "us_shutdown": "policy_decision",
    "fed_policy": "policy_decision",
    "us_china_trade": "policy_decision",
    "oil_price": "asset_price_threshold",
    "metal_price": "asset_price_threshold",
}

#: Layer 4 资格：价格定义事件禁止（框架 §8.2 表）。
INELIGIBLE_EVENT_TYPES: frozenset[str] = frozenset({"asset_price_threshold"})


def build_families(registry: pd.DataFrame) -> pd.DataFrame:
    """登记表 -> 逐市场族与门控标注表。

    Args:
        registry: ``cn_registry*.parquet``（须含 ``condition_id / slug / theme``）。

    Returns:
        ``[condition_id, slug, theme, family_type, family_key, strike, barrier,
        deadline, deadline_ts, event_source_type, layer4_eligible]``，逐市场一行。
    """
    uniq = registry.drop_duplicates("condition_id")[["condition_id", "slug", "theme"]]
    rows: list[dict] = []
    for rec in uniq.itertuples(index=False):
        p = parse_slug(rec.slug)
        etype = THEME_EVENT_TYPE.get(rec.theme, "exogenous_occurrence")
        if p.family_type == "price_ladder":
            etype = "asset_price_threshold"
        deadline_ts = None
        if p.deadline is not None:
            deadline_ts = int(
                datetime(p.deadline.year, p.deadline.month, p.deadline.day,
                         tzinfo=UTC).timestamp()
            ) + 86400
        rows.append(
            {
                "condition_id": rec.condition_id,
                "slug": rec.slug,
                "theme": rec.theme,
                "family_type": p.family_type,
                "family_key": p.family_key,
                "strike": p.strike,
                "barrier": p.barrier,
                "deadline": p.deadline,
                "deadline_ts": deadline_ts,
                "event_source_type": etype,
                "layer4_eligible": etype not in INELIGIBLE_EVENT_TYPES,
            }
        )
    return pd.DataFrame(rows)
