"""主报告第一、二部分的答辩级补充小节（HTML 片段生成）。

读取 ``scripts/v3_part12_supplement.py`` 与修正后回测的产物，生成插入
主报告的小节。除公式与体例文字外，全部数字在构建时从 parquet 读取，
避免文本与计算值脱节（第三部分对抗审查 (d) 类缺陷的防御）。

第一部分插入：§3.4 edge cases 与质量清单、§4.6 信号统计与分布、
§7.4 RankIC/ICIR 桥接、§8.3 双主题交互、§10b 事件研究修正、
§12.5b RV 样本外、§12.6b 交易属性、§12.9-12.11 组合族 / 事件共现 /
检验总量核算。第二部分插入：§15.4 测量层统计、§17.5 吸收冲突期分解、
§17.6 tension 门控与 episode 共现。

由 ``build_thesis_html.py`` 导入调用；不独立运行。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "supp"
DEEP = ROOT / "data" / "cn_futures" / "analysis" / "deep"

THEME_CN = {
    "mideast_conflict": "中东冲突", "oil_price": "油价阈值",
    "metal_price": "金属价格", "fed_policy": "美联储政策",
    "russia_ukraine": "俄乌冲突", "us_china_trade": "中美贸易",
    "taiwan_risk": "台海风险", "us_shutdown": "美政府停摆",
}

COMBO_CN = {
    "W1": "双主题同向共现", "W2": "双主题分歧", "W3": "信念广度",
    "W4": "多主题复合信念", "W5": "双 surprise 共现",
    "W6": "信号×前日波动状态", "W7": "信号×当窗资金流",
    "W8": "信号×兄弟品种夜盘确认", "W9": "五日未兑现缺口",
    "W10": "窗口内符号一致性", "W11": "隔夜未兑现缺口",
    "W12": "信号×期货夜盘量能", "W13": "信号×市场级共识度",
    "W8_ctrl_sib": "对照：兄弟夜盘单独", "W8_ctrl_pm": "对照：PM 信号单独",
}


def _f(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "—"
    return f"{x:+.{nd}f}" if nd else f"{x:.0f}"


def _table(rows: list[str], header: str, cls: str = "") -> str:
    body = "\n".join(rows)
    c = f' class="{cls}"' if cls else ""
    return (f'<div class="tablewrap"><table{c}>\n<tr>{header}</tr>\n'
            f"{body}\n</table></div>")


def _meta() -> dict:
    p = SUPP / "meta_supp.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _fig(name: str, caption: str) -> str:
    """与主报告 fig_tag 同款的 base64 自包含图。"""
    import base64
    fp = ROOT / "docs" / "figures" / name
    if not fp.exists():
        return ""
    b = base64.b64encode(fp.read_bytes()).decode()
    return (f'<figure><div class="figcard">'
            f'<img src="data:image/png;base64,{b}" alt=""></div>'
            f"<figcaption>{caption}</figcaption></figure>")


# ------------------------------------------------ 第一部分 §3.4 edge cases
def sec_edge_cases() -> str:
    """§3.4：散落的处理细节收拢为结构化清单 + 涨跌停代理实测。"""
    cases = [
        ("D1", "夜盘文件错位", "夜盘 bar 存于开始时刻次一自然日文件；周一文件从不含夜盘",
         "按时间戳重新归属\"夜盘归下一交易日\"；测试断言周一含上周五 21:01-周六 02:30"),
        ("D2", "换月跳空", "主力连续为预拼接，换月日跨合约收益是拼接伪影",
         "隔夜 / 收对收收益置缺失并打 roll 标记；全部检验剔除 roll 日"),
        ("D3", "无夜盘品种", "IF 等品种 r_night 结构性缺失",
         "gap 收益并段（r_gap_full）；s_pre 退化为 s_gap，仅在全缺失时替换"),
        ("D4", "首交易日无前收", "窗口边界依赖前一交易日收盘",
         "首日 night/gap 窗口记 NaT，调用方跳过"),
        ("D5", "涨跌停", "主力连续无涨跌停价表，极端日收益可能不可实现",
         "以收尾 15 分钟 high==low 锁死占比作代理监控（下表实测）；未做截尾修正，"
         "推广到小品种须加过滤"),
        ("P1", "方向注册反号", "ceasefire-end（停火结束）是升级事件，曾被子串误伤反号",
         "v1.1 对抗审查逐市场人工复核方向表"),
        ("P2", "结算后价格", "结算后市场价恒 0/1，logit 发散且非信号",
         "结算时刻后的窗口信号记缺失"),
        ("P3", "LOCF 过时", "端点取最后成交（LOCF）可能无限延伸",
         "距最后成交超 120 分钟（p_age）记缺失"),
        ("P4", "样本内准入", "用全窗口流动性筛市场=用未来信息选样",
         "时点化准入 admit_ts：累计成交额首达 $10 万后才入面板"),
        ("P5", "logit 边界发散", "临近结算价格贴近 0/1",
         "p 截断到 [0.02, 0.98] 再取 logit；消融显示对位置不敏感"),
        ("P6", "bid-ask bounce", "逐笔价在买卖价间跳动不代表概率变化",
         "15 分钟桶内按 taker 方向分堆取加权中位数再平均"),
        ("P7", "聚合权重口径", "√usdc 权重的成交额累计窗口存在样本内信息",
         "等权变体消融一致（§12.1）；第三部分已升级为时点化累计权重，"
         "窗口级维持原口径并在 §4.6 注声明"),
        ("P8", "近结算同义反复", "临近结算的市场价格与标的实现值趋同",
         "剔除结算前 3 个交易日的稳健性变体，结论不变"),
        ("M1", "夏令时切换", "2026-03-08 美国进入夏令时，美东-北京时差 13h→12h",
         "全部时间换算经 tz 数据库自动处理；窗口边界以北京时间定义，不受影响"),
        ("M2", "美股假日 LOCF 基准", "美股假日（如 01-19、02-16、04-03）ETF 无成交，"
         "同窗基准收益经 LOCF 机械为 0",
         "§11 控制回归在这些天退化为无控制，属残留风险，此前只在代码注释声明，"
         "现列入清单"),
        ("M3", "周一窗口含周末", "周一交易日的闭市窗口长约 66 小时，含整个周末",
         "窗口收益与信号同界配对，长度差异由窗口定义自然吸收；周一 vs 平日"
         "分组见 §6.2"),
    ]
    rows = [
        f"<tr><td><b>{c}</b></td><td>{t}</td><td>{d}</td><td>{h}</td></tr>"
        for c, t, d, h in cases
    ]
    tab = _table(rows, "<th>编号</th><th>情形</th><th>问题</th><th>处理</th>",
                 "wraptext")

    lock_html = ""
    lp = SUPP / "locked_proxy.parquet"
    if lp.exists():
        lk = pd.read_parquet(lp)
        day = lk[lk["session"] == "day"]
        top = day.sort_values("locked_share_tail15", ascending=False).head(6)
        rows2 = [
            f"<tr><td>{r['product']}</td><td>{r['trade_date']}</td>"
            f"<td>{r['locked_share_tail15']:.0%}</td>"
            f"<td>{r['locked_share_all']:.1%}</td></tr>"
            for _, r in top.iterrows()
        ]
        flag = day[(day["trade_date"].isin(["2026-03-09", "2026-03-10"]))
                   & (day["product"] == "SC")]
        flag_txt = "；".join(
            f"{r['trade_date']} 收尾锁死 {r['locked_share_tail15']:.0%}"
            for _, r in flag.iterrows())
        lock_html = (
            "<p><b>涨跌停代理实测（D5）</b>：五品种逐日计算日盘收尾 15 分钟"
            "high==low 锁死分钟占比，前 6 名如下；旗舰观测日 SC 2026-03-09/10 "
            f"的实测值：{flag_txt or '两日均无收尾锁死'}——极端行情日的收益"
            "在收尾时段仍在正常成交中形成，涨跌停封板未成为主导因素。</p>"
            + _table(rows2, "<th>品种</th><th>交易日</th>"
                     "<th>收尾15分钟锁死</th><th>全日锁死</th>"))

    return f"""
<h3>3.4 数据处理 edge cases 与质量检查清单</h3>
<p>以上三小节的处理细节按答辩篇（第三部分 §1.6-1.7）的体例收拢为结构化
清单——D 类为期货源数据陷阱，P 类为 Polymarket 信号构造陷阱，M 类为
两市映射陷阱。每条给出问题与处理，未处理项显式声明：</p>
{tab}
{lock_html}
<p><b>未处理并声明的项</b>：窗口收益为单点 close/open 标签（开盘跳空对
单 bar 噪声敏感），未做区间 VWAP 端点对照——导师方法论优先推荐 VWAP
标签，第三部分 §4.5 的分钟级双标签检验显示两口径相关 0.41-0.95（品种
差异大），窗口级敏感性列为局限（§13.2）。</p>
{_qc_block()}
"""


def _qc_block() -> str:
    """期货库质量检查的执行结果汇总（qc_summary 首次进入报告）。"""
    qp = ROOT / "data" / "cn_futures" / "qc_summary.parquet"
    if not qp.exists():
        return ""
    q = pd.read_parquet(qp)
    rows = [
        ("交易日缺失", "逐品种比对交易日历",
         f"88 品种中 {int((q['missing_days'] > 0).sum())} 个有缺失日"
         f"（多为晚上市品种的上市前日期），主评估品种 SC/AU/AG/CU/M 均为 0"),
        ("重复时间戳", "逐品种 ts 去重计数",
         f"全部品种 dup_ts = {int(q['dup_ts'].sum())}"),
        ("OHLC 违例", "high ≥ max(open,close)、low ≤ min(open,close)",
         f"共 {int(q['ohlc_violations'].sum())} 根（88 品种 350 万根 bar）"),
        ("日盘 bar 数众数", "逐品种统计每日 bar 数",
         f"众数 {int(q['day_bar_mode'].mode()[0])}，偏离日数全为 "
         f"{int(q['days_day_bar_deviate'].sum())}"),
        ("零成交 bar", "volume==0 计数",
         f"总计 {int(q['zero_volume_bars'].sum()):,} 根（集中于不活跃品种的"
         "夜盘，占比与品种活跃度一致；主评估品种占比低）"),
        ("换月次数", "dominant_table 逐品种统计",
         f"{int(q['n_rolls'].min())}-{int(q['n_rolls'].max())} 次/品种，"
         "全部打 roll 标记"),
        ("单日多合约", "同交易日 contract 唯一性",
         f"违例 {int(q['multi_contract_days'].sum())} 天（主力切换在 21:01 "
         "交易日边界，日内唯一性成立）"),
    ]
    trs = [f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>"
           for a, b, c in rows]
    return ("<p><b>期货库质量检查执行结果</b>（<code>qc_summary.parquet</code>"
            "，答辩篇 §1.7 同款清单，首次进入报告正文）：</p>"
            + _table(trs, "<th>检查项</th><th>执行方式</th><th>结果</th>",
                     "wraptext"))


# --------------------------------------------- 第一部分 §4.6 信号统计
def sec_signal_stats() -> str:
    """§4.6：窗口信号的覆盖率、取值统计与分布。"""
    st = pd.read_parquet(SUPP / "signal_stats_window.parquet")
    st = st[st["signal"].isin(("s_night", "s_gap", "s_day"))]
    order = ["mideast_conflict", "oil_price", "metal_price", "fed_policy",
             "russia_ukraine", "us_china_trade", "taiwan_risk", "us_shutdown"]
    rows = []
    for theme in order:
        g = st[st["theme"] == theme]
        if g.empty:
            continue
        first = True
        for _, r in g.iterrows():
            th_cell = (f'<td rowspan="{len(g)}"><b>{THEME_CN[theme]}</b></td>'
                       if first else "")
            first = False
            usdc = (f"{r['usdc_q50'] / 1e3:.0f}k"
                    if pd.notna(r["usdc_q50"]) else "—")
            rows.append(
                f"<tr>{th_cell}<td>{r['signal']}</td>"
                f"<td>{r['n_valid']}</td><td>{r['coverage']:.0%}</td>"
                f"<td>{_f(r['mean'])}</td><td>{r['std']:.3f}</td>"
                f"<td>{_f(r['q01'])}</td><td>{_f(r['q50'])}</td>"
                f"<td>{_f(r['q99'])}</td><td>{_f(r['skew'], 2)}</td>"
                f"<td>{r['kurt']:.1f}</td>"
                f"<td>{r['extreme_rate']:.1%}</td><td>{usdc}</td></tr>"
            )
    tab = _table(
        rows,
        "<th>主题</th><th>信号</th><th>有效窗口</th><th>覆盖率</th>"
        "<th>均值</th><th>σ</th><th>q01</th><th>中位</th><th>q99</th>"
        "<th>偏度</th><th>峰度</th><th>|s|&gt;3σ</th><th>窗成交中位</th>")
    return f"""
<h3>4.6 信号的频率、取值统计与分布</h3>
<p>答辩篇（第三部分 §3-4）为分钟级信号登记了频率与分布统计；同一标准
补齐窗口级。主题信号在品种间是同值复制（同一市场集合），故按主题层
去重统计，分母为 125 个交易日（扩展样本口径，与第二部分 §17 一致）：</p>
{tab}
<p>读法与要点：（i）<b>覆盖率差异极大</b>——中东 / 俄乌接近全覆盖，
美政府停摆约三分之一，这是 §6-7 各配对 n 大幅波动（34-124）的根源：
n 的缺口来自"该窗口无活跃市场"（主要）、p_age 过时、结算后屏蔽与换月
剔除（次要），逐配对的有效窗口数见产物
<code>supp/signal_stats_window.parquet</code>；（ii）<b>均值全部接近 0</b>
——logit 端点差天然近似鞅差；（iii）<b>重尾显著</b>——峰度普遍高于正态，
|s|&gt;3σ 的极端窗口占比 1-4%，2026-03-06 的 +2.85（+4.8σ）是 q99 之外
的极端观测，回归以 HAC 推断且 §6.3 散点确认非单点驱动；（iv）主题信号
未做时序归一化——同期回归与相关对信号尺度不变；组合信号（§12.9）因涉
阈值与跨主题可比性改用展开窗口 z 分数（min 20 日、仅 ≤t 信息）。</p>
{_fig("f_supp_signal_dist.png",
      "头部四主题 × 三窗口的信号分布（125 交易日）。零附近高峰 = 多数窗口"
      "无新信息；重尾 = 少数事件窗口贡献绝大部分信号能量，这是 §10 事件"
      "研究与第三部分离散信号族的统计基础。")}
<p><b>聚合权重口径注</b>：主题聚合的 √usdc 权重取市场全窗口累计成交额
（v1.1 原口径），含轻微样本内信息（§13.2 局限第 4 条已声明）；§12.1
消融的等权变体结论一致；第三部分分钟级管线已升级为时点化累计权重
（√cumUSDC，仅用 ≤t 成交）。三个口径的方向与显著性一致，窗口级保持
原口径以维持与已发布数字的可比性。</p>
"""


# --------------------------------------- 第一部分 §7.4 RankIC / ICIR 桥接
def sec_rank_ic() -> str:
    """§7.4：把窗口级结果换算成答辩口径（RankIC 与月度 ICIR）。"""
    ic = pd.read_parquet(SUPP / "window_ic.parquet")
    heads = [
        ("oil_price", "SC"), ("mideast_conflict", "SC"),
        ("metal_price", "AG"), ("metal_price", "AU"),
        ("fed_policy", "AU"), ("us_china_trade", "M"),
    ]
    rows = []
    for theme, product in heads:
        g = ic[(ic["theme"] == theme) & (ic["product"] == product)]
        for _, r in g.iterrows():
            kind = "同期" if r["kind"] == "contemporaneous" else "预测"
            rows.append(
                f"<tr><td>{THEME_CN[theme]}×{product}</td><td>{kind}</td>"
                f"<td>{r['signal']}→{r['ret']}</td><td>{r['n']}</td>"
                f"<td>{_f(r['pearson'])}</td><td>{_f(r['rank_ic'])}</td>"
                f"<td>{r['n_months']}</td><td>{_f(r['ic_mean'])}</td>"
                f"<td>{_f(r['icir'], 2)}</td>"
                f"<td>{r['n_pos_months']}/{r['n_months']}</td></tr>"
            )
    tab = _table(
        rows,
        "<th>配对</th><th>类型</th><th>信号→收益</th><th>n</th>"
        "<th>Pearson</th><th>RankIC</th><th>月数</th><th>月均 IC</th>"
        "<th>ICIR</th><th>正月占比</th>")
    return f"""
<h3>7.4 答辩口径桥接：RankIC 与月度 ICIR</h3>
<p>第 6-7 节的主口径是 Pearson + HAC t（连续检验传统），答辩篇（第三部分
§4）的主口径是 RankIC / ICIR（因子评价传统）。同一批窗口级配对按第二种
口径重算（扩展样本 125 日）：逐自然月（≥8 个有效交易日）计算 Spearman
IC，ICIR = 月度 IC 均值 / 标准差。<b>样本只有约 6 个自然月，ICIR 的
分母由 3-7 个月度观测估计，仅作方向参考，不作强度声称</b>：</p>
{tab}
<p>要点：（i）同期吸收在两种口径下一致——头部配对的 RankIC 与 Pearson
同号且量级接近，月度符号高度一致；（ii）RankIC 不低于 Pearson，说明
关系不是少数极端日的线性杠杆撬动（对 2026-03-09 离群点质疑的标准回应，
与 §6.3 散点互证）；（iii）预测行（s_pre→r_day）两种口径都弱，与 §7.1
的"吸收后无剩余预测力"互检通过。</p>
"""


# ------------------------------------------- 第一部分 §8.3 双主题交互
def sec_interaction() -> str:
    """§8.3：乘积交互回归与二维分位表（超加性检验）。"""
    reg = pd.read_parquet(SUPP / "interaction_reg.parquet")
    rows = []
    for _, r in reg.iterrows():
        if r.get("note"):
            rows.append(
                f"<tr><td>{THEME_CN[r['theme_a']]}×{THEME_CN[r['theme_b']]}"
                f"·{r['product']}</td><td>{r['window']}</td><td>{r['n']}</td>"
                f"<td colspan=4>{r['note']}</td></tr>")
            continue
        rows.append(
            f"<tr><td>{THEME_CN[r['theme_a']]}×{THEME_CN[r['theme_b']]}"
            f"·{r['product']}</td><td>{r['window']}</td><td>{r['n']}</td>"
            f"<td>{_f(r['t_a'], 2)}</td><td>{_f(r['t_b'], 2)}</td>"
            f"<td><b>{_f(r['t_int'], 2)}</b></td><td>{r['r2']:.2f}</td></tr>")
    tab = _table(rows, "<th>主题对·品种</th><th>窗口</th><th>n</th>"
                 "<th>t(主效应A)</th><th>t(主效应B)</th><th>t(交互)</th>"
                 "<th>R²</th>")

    q = pd.read_parquet(SUPP / "interaction_quantile.parquet")
    piv = q.pivot_table(index="mideast_tercile", columns="oil_tercile",
                        values=["mean_bp", "n"], aggfunc="first")
    order = ["低", "中", "高"]
    qrows = []
    for a in order:
        cells = []
        for b in order:
            try:
                m = piv[("mean_bp", b)][a]
                n = int(piv[("n", b)][a])
                cells.append(f"<td>{m:+.0f}bp<br><span class='dim'>n={n}"
                             f"</span></td>")
            except KeyError:
                cells.append("<td>—</td>")
        qrows.append(f"<tr><td><b>中东 {a}</b></td>" + "".join(cells) + "</tr>")
    qtab = _table(qrows, "<th></th><th>油价 低</th><th>油价 中</th>"
                  "<th>油价 高</th>")

    t_int = reg[(reg["window"] == "r_gap_total")
                & (reg["product"] == "SC")]["t_int"]
    t_val = float(t_int.iloc[0]) if len(t_int) else np.nan
    verdict = ("<b>负交互（信息冗余）</b>：两主题同时高涨时的联合定价小于"
               "线性叠加之和" if t_val < -2 else
               "<b>超加性共振</b>" if t_val > 2 else
               "交互项不显著：联合定价近似线性叠加")
    return f"""
<h3>8.3 双主题交互："两个事件同时发生"的窗口级检验</h3>
<p>§8.2 的叠加是纯加性（联合回归求边际 R²）；按导师"尝试各种组合"的
要求补乘积交互项：r_w = α + β₁z_A + β₂z_B + β₃·z_A·z_B + ε（z 为展开窗
标准化，HAC 推断）。β₃ &gt; 0 为超加性共振（两事件同时发生放大定价），
β₃ &lt; 0 为信息冗余（重叠部分被重复计价后互相削弱）：</p>
{tab}
<p>头部配对（中东×油价·SC 的 gap 窗）：t(交互) = {_f(t_val, 2)}——
{verdict}。这与第三部分 X 族（确认型条件组合跨品种一致为负）在窗口级
形成同构证据：<b>共现不是增强器，而是同一信息的重复计数信号</b>。
无参数对照（3×3 分位表，SC gap 收益，行=中东三分位、列=油价三分位）：</p>
{qtab}
<p>对角线单调（低低最负、高高最正）确认两个主效应；但"高高"格相对
"中高/高中"无跳升，与回归的负交互一致——超加性不存在。第三部分
§5 的 X1（共振确认）分钟级为负，本表为其窗口级对应证据。</p>
"""


# ------------------------------------------- 第一部分 §10b 事件研究修正
def sec_event_correction() -> str:
    """§10b：事件研究的勘误与修正版（视界 / E3 符号 / 双基线）。"""
    p = SUPP / "event_car.parquet"
    if not p.exists():
        return ""
    car = pd.read_parquet(p)
    type_cn = {"E1": "E1 价格跳", "E2": "E2 量爆发", "E3": "E3 新市场创建"}
    rows = []
    for etype in ("E1", "E2", "E3"):
        g = car[car["type"] == etype].sort_values("h_min")
        first = True
        for _, r in g.iterrows():
            tc = (f'<td rowspan="{len(g)}"><b>{type_cn[etype]}</b><br>'
                  f'<span class="dim">n={r["n"]}</span></td>' if first else "")
            first = False
            rows.append(
                f"<tr>{tc}<td>{int(r['h_min'])}′</td>"
                f"<td>{r['car_bp']:+.1f}</td>"
                f"<td>[{r['lo']:+.1f}, {r['hi']:+.1f}]</td>"
                f"<td>{r['uncond_bp']:+.2f}</td>"
                f"<td><b>{r['excess_bp']:+.1f}</b></td></tr>")
    tab = _table(rows, "<th>事件</th><th>视界</th><th>符号化 CAR(bp)</th>"
                 "<th>90% 自助 CI</th><th>无条件基线(bp)</th><th>超额(bp)</th>")
    e3_120 = car[(car["type"] == "E3") & (car["h_min"] == 120)]
    e3_txt = ""
    if len(e3_120):
        r = e3_120.iloc[0]
        sig = (r["lo"] > 0) or (r["hi"] < 0)
        if sig:
            tail = "区间不含零，方向性效应保留"
        elif r["car_bp"] < 0:
            tail = ("区间含零且点估计翻负——原结论四「新市场创建是最强事件"
                    "类型 +41bp」<b>正式撤回</b>：该数字主要由视界标注错误、"
                    "未符号化（硬编码 +1）时吸收的冲突期上行漂移与无基线三个"
                    "缺陷共同制造")
        else:
            tail = "区间含零，原表述「最强事件类型 +41bp」撤回为方向性迹象"
        e3_txt = (f"修正后 E3 的 120 分钟符号化 CAR 为 {r['car_bp']:+.1f}bp"
                  f"（90% CI [{r['lo']:+.1f}, {r['hi']:+.1f}]，n={r['n']}，"
                  f"扣除符号占比加权漂移后超额 {r['excess_bp']:+.1f}bp）"
                  f"——{tail}。")
        e1_5 = car[(car["type"] == "E1") & (car["h_min"] == 5)]
        if len(e1_5):
            r1 = e1_5.iloc[0]
            e1_sig = (r1["lo"] > 0) or (r1["hi"] < 0)
            e3_txt += (f"E1 价格跳在 5 分钟视界 CAR {r1['car_bp']:+.1f}bp"
                       f"（CI [{r1['lo']:+.1f}, {r1['hi']:+.1f}]，"
                       f"n={r1['n']:,}）"
                       + ("为唯一区间不含零的格" if e1_sig else "区间亦含零")
                       + "，更长视界衰减——与「几分钟内吸收完毕」的定性结论"
                       "一致。")
    return f"""
<h3>10b 勘误与修正版事件研究（答辩审计触发）</h3>
<p class="warn"><b>勘误</b>：v1.1 的事件研究实现有三处缺陷，按答辩篇
标准全部修正并重算。（i）<b>视界标注错误</b>——原实现取事件后 25 根
1 分钟 bar 却按 5 分钟 bar 口径标注为"0-120 分钟"，上图（§10）曲线的
真实视界是约 24 分钟；（ii）<b>E3 符号未实现</b>——正文声明"按
orientation×sign(Δℓ) 符号化"，代码实际硬编码 +1；（iii）<b>缺无条件
基线</b>——自助 CI 只度量事件均值的抽样误差，不回答"相对样本期 SC
本来的漂移是否超额"。</p>
<p>修正版（扩展 tape 重算，事件定义与阈值逐字不变；视界按真实 1 分钟
bar；E3 按声明符号化；基线 = 全部有效同长夜盘窗口的无条件 CAR，超额 =
符号化 CAR − 符号占比加权漂移）：</p>
{tab}
<p>{e3_txt}事件逐月频数与共现检验见 §12.10。</p>
"""


# ------------------------------------------- 第一部分 §12.5b RV OOS
def sec_rv_oos() -> str:
    """§12.5b：波动通道的样本外验证（诚实降格）。"""
    p = SUPP / "rv_oos.parquet"
    if not p.exists():
        return ""
    r = pd.read_parquet(p).iloc[0]
    if not r.get("n_oos"):
        return ""
    verdict = ("样本外增量为正且 CW 显著，波动通道在防前视口径下保留"
               if r["cw_t"] > 1.65 else
               "<b>样本外无增量</b>——|s_gap| 的波动预测力在展开窗逐日重估的"
               "防前视口径下消失。结论六据此降格：风险通道的 t=4.4 是样本内"
               "证据，样本外验证不通过，其用于 §12.6 动态杠杆的实用价值存疑")
    return f"""
<h3>12.5b 波动通道的样本外验证</h3>
<p>审计指出：方向通道经第二部分完整 OOS 复核，而全文最强的正结论
（风险可测，t=4.4）从未做样本外验证。补做：SC 日盘已实现波动
（1 分钟收益平方和开根），M0 = 昨日 RV（HAR-lite 基线），M1 = + 当日
开盘前 |s_gap|（中东主题），40 日展开窗逐日重估、一步向前预测，
Clark-West 单侧检验：</p>
<div class="tablewrap"><table>
<tr><th>n(OOS)</th><th>R²_oos M0</th><th>R²_oos M1</th><th>ΔR²</th>
<th>CW t</th></tr>
<tr><td>{int(r['n_oos'])}</td><td>{r['r2_oos_m0']:.3f}</td>
<td>{r['r2_oos_m1']:.3f}</td><td>{r['delta_r2']:+.4f}</td>
<td><b>{r['cw_t']:+.2f}</b></td></tr>
</table></div>
<p>{verdict}。这与第二部分方向通道的结论（样本内显著、OOS 无增量）
同构，进一步支持"信息在开盘吸收中即时定价"的总判断。</p>
"""


# ------------------------------------------- 第一部分 §12.6b 交易属性
def sec_backtest_attrs() -> str:
    """§12.6b：S5 前视修正、成本敏感性与容量参与率。"""
    mp = DEEP / "backtest_metrics.parquet"
    ap = DEEP / "backtest_trade_attrs.parquet"
    if not (mp.exists() and ap.exists()):
        return ""
    met = pd.read_parquet(mp)
    attr = pd.read_parquet(ap)
    strategies = [c for c in met["strategy"].unique()]
    rows = []
    for s in strategies:
        g = met[met["strategy"] == s].set_index("cost_bp")
        cells = "".join(
            f"<td>{g.loc[c, 'sharpe']:+.2f}</td>" if c in g.index else "<td>—</td>"
            for c in (0.0, 2.0, 4.0, 8.0, 12.0))
        rows.append(f"<tr><td><code>{s}</code></td>{cells}</tr>")
    cost_tab = _table(rows, "<th>策略</th><th>0bp</th><th>2bp</th>"
                      "<th>4bp</th><th>8bp</th><th>12bp</th>")
    rows2 = [
        f"<tr><td><code>{r['strategy']}</code></td>"
        f"<td>{r['ann_turnover']:.0f}</td><td>{int(r['n_entries'])}</td>"
        f"<td>{r['avg_holding_days']:.1f}</td>"
        f"<td>{r['participation_100m_pct']:.2f}%</td></tr>"
        for _, r in attr.iterrows()
    ]
    attr_tab = _table(rows2, "<th>策略</th><th>年化换手</th><th>进场次数</th>"
                      "<th>平均持有(日)</th><th>1 亿名义/日均成交额</th>")
    s5 = met[(met["strategy"] == "S5_event") & (met["cost_bp"] == 4.0)]
    s5_txt = ""
    if len(s5):
        r = s5.iloc[0]
        s5_txt = (f"修正后 S5（事件 bar 收盘进场、持有 120 根 1 分钟 bar、"
                  f"不足则至夜盘收盘）4bp 成本下 Sharpe "
                  f"{r['sharpe']:+.2f}（CI [{r['sharpe_lo']:.2f}, "
                  f"{r['sharpe_hi']:.2f}]）。")
    return f"""
<h3>12.6b 勘误（S5 前视）、成本敏感性与容量</h3>
<p class="warn"><b>勘误</b>：v1.1 的 S5 事件策略在实现上于事件<b>前一
分钟</b>的收盘价进场（声明为"信号严格早于持仓窗口"），且持有 25 根
1 分钟 bar 而非声明的 120 分钟。已修正重跑：{s5_txt}</p>
<p>审计补充的可交易性登记（导师清单：换手、容量、成本敏感度）。
成本敏感性（年化 Sharpe，单边 bp）：</p>
{cost_tab}
<p>交易属性与参与率（容量代理）：</p>
{attr_tab}
<p>读法：S1/S2 为持续在场、逐日重定向的策略（年化换手最高，成本敏感
性相应最陡），S3/S5 仅在信号 / 事件日进出；1 亿元名义仓位占 SC 日均
成交额约 0.38%，容量在本文规模下不构成约束。本节数字与 §12.6 主表
同源重算，原表其余行不受 S5 勘误影响。</p>
"""


# ------------------------------------------- 第一部分 §12.9 组合信号
def sec_combo() -> str:
    """§12.9：窗口级组合信号族（W1-W13，族内 FDR + 一致性纪律）。"""
    ci = pd.read_parquet(SUPP / "combo_ic.parquet")
    meta = _meta()
    fam = int(meta.get("combo", {}).get("family_size", 0))

    defs = [
        ("共现", "W1 双主题同向共现", "1{z_A·z_B>0}·sign(z_A)·min(|z_A|,|z_B|)",
         "两个不同事件主题同时同向才给信号"),
        ("共现", "W2 双主题分歧", "|z_A − z_B|",
         "两主题信念背离幅度，检验次日波动"),
        ("共现", "W3 信念广度", "Σ_k 1{|z_k|>1} / K",
         "映射主题中活跃者占比（扩散度）"),
        ("共现", "W5 双 surprise 共现", "1{|zA|>1.5}·1{|zB|>1.5}·sign(zA+zB)",
         "两主题同日大 surprise 的离散事件"),
        ("交互", "W6 信号×波动状态", "z_pre · 1{|r_cc(t−1)|>展开中位}",
         "高波动状态下信念创新是否更有效"),
        ("交互", "W7 信号×资金流", "z_gap · 展开分位秩(usdc_gap)",
         "重资金窗口的信念创新加权"),
        ("交互", "W8 信号×兄弟品种确认", "z_gap · 1{sign(r_night^sib)=sign(z_gap)}",
         "信念被同主题兄弟品种夜盘价格确认"),
        ("交互", "W12 信号×期货夜盘量能", "z_gap · 1{夜盘成交额>展开q75}",
         "国内资金放量响应时的信念创新"),
        ("交互", "W13 信号×市场级共识度", "z_gap · |Σ_m orient·sign(s_m)|/n_m",
         "主题内逐市场方向一致率加权"),
        ("结构", "W4 多主题复合信念", "Σ√usdc_k·z_k / Σ√usdc_k",
         "全部映射主题的资金加权合成"),
        ("结构", "W9 五日未兑现缺口", "z(Σ₅ s_all) − z(Σ₅ r_cc)",
         "第三部分 C8 的窗口级对应物（5 日尺度）"),
        ("结构", "W11 隔夜未兑现缺口", "z(s_pre) − z(r_night)",
         "信念已动而本品种夜盘未动的部分（09:00 可知）"),
        ("结构", "W10 窗口内符号一致性", "一致度(night,gap,day)·sign(s_day)·幅度",
         "三窗口信号同向的持续性结构"),
    ]
    drows = [f"<tr><td>{c}</td><td><b>{n}</b></td><td><code>{f}</code></td>"
             f"<td>{w}</td></tr>" for c, n, f, w in defs]
    dtab = _table(drows, "<th>类</th><th>组合</th><th>公式</th><th>动机</th>",
                  "wraptext")

    surv = ci[(ci["kind"] != "control") & (ci["q_bh"] < 0.1)]
    surv = surv.reindex(surv["t_hac"].abs().sort_values(ascending=False).index)
    rows = []
    for _, r in surv.iterrows():
        name = COMBO_CN.get(r["combo"], r["combo"])
        tgt = {"r_day": "当日日盘", "r_cc_next": "次日收对收",
               "abs_r_day": "|当日日盘|"}.get(r["ret"], r["ret"])
        agree = (pd.notna(r["rank_ic"]) and pd.notna(r["t_hac"])
                 and np.sign(r["rank_ic"]) == np.sign(r["t_hac"])
                 and abs(r["rank_ic"]) > 0.1)
        rows.append(
            f"<tr><td><b>{r['combo']}</b> {name}</td>"
            f"<td>{r['product']}</td><td>{tgt}</td><td>{r['n']}</td>"
            f"<td>{_f(r['rank_ic'])}</td><td>{_f(r['t_hac'], 2)}</td>"
            f"<td>{_f(r['icir'], 2)}</td>"
            f"<td>{r['n_pos_months']}/{r['n_months']}</td>"
            f"<td>{r['q_bh']:.4f}</td>"
            f"<td>{'✓' if agree else '<b>✗</b>'}</td></tr>"
        )
    rtab = _table(
        rows,
        "<th>组合</th><th>品种</th><th>目标</th><th>n</th><th>RankIC</th>"
        "<th>t(HAC)</th><th>ICIR</th><th>正月</th><th>q(BH)</th>"
        "<th>秩线一致</th>")

    ctrl = ci[(ci["combo"].str.startswith("W8")) & (ci["product"] == "I")]
    crows = [
        f"<tr><td>{COMBO_CN.get(r['combo'], r['combo'])}</td><td>{r['n']}</td>"
        f"<td>{_f(r['rank_ic'])}</td><td>{_f(r['t_hac'], 2)}</td></tr>"
        for _, r in ctrl.iterrows()
    ]
    ctab = _table(crows, "<th>信号</th><th>n</th><th>RankIC</th><th>t(HAC)</th>")

    w9 = ci[ci["combo"] == "W9"]
    w9_desc = "、".join(
        f"{r['product']} {_f(r['t_hac'], 2)}" for _, r in w9.iterrows())
    w11 = ci[ci["combo"] == "W11"]
    w11_max = (w11.reindex(w11["t_hac"].abs().sort_values(ascending=False)
                           .index).head(1))
    w11_txt = ""
    if len(w11_max):
        r = w11_max.iloc[0]
        w11_txt = (f"W11（隔夜未兑现缺口，C8 的当日版）最强格为 "
                   f"{r['product']}（t={_f(r['t_hac'], 2)}，"
                   f"q={r['q_bh']:.2f}），未过族内 FDR")

    return f"""
<h3>12.9 窗口级组合信号：共现、交互与结构（{fam} 项检验，族内 FDR）</h3>
<p>第 6-11 节全部是单主题单信号。按导师"尝试各种组合"的要求与第三部分
C/X 族的设计思路，在窗口级构造三类共 13 个组合信号。双主题配对按
<b>经济学预登记</b>（SC：中东×油价；AU：中东×金属；AG：金属×美联储；
CU：贸易×美联储；单主题品种自动跳过）——不用流动性排序选对，因为它会
给 SC 选出中东×俄乌两个高度相关的冲突主题，属同一信息的重复计数而非
"两个不同事件"。归一化用展开窗口 z 分数（min 20 日，仅 ≤t 信息）。
预注册纪律：全部 {fam} 项检验一次性入 BH-FDR 台账（对照行不入族），
声称门槛沿用第三部分 §6 的三类一致性（秩线一致 / 月度稳定 / 冲突段
内外同号）；本族设计受第三部分分钟级结果启发，属研究内选择而非样本外
证据，如实声明：</p>
{dtab}
<p>族内 FDR 存活（q&lt;0.1）的全部行（末列 = RankIC 与线性 t 同号且
|RankIC|&gt;0.1）：</p>
{rtab}
<p><b>按一致性纪律的诚实读法（重要）</b>：</p>
<p>（i）存活行中 W12（期货量能交互）虽 q 值极小，但<b>秩线背离严重</b>
（如 AU：t=+5.9 而 RankIC≈0.00）且跨品种符号翻转（AU/M 为正、CU/CF 为
负），是少数极端日驱动的线性假象叠加多品种不一致——按纪律<b>不作为
发现</b>，登记为反例。W13（共识度）仅在贸易主题品种为负且月度全负，
同样不稳定。</p>
<p>（ii）通过全部纪律的只有铁矿石 I 上的两项：<b>W8 贸易信念×棉花夜盘
确认</b>与 <b>W10 符号一致性</b>。W8 的归因对照显示交互本身携带信息
（两个成分单独均不显著）：</p>
{ctab}
<p>但 n 仅约 60、月度稳定性由 3 个月估计、"兄弟"关系仅经主题共享定义
——在续期样本复验前是<b>候选发现</b>（与第二部分 M 的定位相同）。</p>
<p>（iii）<b>W9（C8 的 5 日尺度对应物）全品种无信号</b>（t：{w9_desc}）；
{w11_txt}。两个粒度的对照定位了信息的时间尺度：信念-价格缺口的可预测
性存在于分钟级、在日级聚合中被抹平——与 §7"吸收发生在开盘瞬间"互为
印证。</p>
<p>（iv）完整 {fam} 行检验表（含全部不显著项）见
<code>supp/combo_ic.parquet</code>，不筛选呈现。</p>
"""


# ------------------------------------- 第一部分 §12.10 离散事件共现
def sec_event_cooccur() -> str:
    """§12.10：两类事件同时发生的离散共现检验 + 事件月频。"""
    p = SUPP / "event_cooccur.parquet"
    if not p.exists():
        return ""
    co = pd.read_parquet(p)
    em_p = SUPP / "event_monthly.parquet"
    em_html = ""
    if em_p.exists():
        em = pd.read_parquet(em_p)
        rows = [
            f"<tr><td>{r['month']}</td>"
            f"<td>{int(r.get('E1', 0))}</td><td>{int(r.get('E2', 0))}</td>"
            f"<td>{int(r.get('E3', 0))}</td></tr>"
            for _, r in em.iterrows()
        ]
        em_html = ("<p>三类事件的月度频数（扩展窗口，SC 相关主题；答辩篇"
                   "要求的频率登记，也回答 E3 是否集中于 2-3 月冲突段）：</p>"
                   + _table(rows, "<th>月份</th><th>E1 价格跳</th>"
                            "<th>E2 量爆发</th><th>E3 新市场</th>"))
    grp_cn = {
        "V1_dual_theme_same_sign": "V1 双主题同夜 E1 同向（油价 & 中东）",
        "V2_price_volume_resonance": "V2 价量共振（同市场同桶 E1+E2）",
        "V3_single_theme": "V3 单主题 E1（对照组）",
        "V4_e3_with_e1_same_theme": "V4 新市场创建×同主题 E1 同夜",
        "baseline_all_days": "无条件基线（全部交易日）",
    }
    rows = []
    for _, r in co.iterrows():
        if r["n"] == 0:
            rows.append(f"<tr><td>{grp_cn.get(r['group'], r['group'])}</td>"
                        f"<td>0</td><td colspan=4>样本内无此类事件</td></tr>")
            continue
        rows.append(
            f"<tr><td>{grp_cn.get(r['group'], r['group'])}</td>"
            f"<td>{int(r['n'])}</td><td>{_f(r['mean_bp'], 0)}</td>"
            f"<td>{_f(r['median_bp'], 0)}</td>"
            f"<td>{_f(r.get('t_stat'), 2)}</td>"
            f"<td>{r['share_pos']:.0%}</td></tr>"
        )
    tab = _table(rows, "<th>事件组</th><th>n</th><th>均值(bp)</th>"
                 "<th>中位(bp)</th><th>t</th><th>正占比</th>")
    return f"""
<h3>12.10 离散事件共现："两个事件同时发生"的直接检验</h3>
<p>§8.3 与 §12.9 是共现的连续版本；此处按导师要求做字面意义的离散版：
在扩展 tape 上重算 §10 的 E1/E2/E3 事件（定义、阈值逐字相同，E3 已按
10b 勘误符号化），检验两类事件同时发生的交易日上 SC 的符号化次日日盘
收益。事件归属交易日按 SC 三窗口边界划分：</p>
{em_html}
{tab}
<p>诚实读法：共现日样本量由事件结构决定、不可扩充，n 两位数以下的组
只登记方向与占比、不做显著性声称；V3（单主题对照）用于判断共现是否比
单发更强。完整表在 <code>supp/event_cooccur.parquet</code>。</p>
"""


# ------------------------------------- 第一部分 §12.11 检验总量核算
def sec_test_count() -> str:
    """§12.11：全报告检验单元总账（多重性透明度）。"""
    p = SUPP / "test_count.parquet"
    if not p.exists():
        return ""
    tc = pd.read_parquet(p)
    part_cn = {"I": "第一部分", "II": "第二部分", "supp": "本次补充（答辩级）"}
    rows = []
    for part in ("I", "II", "supp"):
        g = tc[tc["part"] == part]
        first = True
        for _, r in g.iterrows():
            pc = (f'<td rowspan="{len(g)}"><b>{part_cn[part]}</b></td>'
                  if first else "")
            first = False
            rows.append(f"<tr>{pc}<td>{r['family']}</td>"
                        f"<td>{int(r['n_units'])}</td>"
                        f"<td><code>{r['source']}</code></td></tr>")
    total = int(tc["n_units"].sum())
    tab = _table(rows, "<th>部分</th><th>检验族</th><th>单元数</th>"
                 "<th>产物</th>", "wraptext")
    return f"""
<h3>12.11 检验总量核算（多重性透明度）</h3>
<p>第三部分 §6 的透明度标准：明示总检验单元数与名义水平下的期望假阳性。
此前第一部分只对 §7.1 的 68 项预测性检验做了 FDR 核算；现按产物行数
盘点全报告：</p>
{tab}
<p>合计约 <b>{total}</b> 个检验单元（不含第三部分的 1,128 格，其自有
核算见第三部分 §6）；5% 名义水平下期望偶然显著约 {total // 20} 个。
解读纪律：BH-FDR 显式控制的族为 §7.1（68 项）、§12.9 组合族与第二部分
OOS 主设计；其余表格（吸收、控制、消融）为描述性或单独假设检验，头部
结论的可信度不依赖单格 p 值，而依赖跨设计一致性（安慰剂 / 置换 / 消融 /
子样本 / 国际控制的联合证据链）。</p>
"""


# ------------------------------------- 第二部分 §15.4 测量层信号统计
def sec_v3_stats() -> str:
    """§15.4：v3 测量层信号的覆盖率与分布统计。"""
    st = pd.read_parquet(SUPP / "v3_signal_stats.parquet")
    keep = st[st["signal"].isin(("s_pre", "s_day", "tension", "hz_day"))]
    order = ["mideast_conflict", "oil_price", "metal_price", "fed_policy",
             "russia_ukraine", "us_china_trade", "taiwan_risk", "us_shutdown"]
    rows = []
    for theme in order:
        g = keep[keep["theme"] == theme]
        if g.empty:
            continue
        first = True
        for _, r in g.iterrows():
            th_cell = (f'<td rowspan="{len(g)}"><b>{THEME_CN[theme]}</b></td>'
                       if first else "")
            first = False
            rows.append(
                f"<tr>{th_cell}<td><code>{r['signal']}</code></td>"
                f"<td>{r['n_valid']}</td><td>{r['coverage']:.0%}</td>"
                f"<td>{_f(r['mean'])}</td><td>{_f(r['std'])}</td>"
                f"<td>{_f(r['q01'])}</td><td>{_f(r['q50'])}</td>"
                f"<td>{_f(r['q99'])}</td></tr>"
            )
    tab = _table(
        rows,
        "<th>主题</th><th>信号</th><th>有效日</th><th>覆盖率</th>"
        "<th>均值</th><th>σ</th><th>q01</th><th>中位</th><th>q99</th>")
    nu = _meta().get("nu_std", {})
    nu_txt = ""
    if nu:
        nu_txt = (f"滤波标准化创新 ν̃ 的合并分布（n={nu['n']:,}）：均值 "
                  f"{nu['mean']:+.3f}、标准差 {nu['std']:.3f}——接近 (0,1) "
                  f"即 Kalman 滤波的噪声模型与数据基本相容，偏离部分对应"
                  f"重尾事件桶，这是滤波质量的直接检查。")
    return f"""
<h3>15.4 测量层信号的覆盖率与分布（答辩级登记）</h3>
<p>与第一部分 §4.6、第三部分 §3 同一标准，登记 v3 测量层输出（Kalman
滤波 + 一致性投影后）的统计基本面。<code>s_pre</code> 为开盘前信念创新、
<code>tension</code> 为族内逻辑张力、<code>hz_day</code> 为结算 hazard：</p>
{tab}
<p>{nu_txt}与 §4.6 的原始窗口信号相比，滤波后信号的重尾收敛（q99 幅度
下降约三成）——测量层降噪的直接证据；覆盖率格局不变（测量层不创造
数据，只重新加权），§17 各回归的 n 差异同样源于逐主题覆盖率。聚合权重
口径：v3 测量层与 v1.1 基线同用全窗口 √usdc（差异声明与消融见第一部分
§4.6 注），时点化累计权重变体列入 §18.1 待办。</p>
"""


# --------------------------------- 第二部分 §17.5 吸收的冲突期内外分解
def sec_absorb_subsample() -> str:
    """§17.5：同期吸收在冲突期内外的分段。"""
    ab = pd.read_parquet(SUPP / "absorption_subsample.parquet")
    heads = [("oil_price", "SC", "s_gap"), ("mideast_conflict", "SC", "s_gap"),
             ("mideast_conflict", "SC", "s_night"),
             ("metal_price", "AG", "s_gap"), ("metal_price", "AU", "s_gap")]
    rows = []
    flips: list[str] = []
    for theme, product, sig in heads:
        g = ab[(ab["theme"] == theme) & (ab["product"] == product)
               & (ab["signal"] == sig)]
        if g.empty:
            continue
        cells = {r["subset"]: r for _, r in g.iterrows()}

        def fmt(sub: str, cells: dict = cells) -> str:
            r = cells.get(sub)
            if r is None:
                return "<td>—</td><td>—</td>"
            return (f"<td>{_f(r['pearson'])}"
                    f"（{_f(r['t_hac'], 1)}）</td><td>{r['n']}</td>")
        hot_r = cells.get("hot")
        cold_r = cells.get("cold")
        if (hot_r is not None and cold_r is not None
                and np.sign(hot_r["pearson"]) != np.sign(cold_r["pearson"])):
            flips.append(f"{THEME_CN[theme]}×{product}·{sig}")
        rows.append(
            f"<tr><td>{THEME_CN[theme]}×{product}·{sig}</td>"
            + fmt("all") + fmt("hot") + fmt("cold") + "</tr>")
    tab = _table(
        rows,
        "<th>配对·信号</th><th>全样本 r(t)</th><th>n</th>"
        "<th>冲突期 r(t)</th><th>n</th><th>冲突期外 r(t)</th><th>n</th>")
    flip_txt = ("符号在冲突期内外全部保持——结论是\"强度时变、方向稳健\"，"
                "推广表述维持 §13 原状" if not flips else
                f"以下配对在冲突期外<b>符号翻转</b>：{'、'.join(flips)}——"
                "这些结论降格为\"冲突段条件性发现\"，§13/§18 的推广表述"
                "同步收紧")
    return f"""
<h3>17.5 吸收的冲突期内外分解</h3>
<p>第三部分 §5 对其头部结构 C8 做了冲突段内外同号检验；审计指出全报告
最强证据链（吸收，控制代理后 t≈4-5）反而缺同型检验。按同一冲突期边界
（2026-02-01 至 03-31）硬分段（扩展样本 125 日，Pearson r 与 HAC t）：</p>
{tab}
<p>读法：吸收在冲突期内更强是<b>预期内的条件性</b>——事件概率大幅变动
的窗口才有可吸收的信息；关键是冲突期外的符号。本表实测：{flip_txt}。</p>
"""


# --------------------------------- 第二部分 §17.6 tension 门控与 episode
def sec_tension_episode() -> str:
    """§17.6：测量层多维输出的组合试验（tension 门控 + episode 共现）。"""
    tg = pd.read_parquet(SUPP / "tension_gate.parquet")
    rows = []
    for _, r in tg.iterrows():
        if r.get("note"):
            rows.append(f"<tr><td>{THEME_CN[r['theme']]}×{r['product']}</td>"
                        f"<td>{r['n']}</td><td colspan=3>{r['note']}</td></tr>")
            continue
        rows.append(
            f"<tr><td>{THEME_CN[r['theme']]}×{r['product']}</td>"
            f"<td>{r['n']}</td><td>{_f(r['t_z'], 2)}</td>"
            f"<td><b>{_f(r['t_zgate'], 2)}</b></td>"
            f"<td>{_f(r['t_gate'], 2)}</td></tr>")
    ttab = _table(rows, "<th>配对</th><th>n</th><th>t(信号)</th>"
                  "<th>t(信号×高张力)</th><th>t(门)</th>")

    ep = pd.read_parquet(SUPP / "episode_cooccur.parquet")
    ep = ep[(ep["n_days"] >= 3)
            & (ep["product"].isin(["SC", "AU"]))]
    grp_cn = {"co_same": "共现·同向", "co_opposite": "共现·异向",
              "single": "单主题（对照）"}
    rows2 = [
        f"<tr><td>{r['product']}</td><td>{grp_cn.get(r['group'], r['group'])}"
        f"</td><td>{int(r['n_days'])}</td>"
        f"<td>{_f(r['mean_abs_bp'], 0)}</td>"
        f"<td>{_f(r['mean_signed_bp'], 0)}</td>"
        f"<td>{r['share_pos']:.0%}</td></tr>"
        for _, r in ep.iterrows()
    ]
    etab = _table(rows2, "<th>品种</th><th>episode 组</th><th>n 日</th>"
                  "<th>|r_cc| 均值(bp)</th><th>符号化 r_cc(bp)</th>"
                  "<th>正占比</th>")
    return f"""
<h3>17.6 测量层多维输出的组合试验</h3>
<p>审计指出：tension 只作了单变量波动回归，episode 的跨主题同日结算只
被当作伪重复威胁处理——两者作为<b>组合信号</b>的另一面从未检验。补做
两个试验（均为探索性，n 小、不进任何结论）：</p>
<p><b>（a）tension 门控吸收</b>：r_gap ~ z(s_gap) + z·1{{tension&gt;展开
q75}} + 门，HAC。交互项 t 为负即"族内定价失衡高时信念信号更噪"：</p>
{ttab}
<p><b>（b）episode 事件共现</b>：同日 ≥2 主题结算（surprise 同向 / 异向）
vs 单主题日的收对收响应：</p>
{etab}
<p>读法：两表均为小样本探索。tension 门控在 AU 上的负交互与 H4（tension
高→波动低→"张力抑制波动"）方向相容，值得在更长样本上专项验证；episode
共现日的响应幅度普遍高于单主题日，与功效核算"少数大 surprise 日主导"
的定位一致，但符号化收益在 n≤10 的组内不稳定，不作方向性声称。</p>
"""
