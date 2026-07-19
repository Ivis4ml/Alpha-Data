"""生成答辩文档 docs/v3_signal_defense.html（v2 完全重构版）。

结构：主文档走主线（结论 -> 数据处理与 edge cases -> 信号构造 -> 统计与
检验 -> 诚实结论），全部解释性内容归入五个附录（定价机制与市场结构 /
数据与方法的已知边界 / 字段词典 / 与导师方法论文档逐条对照 / 复现索引）。

面向读者：懂股票与期货、完全不懂 Polymarket 的评审者。

依赖 ``scripts/v3_defense_build.py`` 的产物（parquet + meta.json）。

用法::

    .venv/bin/python scripts/v3_defense_report.py
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pub_style  # noqa: E402
from pub_style import tex_svg  # noqa: E402
from v3_defense_build import (  # noqa: E402
    ALL_SIGNALS,
    COMBO_SIGNALS,
    HORIZONS,
    K_SIGNALS,
    NUM_SIGNALS,
    PRODUCT_THEMES,
    X_SIGNALS,
)

D = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
OUT = ROOT / "docs" / "v3_signal_defense.html"
P = pub_style.PALETTE

THEME_CN = {
    "mideast_conflict": "中东冲突", "oil_price": "油价阈值",
    "metal_price": "金银阈值", "russia_ukraine": "俄乌冲突",
    "fed_policy": "美联储", "us_china_trade": "中美贸易",
    "taiwan_risk": "台海风险", "us_shutdown": "美政府停摆",
}


# ---------------------------------------------------------------- 基础渲染件
def tex(f: str) -> str:
    return f'<div class="texblock">{tex_svg(f, fontsize=12)}</div>'


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def figure(fig, caption: str) -> str:
    return (f'<figure><div class="figcard"><img src="data:image/png;base64,'
            f'{fig_b64(fig)}" alt=""></div>'
            f"<figcaption>{caption}</figcaption></figure>")


def wrap(head: str, rows: list[str]) -> str:
    return ('<div class="tablewrap"><table>' + head + "".join(rows)
            + "</table></div>")


def wrap_text(head: str, rows: list[str]) -> str:
    """长文本表：允许换行、左对齐（edge cases / 对照表等）。"""
    return ('<div class="tablewrap wraptext"><table>' + head + "".join(rows)
            + "</table></div>")


def _short(addr: str) -> str:
    return f"{addr[:6]}…{addr[-4:]}"


# ---------------------------------------------------------------- 信号元信息
#: (id, 公式, 标题, 机制一句话, 输入, 窗口, 归一化)
SIGNAL_DEFS: list[tuple[str, str, str, str, str, str, str]] = [
    ("N1", r"N1_t = \sum_m \frac{w_m\,[\ell(p_{m,t}) - \ell(p_{m,t^-})]}"
           r"{\sum_m |w_m|},\;\; w_m = \sigma_m\sqrt{\mathrm{USDC}_m}",
     "主题 1 分钟 logit 创新", "事件信念的最小时间单位增量（orientation 加权）",
     "PM 逐笔", "1min", "无（原始）"),
    ("N2", r"N2_t = \sum_{u=t-14}^{t} N1_u",
     "15 分钟累积创新", "与窗口级研究可比的中尺度信念变化",
     "N1", "15min", "无（原始）"),
    ("N3", r"N3_t = z\left(\sum_{u=t-14}^{t}\sum_m \sigma_m D\,"
           r"\mathrm{USDC}\right)",
     "签名（signed）主动流 z", "价格之外的「用钱投票」净强度",
     "PM 逐笔", "15min", "滚动 z"),
    ("N4", r"N4_t = z(N2_t)", "累积创新 z",
     "时序归一化的基准形式（评审要求的 zscore 口径）", "N2", "15min", "滚动 z"),
    ("N5", r"N5_t = \frac{\sum \sigma_m D\,\mathrm{USDC}}"
           r"{\sum \mathrm{USDC}} \in [-1,1]",
     "流不平衡比", "自带归一的方向占比，对量纲稳健",
     "PM 逐笔", "15min", "结构自归一"),
    ("N6", r"N6_t = z\left(\log(1+n^{15}_t)\right)",
     "成交强度 z", "关注度 / 信息到达率代理",
     "PM 笔数", "15min", "log + 滚动 z"),
    ("N7", r"N7_t = z\left(\sum_{\theta}\sum_{u=t-14}^{t} "
           r"N1^{\theta}_u\right)",
     "多主题复合创新 z", "品种全部映射主题合并，降低单主题噪声",
     "全主题 dl", "15min", "滚动 z"),
    ("N8", r"N8_t = z\left(\overline{n^{\mathrm{mkts}}}^{15}_t\right)",
     "活跃市场数 z", "族内广度：多少个市场同时在动",
     "PM 市场数", "15min", "滚动 z"),
    ("N9", r"N9_t = z\left(\mathrm{std}_{60}(N1)\right)",
     "信念波动 z", "事件不确定性的强度维度（非方向）",
     "N1", "60min", "滚动 z"),
    ("N10", r"N10_t = z\left(\sum_{u=t-14}^{t}|N1_u|\right)",
     "绝对创新和 z", "无方向的信息流量",
     "N1", "15min", "滚动 z"),
    ("C1", r"C1_t = N4_t \times z(\mathrm{mom}^{15}_t)",
     "事件 × 动量确认", "两个独立信息源同向时更可信",
     "N4 + 期货", "15min", "成分各自 z"),
    ("C2", r"C2_t = N4_t \times \mathbf{1}\{|z(\mathrm{mom}^{15}_t)|<0.5\}",
     "事件动而价未动", "未兑现信息假说：期货尚未反应",
     "N4 + 期货", "15min", "成分各自 z"),
    ("C3", r"C3_t = N4_t \,/\, (1 + z(\mathrm{RV}^{15}_t)_+)",
     "波动折减事件强度", "高波动时单位创新的信息含量更低",
     "N4 + 期货", "15min", "成分各自 z"),
    ("C4", r"C4_t = N4_t \times z(\mathrm{Vol}^{15}_t)",
     "事件 × 放量确认", "期货放量佐证信息到达",
     "N4 + 期货", "15min", "成分各自 z"),
    ("C5", r"C5_t = N3_t \times \mathbf{1}\{N3_t \cdot "
           r"z(\mathrm{mom}^{15}_t) > 0\}",
     "签名流与价同向门", "只保留方向一致的主动流",
     "N3 + 期货", "15min", "成分各自 z"),
    ("C6", r"C6_t = N4_t \times z(\mathrm{RV}^{15}_t)",
     "事件 × 高波动状态", "波动状态下事件冲击被放大（与 C3 相反的假说）",
     "N4 + 期货", "15min", "成分各自 z"),
    ("C7", r"C7_t = N4_t - \hat\beta_t\, z(\mathrm{mom}^{15}_t),\;\;"
           r"\hat\beta_t = \frac{\mathrm{E}_{4800}[N4 \cdot m]}"
           r"{\mathrm{E}_{4800}[m^2]}",
     "对期货动量的滚动残差", "剥离已被期货价格解释的创新（残差型因子）",
     "N4 + 期货", "4800min 回归", "残差"),
    ("C8", r"C8_t = z\left(\sum_{120} N1\right) - "
           r"z\left(\sum_{120} r_1\right)",
     "120 分钟未兑现缺口",
     "信念累积与价格累积之差（两腿均按交易分钟序跨段滚动 120、"
     "min_periods 30；跨段首分钟收益按 0 计，gap 跳空不入价格腿）",
     "N1 + 期货", "120min（跨段）", "两侧各自 z 后相减"),
    ("C9", r"C9_t = N5_t \times z(\mathrm{Amt}^{15}_t)",
     "不平衡 × 成交额", "厚市场中的方向性下注",
     "N5 + 期货", "15min", "成分各自 z"),
    ("C10", r"C10_t = N9_t - z(\mathrm{RV}^{15}_t)",
     "信念波动-实现波动差", "事件风险相对已实现风险的溢出",
     "N9 + 期货", "60/15min", "两侧各自 z 后相减"),
    ("K1", r"K1_t = \sum_{60}(\mathbf{1}\{E^{up}\} - \mathbf{1}\{E^{dn}\})",
     "60 分钟净事件数", "离散事件的最直接计数聚合",
     "E 事件", "60min", "计数"),
    ("K2", r"K2_t = \frac{c_t - \mu_{hod}(c)}{\sigma_{hod}(c)},\;\;"
           r"c_t = \sum_{60}\mathbf{1}\{E\}",
     "分时基准化频率 z", "剔除时段固有活跃差（时序分桶）",
     "E 事件", "60min × 同小时组内 180 分钟", "分时段 z"),
    ("K3", r"K3_t = \sum_{\theta \in \Theta(j)}\sum_{60}"
           r"\mathbf{1}\{|N1^{\theta}| > \kappa\}",
     "跨主题截面事件计数", "同一宏观驱动下多主题的截面合计",
     "全主题事件", "60min", "计数"),
    ("K4", r"K4_t = \sum_{s \leq t} e^{-(t-s)/30}\,\mathrm{sgn}(E_s)",
     "指数衰减签名强度", "近期事件权重更高的记忆核",
     "E 事件", "τ=30min", "衰减核"),
    ("K5", r"K5_t = \max_{30}\,\mathrm{runlen}(\mathrm{sgn}(E))",
     "同向连发长度", "事件串（升级进行时）识别",
     "E 事件", "30min", "计数"),
    ("K6", r"K6_t = e^{-\Delta t_{last}/60} \cdot \mathrm{sgn}(E_{last})",
     "近因得分", "把「距上次信号的时间」转成有序数值",
     "E 事件", "τ=60min", "衰减核"),
    ("K7", r"K7_t = z\left(\sum_{60}|N1| \cdot "
           r"\mathbf{1}\{|N1| > \kappa\}\right)",
     "尾部幅度和 z", "只统计超阈值大跳的总能量",
     "N1 + κ", "60min", "滚动 z"),
    ("K8", r"K8_t = \frac{\sum_{120}\mathbf{1}\{E^{up}\} - "
           r"\sum_{120}\mathbf{1}\{E^{dn}\}}{\sum_{120}\mathbf{1}\{E\}}",
     "方向一致率", "事件流的方向纯度 [-1, 1]",
     "E 事件", "120min", "结构自归一"),
    ("K9", r"K9_t = \left(1 + \mathrm{med}_5(\Delta t_{inter})\right)^{-1}",
     "到达率热度", "事件间隔的倒数：越密越热",
     "E 事件", "近 5 个间隔", "倒数变换"),
    ("K10", r"K10_t = \mathbf{1}\{\geq 2\;\mathrm{themes\;fired\;in\;}15m\}"
            r"\cdot \mathrm{sgn}(N2_t)",
     "跨主题共振", "多主题同时触发的置信增强",
     "全主题事件", "15min", "符号型"),
    ("X1", r"X1_t = K10_t \cdot \mathbf{1}\{\mathrm{sgn}(K10_t \cdot "
           r"z(\mathrm{mom}^{15}_t)) > 0\}",
     "共振 + 期货同向确认", "两个主题与期货三方同向：最强确认组合",
     "K10 + 期货", "15min", "符号型"),
    ("X2", r"X2_t = K10_t \cdot \mathbf{1}\{|z(\mathrm{mom}^{15}_t)| < 0.5\}",
     "共振 + 期货未动", "多主题都动了、期货还没动：未兑现共振",
     "K10 + 期货", "15min", "符号型"),
    ("X3", r"X3_t = \mathrm{sgn}(E_t) \cdot "
           r"\mathbf{1}\{z(\mathrm{RV}^{15}_t) > 1\}",
     "事件 × 高波动状态", "事件落在期货高波动状态（状态条件事件）",
     "E + 期货", "15min", "符号型"),
    ("X4", r"X4_t = \mathrm{sgn}(E_t) \cdot "
           r"\mathbf{1}\{z(\mathrm{Vol}^{15}_t) > 1\}",
     "事件 × 期货放量", "事件与期货量能同时出现",
     "E + 期货", "15min", "符号型"),
    ("X5", r"X5_t = \mathbf{1}\{E^{up}_t \wedge \sum_{60} E^{up} \geq 2\} - "
           r"\mathbf{1}\{E^{dn}_t \wedge \sum_{60} E^{dn} \geq 2\}",
     "一小时内同向第二击", "同方向一小时内第二次事件（「两个事件相继发生」）",
     "E 事件", "60min", "符号型"),
    ("X6", r"X6_t = \mathrm{sgn}(\mathrm{flow}^{15}_t) \cdot E^{big}_t \cdot "
           r"\mathbf{1}\{|z(\mathrm{mom}^{15}_t)| < 0.5\}",
     "大额流 + 期货未动", "大钱进场、期货未反应：知情流（informed flow）假说",
     "flow + E_big + 期货", "15min", "符号型"),
    ("X7", r"X7_t = \mathrm{sgn}(E_t) \cdot \mathbf{1}\{\mathrm{session}_t = "
           r"\mathrm{night}\}",
     "夜盘事件", "事件 × 时段状态（夜盘 = 美国活跃时段）",
     "E + 时段", "1min", "符号型"),
    ("X8", r"X8_t = \mathrm{sgn}(E_t) \cdot \mathbf{1}\{\mathrm{day\;open\;"
           r"30min}\}",
     "开盘半小时事件", "隔夜积累信息在开盘窗口的即时事件",
     "E + 时段", "30min", "符号型"),
    ("X9", r"X9_t = \mathrm{sgn}(E_{t-3}) \cdot \mathbf{1}\{|r_{t-2..t}| < "
           r"0.3 \cdot \mathrm{RV}^{15}_t\}",
     "事件后 3 分钟未动", "延迟反应假说：事件 3 分钟后期货仍未动再触发",
     "E + 期货", "3min 延迟", "符号型"),
    ("X10", r"X10_t = \mathrm{sgn}(E_t) \cdot \mathbf{1}\{\mathrm{sgn}(N1_t) "
            r"\cdot (N2_t - N1_t) > 0\}",
     "多尺度同向确认", "1 分钟跳与此前 14 分钟累积同向：趋势中的跳",
     "N1 + N2", "1+14min", "符号型"),
]

FAM_TITLE = {
    "N": "N 族　数值信号（信念创新与主动流）",
    "C": "C 族　量价组合信号（PM × 期货，连续值）",
    "K": "K 族　离散事件的复杂统计指标",
    "X": "X 族　组合 / 条件 / 共振信号（离散签名型）",
}


# ================================================================ 主线章节
def guide_box() -> str:
    return """
<div class="callout blue"><b>阅读指南。</b>本文档分主线与附录两部分。
<b>主线（第 0-6 节）</b>只讲结论、数据处理、信号定义与检验结果；
<b>附录是主线的 supplement</b>：不了解 Polymarket 定价机制的读者请先读
<a href="#appA">附录 A（定价机制与市场结构）</a>；数据与方法的已知边界见
<a href="#appB">附录 B</a>，字段定义见
<a href="#appC">附录 C（字段词典）</a>；与《分钟线数据方案（合并版）》
方法论标准的逐条对照见 <a href="#appD">附录 D</a>；复现命令与产物索引见
<a href="#appE">附录 E</a>。文中标注「评审要求口径」处，对应导师预答辩提问
的原文口径。本篇同时以「第三部分」收录于主报告
cn_futures_polymarket_report.html（合订本）。</div>
"""


def summary_section(meta: dict) -> str:
    return f"""
<h2 id="s0">0　执行摘要</h2>
<p><b>研究问题。</b>Polymarket（用真金白银给「事件会不会发生」定价的链上
预测市场）的分钟级信念变化，对中国商品期货的分钟级收益有没有可检验的信息
含量？本篇是该问题的<b>分钟级测量与单信号检验层</b>；日频窗口级的识别框架
与方向性结论见主报告（cn_futures_polymarket_report.html v2.1），两者口径
互补、结论一致。</p>
<p><b>数据。</b>Polymarket 侧：全量链上逐笔成交记录 <b>855,614,453 笔 /
约 265 亿美元 / 约 69 万个市场</b>（2022-11-21 至 2026-07-14；前段来自公开
数据集，后段自建爬虫补齐，两段同构拼接、迁移期同质性实测通过）。期货侧：
聚宽风格主力连续 1 分钟 K 线，评估品种 SC / AU / AG / CU / M，样本
2026-01-05 至 07-13 共 {meta['SC']['n_days']} 个交易日、逐品种
{meta['M']['n_minutes']:,}（M）至 {meta['SC']['n_minutes']:,}（SC）个交易
分钟。</p>
<p><b>方法。</b>把登记主题（中东冲突、油价、金银、美联储、中美贸易等
8 类、492 个市场）的逐笔成交聚合为分钟级信念创新，构造 <b>40 个信号</b>
（10 数值 + 10 量价组合 + 10 离散复杂统计 + 10 组合条件），全部给出显式
公式与时序归一化登记；对 1/2/3/5/10/15 分钟前向收益做 IC / RankIC / ICIR、
按日盘夜盘分层、对离散事件做双基线事件研究（event study）；标签以 close-to-close 为主
口径、未来区间 VWAP 为对照。全流程防前视：信号分钟桶只含严格早于期货 K 线
收盘戳的成交，归一化与聚合权重均只用滚动历史（含市场权重的时点化累计
成交额）。</p>
<div class="finding"><span class="no">一</span><b>数据工程是本篇最扎实的
部分。</b>每一行是什么、主动方向怎么来、p_event 怎么算、时差怎么映射、
22 条边界情形如何处理，全部在 §1 用真实数据逐条给出；方法论文档的数据
检查清单逐项执行（§1.7）。</div>
<div class="finding"><span class="no">二</span><b>全表唯一跨品种稳定的
结构是 C8 复合信号（120 分钟未兑现缺口 = PM 信念累积 − 价格累积，
含期货自身反转腿）。</b>SC / AU / CU 三个
品种六个视界 RankIC 全为正且随视界单调增强（SC：+0.016 → +0.033，ICIR
0.54-0.60；M 亦全正但随视界衰减），数字见 §4.3 重点信号表。量级 0.01-0.03
——真实但小，属分钟级事件信号的正常水平。<b>外部审计降格
（重要）</b>：双腿分解显示该稳定性主要由"−z(过去 120 分钟期货收益)"的
期货自身反转腿承载（反转腿单独 t 8-11），PM 腿单独在按交易日聚类的
推断下于 SC / AU / CU / M 均不显著、AG 显著为负；随后的严格增量检验
（价格基线 + 安慰剂族 + OOS ΔR²）五品种全部未通过预设确认门槛，
<b>C8 作为可交易的 Polymarket 分钟增量已否定</b>，本表读作复合信号
描述，详见主报告 §12.14-12.15。</div>
<div class="finding"><span class="no">三</span><b>离散化组合（X 族）整体
弱于连续值结构——诚实的主要negative结果（§5）。</b>把创新离散成事件再组合
（共振 / 未动 / 时段条件）后，多数组合为负或跨品种混杂：确认型 X1 跨品种
一致为负（触发点在期货已反应之后、随后偏向回吐）；X2 / X9「未动」类仅在
个别品种（SC 短视界 / M）为正；事件研究显示基础事件后的方向收益接近零。
信息在<b>连续的缺口幅度</b>里，离散触发丢掉了它。</div>
<div class="finding"><span class="no">四</span><b>时段与标签稳健性。</b>
日盘 / 夜盘的平均信息强度相当（|RankIC| 均值 0.022 vs 0.020，同号率 52%
——不存在「夜盘普遍更强」）；close 与未来区间 VWAP 双标签下信号排序高度
一致（多数品种相关 0.77-0.95，低活跃的 AG 0.41 例外）。</div>
<div class="finding"><span class="no">五</span><b>多重检验纪律（§6）。</b>
全表约 1,128 个全时段格子（连同时段分层与标签对照约 4,000 个相关系数），
5% 名义水平下分别期望约 56 / 200 个偶然「显著」，单格不构成结论；且三类
一致性证据相互不独立（视界嵌套、SC/AU 共享事件流），解读以 §6 的纪律
为准。</div>
"""


def pm_data_section(meta: dict) -> str:
    pes = pd.read_parquet(D / "p_event_stats.parquet")
    pes_rows = []
    for r in pes.itertuples(index=False):
        name = THEME_CN.get(r.scope, r.scope)
        cls = ' class="base"' if r.scope == "全部二元市场" else ""
        pes_rows.append(
            f"<tr{cls}><td>{name}</td><td>{int(r.n_trades):,}</td>"
            f"<td>{r.mean_eq:.3f}</td><td>{r.mean_usdc_wtd:.3f}</td>"
            f"<td>{r.med:.3f}</td><td>{r.share_extreme:.0%}</td>"
            f"<td>{r.share_mid:.0%}</td><td>{r.usdc_mn:,.0f}</td></tr>")

    raw = pd.read_parquet(D / "example_tx_raw.parquet")
    clean = pd.read_parquet(D / "example_tx_clean.parquet")
    ex_rows = []
    for r in raw.sort_values("log_index").itertuples(index=False):
        cls = ' class="die"' if r.is_relay else ""
        tag = "<b>是（剔除）</b>" if r.is_relay else "否（保留）"
        ex_rows.append(
            f"<tr{cls}><td><code>{r.id}</code></td><td>{_short(r.maker)}</td>"
            f"<td>{_short(r.taker)}</td><td>{r.taker_direction}</td>"
            f"<td>{r.price:.3f}</td><td>{r.token_amount:,.2f}</td>"
            f"<td>{r.usdc_amount:,.2f}</td><td>{tag}</td></tr>")
    cl_rows = [
        f"<tr><td>{r.price:.3f}</td><td>{r.taker_direction}</td>"
        f"<td>{r.usdc_amount:,.2f}</td><td>{r.outcome_label}"
        f"（seq={r.outcome_seq}）</td><td><b>{r.p_event:.3f}</b></td>"
        f"<td><b>{int(r.D):+d}</b></td></tr>"
        for r in clean.itertuples(index=False)
    ]
    fills = raw.loc[~raw["is_relay"]]
    total = float(fills["token_amount"].sum())

    # 公式提升出 f-string（py3.11 的 f-string 反斜杠限制）。
    t_pev = tex(r"p^{event} = \mathrm{price} \cdot \mathbf{1}\{\mathrm{seq}=1\}"
                r" + (1 - \mathrm{price}) \cdot \mathbf{1}\{\mathrm{seq}=2\}")
    t_d = tex(r"D = (\mathbf{1}\{\mathrm{taker\;BUY}\} - "
              r"\mathbf{1}\{\mathrm{taker\;SELL}\}) \times "
              r"(\mathbf{1}\{\mathrm{Yes}\} - \mathbf{1}\{\mathrm{No}\})"
              r" \;\in\; \{-1, +1\}")
    t_align = tex(r"\mathrm{signal}_T = f\left(\mathrm{trades}\;[T-60s,\;T)"
                  r"\right), \qquad \mathrm{fwd}_k(T) = \ln C_{T+k} - \ln C_T")

    return f"""
<h2 id="s1">1　数据层：定义、处理与边界情形</h2>

<h3>1.1 Polymarket 成交数据：每一行是什么</h3>
<p>Polymarket 是建在 Polygon（一条公共区块链，交易记录公开、不可篡改、
任何人可验证）上的二元事件预测市场：每个市场问一件事
（如「美军 4 月 30 日前进入伊朗」），发行 Yes / No 两种代币，结算时押对的
一方每份兑付 1 USDC（美元稳定币，1 USDC ≈ 1 美元）；
<b>代币价格 ∈ (0,1) 即市场隐含概率</b>（机制推导见
附录 A.2）。挂单撮合在运营方链下引擎完成，<b>成交清算上链</b>——我们爬取
的就是链上清算事件 <code>OrderFilled</code>：不可篡改、无采样、全历史
可回溯。</p>
<p><b>每一行 = 一条 maker-taker 成交腿</b>：吃单方（taker）与一个具体
挂单方（maker）在某价位的一次撮合。一个吃单扫过多档挂单产生多行；交易所
以自身为对手方的「中继腿」是对整笔订单的汇总记账，<b>保留会使成交量恰好
翻倍，清洗层剔除</b>。行键 <code>链ID_区块号_日志序号</code> 全局唯一，
重复爬取天然幂等。下面是一笔真实订单的全部链上行（市场：「特朗普 6 月
30 日前宣布美伊停火结束」，UTC 2026-06-27 10:29:14）：</p>
{wrap('<tr><th>id</th><th>maker</th><th>taker</th><th>taker 方向</th>'
      '<th>价格</th><th>份额</th><th>USDC</th><th>中继腿?</th></tr>', ex_rows)}
<div class="texnote">三条成交腿份额 2,196.87 + 26.51 + 4,966.93 =
{total:,.2f}，与中继腿的 {total:,.2f} 恰好相等——同一笔卖单从 0.974 吃到
0.973 两档买盘（订单簿（order book）与滑点机制见附录 A.2）。字段逐一解释见附录 C。</div>

<h3>1.2 p_event 与 D：统一到「事件视角」的换算</h3>
<p>链上行只记录「某种代币值多少钱」；研究需要「事件发生的概率」。二元市场
Yes 价 + No 价恒等于 1（由铸造 / 销毁套利钉住，真实数据见附录 A.3），故
换算只有一条规则；主动方向 D 定义为「taker 这笔交易把事件概率往哪推」，
两层都来自链上资产转移的事实、非推断：</p>
{t_pev}
{t_d}
<div class="texnote">seq = outcome_seq（1 = Yes 即事件本身，2 = No）。D 的
真值表四格：买 Yes = +1、卖 Yes = −1、买 No = −1、卖 No = +1（+1 恒为
「把事件概率推高」）；经 2024-11 全月与公开数据集逐行核对。outcome_seq 由
代币的 ERC-1155 positionId 关联市场元数据取得，非猜测（附录 C）。</div>
<p>上面那笔真实订单换算到清洗层（taker 卖 No @ 0.973-0.974）：</p>
{wrap('<tr><th>price（No 代币）</th><th>taker 方向</th><th>USDC</th>'
      '<th>outcome</th><th>p_event</th><th>D</th></tr>', cl_rows)}
<div class="texnote">卖 No = 押「停火结束」会发生 = 把事件概率从 2.6% 往上
推（D = +1）——经注册方向先验 σ（升级利多原油）后，成为 SC 的正向信号
增量。σ 的注册规则见主报告 §3。</div>

<h3>1.3 时间戳、爬取延迟与时差映射</h3>
{wrap_text('<tr><th>环节</th><th>延迟 / 精度</th><th>说明</th></tr>', [
    "<tr><td>交易发生 → 上链</td><td>约 2 秒</td><td>Polygon 出块间隔约 "
    "2.1s；时间戳 = 出块时间，<b>秒级</b>（同秒可多笔，分钟内次序由 "
    "logIndex 保序）</td></tr>",
    "<tr><td>上链 → 本文抓取上界</td><td>finalized 约 1-2 分钟；保守回退 = "
    "链头 −256 块 ≈ 9 分钟</td><td><b>研究口径</b>选择（防区块重组），非"
    "技术上限；实盘增量可跟链头（端到端 2-5 秒 + 调度间隔）</td></tr>",
    "<tr><td>历史回填</td><td>无延迟概念</td><td>按区块区间幂等重放，断点"
    "续爬</td></tr>",
])}
<p><b>时差映射只有一步且无歧义</b>：链上时间戳是 UTC 秒，北京时间 =
UTC + 8（中国无夏令时，映射常数恒定；美东 DST 只影响主报告的美股 ETF
对齐，与本篇无关）。示例：<code>block_timestamp = 1782556154</code> → UTC
2026-06-27 10:29:14 → 北京 18:29:14（恰为国内闭市时段——闭市事件由主报告
的窗口设计处理，本篇只评估交易分钟）。</p>
<p><b>与期货 K 线的对齐（防前视的关键一步）</b>：期货 1 分钟 K 线的时间戳
是<b>收盘戳</b> T（21:31 的 K 线覆盖 21:30:00-21:30:59）。PM 侧按分钟桶
[T−60s, T) 聚合、标签取右端 T——对齐后，<b>标签 T 的 PM 信号只含严格早于
T 的成交</b>，前向收益从 T 起算，任何信息都不可能穿越 T：</p>
{t_align}

<h3>1.4 Polymarket 侧规模与 p_event 分布</h3>
{wrap('<tr><th>段</th><th>行数</th><th>市场数</th><th>时间覆盖（UTC）</th>'
      '<th>名义额</th></tr>', [
    "<tr><td>公开数据集段（HF daily_aligned）</td><td>601,934,424</td>"
    "<td>688,131</td><td>2022-11-21 19:50:09 … 2026-04-28 11:00:40</td>"
    "<td>$196.2 亿</td></tr>",
    "<tr><td>自建爬虫段（扩展逐笔成交记录）</td><td>253,680,029</td><td>528,540</td>"
    "<td>2026-04-28 11:04:48 … 2026-07-14 16:08:44</td><td>$69.1 亿</td></tr>",
    "<tr class='base'><td>合计</td><td><b>855,614,453</b></td>"
    "<td>约 69 万（两段市场有重叠）</td><td>3.6 年</td><td><b>$265.3 亿</b>"
    "</td></tr>",
])}
<div class="texnote">两段拼接的同质性：v2 新交易所与旧合约并行的 25 天内，
六个抽样窗口的 v2 成交占比全部 &lt; 0.04%（逐窗数字见主报告 §14.1），拼接
边界日共有市场的概率中位差 0.02。</div>
<p><b>p_event 的成交分布</b>（「p_event 的均值是多少」的正式口径——汇总
均值是「哪些市场更活跃」的产物，必须连同分布占比一起读）：</p>
{wrap('<tr><th>范围</th><th>成交笔数</th><th>简单均值</th>'
      '<th>成交额加权均值</th><th>中位数</th><th>极端区占比（&lt;0.05 或 '
      '&gt;0.95）</th><th>中间区占比（0.2-0.8）</th><th>名义额 $mn</th></tr>',
      pes_rows)}
<div class="texnote">三个结构性事实：（i）<b>全平台</b>成交以中间区为主
（64%，均值 ≈ 0.49——体育 / 选举市场在均衡赔率附近交易）；（ii）<b>我们的
事件主题</b>显著偏向低概率区（金银 / 美联储 / 台海中位数 0.06-0.15）——
日期阶梯与阈值阶梯类市场天然在「小概率尾部」交易，这正是 logit 变换 + 截断
[0.02, 0.98] 的动机；（iii）多数主题的成交额加权均值高于简单均值（俄乌
0.73 vs 0.25 最极端）——大额成交偏向高概率 / 临近确认阶段，小额投机偏向
尾部。</div>

<h3>1.5 期货 1 分钟数据、主题映射与交易时段</h3>
<p><b>品种 × 主题映射</b>（准入时注册、不事后调整；σ 为方向先验，+1 = 事件
概率上升利多该品种）：</p>
{wrap_text('<tr><th>品种</th><th>映射主题</th><th>σ 示例</th>'
      '<th>经济逻辑</th><th>入选理由</th></tr>', [
    "<tr><td>SC 原油</td><td>中东冲突、油价阈值</td><td>升级 +1 / 停火 −1"
    "</td><td>INE 可交割油种为中东油，供给风险直接</td><td>主题事件密度最高"
    "+ 夜盘流动性好</td></tr>",
    "<tr><td>AU 黄金</td><td>中东冲突、金银阈值、美联储</td><td>升级 +1、"
    "降息 +1</td><td>避险 + 利率敏感</td><td>避险敞口强，多主题覆盖</td></tr>",
    "<tr><td>AG 白银</td><td>金银阈值、美联储</td><td>触及上方价位 +1</td>"
    "<td>贵金属 + 工业属性</td><td>与 AU 成对照（无中东直接映射）</td></tr>",
    "<tr><td>CU 铜</td><td>美联储、中美贸易</td><td>降息 +1、缓和 +1</td>"
    "<td>全球需求 / 利率敏感</td><td>宏观品种代表</td></tr>",
    "<tr><td>M 豆粕</td><td>中美贸易</td><td>缓和 −1（进口恢复利空）</td>"
    "<td>大豆进口结构</td><td>贸易主题最直接的标的</td></tr>",
])}
<p><b>数据来源</b>：聚宽风格主力连续（XX9999）分钟 CSV 本地重建（88 品种、
350 万根 K 线，与交易所交易日历核对、断言测试覆盖），字段 open / high / low /
close / volume（手）/ money（元）/ open_interest / contract；K 线时间戳为
收盘戳。<b>样本期为什么是 2026-01-05 至 07-13</b>：这是登记主题市场高密度
存在期（映射主题在 2025 年前事件密度不足，见 §4.7 逐月图）与期货分钟库
覆盖期的交集；单一状态期（美伊冲突主导）的外推局限在 §6 声明。源数据的三个陷阱（夜盘文件
归属 / 换月 / 主力拼接）及处理见 §1.6。评估品种的时段结构：</p>
{wrap('<tr><th>品种</th><th>日盘</th><th>夜盘</th><th>交易分钟/日（约）</th>'
      '</tr>', [
    "<tr><td>SC 原油 / AU 黄金 / AG 白银</td>"
    "<td rowspan='3'>09:00-10:15，10:30-11:30，13:30-15:00</td>"
    "<td>21:00-次日 02:30</td><td>555</td></tr>",
    "<tr><td>CU 铜</td><td>21:00-次日 01:00</td><td>465</td></tr>",
    "<tr><td>M 豆粕</td><td>21:00-23:00</td><td>345</td></tr>",
])}
<p><b>时段间隔的统一处理——连续分钟段规则</b>：相邻 K 线时间差 &gt; 1 分钟
即断开为新段；<b>前向收益一律不跨段</b>（置 NaN）。午间休市、10:15 小节、
日夜盘边界、节假日全部被同一条规则自动处理，无需手工日历。逐品种规模：</p>
{wrap('<tr><th>品种</th><th>交易分钟</th><th>交易日</th><th>连续段数</th>'
      '<th>夜盘分钟占比</th><th>fwd15 有效率</th><th>high==low 分钟占比</th>'
      '<th>PM 主题活跃分钟占比</th></tr>',
      [f"<tr><td>{p}</td><td>{m['n_minutes']:,}</td><td>{m['n_days']}</td>"
       f"<td>{m['n_segments']:,}</td><td>{m['night_minute_share']:.0%}</td>"
       f"<td>{m['fwd15_valid_share']:.1%}</td><td>{m['locked_share']:.2%}</td>"
       f"<td>{m['pm_active_minute_share']:.1%}</td></tr>"
       for p, m in meta.items()])}
<div class="texnote">PM 主题活跃分钟占比的品种差异是真实结构：SC / AU 映射
中东主题（样本期 125 个交易日内高强度交易），AG / CU / M 的主题（金银阈值 / 美联储 /
贸易）事件密度低——低活跃品种的信号覆盖天然稀疏，解读其 IC 时须结合 §4.1
的频率表。</div>
"""


def edge_cases_section() -> str:
    pm_cases = [
        ("P1", "中继腿（relay leg，交易所自身为对手方的汇总记账行）",
         "成交量恰好翻倍、所有统计失真",
         "按 taker == 交易所合约地址剔除（§1.1 示例第 4 行）", "无"),
        ("P2", "negRisk 多选一市场（平台把互斥候选项打包发行的类型，如「谁当选」）",
         "p_event = 1 − price 的二元换算不成立",
         "整类剔除（清洗层 WHERE NOT neg_risk）", "损失该类市场信息"),
        ("P3", "非二元市场（结果槽 > 2）",
         "同上", "剔除（仅保留 max(outcome_seq) = 2）", "无"),
        ("P4", "尘埃成交（单笔 &lt; $1）",
         "把噪声当概率变化",
         "不硬性剔除；分钟聚合用 usdc 加权 vwap，权重天然趋零",
         "极稀薄分钟仍可能被小单主导"),
        ("P5", "结算后残余交易",
         "常数价格被当作信号",
         "resolved_at 之后剔除（结算时刻取链上 ConditionResolution 事件，"
         "覆盖 455/492）", "37 个未结算市场本就无需截断"),
        ("P6", "UMA 预言机（oracle，第三方裁决机制，附录 A.4）争议重报（同市场多次结算事件）",
         "结算时刻取错", "取首次 ConditionResolution",
         "争议期内价格反映「预期裁决」"),
        ("P7", "准入前的低流动期（新市场几乎无人交易）",
         "稀薄价格污染信号",
         "时点化准入：累计成交额首达 $10 万后才入面板（admit_ts）",
         "准入前的真实信息被放弃（保守方向）"),
        ("P8", "交易所 v1 → v2 迁移（2026-04-28）",
         "拼接断层",
         "迁移期同质性实测（v2 占比 &lt; 0.04%）+ 边界日交叉核对", "无"),
        ("P9", "价格贴近 0 / 1（logit 发散）",
         "单笔尾部成交产生无穷大创新",
         "p 截断到 [0.02, 0.98] 再取 logit", "截断区内的极端变化被压缩"),
        ("P10", "同秒多笔 / 区块时间非均匀",
         "把区块批次当作均匀采样",
         "分钟桶聚合（usdc 加权 vwap），不假设逐笔等间隔",
         "分钟内次序信息未用"),
        ("P11", "同一现实事件对应几十个市场（日期 / 阈值阶梯族）",
         "一个事件被当成几十个独立样本（伪重复）",
         "分钟层：主题聚合天然合并；事件影响回归层：episode 聚类（主报告 "
         "§16——曾据此撤回一个错误结论）", "族内权重仍按流动性"),
        ("P12", "自成交刷量（同主体左右手互倒）",
         "量类信号（N6 / E_big）虚高",
         "未专门识别（需地址聚类）；方向类信号天然免疫（自成交方向对冲）",
         "量类信号可能偏高，结论中已降权"),
    ]
    fut_cases = [
        ("F1", "夜盘 K 线存放在开始时刻次一自然日的文件里",
         "周一文件永远无夜盘 → 误判缺数据",
         "重建时按时间戳重归属（夜盘归属下一交易日）+ 断言测试", "无"),
        ("F2", "午间休市 / 10:15 小节 / 日夜盘间隔",
         "跨间隔算收益 = 把休市当 1 分钟",
         "连续分钟段规则（相邻 K 线 > 1 分钟断段，前向不跨段）", "无"),
        ("F3", "主力合约换月（21:01 切换）",
         "跨合约拼接跳空被当作收益",
         "交易日内合约唯一；跨 21:00 边界的前向窗口已被段规则截断；日频侧"
         "另剔换月日", "无"),
        ("F4", "涨跌停",
         "封板价上信号不可成交、收益被截尾",
         "主力连续无涨跌停价表；以 high == low 锁死分钟占比作代理监控"
         "（0.03%-2.43%，多为清淡分钟）；收益截尾属保守方向",
         "无法精确区分「清淡」与「封板」——本库明示局限"),
        ("F5", "零成交分钟",
         "vwap 除零、量类特征失真",
         "vwap 置 NaN 不前向填充（方法论清单原则）；量类窗口统计跳过 NaN",
         "无"),
        ("F6", "节假日 / 周末（期货休市而 PM 7×24 在交易）",
         "闭市期间信息丢失或错配",
         "本篇只评估交易分钟；闭市累积信息由主报告窗口设计（night/gap/day）"
         "覆盖", "两套设计需对照阅读"),
        ("F7", "开盘 / 收盘分钟的撮合特殊性（集合竞价）",
         "开盘 K 线含隔夜跳空信息",
         "X8 把开盘 30 分钟做成显式条件信号单独检验，不混入全样本",
         "首分钟未单独剔除"),
    ]
    align_cases = [
        ("A1", "时区（UTC vs 北京）", "错 8 小时 = 信号错到另一个时段",
         "PM 为 UTC 秒 + 8h 常数映射；中国无夏令时", "无"),
        ("A2", "K 线收盘戳与起始戳约定", "错 1 分钟 = 前视",
         "本库 K 线为收盘戳（已核对）；PM 桶右端标签与之对齐，信号严格早于"
         "前向窗", "无"),
        ("A3", "归一化窗口跨时段 / 跨日", "z 分数被时段结构污染",
         "z 窗口 4800 分钟跨段连续（约 10 个交易日）；另设 K2 分时段基准化"
         "对照；§4.4 分层检验兜底", "无"),
    ]

    def case_rows(cases):
        return [
            f"<tr><td><b>{cid}</b></td><td>{what}</td><td>{risk}</td>"
            f"<td>{how}</td><td>{resid}</td></tr>"
            for cid, what, risk, how, resid in cases
        ]

    head = ('<tr><th>#</th><th>现象</th><th>不处理的后果</th>'
            '<th>我们的处理</th><th>残留风险</th></tr>')
    return f"""
<h3 id="s16">1.6 Edge cases 完备清单（22 条）</h3>
<p>数据处理的可信度不在「主流程对」，在 corner cases 全部被显式处理或显式
声明。三张表分别覆盖 Polymarket 侧、期货侧与对齐侧：</p>
<h4>Polymarket 侧（12 条）</h4>
{wrap_text(head, case_rows(pm_cases))}
<h4>期货侧（7 条）</h4>
{wrap_text(head, case_rows(fut_cases))}
<h4>对齐侧（3 条）</h4>
{wrap_text(head, case_rows(align_cases))}
"""


def quality_section(meta: dict) -> str:
    checks = [
        ("交易日数量是否完整",
         f"{meta['SC']['n_days']} 个交易日与交易所日历核对（假日无 K 线为"
         "预期）", "通过"),
        ("每日分钟数是否完整",
         f"SC 满额 555 分钟/日，实测日均 "
         f"{meta['SC']['n_minutes'] // meta['SC']['n_days']}——差额来自"
         "节假日前夜盘停市的交易日；缺失即断段、显式可见", "通过"),
        ("每标的记录数是否异常",
         "5 品种分钟数与其时段结构成比例（§1.5 表）", "通过"),
        ("价格字段逻辑错误",
         "high ≥ max(open, close)、low ≤ min(open, close) 全量断言", "通过"),
        ("成交量 / 成交额负值", "全量扫描无负值", "通过"),
        ("复权断点",
         "期货主力连续无复权概念；换月跳空按 F3 处理", "不适用 + 替代处理"),
        ("标的池变更正确生效",
         "登记表版本固定（cn_registry_v3，492 市场）+ 时点化准入", "通过"),
        ("标签未来函数",
         "前向收益 shift(-k) + 段内约束；信号桶右端标签；PIT 单元测试断言",
         "通过"),
        ("午休 / 停牌 / 涨跌停一致性",
         "段规则统一处理全部间隔；涨跌停为代理监控（F4）",
         "通过（F4 有声明局限）"),
        ("样本覆盖率突变",
         "PM 主题活跃分钟占比逐品种监控（§1.5），AG/CU/M 低活跃如实呈现",
         "通过"),
        ("行级可追溯",
         "PM 行键 = 链ID_区块_日志序号；期货 K 线以 (contract, ts) 唯一",
         "通过"),
        ("全流程可复现",
         "任一表可由附录 E 命令重算；数据版本 = 爬虫游标 + 登记表版本",
         "通过"),
    ]
    rows = [
        f"<tr><td>{c}</td><td>{how}</td><td><b>{res}</b></td></tr>"
        for c, how, res in checks
    ]
    return f"""
<h3>1.7 数据质量检查清单（对照方法论标准逐项执行）</h3>
{wrap_text('<tr><th>检查项（方法论文档 §数据规范）</th><th>执行方式</th>'
      '<th>结果</th></tr>', rows)}
"""


# ---------------------------------------------------------------- §2 信号构造
def construction_section() -> str:
    t_z = tex(r"z_t(x) = \frac{x_t - \mu_t}{\sigma_t}, \qquad \mu_t, "
              r"\sigma_t = \mathrm{mean/std}\left(x_{t-4799..t}\right),"
              r"\;\; \geq 960\;\mathrm{obs}")
    t_k = tex(r"\kappa_t = \max\left(0.05,\; 3 \times 1.4826 \times "
              r"\mathrm{MAD}_{4800}(N1_{\neq 0})\right)")
    return f"""
<h2 id="s2">2　信号构造：归一化方法论与四族设计</h2>

<h3>2.1 归一化：为什么、怎么做、登记在哪</h3>
<p>方法论文档把「因子未做归一化」列为算法层风险（量纲失真、极端值放大、
跨时点不可比），推荐口径是<b>横截面</b>稳健归一化（同一时点跨股票的
median-MAD）。本篇是<b>单品种时序</b>场景（每个品种一条信号序列，无横
截面），对应物是<b>时序滚动归一化</b>——同一原则（可比性、稳健性、无
前视）在时间轴上的实现：</p>
{t_z}
<div class="texnote">窗口 4800 交易分钟 ≈ 9 个 SC 交易日：短到能适应事件
期 / 平静期的水平迁移，长到日内结构不主导估计。只用截至当刻的历史——不用
全样本统计量，否则 1 月的信号会「知道」6 月的波动水平（时间前视）。
滚动统计设最小样本 960（约 2 个交易日）：不足 960 个历史观测时 z 不输出
（预热期，按缺测 = 0 处理），960-4800 之间按实际可得历史计算。</div>
<p>离散事件阈值用<b>稳健尺度</b>而非标准差（事件分布重尾，std 被自身尾部
抬高会漏掉真事件）：</p>
{t_k}
<div class="texnote">MAD 对<b>带符号</b>非零创新围绕其滚动中位数计算
（非对绝对值序列）；1.4826 把 MAD 换算为正态尺度下的尺度等价物；下限
0.05
（logit 单位）防止长平静期阈值塌缩。K2 另做同小时分桶、组内滚动 180 个
同时段分钟（约 3 个交易日的同小时样本）基准化，处理小时级活跃结构
（美国时段 PM 天然更活跃）。</div>
<p><b>记号约定</b>：σ<sub>m</sub> 在本文专指注册方向先验（±1，§1.2 /
§1.5 映射表）；上段「正态尺度等价物」中的尺度记号与其语境可分，
不混用。</p>
<p><b>登记纪律</b>（方法论要求：归一化方式入元信息、变归一化即变版本）：
每个信号的输入、窗口、归一化方式在 §3.0 元信息总表逐一登记；原始值（N1 /
N2）与归一化值（N4 等）是不同的信号 id，不混用。</p>

<h3>2.2 基础派生量</h3>
{wrap_text('<tr><th>侧</th><th>派生量</th><th>定义</th></tr>', [
    "<tr><td rowspan='5'>Polymarket（主题级，分钟）</td><td>dl</td>"
    "<td>orientation 加权 logit 创新：市场级 usdc 加权 vwap → logit →"
    "对自身上一活跃分钟差分 → σ·√USDC 加权聚合</td></tr>",
    "<tr><td>flow</td><td>σ · D · USDC 签名主动流</td></tr>",
    "<tr><td>usdc</td><td>分钟成交额</td></tr>",
    "<tr><td>n</td><td>分钟笔数</td></tr>",
    "<tr><td>n_mkts</td><td>分钟活跃市场数（族内广度）</td></tr>",
    "<tr><td rowspan='5'>期货（品种级，分钟）</td><td>r1</td>"
    "<td>log close 差分（段内）</td></tr>",
    "<tr><td>mom15</td><td>15 分钟累计收益</td></tr>",
    "<tr><td>rv15</td><td>15 分钟已实现波动 √Σr²</td></tr>",
    "<tr><td>volu15 / amt15</td><td>15 分钟量 / 额</td></tr>",
    "<tr><td>vwap 代理</td><td>money / volume（VWAP 标签用；比值型收益中"
    "合约乘数抵消）</td></tr>",
])}

<h3>2.3 四族设计逻辑</h3>
<p><b>N 族</b>回答「Polymarket 此刻在说什么」（信念创新、主动流、强度、
广度、不确定性）；<b>C 族</b>把它与期货量价做连续值组合（确认型 / 背离型 /
状态调制型 / 残差型——C7 对应方法论文档的残差因子）；<b>K 族</b>先把连续
创新离散化成事件（E<sup>up</sup> / E<sup>dn</sup> / E<sup>big</sup>），再做
计数、频率分桶、指数衰减、连发、截面合计等复杂统计（「时序分桶 / 统计频率
/ 跳动超阈值 / 截面统计」的逐条落实）；<b>X 族</b>是组合 / 条件信号：事件
与事件（共振、第二击）、事件与期货状态（未动 / 放量 / 高波动）、事件与
时段（夜盘 / 开盘）的联合触发，回答「两个事件同时发生之类的组合」——好坏
结果在 §5 并陈。</p>
"""


def signal_defs_section() -> str:
    meta_rows = [
        f"<tr><td><b>{sid}</b></td><td>{title}</td><td>{inputs}</td>"
        f"<td>{window}</td><td>{norm}</td></tr>"
        for sid, _f, title, _w, inputs, window, norm in SIGNAL_DEFS
    ]
    blocks = [
        "<h3>3.0 信号元信息总表（40 个）</h3>",
        wrap('<tr><th>id</th><th>名称</th><th>输入</th><th>窗口</th>'
             '<th>归一化</th></tr>', meta_rows),
    ]
    cur = ""
    idx = {"N": "3.1", "C": "3.2", "K": "3.3", "X": "3.4"}
    for sid, formula, title, why, _i, _w, _n in SIGNAL_DEFS:
        fam = sid[0]
        if fam != cur:
            cur = fam
            blocks.append(f"<h3>{idx[fam]} {FAM_TITLE[fam]}</h3>")
            if fam == "K":
                blocks.append(
                    "<p>原始离散事件（K / X 两族的构件）：利多事件 "
                    "E<sup>up</sup>: N1<sub>t</sub> &gt; κ<sub>t</sub>；利空 "
                    "E<sup>dn</sup> 对称；美元量爆发 E<sup>big</sup>: 分钟"
                    "成交额 &gt; 滚动 99 分位。阈值 κ 定义见 §2.1。</p>")
            if fam == "X":
                blocks.append(
                    "<p>X 族构造原则：每个信号都是「事件 × 条件」的联合"
                    "触发，条件来自另一主题（共振）、期货状态（动量 / 波动 /"
                    "量能）或时段——即答辩要求的「各种组合」。</p>")
        blocks.append(
            f"<div class='sigdef'><b>{sid}　{title}</b>{tex(formula)}"
            f"<div class='why'>{why}</div></div>")
    return "<h2 id='s3'>3　信号定义全集（公式）</h2>" + "".join(blocks)


# ---------------------------------------------------------------- §4 评估
def stats_section() -> str:
    st = pd.read_parquet(D / "signal_stats.parquet")
    sub = st[st["product"] == "SC"].set_index("signal").reindex(ALL_SIGNALS)
    head = ("<tr><th>信号</th><th>非零占比</th><th>日均非零</th>"
            "<th>缺失率</th><th>极值率</th><th>均值</th><th>标准差</th>"
            "<th>偏度</th><th>q1</th><th>中位</th><th>q99</th></tr>")
    rows = []
    for s, r in sub.iterrows():
        if pd.isna(r["n_minutes"]):
            continue
        ext = "" if pd.isna(r["extreme_rate"]) else f"{r['extreme_rate']:.1%}"
        def fm(v: float) -> str:
            return "0" if abs(v) < 1e-10 else f"{v:+.3g}"
        rows.append(
            f"<tr><td>{s}</td><td>{r['nonzero_rate']:.1%}</td>"
            f"<td>{r['per_day']:.0f}</td><td>{r['missing_rate']:.1%}</td>"
            f"<td>{ext}</td>"
            f"<td>{fm(r['mean'])}</td><td>{r['std']:.3g}</td>"
            f"<td>{r['skew']:+.2f}</td><td>{fm(r['q1'])}</td>"
            f"<td>{fm(r['q50'])}</td><td>{fm(r['q99'])}</td></tr>")
    figs = []
    panel = pd.read_parquet(D / "panel_SC.parquet", columns=ALL_SIGNALS)
    for fam, sigs in (("N", NUM_SIGNALS), ("C", COMBO_SIGNALS),
                      ("K", K_SIGNALS), ("X", X_SIGNALS)):
        fig, axes = plt.subplots(2, 5, figsize=(10, 3.6))
        for ax, s in zip(axes.ravel(), sigs, strict=True):
            v = panel[s].replace([np.inf, -np.inf], np.nan).dropna()
            v = v[v != 0]
            if len(v) > 50:
                lo, hi = np.percentile(v, [0.5, 99.5])
                ax.hist(v.clip(lo, hi), bins=40, color=P["blue"])
            ax.set_yscale("log")
            ax.set_title(s, fontsize=8)
            ax.tick_params(labelsize=6)
        fig.tight_layout()
        figs.append(figure(
            fig, f"{fam} 族信号非零取值分布（SC，截尾 [0.5%, 99.5%]，"
                 "对数频数轴）。"
                 + ("X 族取值集中在 ±1（签名离散型）。" if fam == "X" else "")))
    return ("<h2 id='s4'>4　单信号评估：统计基本功</h2>"
            "<h3>4.1 频率、取值统计、缺失率与极值率</h3>"
            "<p>下表为 SC；全部 5 品种 × 40 信号见 "
            "<code>signal_stats.parquet</code>。缺失率 = z 预热期占比（缺测"
            "编码为 0），极值率 = 非零值中偏离均值 3σ 以上占比：</p>"
            + wrap(head, rows)
            + "<h3>4.2 取值分布</h3>" + "".join(figs))


def ic_section() -> str:
    ic = pd.read_parquet(D / "ic_table.parquet")
    ic_all = ic[ic["scope"] == "all"]
    figs, parts = [], []

    sub = ic_all[ic_all["product"] == "SC"]
    piv = sub.pivot_table(index="signal", columns="horizon",
                          values="rank_ic").reindex(ALL_SIGNALS)
    piv_ir = sub.pivot_table(index="signal", columns="horizon",
                             values="icir").reindex(ALL_SIGNALS)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 7.4))
    for ax, (mat, ttl, vmax) in zip(
        axes, ((piv, "RankIC", 0.06), (piv_ir, "ICIR（逐日）", 0.6)),
        strict=True,
    ):
        im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)),
                      [f"{h}m" for h in mat.columns], fontsize=7)
        ax.set_yticks(range(len(mat.index)), mat.index, fontsize=6)
        ax.set_title(f"SC · {ttl}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    figs.append(figure(fig, "SC 的 40 信号 × 6 视界 RankIC 与 ICIR 热图"
                            "（红 = 正、蓝 = 负；ICIR 空缺见下文条件）。"))

    # 重点数值信号表：C8 / C2 / C7 / N4 全品种全视界（回应「C8 数字在哪」）
    focus_rows = []
    for sig in ("C8", "C2", "C7", "N4"):
        for prod in PRODUCT_THEMES:
            g = ic_all[(ic_all["signal"] == sig)
                       & (ic_all["product"] == prod)].sort_values("horizon")
            if g.empty:
                continue
            cells = "".join(f"<td>{r.rank_ic:+.4f}</td>"
                            for r in g.itertuples(index=False))
            ir15 = g[g["horizon"] == 15]["icir"]
            ir = ("" if ir15.empty or pd.isna(ir15.iloc[0])
                  else f"{ir15.iloc[0]:+.2f}")
            npos = int((g["rank_ic"] > 0).sum())
            cls = ' class="live"' if npos == 6 else (
                ' class="die"' if npos == 0 else "")
            focus_rows.append(
                f"<tr{cls}><td>{sig}</td><td>{prod}</td>{cells}"
                f"<td>{ir}</td><td>{npos}/6</td></tr>")
    focus_head = ("<tr><th>信号</th><th>品种</th>"
                  + "".join(f"<th>{k}m</th>" for k in HORIZONS)
                  + "<th>ICIR@15m</th><th>为正视界</th></tr>")
    focus_html = (
        "<h4>重点数值信号表（C8 / C2 / C7 / N4，全品种 × 全视界 RankIC）</h4>"
        + wrap(focus_head, focus_rows)
        + "<div class='texnote'>绿行 = 六视界全正。C8 在 SC / AU / CU 六视界"
          "全正且随视界单调增强（M 全正但衰减），是全表唯一跨品种一致的"
          "结构；C2 / C7 / N4 在 SC / M 为正、CU 为负，跨品种不一致，按 §6 "
          "纪律只列事实不作结论。</div>")

    # 衰减曲线：C8 + 高覆盖信号
    show = ["C8", "C7", "N4", "C2", "K4", "K8"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    colors = [P[c] for c in ("blue", "aqua", "orange", "violet", "red",
                             "green")]
    for i, sg in enumerate(show):
        d = sub[sub["signal"] == sg].sort_values("horizon")
        if d.empty:
            continue
        ax.plot(d["horizon"], d["rank_ic"], marker="o", ms=3, label=sg,
                color=colors[i % 6], lw=1.2)
    ax.axhline(0, color=P["ink"], lw=0.6)
    ax.set_xlabel("前向视界（分钟）")
    ax.set_ylabel("RankIC")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    pub_style.soft_grid(ax)
    fig.tight_layout()
    figs.append(figure(fig, "衰减曲线（SC，高覆盖信号）：C8 的 RankIC 随"
                            "视界单调增强、15 分钟内未见衰减拐点——信息在"
                            "事件后持续渗入价格；其余高覆盖信号幅度小且"
                            "形态平缓。稀疏触发信号（n 数百）的衰减形态"
                            "不可靠，不在此图呈现。"))

    # 每品种 top-8（加可靠性门槛）
    for prod in PRODUCT_THEMES:
        s2 = ic_all[(ic_all["product"] == prod) & (ic_all["n"] >= 1000)
                    & (ic_all["n_days"] >= 20)]
        if s2.empty:
            continue
        best = (s2.assign(a=lambda x: x["rank_ic"].abs())
                .sort_values("a", ascending=False).head(8))
        head = ("<tr><th>信号</th><th>视界</th><th>n</th><th>IC</th>"
                "<th>RankIC</th><th>ICIR</th><th>IC 日数</th></tr>")
        rows = [
            f"<tr><td>{r.signal}</td><td>{r.horizon}m</td><td>{r.n:,}</td>"
            f"<td>{r.ic:+.4f}</td><td><b>{r.rank_ic:+.4f}</b></td>"
            f"<td>{'' if pd.isna(r.icir) else f'{r.icir:+.2f}'}</td>"
            f"<td>{r.n_days}</td></tr>"
            for r in best.itertuples(index=False)
        ]
        parts.append(
            f"<h4>{prod}（|RankIC| 前 8；门槛 n ≥ 1,000 且 IC 日数 ≥ 20）"
            f"</h4>" + wrap(head, rows))

    # 时段分层（计算驱动的中性表述）
    sess_html = ""
    if "night" in set(ic["scope"]):
        piv_s = (ic[ic["horizon"] == 15]
                 .pivot_table(index=["product", "signal"], columns="scope",
                              values="rank_ic").reset_index())
        piv_s = piv_s.dropna(subset=["day", "night"])
        if not piv_s.empty:
            fig, ax = plt.subplots(figsize=(4.6, 4.0))
            for prod, c in (("SC", P["blue"]), ("AU", P["orange"]),
                            ("AG", P["aqua"]), ("CU", P["violet"]),
                            ("M", P["green"])):
                d = piv_s[piv_s["product"] == prod]
                if d.empty:
                    continue
                ax.scatter(d["day"], d["night"], s=12, label=prod, color=c,
                           alpha=0.75)
            lim = 0.1
            ax.plot([-lim, lim], [-lim, lim], color=P["ink2"], lw=0.6,
                    ls="--")
            ax.axhline(0, color=P["ink"], lw=0.5)
            ax.axvline(0, color=P["ink"], lw=0.5)
            ax.set_xlabel("日盘 RankIC（15m）")
            ax.set_ylabel("夜盘 RankIC（15m）")
            ax.legend(fontsize=7, frameon=False)
            pub_style.soft_grid(ax)
            fig.tight_layout()
            cons = float((np.sign(piv_s["day"]) == np.sign(piv_s["night"]))
                         .mean())
            mean_d = float(piv_s["day"].abs().mean())
            mean_n = float(piv_s["night"].abs().mean())
            sess_html = (
                "<h3>4.4 时段分层：日盘 vs 夜盘</h3>"
                "<p>分钟信号的时段依赖是方法论必检项（「时段状态」）。对每个"
                "（品种, 信号）分别在日盘与夜盘估计 15 分钟 RankIC：</p>"
                + figure(fig,
                         f"日盘 vs 夜盘 RankIC 散点（15m）。同号率 "
                         f"{cons:.0%}（接近随机）；|RankIC| 均值日盘 "
                         f"{mean_d:.3f} vs 夜盘 {mean_n:.3f}——汇总层面"
                         "未见系统性时段不对称，「信息在其到达时段更有效」"
                         "的假说在平均意义上不成立；时段结构证据只在个别"
                         "（品种, 信号）上出现（如 X7 在 SC，见 §5），"
                         "逐格数字见 ic_table.parquet 的 scope 列。"))

    # 标签稳健性（全品种）
    lab = pd.read_parquet(D / "label_robustness.parquet")
    l15 = lab[lab["horizon"] == 15].dropna()
    corr_by = {p_: round(float(g["rank_ic_close"].corr(g["rank_ic_vwap"])), 2)
               for p_, g in l15.groupby("product")}
    lab_html = (
        "<h3>4.5 标签口径稳健性：close vs 未来区间 VWAP</h3>"
        "<p>方法论文档推荐未来区间 VWAP 收益作标签（降单点噪声、贴近执行）。"
        "本篇主口径为 close-to-close，另算 VWAP 标签全表对照：15 分钟 "
        "RankIC 在两种标签下的横向相关为 "
        f"AU {corr_by.get('AU')}、M {corr_by.get('M')}、CU {corr_by.get('CU')}、"
        f"SC {corr_by.get('SC')}——排序结论对标签口径稳健；例外是 AG"
        f"（{corr_by.get('AG')}），与其 PM 活跃分钟占比低（4.3%）、逐格样本"
        "稀疏一致，标签稳健性结论限于 PM 活跃品种（全表 "
        "<code>label_robustness.parquet</code>）。</p>")

    # 子样本稳健性：冲突段 vs 其余（SC 上 C8 / N4 / X7）
    panel = pd.read_parquet(D / "panel_SC.parquet",
                            columns=["trade_date", "fwd_15", "C8", "N4",
                                     "X7"])
    hot = (panel["trade_date"] >= "2026-02-01") & \
        (panel["trade_date"] <= "2026-03-31")
    sub_rows = []
    for sg in ("C8", "N4", "X7"):
        row = {"sig": sg}
        for name, mask in (("冲突段 2-3 月", hot), ("其余月份", ~hot)):
            d = panel[mask]
            ok = (d[sg] != 0) & d[sg].notna() & d["fwd_15"].notna()
            row[name] = (float(d.loc[ok, sg].rank()
                               .corr(d.loc[ok, "fwd_15"].rank()))
                         if ok.sum() > 500 else np.nan)
        sub_rows.append(row)
    sub_html_rows = [
        f"<tr><td>{r['sig']}</td>"
        f"<td>{r['冲突段 2-3 月']:+.4f}</td><td>{r['其余月份']:+.4f}</td></tr>"
        for r in sub_rows if np.isfinite(r.get("冲突段 2-3 月", np.nan))
    ]
    subsample_html = (
        "<h3>4.5b 子样本稳健性：美伊冲突段 vs 其余月份（SC，15m RankIC）</h3>"
        "<p>样本期由单一高强度事件段主导（§4.7），必须检验结论是否只来自"
        "该段：</p>"
        + wrap("<tr><th>信号</th><th>冲突段（2026-02 至 03）</th>"
               "<th>其余月份（01、04-07）</th></tr>", sub_html_rows)
        + "<div class='texnote'>C8 在两个子样本中同为正——「未兑现缺口」"
          "结构不只由冲突段驱动；幅度差异如实呈现，事件密度低的月份估计"
          "噪声更大。</div>")

    note = (
        "<div class='callout'><b>读法与量级预期。</b>分钟级事件信号 RankIC "
        "量级 0.01-0.05 属正常水平（此类对比仅作数量级直觉：与日频横截面"
        "因子的口径不同，不可直接比较）。解读纪律与检验总量核算见 §6，"
        "本节不依据任何单一格结果作结论。</div>")
    return ("<h3 id='s43'>4.3　IC / RankIC / ICIR 与衰减曲线</h3>"
            "<p>定义：对每个信号与视界 k，取信号非零的分钟，计算信号值与 k "
            "分钟前向收益的 Pearson IC 与 Spearman RankIC；ICIR = 逐交易日 "
            "Pearson IC 序列的 mean / std——日内 ≥ 10 个有效分钟才计入该日，"
            "有效日数 ≥ 20 才报告 ICIR（表中空缺即此情形，多为稀疏触发"
            "信号）。</p>"
            + "".join(figs) + focus_html + "".join(parts) + note
            + sess_html + lab_html + subsample_html)


def event_section(meta: dict) -> str:
    ev = pd.read_parquet(D / "event_study.parquet")
    tables = []
    for prod in PRODUCT_THEMES:
        sub = ev[ev["product"] == prod]
        if sub.empty:
            continue
        base = meta[prod]["baseline_p_move"]
        bret = meta[prod]["baseline_ret_bp"]
        head = ("<tr><th>事件</th><th>n</th><th>月均</th>"
                + "".join(f"<th>提升 {k}m</th>" for k in HORIZONS)
                + "<th>15m 签名收益 bp</th>"
                + "<th>直到下一信号 bp（中位间隔）</th></tr>")
        rows = []
        for r in sub.itertuples(index=False):
            cells = ""
            for k in HORIZONS:
                lf = getattr(r, f"lift_{k}", np.nan)
                cells += (f"<td>{lf:.2f}</td>" if np.isfinite(lf)
                          else "<td>—</td>")
            rb15 = getattr(r, "ret_bp_15", np.nan)
            rb15s = f"{rb15:+.1f}" if np.isfinite(rb15) else "—"
            nx = getattr(r, "ret_bp_next", np.nan)
            sp = getattr(r, "med_span_min", np.nan)
            nxs = f"{nx:+.1f}（{sp:.0f}min）" if np.isfinite(nx) else "—"
            rows.append(
                f"<tr><td>{r.event}</td><td>{r.n}</td>"
                f"<td>{r.per_month:.0f}</td>{cells}<td>{rb15s}</td>"
                f"<td>{nxs}</td></tr>")
        base_str = "，".join(f"{k}m: {base[str(k)]:.0%}"
                             for k in (1, 5, 15))
        tables.append(
            f"<h4>{prod}（基线一收益 @15m = {bret['15']:.2f} bp；基线二占比："
            f"{base_str}）</h4>" + wrap(head, rows))

    reading = """
<div class="callout"><b>读数（修正基线后的诚实结果）。</b>
（一）<b>基础事件本身几乎没有即时方向收益</b>：E_up / E_dn 的 15 分钟签名
收益接近零甚至为负，短视界「提升」普遍低于 1（事件后价格反而比无条件基线
更不动）——一个机制假说是事件分钟多落在 PM 活跃、国内相对清淡的时段。
方向信息不在「事件发生」这一离散事实本身，而在「事件发生了、期货价格还
没跟上」的<b>连续缺口幅度</b>里（C8 复合口径，PM 腿增量未过确认门槛，§4.3 与主报告 §12.15）——两节互为印证。
（二）E_big（美元量爆发）是唯一提升稳定大于 1 的事件（各品种 1m 提升
1.1-1.9）：大额资金进场处波动确实更高，但其签名收益同样不稳定，「有波动、
无方向」。（三）收益基线（评审要求口径）：品种平均分钟收益 × k 在 15 分钟
上仅约 0.1 bp（表头逐品种给出），相对事件收益可忽略——本库期货数据自
2026-01 起，无法按「过去一年」计算，用样本期均值替代并如实声明。</div>"""

    hist_html = ""
    hp = D / "history_monthly.parquet"
    if hp.exists():
        h = pd.read_parquet(hp)
        piv = (h.pivot_table(index="month", columns="theme",
                             values="n_events", aggfunc="sum").fillna(0.0))
        fig, ax = plt.subplots(figsize=(9.0, 3.2))
        x = range(len(piv.index))
        bottom = np.zeros(len(piv))
        colors = [P[c] for c in ("blue", "aqua", "yellow", "green", "violet",
                                 "red", "magenta", "orange")]
        for i, c in enumerate(piv.columns):
            ax.bar(x, piv[c].to_numpy(), bottom=bottom, label=c,
                   color=colors[i % len(colors)], width=0.85)
            bottom += piv[c].to_numpy()
        step = max(len(piv) // 12, 1)
        ax.set_xticks(list(x)[::step], list(piv.index)[::step],
                      rotation=45, fontsize=6.5)
        ax.legend(fontsize=6, ncol=4, frameon=False)
        ax.set_ylabel("|Δlogit|>0.10 分钟事件数 / 月")
        pub_style.soft_grid(ax)
        fig.tight_layout()
        mon_rows = []
        for theme, g in h.groupby("theme"):
            months = g["month"].nunique()
            first = g["month"].min()
            total = int(g["n_events"].sum())
            peak_mkts = int(g["n_markets"].max())
            mon_rows.append(
                f"<tr><td>{THEME_CN.get(theme, theme)}</td>"
                f"<td>{first}</td><td>{months}</td>"
                f"<td>{total / max(h['month'].nunique(), 1):.0f}</td>"
                f"<td>{total / max(months, 1):.0f}</td>"
                f"<td>{peak_mkts}</td></tr>")
        hist_html = (
            "<h3>4.7 全历史逐月频率（2023-09 至 2026-07）</h3>"
            + figure(fig, "规则匹配的全历史主题市场（1,519 个市场，其中在"
                          "各自月份产生过成交的进入统计）每月 |Δlogit| > "
                          "0.10 的分钟事件数（统一阈值——与评估用的滚动 κ "
                          "阈值不同口径，前者便于跨年比较，后者自适应）。"
                          "频率高度事件驱动：2026 年 1-4 月为高峰平台"
                          "（单月峰值 2026-03 约 7.5 万次），主要由美伊冲突"
                          "相关市场驱动；匹配市场自 2023-09 起才出现。")
            + wrap("<tr><th>主题</th><th>首个活跃月</th><th>活跃月数</th>"
                   "<th>全窗月均事件</th><th>活跃月均事件</th>"
                   "<th>单月峰值活跃市场数</th></tr>", mon_rows)
            + "<div class='texnote'>「全窗月均」分母为数据存在的全部月份，"
              "「活跃月均」分母为该主题有市场的月份——金银 / 油价阈值等"
              "族在 2025 年末才批量出现，两个口径差异大，答辩引用时须"
              "注明。单月峰值活跃市场数为该主题单月去重市场数的最大值"
              "（各主题全历史去重合计 = 1,519）。</div>")
    return ("<h3 id='s46'>4.6 离散信号事件研究（双基线）</h3>"
            "<p>评审要求口径：事件后 1、2、3、5、10、15 分钟及「直到下一次"
            "信号」的价格变化，统计变动比例超过千分之一的占比。<b>基线一"
            "（要求口径）</b>：品种平均分钟收益按时间缩放到 k 分钟"
            "（本库覆盖自 2026-01，无法回溯一年，用样本期均值声明替代）；"
            "<b>基线二（占比基线）</b>：该品种全部具有有效 k 分钟前向收益的"
            "交易分钟中 |Δp| &gt; 0.1% 的无条件占比，「提升」= 事件后占比 / "
            "基线二。签名收益按事件方向符号化；E_big / E_burst 无先验方向，"
            "取当刻 N1 符号。全部视界的收益与占比明细在 "
            "event_study.parquet。</p>"
            + "".join(tables) + reading + hist_html)


def combo_section() -> str:
    ic = pd.read_parquet(D / "ic_table.parquet")
    sub = ic[(ic["scope"] == "all") & (ic["signal"].isin(X_SIGNALS))]
    piv = sub.pivot_table(index=["signal", "product"], columns="horizon",
                          values="rank_ic")
    st = pd.read_parquet(D / "signal_stats.parquet")
    freq = st[st["signal"].isin(X_SIGNALS)].set_index(["signal", "product"])
    head = ("<tr><th>组合</th><th>品种</th><th>日均触发</th>"
            + "".join(f"<th>{k}m</th>" for k in HORIZONS) + "</tr>")
    rows = []
    for sig in X_SIGNALS:
        for prod in PRODUCT_THEMES:
            try:
                f = float(freq.loc[(sig, prod), "per_day"])
            except KeyError:
                f = 0.0
            in_ic = (sig, prod) in piv.index
            if not in_ic:
                note = ("结构性不适用" if (prod == "M" and sig in
                        ("X1", "X2")) else "触发不足")
                rows.append(
                    f"<tr class='die'><td>{sig}</td><td>{prod}</td>"
                    f"<td>{f:.0f}</td><td colspan='6'>—（{note}，n &lt; "
                    f"200）</td></tr>")
                continue
            r = piv.loc[(sig, prod)]
            cells = "".join(
                (f"<td>{r.get(k, np.nan):+.3f}</td>"
                 if np.isfinite(r.get(k, np.nan)) else "<td>—</td>")
                for k in HORIZONS)
            vals = [r.get(k, np.nan) for k in HORIZONS]
            pos = int(np.nansum([v > 0 for v in vals]))
            n_fin = int(np.sum([np.isfinite(v) for v in vals]))
            cls = ""
            if n_fin >= 4:
                if pos >= n_fin - 1:
                    cls = ' class="live"'
                elif pos <= 1:
                    cls = ' class="die"'
            rows.append(f"<tr{cls}><td>{sig}</td><td>{prod}</td>"
                        f"<td>{f:.0f}</td>{cells}</tr>")
    return f"""
<h2 id="s5">5　组合信号结果：完整呈现正负结果</h2>
<p>X 族全部 10 个组合 × 5 品种的 50 行完整列出（无法计算的行以「—」占位并
注明原因；M 只映射单一主题，共振类 X1 / X2 结构性不适用）。绿行 = 几乎
全部视界为正，灰行 = 几乎全部为负或不可计算。<b>不挑好看的报</b>：</p>
{wrap(head, rows)}
<div class="callout"><b>读数（修正口径后，以跨品种一致性为准）。</b>
（一）<b>X 族的主要结果是 negative：离散化组合整体弱于连续值结构。</b>
修正基线与权重后，多数组合跨品种为负或混杂——与 §4.3 中连续值 C8 的稳健
形成对照：把「缺口」离散成「触发 / 不触发」丢掉了幅度信息，样本又薄。
（二）<b>确认型 X1 跨品种一致为负</b>（SC / AU 六视界全负）——这本身是
有信息的结构：三方同向的触发点在期货已反应之后，随后偏向回吐；「追确认」
在分钟尺度是反向指标。（三）「未动」类的离散版（X2 / X9）只在个别品种
为正（X2 于 SC 短视界、X9 于 M），远弱于其连续版 C8——同一假说，连续
表达稳健、离散表达脆弱。（四）时段条件 X7 在 SC 六视界中五个为正（与 C8
主线品种一致），在 AU / CU 为负；X3（事件 × 高波动）在 AU 长视界显著为负、
在 M 为正但日均仅约 2 次触发——均不满足跨品种一致，不作结论。</div>
<p>方法论意义：组合实验的价值不在找到「更强的信号」，而在<b>定位信息的
载体</b>——本轮 40 信号 × 5 品种的完整矩阵指向同一个答案：信息在连续的
未兑现缺口幅度（C8）里，不在离散事件及其组合里。这与主报告的窗口级结论
（闭市吸收强、事后无剩余）一致。</p>
"""


def jump_section() -> str:
    """§5b：J 族跳变因子（套利钉住恒等式后，水平跳变的信息含量）。"""
    jdir = D.parent / "jump"
    ic_p = jdir / "ic.parquet"
    if not ic_p.exists():
        return ""
    ic = pd.read_parquet(ic_p)
    meta = json.loads((jdir / "meta.json").read_text())
    ev = pd.read_parquet(jdir / "event_groups.parquet")

    defs = [
        ("J1", "跳强度", "过去 120 分钟映射主题的跳数", "信息到达的频率"),
        ("J2", "带方向跳幅和", "Σ orientation×Δℓ（120 分钟内的跳）",
         "信念修正的净方向"),
        ("J3", "跳能量", "Σ J²（240 分钟）", "信念变化中跳的贡献强度"),
        ("J4", "前导流加权跳", "Σ J×preflow（跳桶前 60 分钟带方向净流比率）",
         "知情流通道：有人先动手的跳"),
        ("J5", "孤立跳幅和", "Σ J×1{同桶无同主题共跳}",
         "私有信息候选：只有一个市场动"),
        ("J6", "同步跳幅和", "Σ J×1{同桶 ≥1 共跳}",
         "公共新闻通道：全族同时动"),
        ("J7", "跳方向偏度", "(升级跳数−降级跳数)/总跳数（240 分钟）",
         "跳的不对称"),
    ]
    drows = [f"<tr><td><b>{a}</b> {b}</td><td><code>{c}</code></td>"
             f"<td>{d}</td></tr>" for a, b, c, d in defs]
    dtab = wrap("<tr><th>因子</th><th>定义</th><th>动机</th></tr>", drows)

    surv = ic[ic["q_bh"] < 0.1].copy()
    surv = surv.reindex(surv["rank_ic"].abs()
                        .sort_values(ascending=False).index)
    consistent = []
    for (product, sig), g in ic.groupby(["product", "signal"]):
        if len(g) >= 5 and (np.sign(g["rank_ic"]) == np.sign(
                g["rank_ic"].iloc[0])).all() and g["rank_ic"].abs().min() > 0:
            both = g.dropna(subset=["rank_ic_hot", "rank_ic_cold"])
            hot_ok = bool(len(both) > 0 and (
                np.sign(both["rank_ic_hot"])
                == np.sign(both["rank_ic_cold"])).all())
            consistent.append((product, sig, float(g["rank_ic"].mean()),
                               hot_ok))

    def _f2(x: float, nd: int = 2) -> str:
        return "—" if pd.isna(x) else f"{x:+.{nd}f}"

    rows = []
    for _, r in surv.head(14).iterrows():
        if pd.isna(r["rank_ic_hot"]) or pd.isna(r["rank_ic_cold"]):
            hc = (f"{_f2(r['rank_ic_hot'], 3)} / {_f2(r['rank_ic_cold'], 3)}"
                  "（单侧样本不足）")
        else:
            same = np.sign(r["rank_ic_hot"]) == np.sign(r["rank_ic_cold"])
            hc = (f"{r['rank_ic_hot']:+.3f} / {r['rank_ic_cold']:+.3f}"
                  f"{'' if same else '<b>（反号）</b>'}")
        rows.append(
            f"<tr><td><b>{r['signal']}</b>×{r['product']}</td>"
            f"<td>{int(r['h'])}′</td><td>{r['n']:,}</td>"
            f"<td>{r['rank_ic']:+.4f}</td><td>{_f2(r['icir'])}</td>"
            f"<td>{hc}</td>"
            f"<td>{r['q_bh']:.4f}</td></tr>")
    stab = wrap("<tr><th>因子×品种</th><th>视界</th><th>n</th>"
                "<th>RankIC</th><th>ICIR</th><th>冲突期内/外</th>"
                "<th>q(BH)</th></tr>", rows)

    ev5 = ev[ev["h"] == 15]
    erows = [
        f"<tr><td>{r['group']}</td><td>{int(r['n'])}</td>"
        f"<td>{r['signed_bp']:+.2f}</td><td>{r['share_pos']:.0%}</td></tr>"
        for _, r in ev5.iterrows() if pd.notna(r["signed_bp"])
    ]
    etab = wrap("<tr><th>跳分组（SC，事后属性）</th><th>n</th>"
                "<th>15 分钟符号化响应(bp)</th><th>正占比</th></tr>", erows)

    strong = [(p, s, m, ok) for p, s, m, ok in consistent
              if abs(m) >= 0.01]
    strong.sort(key=lambda x: -abs(x[2]))
    cons_txt = "、".join(f"{s}×{p}（均值 {m:+.3f}"
                        f"{'，冲突期内外同号' if ok else ''}）"
                        for p, s, m, ok in strong[:8]) or "无"
    share_iso = meta["attr_stats"]["share_isolated"]
    return f"""
<h2 id="s5b">5b　J 族：跳变因子（审计后新增）</h2>
<p class="warn"><b>第五轮外部审计降格（先读）</b>：本节 IC 表把滚动
保留 120/240 分钟的因子按全部分钟行数推断（如 J5 × M 15 分钟格名义
n = 18,163，背后独立事件仅数十次、集中在数十个交易日），前向收益
窗口又高度重叠，表中 q 值<b>不作为确认证据</b>；SC 盘中日盘跳的
+9.8 bp 经事件桶折叠 + 交易日聚类后为 +6.9 bp（HAC t = 1.68）；
正确推断已执行：HAC 一致的 wild cluster bootstrap p = 0.17 不支持，
严格枚举的日期块置换 p &lt; 0.008 支持，两法分歧，仅为<b>置换单方法
支持的探索候选</b>，且毛响应低于两倍成本门槛；时段分类原版未用真实
期货交易日历（约 58% 精确映射），已按真实分钟网格修正。详见主报告
§12.14-12.15。</p>
<p>动机来自一个机制观察（正文 §2 与附录 A.3）：套利与撮合器内建的
mint / merge 把 Yes + No 恒等式钉住，但对价格<b>水平</b>没有约束力，
因此水平的跳变只剩两种解释——公共信息到达（信念突变）或私有信息入场
（知情流）。跳是解释上最干净的对象，值得单独成族。跳检测沿用 E1 的
注册口径（不引入新探测器自由度），共检出 <b>{meta['n_jumps']:,}</b>
个跳（{meta['n_markets']} 个市场；孤立跳占比 {share_iso:.0%}）；每个
跳带一个属性向量，其中前导流与同步性在跳发生时刻即可知（进实时因子），
持续性与结算临近度用到事后信息（只进分组事件研究）：</p>
{dtab}
<p>归一化与检验协议与 N/C 族完全一致（滚动 z、逐日 IC 的 ICIR、
{meta['family_cells']} 个格子作为一族做 BH-FDR 并计入检验总账）。
族内 FDR 存活（q&lt;0.1）的前 14 行：</p>
{stab}
<p>六视界符号全部一致的（因子×品种）组合：{cons_txt}。</p>
<p>按事后属性分组的跳后 SC 响应（双基线口径，基线为无条件
|fwd|均值）：</p>
{etab}
<p><b>读法纪律</b>：J 族是在 C8 之后新增的第二个结构族，其全部格子
已并入检验总账；单格 q 值不作结论，按"六视界同号 + 冲突期内外同号 +
秩线一致"三条纪律筛选后余下的组合才进入候选；分组事件研究中
"持续跳 / 回吐跳"与"临近结算"用了事后信息，只用于机制归因，不构成
可交易声称。</p>
{_jump_stage2()}
"""


def _fig_file(name: str, caption: str) -> str:
    """磁盘 PNG 的 base64 内嵌（与主报告 fig_tag 同款）。"""
    fp = ROOT / "docs" / "figures" / "v3" / name
    if not fp.exists():
        return ""
    b = base64.b64encode(fp.read_bytes()).decode()
    return (f'<figure><div class="figcard">'
            f'<img src="data:image/png;base64,{b}" alt=""></div>'
            f"<figcaption>{caption}</figcaption></figure>")


def _jump_stage2() -> str:
    """§5b.1-5b.2：时段剖面、同质化归并与跨族复合（第二阶段）。"""
    jdir = D.parent / "jump"
    sp = jdir / "session_response.parquet"
    if not sp.exists():
        return ""
    sess = pd.read_parquet(sp)
    comp = pd.read_parquet(jdir / "composite_ic.parquet")
    corr = pd.read_parquet(jdir / "factor_corr.parquet")
    meta2 = json.loads((jdir / "meta_integration.json").read_text())

    order = ["盘中·日盘", "盘中·夜盘", "闭市·傍晚", "闭市·凌晨",
             "闭市·周末"]
    rows = []
    for product in ("SC", "AU", "M"):
        d = sess[(sess["product"] == product) & (sess["h"] == 15)]
        d = d.set_index("session").reindex(
            [s for s in order if s in set(d["session"])])
        first = True
        for sname, r in d.iterrows():
            pc = (f'<td rowspan="{len(d)}"><b>{product}</b></td>'
                  if first else "")
            first = False
            sig = (r["lo"] > 0) or (r["hi"] < 0)
            star = " <b>✦</b>" if sig else ""
            rows.append(
                f"<tr>{pc}<td>{sname}{star}</td><td>{int(r['n'])}</td>"
                f"<td>{r['signed_bp']:+.2f}</td>"
                f"<td>[{r['lo']:+.2f}, {r['hi']:+.2f}]</td>"
                f"<td>{r['share_pos']:.0%}</td></tr>")
    stab = wrap("<tr><th>品种</th><th>跳所在时段</th><th>n</th>"
                "<th>15′ 符号化响应(bp)</th><th>90% 自助（bootstrap）CI</th>"
                "<th>正占比</th></tr>", rows)

    cm = corr[corr["product"] == "M"].set_index("row")
    j2n1 = float(cm.loc["J2", "N1"])
    j2c8 = float(cm.loc["J2", "C8"])
    xf_m = comp[(comp["product"] == "M") & (comp["h"] == 15)] \
        .set_index("signal")["rank_ic"]
    xf_sc = comp[(comp["product"] == "SC") & (comp["h"] == 15)] \
        .set_index("signal")["rank_ic"]

    return f"""
<h3>5b.1 时段剖面：跳发生在盘中还是闭市，决定信息的变现路径</h3>
<p>每个跳按品种交易时段分类（盘中日盘 / 盘中夜盘 / 闭市傍晚 / 闭市
凌晨 / 闭市周末；夜盘收盘档位逐品种取实际值）。盘中跳的响应从下一
分钟起测，闭市跳的响应从下一开盘分钟起测——后者度量的是"开盘跳空
吸收之后还剩多少"：</p>
{_fig_file("f_jump_panorama.png",
           "J 族全景：(a) 逐月跳数按主题堆叠——1-4 月冲突高峰主导，"
           "5 月后金属与美联储主题接棒；(b) 跳幅重尾分布；(c) 跳的"
           "日内时刻分布（对 SC 时段着色）——跳集中于美东活跃时段，"
           "恰落在国内闭市与夜盘；(d) SC 映射跳的时段构成。")}
{stab}
<p>（✦ = 90% 自助 CI 不含零；<b>勘误</b>：该 CI 在跳层面重抽、未按
交易日聚类，日聚类后本表唯一显著格降为 t = 1.68、HAC 一致 wild
p = 0.17，仅日期块置换支持（p &lt; 0.008），解读以主报告 §12.15 为准。）
三个结构按原口径可读：（i）<b>SC 盘中·
日盘跳 +9.75bp [3.7, 16.2]</b>——日盘进行中的 PM 跳来不及被跳空
吸收，留下显著的分钟级正残余，是 J 族唯一 CI 不含零的正时段格；
（ii）SC 的闭市跳（凌晨 / 周末）响应贴近零——闭市积累的跳在开盘
跳空中定价殆尽，与第一部分"吸收发生在开盘瞬间"在跳级精确互证；
（iii）AU 的盘中跳显著为<b>负</b>（夜盘 −2.31、日盘 −3.93，CI 均
不含零）——金属主题以价格阈值类市场为主，跳是对期货已实现行情的
记账确认（附录 §9 反向传导），确认之后期货小幅回吐。</p>
{_fig_file("f_jump_session.png",
           "SC 与 M 的跳后 15 分钟符号化响应（按时段，自助 CI）。")}

<h3>5b.2 同质化归并与跨族复合：整合的正确形态</h3>
<p><b>事件级归并</b>：同主题同桶共跳是同一事件的期限结构（折叠比
{meta2['collapse_ratio']:.2f}，即平均每个事件 {meta2['collapse_ratio']:.1f}
个市场同跳；多市场事件占 {meta2['multi_market_share']:.0%}）。把
(主题, 桶) 折叠为事件级跳后重建因子，IC 与市场级几乎完全一致——
J 族结果不依赖重复计数的膨胀，归并作为稳健性检验通过。</p>
<p><b>因子相关结构</b>：J 族与 N/C 族在分钟粒度<b>接近正交</b>
（M 面板 J2 与 N1 相关 {j2n1:+.3f}、与 C8 相关 {j2c8:+.2f}）——
跳是稀疏事件、N1 是连续流，两者携带不同的信息维度；市场层面的
同质化（共跳、期限结构抱团）在因子层面已被聚合与归一化吸收：</p>
{_fig_file("f_jump_corr.png",
           "SC 与 M 面板的因子相关矩阵（N/C/J 代表信号）。跨族相关"
           "普遍低；族内（如 J1 与 J2、J5）相关高——复合应跨族做、"
           "族内择一。")}
<p><b>跨族等权复合 XF = mean(z(C8), z(N4), z(J2))</b>（成分为三族
既有头部、等权、无样本内拟合）：15 分钟 RankIC 在 M 上为
{xf_m.get('XF', float('nan')):+.3f}（最强单因子 J2
{xf_m.get('J2', float('nan')):+.3f}），SC 上为
{xf_sc.get('XF', float('nan')):+.3f}（最强单因子 C8
{xf_sc.get('C8', float('nan')):+.3f}）——<b>等权复合在两个品种上都
不敌各自的头部单因子</b>，因为最优结构因品种而异（SC 的信息在缺口
结构、M 的信息在跳）。诚实结论：整合的正确形态不是等权平均，而是
按品种选择结构；而"选哪个"若在样本内做就是过拟合，须交给第二部分的
ex-ante 门控 OOS 框架（§18.1 待办 (d) 扩充为含 J 族的结构选择）。</p>
{_fig_file("f_jump_composite.png",
           "左：M 上复合 XF 与成分的 IC 视界曲线——XF 被弱成分稀释，"
           "不及 J2；右：M 上事件级折叠（JE2）与市场级（J2）几乎重合"
           "——归并不改变结论。")}
"""


def honest_section() -> str:
    return """
<h2 id="s6">6　多重检验与诚实结论</h2>
<p><b>检验总量核算</b>：40 信号 × 6 视界 × 5 品种 = 1,200 个设计格子（实际
可计算 1,128，其余触发不足）；加上日盘 / 夜盘分层（ic_table 共 3,254 行）
与标签对照（752 个相关），全文约 4,000 个相关系数。5% 名义水平下期望约
200 个偶然「显著」。因此本文档不以任何单格显著性作结论，只认三类证据：
（i）同一信号跨视界符号一致；（ii）跨品种符号一致；（iii）|ICIR| 持续。</p>
<p><b>三类证据相互不独立，权重有序</b>：六个视界的前向收益相互嵌套（1 分钟
收益是 15 分钟收益的一部分），跨视界一致性偏乐观；SC 与 AU 共享同一事件流
（中东主题映射两者），其跨品种一致不是独立复验——跨品种证据以事件流不
重叠的品种对为准（如 SC 对 M）；|ICIR|（逐日独立估计的稳定性）最接近独立
复验，权重最高。C8 同时满足三类且含 SC-M 对，是唯一按此纪律站住的
结构；但第五轮外部审计的双腿分解表明该结构的强度主要由期货自身反转腿
承载、Polymarket 腿单独在日聚类下不显著（§0 发现二、主报告 §12.14），
故"站住"的是复合信号，不是 Polymarket 的分钟级增量。</p>
<p><b>与主报告的关系</b>：本篇是测量与单信号层——回答「数据处理是否扎实、
信号有没有信息含量、集中在什么结构」。「能否构成可交易 alpha、控制境外
市场后是否仍有增量」由主报告（v2.1）的识别与功效框架回答；那边的结论
（同期吸收强、次日方向样本外（out-of-sample, OOS）增量整体不为正、
影响系数当前不可识别）不因本篇
的分钟级正 IC 而改变——分钟级 IC 反映「事件信息在分钟尺度渗入价格的
过程」，与「隔日方向可预测」是两个命题，拼起来恰是完整故事：信息很快被
吃掉，所以留不到明天。</p>
<p><b>本篇可辩护的结论</b>：（一）数据层：8.55 亿行链上逐笔成交记录的行级定义、
主动方向、概率换算、时差映射与 22 条边界情形全部显式处理或显式声明；
（二）信号层：40 个信号全部有公式、归一化登记、频率 / 分布 / 缺失率 /
极值率；（三）检验层：唯一跨品种稳定的结构是 C8 复合信号（含期货反转腿；
PM 腿经严格增量检验未过确认门槛，不作为 Polymarket 分钟增量，主报告
§12.14-12.15），量级 0.01-0.03；时段与标签
稳健性中性偏稳；（四）组合层：离散化组合整体弱于连续值结构，确认型组合
跨品种一致为负——negative 结果完整保留。</p>
"""


# ================================================================ 附录
def appendix_a() -> str:
    arb_rows = [
        ("10:09:58", "NO", "0.974-0.979", "0.021-0.026", ""),
        ("<b>10:29:14</b>", "<b>NO（§1.1 示例大单）</b>",
         "<b>0.974 → 0.973</b>", "<b>推到 0.026-0.027</b>", "卖单吃穿两档"),
        ("10:29:17", "YES", "0.021", "0.021", "3 秒后：旧挂单还没跟上"),
        ("10:32:37", "YES", "0.028", "0.028", "3 分钟后：开始对齐"),
        ("10:42:11", "YES + NO", "0.028 / 0.972", "1.000 / 0.999",
         "13 分钟后：两侧重新互补"),
    ]
    return f"""
<h2 id="appA">附录 A　Polymarket 定价机制与市场结构</h2>

<h3>A.1 与股票 / 期货的对照</h3>
{wrap_text('<tr><th>维度</th><th>股票（A 股）</th><th>商品期货</th>'
      '<th>Polymarket</th></tr>', [
    "<tr><td>标的</td><td>公司权益</td><td>商品合约</td>"
    "<td><b>一个事件的结果</b>（二元）</td></tr>",
    "<tr><td>价格含义</td><td>贴现现金流</td><td>现货预期 + 持有成本</td>"
    "<td><b>事件发生概率</b>（0-1 美元）</td></tr>",
    "<tr><td>交易时间</td><td>4 小时/日</td><td>日盘 + 夜盘</td>"
    "<td><b>7×24 不间断</b></td></tr>",
    "<tr><td>涨跌停</td><td>±10%</td><td>各品种不同</td>"
    "<td>无；价格天然限于 (0,1)</td></tr>",
    "<tr><td>做空</td><td>融券（受限）</td><td>卖开仓</td>"
    "<td><b>买 No 代币即做空事件</b></td></tr>",
    "<tr><td>杠杆 / 保证金</td><td>融资融券</td><td>保证金约 10%</td>"
    "<td><b>无杠杆，全额 USDC 抵押</b>，无强平</td></tr>",
    "<tr><td>结算</td><td>—</td><td>交割 / 现金</td>"
    "<td>事件判定：押对方每份兑 1 USDC，押错归零</td></tr>",
    "<tr><td>T+N</td><td>T+1</td><td>T+0</td><td>T+0，链上即时清算</td></tr>",
    "<tr><td>撮合</td><td>交易所 CLOB</td><td>交易所 CLOB</td>"
    "<td>链下 CLOB 撮合 + <b>链上清算</b>（成交不可篡改）</td></tr>",
    "<tr><td>费率</td><td>佣金 + 印花税</td><td>手续费</td>"
    "<td>基础费率多为 0（taker 费在部分市场）</td></tr>",
])}

<p><b>实例</b>：以美联储议息决议市场为例（
<a href="https://polymarket.com/event/fed-decision-in-september-762"
target="_blank" rel="noopener">polymarket.com/event/fed-decision-in-september-762</a>，
截图日期 2026-07-18，行情随时变动，仅作结构示意）。「9 月美联储决议」是
一个<b>事件（event）</b>，其下拆成五个互斥结果各自独立的二元市场：不变息、
降 25bp、降 50bp 以上、加 25bp、加 50bp 以上，这正是 A.1 表中「标的 =
一个事件的结果」的具体样子，而非单一价格标的：</p>
{_fig_file("f_polymarket_ui_fed_chart.png",
     "「Fed Decision in September?」事件页：五个结果各自的 Yes 价格随时间"
     "演化即隐含概率的时间序列（示例区间「不变」价格约 59%、「降 25bp」"
     "约 37%，此消彼长）；右侧为下单面板（Buy Yes / Buy No，限价单）。")}
<p>展开逐结果列表更直观地看到「价格即概率」与近似互补：</p>
{_fig_file("f_polymarket_ui_fed_outcomes.png",
     "五个互斥结果的 Buy Yes / Buy No 报价（美分）。「不变」Yes 59¢ 意味着"
     "市场认为不变息概率约 59%；同一结果 Yes 价 + No 价接近 100¢（如「降"
     "50bp 以上」2.2¢ + 97.9¢），价差主要来自买卖盘口点差，而非跨结果的"
     "Yes 价直接求和：五个结果的 Yes 价加总本身即市场对「本次会议究竟落在"
     "哪一档」的完整概率分布（约 2%+3%+59%+37%+1%≈102%，超出 100% 的部分"
     "是做市商价差留出的套利空间，§A.3 铸造 / 合并机制会把明显偏离拉回）。")}

<h3>A.2 价格形成与价格即概率</h3>
<p>与期货同构：任意时刻存在买卖挂单阶梯（限价单簿），<b>maker</b> 挂单
提供流动性、<b>taker</b> 吃单扫过盘口，吃单量超过最优价档位深度时按序
路由到下一档更差的价格。§1.1 的真实订单即此机制的样本：卖 7,190 份，
0.974 一档只有 2,223 份买盘，剩余 4,967 份被迫吃到 0.973。与期货的差异
是挂单为链下签名消息（免 gas、可随时撤），成交清算上链。</p>
<p>一份 Yes 代币的结算支付为事件发生兑 1 USDC、不发生兑 0（合约硬编码）。
若事件概率为 p，其期望价值即为 p，价格偏离 p 的一侧下注方期望亏损，双向
下注把价格推向 p，因此<b>价格直接可读为概率，无需定价模型换算</b>（对比
期权价格需反解隐含波动率）。已知系统偏差：小概率结果价格偏高、大概率
结果价格偏低（favorite-longshot bias，见附录 B）；本研究以概率的逐分钟
变化（logit 差分）构造信号而非概率水平，多数水平偏差在差分中相互抵消。</p>

<h3>A.3 Yes + No = 1：套利如何维持、联动有多快</h3>
<p>Yes 和 No 是<b>两本独立的订单簿</b>——一笔 No 成交不会自动改动 Yes 的
价格。维持互补的是套利：任何人可用 1 USDC 铸造 1 Yes + 1 No（split），或
反向销毁换回 1 USDC（merge）。若 Yes + No &gt; 1：铸造后双卖；&lt; 1：
双买后销毁。套利需要真人执行，存在分钟级时间差。真实数据（§1.1 同一市场，
2026-06-27）完整记录了一次「大单推动 → 另一侧滞后 → 套利对齐」：</p>
{wrap('<tr><th>时刻（UTC）</th><th>成交侧</th><th>成交价</th>'
      '<th>隐含 p_event</th><th>备注</th></tr>',
      [f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td>"
       f"</tr>" for a, b, c, d, e in arb_rows])}
<div class="texnote">最终 Yes(0.028) + No(0.972) = 1.000。对研究的含义：
分钟聚合时两侧成交都换算到 p_event 后混合，套利时间差表现为分钟内噪声，由
usdc 加权 vwap 与 logit 截断吸收。</div>

<h3>A.4 市场生命周期与参与者</h3>
<p><b>创建与结算</b>：Polymarket 按事件热点创建市场并撰写结算条款（判定
标准、数据源、截止时刻）；同一现实事件常被做成一族市场（不同截止日 / 不同
阈值），这是 §1.6 P11 伪重复问题的来源。结算由 UMA 乐观预言机裁决：任何人
可提交结果并质押保证金，争议期内无人挑战即生效，有挑战则升级投票；裁决
写入链上 <code>ConditionResolution</code> 事件（本文结算时刻的来源，覆盖
455/492 个登记市场）。极少数市场重报，本文取首次事件（P6）。</p>
<p><b>参与者</b>：匿名链上地址，包括做市商程序、方向性投机者与事件知情者；
预测市场不限制知情资金入场，知情流正是价格信息含量的来源。价格可被短时
推离真值，但成本随之上升（推离头寸即是给套利者与知情者的补贴，结算按
真实结果清算）；本文的缓解措施为 10 万美元准入门槛、流动性加权与分钟
vwap 聚合，另见正文的置换检验、安慰剂检验与控制变量族。</p>
"""


def appendix_b() -> str:
    return """
<h2 id="appB">附录 B　数据与方法的已知边界</h2>
<p>价格读作概率存在已知系统偏差：小概率结果价格偏高、大概率结果价格偏低
（favorite-longshot bias，多类预测市场中稳健存在）。本研究以概率的逐分钟
变化（logit 差分）构造信号而非概率水平，多数水平偏差在差分中相互抵消。</p>
<p>链上仅记录成交、不记录历史挂单，订单簿状态不可回溯；点差以同分钟买卖
两向成交的中位价差代理，头部主题市场约 0.001-0.005（0.1-0.5 个百分点
概率）。头部事件市场（中东、俄乌）单市场累计成交可达数千万至上亿美元，
长尾市场很薄，本文以 10 万美元累计成交为准入门槛并做流动性加权处理这一
差异。</p>
<p>数据采集采用自建链上爬虫而非官方 REST API：官方接口不承诺完整历史
逐笔，链上原始日志可独立复算、无速率限制，代价是需自行处理中继腿等
清洗（§1.6）。成交流清洗已剔除中继腿（记账重复）与铸造 / 合并（不代表
方向性观点的机械成交），但未专门识别同一实际主体跨地址的自成交；据此，
量类信号（成交笔数、名义额）解读从严，方向类信号（logit 差分符号）对
该问题天然不敏感（§1.6 P12）。</p>
<p>采用预测市场而非新闻文本作为事件信号来源：新闻缺乏可比的概率读数、
无金钱代价、时间戳口径不统一；预测市场把事件信息转换为带出价的连续
时间序列，这是本研究方法论的前提。</p>
"""


def appendix_c() -> str:
    return f"""
<h2 id="appC">附录 C　字段词典（三层）</h2>
<h3>C.1 链上原始行（爬虫层）</h3>
{wrap_text('<tr><th>字段</th><th>含义</th></tr>', [
    "<tr><td><code>id</code></td><td>链ID_区块号_日志序号（137 = Polygon；"
    "logIndex 为事件在区块全部日志中的序号，同区块其他事件占中间序号故不"
    "连续）。全局唯一 → 幂等去重键</td></tr>",
    "<tr><td><code>maker / taker</code></td><td>挂单方 / 吃单方钱包地址。"
    "中继腿上 taker = 交易所合约地址（is_relay 判据）</td></tr>",
    "<tr><td><code>maker / taker_direction</code></td><td>同笔两视角"
    "（镜像）。由「谁付出哪种资产」推出：资产 id 0 = USDC、非 0 = 结果"
    "代币，maker 付 USDC 即 maker 买——来自资产转移事实，非 Lee-Ready 式"
    "推断</td></tr>",
    "<tr><td><code>price</code></td><td>USDC / 份 ∈ (0,1) = 该代币对应结果"
    "的隐含概率</td></tr>",
    "<tr><td><code>token_amount / usdc_amount</code></td><td>份数与名义额，"
    "usdc = price × token</td></tr>",
    "<tr><td><code>fee_usdc</code></td><td>协议费（taker 侧，常挂中继腿）"
    "</td></tr>",
    "<tr><td><code>token_asset_id</code></td><td>ERC-1155 positionId（由"
    "抵押品、conditionId、结果集合两层 keccak 派生）——关联市场元数据的"
    "外键；同市场 Yes / No 是两个不同 id</td></tr>",
    "<tr><td><code>is_relay / venue_class</code></td><td>中继腿标记 / 场馆"
    "白名单分类</td></tr>",
])}
<h3>C.2 清洗层（daily_aligned 同构）</h3>
{wrap_text('<tr><th>字段</th><th>含义</th></tr>', [
    "<tr><td><code>condition_id</code></td><td>市场唯一标识（一个市场 = 一"
    "个 condition）</td></tr>",
    "<tr><td><code>outcome_seq / outcome_label</code></td><td>1 = Yes（事件"
    "本身）、2 = No；标签为条款文本</td></tr>",
    "<tr><td><code>p_event</code></td><td>事件（Yes）发生的市场隐含概率："
    "seq = 1 取 price，seq = 2 取 1 − price（§1.2）</td></tr>",
    "<tr><td><code>D</code></td><td>该笔主动成交对 p_event 的推动方向 ±1"
    "（四格真值表，§1.2）</td></tr>",
    "<tr><td><code>resolved_at</code></td><td>结算时刻（链上 "
    "ConditionResolution 首次事件）</td></tr>",
    "<tr><td><code>market_slug</code></td><td>市场 URL 名（主题规则匹配的"
    "输入）</td></tr>",
])}
<h3>C.3 研究面板（本篇，逐品种逐分钟）</h3>
{wrap_text('<tr><th>字段</th><th>含义</th></tr>', [
    "<tr><td><code>ts / trade_date / session / seg</code></td><td>K 线收盘戳"
    "（北京）/ 交易日 / 日夜盘 / 连续分钟段号</td></tr>",
    "<tr><td><code>r1 / fwd_k / fwd_vwap_k</code></td><td>1 分钟收益 / k 分"
    "钟前向收益（close 与 VWAP 双标签，段内有效）</td></tr>",
    "<tr><td><code>mom15_z / rv15_z / volu15_z / amt15_z</code></td><td>期货"
    "量价特征（滚动 z）</td></tr>",
    "<tr><td><code>dl_θ / flow_θ / usdc_θ / n_θ / n_mkts_θ</code></td><td>主"
    "题 θ 的 PM 分钟聚合（§2.2）</td></tr>",
    "<tr><td><code>N1..N10 / C1..C10 / K1..K10 / X1..X10</code></td><td>40 "
    "个信号（§3）</td></tr>",
    "<tr><td><code>E_up / E_dn / E_big / kappa</code></td><td>原始离散事件与"
    "稳健阈值</td></tr>",
    "<tr><td><code>locked</code></td><td>high == low 锁死分钟标记（涨跌停"
    "代理监控）</td></tr>",
])}
"""


def appendix_d() -> str:
    rows = [
        ("数据规范：标准字段（OHLCV + vwap + 状态标签）",
         "面板含 OHLCV / money + session / seg / locked + vwap 代理；PM 侧"
         "另有五个主题聚合量", "落实"),
        ("时间轴统一：分钟索引；休市不当连续时间差",
         "连续分钟段规则（§1.5）；所有窗口按 K 线顺序而非墙钟时间", "落实"),
        ("缺失值：不前向填充成交量；缺失打标签",
         "期货零成交 vwap 置 NaN；PM 无成交分钟创新 = 0（语义：无新信息）并"
         "有活跃占比监控", "落实（语义差异已说明）"),
        ("涨跌停：is_limit 标签 + 保守成交假设",
         "主力连续无涨跌停价表 → high == low 代理监控 + 明示局限（F4）",
         "部分落实（数据边界）"),
        ("复权一致性", "期货主力连续无复权；换月按 F3 处理", "不适用 + 替代"),
        ("标签：未来区间 VWAP 优先于单点 close",
         "close 主口径 + VWAP 标签全表对照（§4.5，排序高度一致）",
         "落实（双口径）"),
        ("严禁未来函数",
         "桶右端标签 / shift(-k) / 滚动统计 / PIT 单元测试", "落实"),
        ("切分：严禁随机打散，按时间滚动",
         "本篇为单信号测量（无训练步）；主报告第五层用展开窗口样本外检验",
         "落实"),
        ("因子：显式公式 + 参数 + 元信息登记",
         "40 信号全部有公式卡片 + 元信息总表（§3.0）", "落实"),
        ("因子标准化：横截面稳健归一（median-MAD）",
         "单品种时序场景 → 滚动 z + MAD 稳健阈值（§2.1 说明对应关系）",
         "落实（时序对应物）"),
        ("八大类因子结构",
         "按 N / C / K / X 四族组织；与老师八类中的量价 / 波动 / 流动性 / "
         "资金流 / 组合残差类一一对应，另扩展事件类", "结构对应"),
        ("组合型 / 残差型因子",
         "C7 滚动残差 + X 族 10 个条件组合 + 主报告 LOFO 正交化", "落实"),
        ("评价指标：IC / RankIC / ICIR / 衰减 / 缺失率 / 极值率",
         "全部输出（§4.1 / 4.3）；衰减 = 六视界曲线", "落实"),
        ("分层收益 / 多空组合 / 换手 / 成本后收益",
         "本篇为信号测量层、不含组合构建；成本后回测见主报告 §12.6",
         "由主报告承接"),
        ("分状态表现（时段等）",
         "日盘 / 夜盘分层 IC（§4.4）+ X7 / X8 时段条件信号", "落实"),
        ("不依赖极少数时段 / 样本",
         "事件研究报告 n 与月频；主报告对候选品种做影响点删除检验", "落实"),
        ("数据质量检查清单（12 条）", "§1.7 逐条执行", "落实"),
        ("过拟合治理 / 多重检验",
         "§6 检验总量核算 + 不以单格作结论的纪律 + 主报告 BH-FDR / 统计功效门槛",
         "落实"),
        ("工程：模块解耦、可复现、可追溯",
         "build（数据 + 信号 + 检验）与 report（渲染）分离；产物全 parquet "
         "可复算；行级可追溯（§1.7）", "落实"),
        ("可交易性 / T+1 / 卖出框架",
         "期货 T+0 无此约束；可交易性仅做涨跌停代理监控；执行层超出本篇范围",
         "范围外（已声明）"),
    ]
    body = [
        f"<tr><td>{a}</td><td>{b}</td><td><b>{c}</b></td></tr>"
        for a, b, c in rows
    ]
    return f"""
<h2 id="appD">附录 D　与《分钟线数据方案（合并版）》的逐条对照</h2>
<p>老师的方法论文档面向 A 股横截面选股；本项目是「另类数据 × 单品种时序」
——对象不同但纪律通用。下表逐条对照其要求与本项目的落实（含明示的偏离与
原因）：</p>
{wrap_text('<tr><th>方法论要求（原文档）</th><th>本项目落实</th>'
      '<th>状态</th></tr>', body)}
"""


def appendix_e() -> str:
    return """
<h2 id="appE">附录 E　复现与产物索引</h2>
<pre class="formula">.venv/bin/python scripts/v3_defense_build.py    # 面板+信号+检验（约 2 分钟）
.venv/bin/python scripts/v3_defense_report.py   # 本文档
# 上游：crawl_polymarket_chain.py（链上爬虫）→ v3_build_tape.py（统一 tape）
#       build_cn_futures_db.py（期货分钟库）→ select_polymarket_markets.py（登记表）</pre>
<div class="tablewrap"><table>
<tr><th>产物（data/cn_futures/analysis/v3/defense/）</th><th>内容</th></tr>
<tr><td><code>panel_{SC,AU,AG,CU,M}.parquet</code></td><td>分钟研究面板
（字段见附录 C.3）</td></tr>
<tr><td><code>ic_table.parquet</code></td><td>40 信号 × 6 视界 × 3 时段域 ×
5 品种的 IC / RankIC / ICIR</td></tr>
<tr><td><code>signal_stats.parquet</code></td><td>频率 / 统计量 / 缺失率 /
极值率全表</td></tr>
<tr><td><code>event_study.parquet</code></td><td>E 族 + X 族事件研究（对
基线）</td></tr>
<tr><td><code>label_robustness.parquet</code></td><td>close vs VWAP 标签
RankIC 对照</td></tr>
<tr><td><code>p_event_stats.parquet</code></td><td>p_event 成交分布（全部逐笔成交
+ 逐主题）</td></tr>
<tr><td><code>history_monthly.parquet</code></td><td>2022-11 起逐月主题事件
频率</td></tr>
<tr><td><code>example_tx_{raw,clean}.parquet</code></td><td>§1.1 真实示例
交易</td></tr>
<tr><td><code>meta.json</code></td><td>逐品种面板元信息</td></tr>
</table></div>
<div class="footer">Alpha-Data · feat/cn-futures-polymarket · 与主报告
cn_futures_polymarket_report.html（v2.1）互补 · 附录 A.3 套利时间线的复现：
对示例市场 condition_id 取 block_timestamp ∈ [10:29:14 ± 1h] 的全部成交按
Yes / No 侧分列即得。</div>
"""


STYLE_EXTRA = """
.qa { background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; padding: .7rem 1rem; margin: .6rem 0;
  font-size: .93rem; }
.qa b { color: var(--accent); }
.sigdef { background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; padding: .6rem 1rem; margin: .55rem 0;
  font-size: .92rem; }
.sigdef .why { color: var(--ink-soft); font-size: .85rem; }
.wraptext table th, .wraptext table td { white-space: normal;
  text-align: left; }
"""


def build() -> str:
    from build_thesis_html import STYLE
    meta = json.loads((D / "meta.json").read_text())
    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Polymarket 事件数据 × 中国商品期货：数据、信号与检验（答辩版）</title>
<style>{STYLE}{STYLE_EXTRA}</style>
<main>
<div class="eyebrow">Alpha-Data 答辩文档 v2 · 2026-07-16 ·
主线 + 附录（supplement）结构</div>
<h1>Polymarket 事件数据 × 中国商品期货：<br>
数据处理、信号库与分钟级检验</h1>
{guide_box()}
<div class="toc"><b>目录</b><br>
<b>主线</b><br>
<a href="#s0">0 执行摘要</a><br>
<a href="#s1">1 数据层：定义、处理与边界情形</a><br>
<a href="#s2">2 信号构造：归一化与四族设计</a><br>
<a href="#s3">3 信号定义全集（40 个公式）</a><br>
<a href="#s4">4 单信号评估：统计基本功</a><br>
<a href="#s5">5 组合信号结果（完整呈现正负结果）</a><br>
<a href="#s6">6 多重检验与诚实结论</a><br>
<b>附录（supplement）</b><br>
<a href="#appA">A 定价机制与市场结构</a><br>
<a href="#appB">B 数据与方法的已知边界</a><br>
<a href="#appC">C 字段词典</a><br>
<a href="#appD">D 与方法论文档逐条对照</a><br>
<a href="#appE">E 复现与产物索引</a></div>
{summary_section(meta)}
{pm_data_section(meta)}
{edge_cases_section()}
{quality_section(meta)}
{construction_section()}
{signal_defs_section()}
{stats_section()}
{ic_section()}
{event_section(meta)}
{combo_section()}
{jump_section()}
{honest_section()}
{appendix_a()}
{appendix_b()}
{appendix_c()}
{appendix_d()}
{appendix_e()}
</main>
"""


PART3_BANNER = """
<div class="part" id="part3"><div class="kicker">第三部分 · 分钟级信号检验篇</div>
<div class="pt">Polymarket 事件数据的分钟级信号库、统计与 IC 检验</div>
<p>第一部分回答「窗口级传导结构是什么」，第二部分回答「哪些效应可识别」，
本篇下沉到<b>分钟粒度</b>：数据逐行定义与 22 条边界情形、40 个信号的
显式公式与归一化登记、频率 / 分布 / IC / ICIR / 事件研究的全套统计基本功，
以及组合信号的完整正负结果。<b>本篇为自包含单元：篇内的节号
（§0-§6）与附录号（附录 A-E）均指本篇内部</b>；全报告层面的参考文献与
总附录（窗口边界 / 术语表 / v3 产物）在本篇之后。本篇亦有独立版本
（v3_signal_defense.html），内容同源生成。口径注：本篇主题聚合权重为
<b>时点化累计</b>（√cumUSDC，仅用 ≤t 成交），区别于第一、二部分沿用的
静态全窗口权重（其等权消融见第一部分 §12.1、口径声明见 §4.6 注）。</p></div>
"""


def build_embedded() -> str:
    """供主报告嵌入的第三部分：锚点加 m 前缀、交叉引用改写为部内引用。"""
    pub_style.setup(cn_font=True)
    meta = json.loads((D / "meta.json").read_text())
    body = (
        PART3_BANNER
        + summary_section(meta)
        + pm_data_section(meta)
        + edge_cases_section()
        + quality_section(meta)
        + construction_section()
        + signal_defs_section()
        + stats_section()
        + ic_section()
        + event_section(meta)
        + combo_section()
        + jump_section()
        + honest_section()
        + appendix_a()
        + appendix_b()
        + appendix_c()
        + appendix_d()
        + appendix_e()
    )
    # 独立版收录说明在合订本中不需要（须在“主报告”改写之前删除）。
    body = body.replace(
        "本篇同时以「第三部分」收录于主报告\n"
        "cn_futures_polymarket_report.html（合订本）。", "")
    # 锚点加前缀（双引号与单引号两种写法），防止与主报告冲突。
    for q in ('"', "'"):
        body = body.replace(f'id={q}s', f'id={q}ms')
        body = body.replace(f'href={q}#s', f'href={q}#ms')
        body = body.replace(f'id={q}app', f'id={q}mapp')
        body = body.replace(f'href={q}#app', f'href={q}#mapp')
    # 「主报告」引用在合订本语境下改写为部内引用。
    body = body.replace("主报告 §16", '第二部分 <a href="#s16">§16</a>')
    body = body.replace("主报告 §14.1", '第二部分 <a href="#s14">§14.1</a>')
    body = body.replace("主报告 §12.6", '第一部分 <a href="#s12">§12.6</a>')
    body = body.replace("主报告 §3", '第一部分 <a href="#s3">§3</a>')
    body = body.replace("主报告（cn_futures_polymarket_report.html v2.1）",
                        "本报告第一、二部分")
    body = body.replace("主报告（v2.1）", "第二部分（v3.1）")
    body = body.replace("见其 §16", '见第二部分 <a href="#s16">§16</a>')
    body = body.replace("主报告", "第一、二部分")
    # 本篇执行摘要标题在合订本中改名。
    body = body.replace(">0　执行摘要</h2>", ">0　本篇导读（执行摘要）</h2>")
    return body


def main() -> int:
    pub_style.setup(cn_font=True)
    html = build()
    OUT.write_text(html)
    print(f"written {OUT} ({len(html) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
