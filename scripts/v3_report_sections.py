"""论文级 HTML 报告第二部分（v3 识别与统计功效框架）章节生成。

被 ``build_thesis_html.py`` 调用：:func:`build_part2` 返回第 14-18 节 HTML，
:func:`build_appendix_c` 返回附录 C。动态表格从
``data/cn_futures/analysis/v3/`` 的产物读取，图从 ``docs/figures/v3/`` 内嵌。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data" / "cn_futures" / "analysis" / "v3"

#: 主题中文短名（表格显示用）。
THEME_CN = {
    "mideast_conflict": "中东冲突", "oil_price": "油价阈值", "metal_price": "金银阈值",
    "russia_ukraine": "俄乌冲突", "fed_policy": "美联储", "us_china_trade": "中美贸易",
    "taiwan_risk": "台海风险", "us_shutdown": "美政府停摆",
}


def _wrap(head: str, rows: list[str]) -> str:
    return ('<div class="tablewrap"><table>' + head + "".join(rows)
            + "</table></div>")


def migration_table_html() -> str:
    """迁移重叠期 v2 / v1 成交量抽样（六个 3,000 块窗口，实测）。"""
    data = [
        ("85,050,371", "04-03 12:52", 6, 741_817),
        ("85,300,000", "04-09 07:33", 0, 570_090),
        ("85,900,000", "04-23 04:54", 37, 460_128),
        ("86,050,000", "04-26 16:14", 56, 509_194),
        ("86,100,000", "04-27 20:00", 108, 491_462),
        ("86,120,000", "04-28 07:07", 157, 424_817),
    ]
    head = ("<tr><th>起始区块</th><th>UTC 时刻</th><th>v2 成交事件</th>"
            "<th>v1 成交事件</th><th>v2 占比</th></tr>")
    rows = [
        f"<tr><td>{blk}</td><td>{t}</td><td>{v2}</td><td>{v1:,}</td>"
        f"<td>{v2 / (v1 + v2):.4%}</td></tr>"
        for blk, t, v2, v1 in data
    ]
    return _wrap(head, rows)


def coverage_table_html() -> str:
    """结算数据完整性对 Power Gate 的影响（对照实验，实测）。"""
    head = ("<tr><th>结算时刻来源</th><th>登记市场覆盖</th>"
            "<th>可计算 surprise 的合格事件</th><th>通过功效门控的组合</th></tr>")
    rows = [
        "<tr class='die'><td>仅 HF 数据集元数据</td><td>56 / 425</td>"
        "<td>79</td><td><b>0 / 14</b>（全部降级 sign-only）</td></tr>",
        "<tr class='live'><td>+ 链上 ConditionResolution（自爬两段）</td>"
        "<td>455 / 492</td><td>278</td><td><b>4 / 14</b>"
        "（mideast / russia × SC / AU）</td></tr>",
    ]
    return _wrap(head, rows)


def power_table_html() -> str:
    tab = pd.read_parquet(V3 / "v3_power_table.parquet")
    tab = tab.sort_values(["decision", "mde"])
    head = ("<tr><th>主题 × 品种</th><th>事件数</th><th>聚类数</th>"
            "<th>Σ(s−s̄)²</th><th>设计效应</th><th>有效样本</th>"
            "<th>MDE</th><th>经济上限 δ</th><th>决策</th></tr>")
    rows = []
    for r in tab.itertuples(index=False):
        live = r.decision == "category_beta"
        cls = ' class="live"' if live else ' class="die"'
        mde = "∞" if np.isinf(r.mde) else f"{r.mde:.3f}"
        dec = "<b>估计 β</b>" if live else "sign-only"
        rows.append(
            f"<tr{cls}><td>{THEME_CN.get(r.theme, r.theme)} × {r.product}</td>"
            f"<td>{r.n_events}</td><td>{r.n_clusters}</td><td>{r.sum_s2:.2f}</td>"
            f"<td>{r.design_effect:.2f}</td><td>{r.n_eff:.1f}</td>"
            f"<td><b>{mde}</b></td><td>{r.delta_econ:.3f}</td><td>{dec}</td></tr>")
    return _wrap(head, rows)


def impact_table_html() -> str:
    tab = pd.read_parquet(V3 / "v3_impact_betas.parquet")
    tab = tab[tab["mode"] == "category_beta"].sort_values("t_stat", ascending=False)
    head = ("<tr><th>主题 × 品种</th><th>β（收益 / 单位 surprise）</th>"
            "<th>t</th><th>训练事件数</th><th>解读</th></tr>")
    rows = []
    for r in tab.itertuples(index=False):
        sig = abs(r.t_stat) >= 2
        cls = ' class="live"' if sig else ' class="die"'
        note = ("满 surprise ≈ 次日 "
                f"{r.beta * 100:+.1f}% 收益" if sig else "功效充分下的真实阴性")
        rows.append(
            f"<tr{cls}><td>{THEME_CN.get(r.theme, r.theme)} × {r.product}</td>"
            f"<td><b>{r.beta:+.4f}</b></td><td>{r.t_stat:+.2f}</td>"
            f"<td>{r.n_train}</td><td>{note}</td></tr>")
    return _wrap(head, rows)


def oos_table_html() -> str:
    tab = pd.read_parquet(V3 / "v3_oos_results.parquet")
    head = ("<tr><th>品种</th><th>OOS 天数</th><th>v1.1 基线</th>"
            "<th>v3 精简（同维度）</th><th>v3 全测量层</th>"
            "<th>CW t（基线 / 精简）</th></tr>")
    rows = []
    for r in tab.itertuples(index=False):
        base, slim, full = r.dR2_M1_base, r.dR2_M1_v3slim, r.dR2_M1_v3

        def fmt(v: float, best: bool, cw: float) -> str:
            star = "*" if np.isfinite(cw) and abs(cw) >= 1.645 else ""
            s = f"{v * 100:+.1f}%{star}"
            return f"<b>{s}</b>" if best else s

        slim_best = slim >= base
        rows.append(
            f"<tr><td>{r.product}</td><td>{r.n_oos}</td>"
            f"<td>{fmt(base, not slim_best, r.cw_t_M1_base)}</td>"
            f"<td>{fmt(slim, slim_best, r.cw_t_M1_v3slim)}</td>"
            f"<td>{full * 100:+.1f}%</td>"
            f"<td>{r.cw_t_M1_base:+.2f} / {r.cw_t_M1_v3slim:+.2f}</td></tr>")
    return _wrap(head, rows)


def absorption_table_html() -> str:
    tab = pd.read_parquet(V3 / "v3_absorption.parquet")
    piv: dict[tuple[str, str], dict[str, object]] = {}
    for r in tab.itertuples(index=False):
        piv.setdefault((r.theme, r.product), {})[r.signal] = r
    head = ("<tr><th>主题 × 品种</th><th colspan='3'>v1.1 基线信号</th>"
            "<th colspan='3'>v3 coherent 信号</th></tr>"
            "<tr><th></th><th>n</th><th>corr</th><th>t 控制基准后</th>"
            "<th>n</th><th>corr</th><th>t 控制基准后</th></tr>")
    rows = []
    for (theme, product), d in piv.items():
        b, v = d.get("base"), d.get("v3")

        def cell(r: object) -> str:
            if r is None:
                return "<td>—</td><td>—</td><td>—</td>"
            t_ctl = getattr(r, "t_ctl", float("nan"))
            tc = f"<b>{t_ctl:+.2f}</b>" if pd.notna(t_ctl) else "—"
            return f"<td>{r.n}</td><td>{r.corr:+.2f}</td><td>{tc}</td>"

        rows.append(
            f"<tr><td>{THEME_CN.get(theme, theme)} × {product}</td>"
            f"{cell(b)}{cell(v)}</tr>")
    return _wrap(head, rows)


def h4_table_html() -> str:
    tab = pd.read_parquet(V3 / "v3_h4_tension_vol.parquet")
    tab = tab[tab["t_tension"].notna() & (tab["beta_tension"] != 0)]
    head = ("<tr><th>品种</th><th>n</th><th>t(tension)</th><th>解读</th></tr>")
    rows = []
    for r in tab.itertuples(index=False):
        sig = abs(r.t_tension) >= 2
        cls = ' class="live"' if sig else ""
        note = ("显著为负：失衡持续 = 低关注时段，次日波动更低"
                if (sig and r.t_tension < 0) else "不显著")
        rows.append(
            f"<tr{cls}><td>{r.product}</td><td>{r.n}</td>"
            f"<td><b>{r.t_tension:+.2f}</b></td><td>{note}</td></tr>")
    return _wrap(head, rows)


def registry_growth_table_html() -> str:
    """登记表 v1.1 -> v3 的主题构成（唯一市场数）。"""
    reg = pd.read_parquet(ROOT / "data/polymarket/features/cn_registry_v3.parquet")
    v3 = reg.drop_duplicates("condition_id").groupby("theme").size()
    v11 = pd.Series(  # v1.1 报告数字（docs/cn_futures_polymarket_report.md）
        {"fed_policy": 9, "metal_price": 58, "mideast_conflict": 172,
         "oil_price": 79, "russia_ukraine": 34, "taiwan_risk": 7,
         "us_china_trade": 11, "us_shutdown": 18})
    head = ("<tr><th>主题</th><th>v1.1（至 04-28）</th><th>v3（至 07-13）</th>"
            "<th>净增</th></tr>")
    rows = []
    for theme in v3.index:
        a, b = int(v11.get(theme, 0)), int(v3[theme])
        rows.append(
            f"<tr><td>{THEME_CN.get(theme, theme)}</td><td>{a}</td>"
            f"<td><b>{b}</b></td><td>{b - a:+d}</td></tr>")
    rows.append(
        f"<tr class='base'><td>合计</td><td>{int(v11.sum())}</td>"
        f"<td>{int(v3.sum())}</td><td>{int(v3.sum() - v11.sum()):+d}</td></tr>")
    return _wrap(head, rows)


def build_part2(tex: Callable[[str], str], fig_tag: Callable[[str, str], str]) -> str:
    """第二部分（第 14-18 节）HTML。"""
    val = json.loads((V3 / "v3_validation.json").read_text())
    h1r = val["h1_lite"]["ratio"]
    h2n = val["h2_lite"]["n_violations"]
    h2r = val["h2_lite"]["repair_rate"]

    T = [
        # 0 观测方程与噪声模型
        tex(r"y_k \;=\; z_k + \varepsilon_k, \qquad \varepsilon_k \sim "
            r"N(0,\, R_k), \qquad R_k \;=\; "
            r"\left(\frac{\mathrm{spread}_k}{2}\right)^2 + \frac{r_0^2}{n_k}"),
        # 1 状态方程
        tex(r"z_k \;=\; z_{k-1} + \eta_k, \qquad \eta_k \sim "
            r"N\left(0,\; q \cdot \Delta t_k \cdot m(\tau_k)\right)"),
        # 2 标准化创新
        tex(r"\nu_k \;=\; y_k - \hat z_{k|k-1}, \qquad "
            r"\tilde\nu_k \;=\; \nu_k \,/\, \sqrt{S_k}, \qquad "
            r"S_k = P_{k|k-1} + R_k"),
        # 3 PAVA 投影与 tension
        tex(r"q^* \;=\; \min_{q\,\uparrow}\; \sum_i w_i\,(q_i - p_i)^2,"
            r"\qquad \mathrm{Tension} \;=\; \frac{\sum_i w_i (p_i - q^*_i)^2}"
            r"{\sum_i w_i}"),
        # 4 hazard
        tex(r"\Lambda_t(T_k) \;=\; -\ln\left(1 - F_t(T_k)\right), \qquad "
            r"\mathrm{hz}_w \;=\; \Lambda_{t_1}(T_{\mathrm{front}}) - "
            r"\Lambda_{t_0}(T_{\mathrm{front}})"),
        # 5 价格阈值分布
        tex(r"S_t(K) \;=\; P_t\left(\max_{u \leq T} X_u \geq K\right), \qquad "
            r"\mathrm{med} = S_t^{-1}(0.5), \quad \mathrm{tail} = S_t(K_{max})"),
        # 6 公共因子与正交化
        tex(r"F^{\mathrm{PM}}_t \;=\; \frac{\sum_c w_c\, \tilde S_{c,t}}"
            r"{\sum_c w_c}, \qquad S^{\perp}_{c,t} \;=\; S_{c,t} - "
            r"\hat b_{c,<t}\, F^{\mathrm{PM}}_t"),
        # 7 surprise
        tex(r"s_i \;=\; Y_i \;-\; \sigma\left(\hat z(\mathrm{res}_i - "
            r"\mathrm{24h})\right), \qquad Y_i \in \{0, 1\}"),
        # 8 MDE
        tex(r"\mathrm{MDE} \;=\; \left(z_{1-\alpha/2} + z_{1-\gamma}\right)\,"
            r"\frac{\sigma_R}{\sqrt{\sum_i (s_i - \bar s)^2 \,/\, \mathrm{DE}}},"
            r"\qquad \mathrm{DE} = 1 + (\bar m - 1)\rho"),
        # 9 影响系数
        tex(r"R_{j,\,\mathrm{next}(i)} \;=\; \alpha_j + \beta_{j,c}\, s_i + "
            r"\epsilon_{j,i}"),
        # 10 嵌套模型
        tex(r"M_0:\; \Gamma' X_t \qquad M_1:\; M_0 + \theta' S^{\perp}_t "
            r"\qquad M_2:\; M_1 + \varphi'\,\hat\beta_c S^{\perp}_{c,t}"),
        # 11 Clark-West
        tex(r"f_t \;=\; e_{0,t}^2 - e_{1,t}^2 + \left(\hat y_{0,t} - "
            r"\hat y_{1,t}\right)^2, \qquad t_{CW} = "
            r"\frac{\bar f}{\mathrm{se}_{HAC}(\bar f)}"),
    ]

    return f"""
<div class="part" id="part2"><div class="kicker">第二部分</div>
<div class="pt">v3 识别与统计功效框架：门控式管道与扩展样本</div>
<p>第一部分止于三个如实承认的局限：样本只有 75 个重叠交易日且被单一事件主导；
"事件概率变化值多少收益"（影响系数）从未被正式估计；预测性检验没有功效核算，
"不显著"无法区分"不存在"与"测不出"。第二部分按研究规范
《Polymarket 中国期货预测：识别与统计功效 v3》重建整个管道来回应这三点。
框架的核心不是增加特征，而是给每个复杂模块加上<b>资格条件与失败退路</b>：
稳定主干（数据层 → 潜在信念 → 逻辑一致性 → 事件几何 → 增量预测）永远成立，
而"把概率换算成收益"的影响层只有在事件定义合格<b>且</b>统计功效充分时才启用，
否则按预注册阶梯降级。全程执行数据隔离协议：测量层全部超参数只用
Polymarket 内部目标选择，期货数据首次出现于功效核算与影响层训练窗。</p></div>

<h2 id="s14">14　数据扩展：自爬链上管线把样本延长 70%</h2>

<h3>14.1 统一 tape：两段数据的无缝拼接</h3>
<p>第一部分依赖的 HuggingFace 数据集止于 2026-04-28——这不是作者停更，而是
Polymarket 旧交易所合约在该日 11:00:40 UTC（区块 86,126,998）产生最后一笔成交
后被 v2 合约取代（详见 <code>docs/POLYMARKET_CRAWL.md</code>）。我们的链上爬虫
从区块 86,127,000 起自建同构数据：按 topic 扫描新旧两版 OrderFilled 事件、
剔除中继腿、只保留已核验 Polymarket 场馆、剔除 negRisk 市场、用与公开数据集
完全一致的真值表计算事件概率与方向。两段按区块号天然不重叠、不留缝。</p>
<p><b>唯一需要担心的同质性问题</b>：v2 交易所 4 月 3 日就已上线，与旧合约并行
了 25 天——HF 段只含 v1 成交，若 v2 在此期间已有可观流量，拼接处会有流量
断层。直接抽样六个 3,000 块窗口实测：</p>
{migration_table_html()}
<p>迁移重叠期 v2 占比不足 0.04%（用户实际迁移发生在 4 月 28 日切换时刻），
HF 段在其覆盖范围内实质完整，拼接成立。边界日交叉核对：两段共有市场 1,403
个，HF 末笔与扩展段首笔的概率中位差 0.02。</p>
<p>拼接后与期货库（至 07-13）的重叠达 <b>127 个交易日</b>（第一部分 75 个），
且样本不再由美伊冲突单独主导——五至七月包含冲突降温、贸易谈判反复与
贵金属新一轮阶梯市场。同一套事前注册规则（v1.1 的 slug 模式与排除词原样
不动）自动命中扩展期的新市场：</p>
{registry_growth_table_html()}
<p>扩展期新市场没有作者派生的 <code>category_refined</code> 列（对 NULL 类别
放行，实际选择完全由 slug 模式承担；抽查确认新增命中均为既有族的新月份
轮次）。市场级准入仍是时点化的（累计成交额首次达 $10 万的时刻
<code>admit_ts</code>）。</p>

<h3>14.2 结算事件：Power Gate 的数据前提</h3>
<p>功效核算需要每个事件的三元组（结算时刻，事前概率，实现结果）。这里遇到
本轮最大的数据缺口：<b>HF 数据集的 <code>resolved_at</code> 元数据只覆盖
登记市场的 56/425</b>。解决办法仍是链上：ConditionTokens 合约的
<code>ConditionResolution</code> 事件不可篡改地记录每次结算，补爬两段
（2025-12 至 04-28 区间 790,799 行、04-28 之后 880,309 行；事件本身无时间戳列，
区块时刻用成交事件的（区块, 时间）对做最近邻回填，误差秒级）。结算时刻覆盖
升到 455/492（其余 37 个尚未结算）。这一步不是锦上添花——它直接决定了
第 16 节功效门控的判定结果：</p>
{coverage_table_html()}
<div class="callout">对照实验的含义：同一套门控规则，在劣质结算数据下正确地
全线拦截（避免估计不可识别的参数），在数据补齐后精确放行样本最厚的两个主题。
<b>自爬管线在此不可替代</b>——公开数据集里根本没有可用的结算时刻。</div>

<h2 id="s15">15　测量层：从成交流到潜在信念结构</h2>

<h3>15.1 Layer 1：logit 状态空间滤波</h3>
<p>第一部分的信号是"15 分钟桶两向加权中位数的 logit 之差"——直接、稳健，
但每个市场的创新没有统一的噪声尺度，跨市场聚合时只能靠流动性权重近似。
v3 把它升级为状态空间模型：观测是桶价的 logit，真实信念是不可观测的状态，
两者之差是微观结构噪声，其方差由桶内可观测量决定：</p>
{T[0]}
<div class="texnote">spread_k = 买方向与卖方向中位数之差换算到 logit 域（有效
点差代理）；n_k = 桶内笔数（中位数抽样误差随 1/n 收缩）；r_0 为市场级残余
噪声尺度。无订单簿数据源，这是点差信息在成交 tape 上的最优可得代理。</div>
{T[1]}
<div class="texnote">局部水平（随机游走）状态方程；q 为每小时状态方差，
Δt_k 为距上一观测的小时数——不活跃时段状态不确定度自动累积。m(τ) 为剩余
期限乘子（下文）。(q, r_0) 逐市场网格 MLE，956 个市场（登记 492 个 + 平台
公共因子宇宙 600 个头部市场，去重）在共用桶网格上向量化滤波，126 万桶观测
一轮 2 分钟。</div>
{T[2]}
<div class="texnote">ν̃ 是一步预测标准化创新——跨市场、跨期限可比的
"信息增量"，正是主题聚合与公共因子需要的原料。</div>
<p><b>期限异方差是真实的</b>（框架 §5.3 的预判）：第一轮滤波后按剩余期限
分桶估计标准化创新的方差比，呈 U 形——距截止一个月以上 0.97、7-30 天 0.79、
2-7 天 0.73、<b>最后两天跳升至 1.29</b>（机械收敛与事件结算爆发段）。第二轮
以此曲线作 Q 乘子重估。</p>
{fig_tag("v3/f_v3_term.png",
         "标准化创新方差比按剩余期限分桶（第一轮滤波，全市场合并）。临近截止"
         "两天内方差跳升三成——不处理期限异方差会把「快到期」误读成「信息多」。"
         "图内文字：两项内部验证的数值。")}
<p><b>诚实的内部验证</b>（全部用 Polymarket 自身数据，不涉期货）：
（i）H1-lite：滤波一步预测创新方差 / 原始桶间差分方差 = {h1r:.2f}——
两向中位数桶价本身已足够干净，<b>滤波在一步预测上并不优于 LOCF</b>。它的
实际价值在于给出创新的条件方差尺度（标准化）、边界处的状态不确定度、以及
族投影的平滑输入，而不是"预测得更准"。（ii）H2-lite：族内单调性违反
{h2n} 处（约占投影快照 0.1%），违反后价格向投影方向修复的比率 {h2r:.3f}
——方向正确但接近随机。阶梯族在 15 分钟粒度上定价已高度一致，投影是护栏
而非信息源。</p>

<h3>15.2 Layer 2：族识别与逻辑一致性投影</h3>
<p>Polymarket 的市场不是孤立的——"金价 6 月底前触及 3500/3600/3700 美元"是
同一分布的三个切片，"美军 2 月底 / 3 月底 / 4 月底前打击伊朗"是同一事件时刻
的累计分布。从 slug 结构自动解析出两类族（价格阈值族按行权价、日期阶梯族按
截止日），族内施加程序可验证的单调约束，用流动性加权的等张回归（PAVA）投影：</p>
{T[3]}
<div class="texnote">w 取流动性平方根——流动市场的价格更可信，被移动得更少。
Tension 是族内定价失衡的强度，作为独立的测量输出（第 17.4 节）。这是第一部分
§8.3 观察到的"日期阶梯族抱团"现象的约束化利用。</div>

<h3>15.3 Layer 3：事件几何——hazard、分布特征与平台公共因子</h3>
<p><b>日期族恢复 hazard 而非直接用累计概率</b>：累计概率变化在概率接近 1 时
被机械压缩（0.90→0.95 与 0.50→0.55 的"距离"完全不同），换到累计强度坐标：</p>
{T[4]}
<div class="texnote">窗口 hazard 信号以窗口结束时最近的未到期截止（front）为
基准、在窗口两端对同一截止求 Λ 之差——避免 front 滚动造成机械跳变。远端
阶梯常稀薄，只要求 front 有效。</div>
<p><b>价格阈值族只做分布恢复、禁入影响层</b>：hit-high 阶梯给出期内最大值的
生存函数，三个特征刻画其形态与变化：</p>
{T[5]}
<div class="texnote">隐含中位数取对数差后与期货对数收益同量纲。第一部分
§11 已证明这类市场是 USO/GLD 的镜像——它们的"影响系数"是机械映射，v3 把
这一教训制度化为资格规则（第 16.1 节）。</div>
<p><b>平台公共因子</b>：全平台前 600 个高流动市场（不限于国内相关登记表，
含加密、体育、选举等全部类别）的窗口创新按族聚合、展开窗口标准化后取流动性
加权截面均值——这近似"整个平台此刻的方向性资金冲击"。国内主题信号对它做
<b>展开窗口</b>回归取残差（系数只用严格早于当期的观测，无前视）：</p>
{T[6]}
<div class="texnote">动机来自第一部分 §12.2 的安慰剂发现：原始吸收含大量宏观
共同因子。正交化在信号构造阶段就剥离平台层面的共同创新，是"多种类市场组合
降噪"的直接实现。</div>

<h2 id="s16">16　双门控与影响层：什么时候允许把概率换算成收益</h2>

<h3>16.1 事件资格门</h3>
<p>每个市场按 slug 结构自动标注事件类型：价格阈值（oil_price / metal_price
全部市场）标记 <code>asset_price_threshold</code>、<b>禁入影响层</b>——事件
结果是标的价格的确定函数，回归系数退化为机械映射，还会在聚合信号中以巨大的
伪"影响"淹没真正的外生事件。外生事件（冲突、封锁）与政策决策（加息、关税、
停摆）合格。这条规则不是审美选择：第一部分 §11 的 H1 偏零检验（金价类市场
控制 GLD 后 t 2.24→0.72）就是它的实证依据。</p>

<h3>16.2 Power Gate：先算最小可检测效应，再决定是否估计</h3>
<p>对每个已结算的合格事件定义 surprise——实现结果减去结算前 24 小时的滤波
概率：</p>
{T[7]}
<p>识别力来自 surprise 的横截面变差而非事件数量：事前概率已收敛到 0 或 1
且按预期实现的事件，surprise ≈ 0，对回归不贡献信息。同族同周结算的事件共享
信息窗口，按聚类修正有效样本，然后核算 80% 功效下的最小可检测效应：</p>
{T[8]}
<div class="texnote">ρ 预注册取 0.5（保守）；σ_R 为品种日收益标准差（训练窗）。
预注册经济上限 δ = 2σ_R——单位 surprise 的真实效应若超过两倍日波动早就
肉眼可见，MDE 超过它意味着"这个组合只能检测出不可能存在的效应"，
按阶梯降级：市场级 β → 类别级 β → sign-only → 关闭。</div>
{power_table_html()}
{fig_tag("v3/f_v3_power.png",
         "MDE（条形）对预注册经济上限（竖线）：绿色 = 通过（允许估计类别级 β），"
         "红色 = 降级 sign-only。taiwan_risk 的三个事件全部按事前概率实现，"
         "surprise 变差恰为零，MDE 为无穷——形式化地证实了该主题在当前样本"
         "不可检验。")}

<h3>16.3 Layer 4：类别级影响系数（训练窗冻结）</h3>
<p>通过双门控的组合，在训练窗（2026-05-15 之前结算的事件）内做事件研究——
结果变量是品种在结算时刻后<b>下一个可交易日</b>的收盘收益（北京 09:00 前
结算算当日，其后算次日；严格 next-tradable 对齐，杜绝把同期共动记作预测）：</p>
{T[9]}
{impact_table_html()}
<p><b>这是两部分研究中第一次得到可辩护的影响系数</b>：中东冲突类事件的满
surprise（市场认为不会发生的事发生了）对应 SC 次日约 +6.0%、AU 约 +4.4% 的
收益，训练窗 t 值均超过 4——量级与 2 月底冲突升级时 SC 的实际走势一致。
俄乌主题通过了功效门控但系数不显著（t ≈ 0.4 / −0.6）：功效充分时的不显著
是<b>真实阴性</b>（该主题事件对国内资产无稳定的次日效应），与"测不出"有
本质区别——这正是先算功效再估计的意义。未通过门控的组合回退 sign-only
（方向先验 × 标准化信号），不假装有数值精度。</p>

<h2 id="s17">17　增量预测与吸收复核：v3 对 v1.1 的正面对比</h2>

<h3>17.1 设计：嵌套模型 + 展开窗口 OOS + Clark-West</h3>
<p>目标是框架的核心假设 H0: θ = 0——控制"截至开盘你能免费看到的一切"之后，
Polymarket 测量信号是否还有增量。预测目标为当日日盘收益 r_day(t)；信号窗
[前收盘 15:00, 当日 09:00) 严格早于目标窗。三个嵌套模型：</p>
{T[10]}
<div class="texnote">X = 品种直连境外 ETF（USO/GLD/SLV/CPER/SOYB/XME/ASHR）同
窗口收益 + SPY + UUP（美元）+ 自身滞后收益 + 平台公共因子。评估为展开窗口
一步 OOS（最少 40 个训练观测），每个品种约 77-82 个 OOS 交易日。嵌套模型的
MSPE 比较用 Clark-West 修正（M1 包含 M0，参数估计噪声天然使 M1 的原始 MSPE
偏大）：</div>
{T[11]}
<p>基线对照：同一评估框架下，把 v3 测量信号换成第一部分的 v1.1 信号
（相同主题集合、相同控制、相同样本），并另设"v3 精简"变体（只保留正交化
主题创新，与基线维度一致）以排除"参数多寡"的干扰。</p>

<h3>17.2 结果：降噪有效，但次日方向的可预测成分本身就小</h3>
{oos_table_html()}
{fig_tag("v3/f_v3_oos.png",
         "OOS 增量 R²（相对控制集 M0，%）。* 为 Clark-West |t| ≥ 1.645。"
         "多数品种为负——把信号加入控制集在样本外反而放大预测误差；"
         "唯一稳定为正的是豆粕 M（中美贸易主题）。同维度对比下 v3 精简（绿）"
         "在 6/8 品种上优于基线（灰）；全测量层（蓝）因过参数化普遍更差。")}
<p>三个读数，按重要性排序：</p>
<ul>
<li><b>绝对水平：两套信号对次日方向的 OOS 增量预测力整体为负。</b>唯一例外
是豆粕 M（+3.2% → +5.4%，CW t ≈ 2.0）。这与第一部分 §7 的"开盘后无剩余方向
信息"完全一致，但现在是带功效核算与嵌套检验的正式结论，而非小样本印象。
信息在闭市窗口内已被定价——它值钱的地方是同期吸收（下节），不是隔日预测。</li>
<li><b>相对改善：同维度对比下 v3 在 6/8 品种上损失更小或增益更大</b>（AU 的
OOS 损失从 −39% 收窄到 −24%，M 的正增益从 +3.2% 扩大到 +5.4%）。滤波 +
正交化确实降低了信号噪声——"多种类组合降噪"成立，只是降噪之后剩下的
可预测成分不大。</li>
<li><b>全测量层变体普遍差于精简版</b>：约 120 个训练观测承载不起 10-14 个
回归元，hazard 与分布特征的加入在此框架下是过参数化。它们应作为专项检验
对象（如波动率通道），不应默认并入预测面板。</li>
</ul>
<p>影响信号的 M2 在通过门控的品种上与 M1 拟合恒等——单一类别时
impact = β̂ × S⊥ 是测量信号的标量倍数，不含新信息。影响层的价值是<b>经济
单位换算</b>（surprise → 收益幅度，服务于事件驱动的仓位决策），当多个类别
同时通过门控时才可能贡献额外预测力。</p>

<h3>17.3 同期吸收复核：样本翻倍后头部结论增强</h3>
{absorption_table_html()}
{fig_tag("v3/f_v3_absorption.png",
         "扩展样本（127 交易日）上的闭市 gap 同期吸收，v1.1 基线信号与 v3 "
         "coherent 信号并列。第一部分的头部结论全部保持；mideast×SC 的 v3 信号"
         "相关（0.52）高于基线（0.48）。")}
<p>关键行是 <b>mideast×SC：控制同窗 USO 后基线 t = 3.79、v3 信号 t = 4.94</b>
（n = 114，第一部分为 t = 3.68、n = 68）——第一部分最重要的幸存证据（中东
事件概率对 SC 的独立增量）在样本翻倍后不仅保持而且增强，且 v3 信号的版本
更强。诚实标注反例：metal_price×AU 控制 GLD 后 v3 信号（t = 2.04）弱于基线
（t = 6.44）——金价阶梯族的投影平滑掉的成分里可能既有噪声也有 GLD 之外的
边际信息，且两版样本构成不同（n 43 vs 41），此差异待更长样本判别。</p>

<h3>17.4 H4：narrative tension 与波动</h3>
<p>tension（族内定价失衡强度）是 v3 新增的测量对象。对次日日盘绝对收益
回归（控制昨日绝对收益）：</p>
{h4_table_html()}
<p>SC 上显著为<b>负</b>（t = −4.0）：阶梯族失衡越严重，次日波动越低。一个
可检验的解释：失衡只有在套利者缺位（关注度低）的时段才能持续存在，而低
关注恰与低后续波动同现——tension 实际上度量的是"市场此刻没人盯着"。注意
它与第一部分 §12.5 的发现（|信号| 预测高波动，t = 4.4）并不矛盾：|s| 度量
信息流强度，tension 度量定价失衡，两者是不同的对象，符号相反反而说明各自
携带独立内容。列为待复核现象，不作机制声称。</p>

<h2 id="s18">18　第二部分结论：框架给出了什么、没有给出什么</h2>
<div class="finding"><span class="no">一</span><b>样本扩展后第一部分的头部结论
全部保持并增强。</b>吸收结构（oil×SC 0.74、mideast×SC 0.52、metal×AU 0.54）
在 127 个交易日上复现；mideast×SC 控制 USO 后的独立增量从 t=3.68（n=68）升到
t=4.94（n=114）。这是对第一部分最有价值的稳健性升级——结论不再依赖单一
事件期。</div>
<div class="finding"><span class="no">二</span><b>影响系数第一次可识别，且只在
数据补齐后。</b>链上结算事件把可用 surprise 样本从 79 扩到 278，功效门控放行
mideast / russia × SC / AU 四个组合；mideast 的类别级 β 为 SC +6.0%、AU +4.4%
每单位 surprise（训练窗 t &gt; 4）。俄乌主题为功效充分下的真实阴性。</div>
<div class="finding"><span class="no">三</span><b>组合降噪有可度量的收益，但
不翻转"次日方向 OOS ≈ 0"。</b>同维度下 v3 信号在 6/8 品种优于基线；唯一
稳定为正的品种是豆粕 M（+5.4%，CW t ≈ 2.0）。Polymarket 信息的变现通道是
闭市时段的同期吸收与事件驱动的风险管理，不是隔日方向预测。</div>
<div class="finding"><span class="no">四</span><b>门控纪律的价值在对照实验中
直接可见。</b>同一规则在劣质结算数据下全线拦截（79 个 surprise、全部降级）、
在数据补齐后精确放行——若跳过功效核算直接估计市场级 β，得到的将是不可
识别参数的噪声实现，并可能以"发现"的名义进入结论。</div>
<div class="finding"><span class="no">五</span><b>测量层的否定结果同样成立。</b>
滤波不改进一步预测（比率 1.11）；投影修复率 0.51 接近随机；全测量层并入
预测面板属过参数化。框架的贡献不是"更多特征更好"，而是知道每个模块的
边界在哪。</div>
<h3>18.1 局限与下一步</h3>
<ul>
<li>ETF 控制集止于 06-29（美股库右端），7 月上旬 OOS 行按缺控制剔除；
CF 无直连基准（BAL 已退市），控制集退化为 SPY+UUP。</li>
<li>影响系数为单次训练窗切分（滚动 cross-fitting 需更多事件）；跨类别外推
无依据。事件类型标注未做人工复核轮。</li>
<li>下一步：（a）爬虫转日常增量（游标就位），样本每月自然 +20 交易日；
（b）mideast β 按 surprise 分层（中段概率 vs 尾部）直接检验哪类事件携带
增量；（c）波动率通道（§12.5 与 §17.4 合并规格）独立成文；（d）多类别
同时通过门控后重估 M2 的边际预测力。</li>
</ul>
"""


def build_appendix_c() -> str:
    """附录 C：v3 术语与产物索引。"""
    return """
<h2 id="appC">附录 C　第二部分术语与产物索引</h2>
<div class="tablewrap"><table>
<tr><th>术语</th><th>含义</th></tr>
<tr><td>状态空间 / Kalman 滤波</td><td>把观测拆成潜在状态 + 噪声的递推估计；
本文状态为 logit 信念，观测为桶价 logit</td></tr>
<tr><td>标准化创新 ν̃</td><td>一步预测误差除以其条件标准差，跨市场可比的
信息增量单位</td></tr>
<tr><td>期限乘子 m(τ)</td><td>状态方差随剩余期限的比例修正（临近截止两天
方差 +29%）</td></tr>
<tr><td>PAVA / 等张回归</td><td>在保持单调约束下对序列的加权最小二乘投影，
有精确解</td></tr>
<tr><td>tension</td><td>族内价格到单调可行集的加权均方距离，定价失衡强度</td></tr>
<tr><td>hazard / 累计强度 Λ</td><td>−ln(1−F)；日期阶梯族的跨期限可比坐标</td></tr>
<tr><td>平台公共因子 F_PM</td><td>全平台头部市场族级创新的流动性加权截面
均值，近似平台方向性资金冲击</td></tr>
<tr><td>surprise s_i</td><td>事件实现结果减结算前 24h 滤波概率</td></tr>
<tr><td>MDE</td><td>给定显著性与功效下可检测的最小真实效应；超过预注册
经济上限即降级</td></tr>
<tr><td>设计效应 DE</td><td>聚类（同族同周结算）导致的有效样本缩减系数</td></tr>
<tr><td>sign-only</td><td>功效不足时的回退：只用方向先验 × 标准化信号，
不赋数值精度</td></tr>
<tr><td>next-tradable 对齐</td><td>影响回归的结果变量从结算后国内下一个
可交易时点起算，排除同期共动</td></tr>
<tr><td>Clark-West</td><td>嵌套模型 MSPE 比较的修正检验（扣除大模型的参数
估计噪声）</td></tr>
<tr><td>展开窗口（expanding）</td><td>每一步只用严格早于当期的观测重估参数，
防前视</td></tr>
</table></div>
<div class="tablewrap"><table>
<tr><th>产物</th><th>内容</th></tr>
<tr><td><code>alpha_data/polymarket/v3/</code></td><td>框架实现（tape / latent /
families / coherence / geometry / story / gates / impact 八个模块）</td></tr>
<tr><td><code>scripts/v3_build_tape.py</code></td><td>统一 tape 构建 + 登记表
（窗口延至 07-13）</td></tr>
<tr><td><code>scripts/v3_measurement.py</code></td><td>Layer 1-3 测量层管道
（约 2 分钟）</td></tr>
<tr><td><code>scripts/v3_baseline_extended.py</code></td><td>v1.1 基线信号在
扩展样本上的重建（约 22 分钟）</td></tr>
<tr><td><code>scripts/v3_layer5.py</code></td><td>双门控 + Layer 4/5 评估</td></tr>
<tr><td><code>data/cn_futures/analysis/v3/</code></td><td>信号面板、功效表、
影响系数、OOS 预测、吸收复核、H4 等 15 个产物表</td></tr>
<tr><td><code>docs/v3_framework_results.md</code></td><td>本部分的独立
Markdown 版结果文档</td></tr>
<tr><td><code>tests/test_v3_framework.py</code></td><td>19 项离线单元测试
（族解析 / PAVA / hazard / 滤波 / 无前视 / MDE / 对齐 / 时间换算）</td></tr>
</table></div>
"""
