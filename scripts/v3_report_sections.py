"""论文级 HTML 报告第二部分（v3 识别与统计功效框架，v3.1 修订版）章节生成。

被 ``build_thesis_html.py`` 调用：:func:`build_part2` 返回第 14-18 节 HTML，
:func:`build_appendix_c` 返回附录 C。动态表格从
``data/cn_futures/analysis/v3/`` 的产物读取，图从 ``docs/figures/v3/`` 内嵌。

v3.1：外部评审后的修订版——episode 聚类推翻影响系数可识别性、point-in-time
参数协议、ex-ante/ex-post 门控分离、LOFO 公共因子、多重检验与时段拆分。
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


def registry_growth_table_html() -> str:
    """登记表 v1.1 -> v3 的主题构成（唯一市场数）。"""
    reg = pd.read_parquet(ROOT / "data/polymarket/features/cn_registry_v3.parquet")
    v3 = reg.drop_duplicates("condition_id").groupby("theme").size()
    v11 = pd.Series(
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


def pseudo_replication_table_html() -> str:
    """伪重复的量化：市场级 vs episode 级（mideast×SC，实测）。"""
    head = ("<tr><th>口径</th><th>样本单位数</th><th>Σ(s−s̄)²（识别变差）</th>"
            "<th>MDE</th><th>经济上限 δ</th><th>判定</th></tr>")
    rows = [
        "<tr class='die'><td>市场级（初版；family×周聚类，ρ=0.5）</td>"
        "<td>191 个市场</td><td>13.54</td><td>0.047</td><td>0.099</td>"
        "<td><b>误放行</b></td></tr>",
        "<tr><td>episode 级，ex-post（全样本）</td><td>72 个收益日</td>"
        "<td><b>1.02</b></td><td>0.136</td><td>0.099</td><td>拦截</td></tr>",
        "<tr class='base'><td>episode 级，ex-ante（&lt; 05-15，OOS 用）</td>"
        "<td>56 个收益日</td><td>0.99</td><td>0.139</td><td>0.099</td>"
        "<td>拦截</td></tr>",
    ]
    return _wrap(head, rows)


def power_table_html() -> str:
    tab = pd.read_parquet(V3 / "v3_power_table_expost.parquet")
    ante = pd.read_parquet(V3 / "v3_power_table_exante.parquet").set_index(
        ["theme", "product"])
    tab = tab.sort_values("mde")
    head = ("<tr><th>主题 × 品种</th><th>市场数</th><th>episode 数</th>"
            "<th>Σ(s−s̄)²</th><th>MDE（ex-post）</th><th>MDE（ex-ante）</th>"
            "<th>上限 δ</th><th>决策</th></tr>")
    rows = []
    for r in tab.itertuples(index=False):
        try:
            mde_a = ante.loc[(r.theme, r.product), "mde"]
            mde_a = "∞" if np.isinf(mde_a) else f"{mde_a:.3f}"
        except KeyError:
            mde_a = "—"
        mde = "∞" if np.isinf(r.mde) else f"{r.mde:.3f}"
        rows.append(
            f"<tr class='die'><td>{THEME_CN.get(r.theme, r.theme)} × "
            f"{r.product}</td><td>{r.n_markets}</td><td>{r.n_episodes}</td>"
            f"<td>{r.sum_s2:.2f}</td><td><b>{mde}</b></td><td>{mde_a}</td>"
            f"<td>{r.delta_econ:.3f}</td><td>sign-only</td></tr>")
    return _wrap(head, rows)


def oos_table_html() -> str:
    tab = pd.read_parquet(V3 / "v3_oos_results.parquet")
    rob = pd.read_parquet(V3 / "v3_oos_robustness.parquet").set_index("product")
    head = ("<tr><th>品种</th><th>OOS 天数</th><th>v1.1 基线</th>"
            "<th>v3 精简（同维度）</th><th>v3 全测量层</th>"
            "<th>CW t（精简）</th><th>BH-FDR q</th><th>块自助 p</th>"
            "<th>前 / 后半段</th></tr>")
    rows = []
    for r in tab.itertuples(index=False):
        base, slim, full = r.dR2_M1_base, r.dR2_M1_v3slim, r.dR2_M1_v3
        slim_best = slim >= base
        b_s = f"{base * 100:+.1f}%"
        s_s = f"{slim * 100:+.1f}%"
        if slim_best:
            s_s = f"<b>{s_s}</b>"
        else:
            b_s = f"<b>{b_s}</b>"
        try:
            rb = rob.loc[r.product]
            halves = (f"{rb['dR2_v3slim_h1'] * 100:+.1f}% / "
                      f"{rb['dR2_v3slim_h2'] * 100:+.1f}%")
            pmbb = f"{rb['p_mbb_cw_v3slim']:.3f}"
        except KeyError:
            halves, pmbb = "—", "—"
        cls = ' class="live"' if (slim > 0 and r.cw_t_M1_v3slim > 1.645) else ""
        rows.append(
            f"<tr{cls}><td>{r.product}</td><td>{r.n_oos}</td><td>{b_s}</td>"
            f"<td>{s_s}</td><td>{full * 100:+.1f}%</td>"
            f"<td>{r.cw_t_M1_v3slim:+.2f}</td>"
            f"<td>{r.q_cw_M1_v3slim:.2f}</td><td>{pmbb}</td>"
            f"<td>{halves}</td></tr>")
    return _wrap(head, rows)


def sessions_table_html() -> str:
    tab = pd.read_parquet(V3 / "v3_oos_sessions.parquet")
    name_cn = {"close_to_night": "傍晚信号 → 夜盘（控同窗 ETF）",
               "postnight_to_day": "凌晨信号 → 日盘（控夜盘收益）"}
    head = "<tr><th>设计</th><th>品种</th><th>n</th><th>ΔR²</th><th>CW t</th></tr>"
    rows = [
        f"<tr><td>{name_cn.get(r.design, r.design)}</td><td>{r.product}</td>"
        f"<td>{r.n_oos}</td><td>{r.dR2 * 100:+.1f}%</td><td>{r.cw_t:+.2f}</td></tr>"
        for r in tab.itertuples(index=False)
    ]
    return _wrap(head, rows)


def absorption_table_html() -> str:
    tab = pd.read_parquet(V3 / "v3_absorption.parquet")
    piv: dict[tuple[str, str], dict[str, object]] = {}
    for r in tab.itertuples(index=False):
        piv.setdefault((r.theme, r.product), {})[r.signal] = r
    head = ("<tr><th>主题 × 品种</th><th colspan='3'>v1.1 基线信号</th>"
            "<th colspan='3'>v3 coherent 信号</th></tr>"
            "<tr><th></th><th>n</th><th>corr</th><th>t 控制代理后</th>"
            "<th>n</th><th>corr</th><th>t 控制代理后</th></tr>")
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
    head = ("<tr><th>品种</th><th>n</th><th>t(tension)</th>"
            "<th>t(tension) 加活动度控制</th><th>t(活跃市场数)</th>"
            "<th>t(log 成交额)</th></tr>")
    rows = []
    for r in tab.itertuples(index=False):
        sig = pd.notna(r.t_tension_actrl) and abs(r.t_tension_actrl) >= 2
        cls = ' class="live"' if sig else ""

        def f(v: float) -> str:
            return f"{v:+.2f}" if pd.notna(v) else "—"

        rows.append(
            f"<tr{cls}><td>{r.product}</td><td>{r.n}</td>"
            f"<td>{f(r.t_tension)}</td><td><b>{f(r.t_tension_actrl)}</b></td>"
            f"<td>{f(r.t_n_active)}</td><td>{f(r.t_log_usdc)}</td></tr>")
    return _wrap(head, rows)


def build_part2(tex: Callable[[str], str], fig_tag: Callable[[str, str], str]) -> str:
    """第二部分（第 14-18 节）HTML。"""
    import v3_supp_sections
    SUPP_V3_STATS = v3_supp_sections.sec_v3_stats()
    SUPP_ABSORB_SUB = v3_supp_sections.sec_absorb_subsample()
    SUPP_TENSION_EP = v3_supp_sections.sec_tension_episode()
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
        # 5 LOFO 公共因子
        tex(r"F^{-c}_t \;=\; \frac{\sum_{k \neq c} w_k\, \tilde S_{k,t}}"
            r"{\sum_{k \neq c} w_k}, \qquad S^{\perp}_{c,t} \;=\; S_{c,t} - "
            r"\hat b_{c,<t}\, F^{-c}_t"),
        # 6 surprise
        tex(r"s_i \;=\; Y_i \;-\; \sigma\left(\hat z(\mathrm{res}_i - "
            r"\mathrm{24h})\right), \qquad Y_i \in \{0, 1\}"),
        # 7 episode 聚合
        tex(r"s_{d} \;=\; \frac{1}{|E_d|}\sum_{i \in E_d} s_i, \qquad "
            r"E_d = \{i:\; \mathrm{next\_tradable}(i) = d\}"),
        # 8 MDE（episode 设计）
        tex(r"\mathrm{MDE} \;=\; \left(z_{1-\alpha/2} + z_{1-\gamma}\right)\,"
            r"\frac{\sigma_R}{\sqrt{\sum_d (s_d - \bar s)^2}}"),
        # 9 嵌套模型
        tex(r"M_0:\; \Gamma' X_t \qquad M_1:\; M_0 + \theta' S^{\perp}_t "
            r"\qquad X_t \ni r^{\mathrm{night}}_{j,t}"),
        # 10 Clark-West
        tex(r"f_t \;=\; e_{0,t}^2 - e_{1,t}^2 + \left(\hat y_{0,t} - "
            r"\hat y_{1,t}\right)^2, \qquad t_{CW} = "
            r"\frac{\bar f}{\mathrm{se}_{HAC}(\bar f)}"),
    ]

    return f"""
<div class="part" id="part2"><div class="kicker">第二部分 · v3.1 修订版</div>
<div class="pt">v3 识别与统计功效框架：分阶段准入、扩展样本与一次真实的证伪</div>
<p>第一部分止于三个如实承认的局限：样本只有 75 个重叠交易日且被单一事件主导；
"事件概率变化值多少收益"（影响系数）从未被正式估计；预测性检验没有功效核算。
第二部分按研究规范《识别与统计功效 v3》重建整个研究流程来回应这三点。本部分为
<b>v3.1 修订版</b>：初版曾报告"中东冲突影响系数可识别（t &gt; 4）"，外部评审
指出事件伪重复、测量参数时间前视、事前与事后功效门槛混用，以及公共因子含自身
族四类实质问题；全部修复重跑后，<b>影响系数的"可识别"结论被推翻</b>（第 16
节保留了修订前后的完整对照——它是本文最有教学价值的部分），其余主要结论
在更严格的协议下仍然成立。全程执行数据隔离与时点化（point-in-time）双协议：测量层超参数
只用 Polymarket 内部目标、且只用训练截止（2026-05-15）前的数据估计。</p></div>

<h2 id="s14">14　数据扩展：自建链上采集流程把样本延长 70%</h2>

<h3>14.1 统一逐笔成交记录：两段数据的无缝拼接</h3>
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
<p>拼接后的统一逐笔成交记录覆盖 127 个国内交易日（信号侧至 07-15），与期货库
（至 07-13）的<b>重叠为 125 个交易日</b>（第一部分 75 个；此前"127"的表述
混用了两个分母，已统一），且样本不再由美伊冲突单独主导——冲突段内外的
硬分段证据见 §17.5。同一套事前注册规则（v1.1 的 slug 模式与排除词
原样不动）自动命中扩展期的新市场：</p>
{registry_growth_table_html()}
<p>扩展期新市场没有作者派生的 <code>category_refined</code> 列（对 NULL 类别
放行，实际选择完全由 slug 模式承担；抽查确认新增命中均为既有族的新月份
轮次）。市场级准入仍是时点化的（累计成交额首次达 $10 万的时刻
<code>admit_ts</code>）。</p>

<h3>14.2 结算事件：统计功效门槛的数据前提</h3>
<p>功效核算需要每个事件的三元组（结算时刻，事前概率，实现结果）。这里遇到
本轮最大的数据缺口：<b>HF 数据集的 <code>resolved_at</code> 元数据只覆盖
登记市场的 56/425</b>（分母定义：492 个 v3 登记市场中 425 个在 HF 段
（≤04-28）内已有成交；56 为其中带 HF 元数据 resolved_at 者；第一部分的
388 为 v1.1 规则在 HF 段的命中数，三个分母口径不同，此处统一交代）。
解决办法仍是链上：ConditionTokens 合约的
<code>ConditionResolution</code> 事件不可篡改地记录每次结算，补爬两段
（2025-12 至 04-28 区间 790,799 行、04-28 之后 880,309 行；事件本身无时间戳
列，区块时刻用成交事件的（区块, 时间）对做最近邻回填，误差秒级）。结算时刻
覆盖升到 455/492，可计算 surprise 的合格市场事件从 79 个增至 278 个。这决定
了功效核算是否可能——尽管第 16 节将表明，修正伪重复后当前样本仍不足以识别
影响系数，但"能核算"与"核算后判定不足"是两种完全不同的科学状态。</p>

<h2 id="s15">15　测量层：从成交流到潜在信念结构（时点化协议）</h2>

<h3>15.1 第一层：logit 状态空间滤波</h3>
<p>第一部分的信号是"15 分钟桶两向加权中位数的 logit 之差"——直接、稳健，
但每个市场的创新没有统一的噪声尺度。v3 把它升级为状态空间模型：观测是桶价
的 logit，真实信念是不可观测的状态，两者之差是微观结构噪声，其方差由桶内
可观测量决定：</p>
{T[0]}
<div class="texnote">spread_k = 买卖两向中位数差换算到 logit 域（有效点差
代理）；n_k = 桶内笔数；r_0 为市场级残余噪声尺度。</div>
{T[1]}
<div class="texnote">局部水平状态方程；q 为每小时状态方差，Δt_k 为距上一
观测的小时数，m(τ) 为剩余期限乘子。</div>
{T[2]}
<div class="texnote">ν̃ 是一步预测标准化创新——跨市场、跨期限可比的
"信息增量"单位。</div>
<p><b>时点化协议（point-in-time，v3.1 修订）</b>：初版用每个市场完整生命周期估计
(q, r_0)，再回头生成早期滤波状态——即使从不接触期货收益，也让 5 月的信号
构造"知道"了 6 月的 Polymarket 数据（时间前视，区别于目标泄漏）。</p>
<p>修订后
参数与期限乘子只用 <b>2026-05-15 前</b>的观测做网格 MLE，随后冻结、对全样本
做因果滤波（Kalman 递推本身因果）；训练截止后诞生的市场（182 / 956 个）
回退训练窗参数的中位数。期限乘子曲线（训练窗）仍呈 U 形：距截止一个月以上
0.97、7-30 天 0.79、2-7 天 0.77、<b>最后两天 1.32</b>。</p>
{fig_tag("v3/f_v3_term.png",
         "标准化创新方差比按剩余期限分桶（训练窗估计）。临近截止两天内方差"
         "跳升三成——不处理期限异方差会把「快到期」误读成「信息多」。"
         "图内文字：两项内部验证的数值。")}
<p><b>诚实的内部验证</b>：（i）H1-lite：滤波一步预测创新方差 / 原始桶间差分
方差 = {h1r:.2f}——参数冻结后比初版（1.11）更差，进一步确认<b>滤波在一步
预测上不优于 LOCF</b>；它的价值在条件方差尺度、边界不确定度与投影输入。
（ii）H2-lite：族内单调性违反 {h2n} 处（约占投影快照 0.1%），违反后价格向
投影方向修复的比率 {h2r:.3f}——接近随机，投影是护栏而非信息源。</p>

<h3>15.2 第二层：事件族识别与逻辑一致性投影</h3>
<p>从 slug 结构自动解析价格阈值族（按行权价）与日期阶梯族（按截止日），
族内施加程序可验证的单调约束，用流动性加权等张回归（PAVA）投影：</p>
{T[3]}
<div class="texnote">w 取流动性平方根。Tension 是族内定价失衡强度，作为独立
测量输出（第 17.4 节）。</div>

<h3>15.3 第三层：结算风险率、分布特征与留一主题平台公共因子</h3>
<p>日期族换到累计强度坐标（概率接近 1 时累计概率变化被机械压缩）：</p>
{T[4]}
<p>价格阈值族只做分布恢复（隐含中位数 / 尾部质量 / 熵）、禁入影响层——
第一部分 §11 的 H1 偏零检验（金价类市场控制 GLD 后 t 2.24→0.72）是这条
资格规则的实证依据。</p>
<p><b>平台公共因子（v3.1 改为 leave-one-family-out）</b>：全平台前 600 个
高流动市场的族级创新加权截面均值近似"平台此刻的方向性资金冲击"。初版用
全族因子给所有主题做正交化——但 mideast 族本身在平台权重中占比很大，
"对含自身的因子取残差"会机械地从信号中减掉自身的一部分。修订后每个主题
使用<b>剔除自身全部族</b>的因子：</p>
{T[5]}
<div class="texnote">系数 b 只用严格早于当期的观测（展开窗口，无前视）。
全族全局因子仍单独输出，仅作第五层控制变量（控制变量含自身无碍）。</div>

{SUPP_V3_STATS}

<h2 id="s16">16　双重功效门槛与影响层：一次真实的证伪</h2>

<h3>16.1 事件资格门</h3>
<p>每个市场按 slug 结构自动标注事件类型：价格阈值（oil_price / metal_price
全部市场）标记 <code>asset_price_threshold</code>、禁入影响层；外生事件与
政策决策合格。surprise 定义为实现结果减结算前 24 小时的滤波概率：</p>
{T[6]}

<h3>16.2 伪重复：市场不是样本单位</h3>
<p>初版在市场级估计影响系数，得到 mideast×SC 的 beta = 0.060（t = 4.58，
191 个"事件"）——数字漂亮，但外部评审一针见血：<b>同一现实事件让整族日期
阶梯同时结算</b>。2 月底美军打击伊朗，一夜之间 25+ 个 "by-X 日" 市场全部
结算为 YES，它们共享<b>同一个</b>次日 SC 收益；跨族的 khamenei-out、
regime-fall 也在同一收益窗。</p>
<p>市场级回归把一个收益观测复制了 25 次，
family×周聚类（初版的设计效应修正）既漏掉跨族重复、又低估了同收益日内
相关（恒为 1 而非 0.5）。</p>
<p>修订：样本单位改为 <b>episode = (品种, 下一可交易日)</b>，episode 内
surprise 取族内均值（同收益日聚合等价于最保守的 ρ=1 处理），功效核算直接在
episode 设计上进行：</p>
{T[7]}
{T[8]}
<p>后果是决定性的：</p>
{pseudo_replication_table_html()}
<div class="callout"><b>识别变差的 92% 是伪重复。</b>191 个市场只对应 72 个
独立收益日，最大单 episode 有 27 个市场；Σ(s−s̄)² 从 13.5 塌缩到 1.02，
MDE 升到经济上限的 1.4 倍。<b>episode 口径下全部 14 个（主题, 品种）组合、
事前与事后两套功效门槛一致拦截，第四层整体降级为仅保留方向（sign-only）。</b>初版
"影响系数可识别、beta = 6%（t &gt; 4）"是统计错觉，本修订版正式撤回。</div>
{power_table_html()}
{fig_tag("v3/f_v3_power.png",
         "episode 口径（ex-post）的 MDE 对预注册经济上限（竖线）：全部超限。"
         "最接近的 mideast×AU 差 38%。")}
{fig_tag("v3/f_v3_episode.png",
         "(a) mideast×SC 各 episode 共享同一收益日的市场数（最大 27）——伪重复"
         "的直观呈现；(b) episode 级 surprise 分布：SD 0.12、IQR 0.021，"
         "|s|>0.2 的事件批次仅 6 个——即便功效门槛放行，系数也几乎由这 6 天决定。")}
<p><b>统计功效门槛分事前与事后两套</b>（评审第 4 点）：决定样本外模型
结构的事前门槛只用训练截止（05-15）前的事件批次；全样本版本仅回答"以当前数据量
哪些参数可研究"，不得反向影响模型选择。本轮两套判定一致（都拦截），随爬虫
日常增量与新事件周期积累，事后功效表按月复核——影响系数何时可识别由数据
说话，功效门槛继续作为研究流程的必经环节。</p>
<p><b>已知的剩余粗糙度</b>：episode 键按"北京 09:00 前算当日、其后算次一
交易日"归并到<b>日</b>粒度；对有夜盘品种，严格的下一可交易时点常是当晚
21:00 而非次日 09:00，事件批次收益窗因此混合了夜盘与日盘吸收。在功效门槛全部
拦截的现状下该细化不改变任何决策（它只会进一步收紧样本），第五层的时段
拆分设计（§17.2）已沿真实交易序列分解了吸收位置；待 episode 数积累到接近
功效门槛时，此对齐须先于任何 β 估计完成。</p>

<h2 id="s17">17　增量预测与吸收复核：修订协议下的对比</h2>

<h3>17.1 设计</h3>
<p>目标不变：控制"截至开盘你能免费看到的一切"之后，Polymarket 测量信号是否
还有增量。v3.1 的控制集新增品种<b>自身夜盘收益</b>（09:00 时已实现——信号窗
[前收盘 15:00, 当日 09:00) 内本品种夜盘开市，其间信息可能已被自身吸收，
评审第 7 点）：</p>
{T[9]}
{T[10]}
<div class="texnote">X 另含品种直连 ETF（USO/GLD/SLV/CPER/SOYB/XME/ASHR）同
窗口收益 + SPY + UUP + 自身滞后 + 平台全局因子。采用展开窗口的一步样本外检验，最少 40
个训练观测。多重检验：跨 8 品种 BH-FDR、逐品种循环块自助（块长 10）、
前后半段一致性。</div>

<p><b>品种集合说明（审计补）</b>：第二部分固定 8 品种（AG/AU/CF/CU/I/IF/
M/SC）——入选标准是"有直连或近缘 ETF 代理可作控制"。第一部分 §12.7-12.8
的探索性头部结果之一 mideast×IM（控制后 t=2.15-2.62）未入本框架，原因是
IM（中证 1000 股指）无直连 ETF 代理、控制集不可比；该线索的复核列入
§18.1 待办。</p>

<h3>17.2 结果：M 是候选发现；v3 降噪为方向性改善</h3>
{oos_table_html()}
{fig_tag("v3/f_v3_oos.png",
         "样本外增量 R²（相对控制集 M0，%）。Clark-West 为单侧检验：▲ = t ≥ 1.645"
         "（扩展模型显著改善），▼ = t ≤ −1.645（显著恶化），二者不共用记号。"
         "多数品种为负；唯一显示正向样本外增量的候选品种是豆粕 M。同维度对比下 "
         "v3 精简（绿）在 6/8 品种上优于基线（灰）；全测量层（蓝）过参数化更差。")}
<ul>
<li><b>绝对水平：两套信号对次日方向的样本外增量预测力整体为负。</b>唯一显示
正向样本外增量的候选品种是豆粕 M：v3 信号 +4.9%（CW 单侧 t = 1.86，块自助
p ≈ 0.001，前后半段 +4.2% / +6.2% 同号，对评估起点稳健），对全部六项修复
稳健。但<b>影响点检查</b>显示增量集中于少数日子：删除贡献最大的 1 天降至
+1.4%，删除前 3 天即转负（−1.1%）——事件驱动信号在事件日获得收益并不意外，
但这意味着显著性依托于极少数观测。叠加跨 8 品种 BH-FDR q = 0.12 与
SOYB（大豆 ETF）代理豆粕、缺 USDCNH——定性为<b>"少数事件日的定价滞后"
候选，而非日频 alpha</b>。另注意 Clark-West 的解读是单侧的：AG 基线的
t = −2.18 属"扩展模型显著恶化"，与 M 的正 t 不是同一种"显著"。</li>
<li><b>相对改善：v3 精简在 6/8 品种上优于基线</b>（修复时间前视与 LOFO 后
仍成立），但跨品种合并的标准化损失差块自助单侧 p = 0.147——<b>方向性
改善，统计上不足为证</b>。6/8 的符号计数本身不构成显著性结论。</li>
<li>CF 的 CW t = +1.99 与负 ΔR² 并存：扣除参数噪声后信号存在、但估计成本
超过收益的典型 Clark-West 形态，不作方向结论。</li>
<li>全测量层变体仍普遍差于精简版：约 120 个训练观测承载不起 10-14 个
回归元。</li>
</ul>
<p><b>时段拆分</b>（评审第 7 点：夜盘才是多数品种真正的下一可交易时点）：
傍晚信号（15:00-21:00，夜盘开盘前已知）预测夜盘、凌晨信号（夜盘收盘后）
预测日盘并控制夜盘收益，两套设计全部 ΔR² &lt; 0（CW 最高 1.85，边际）：</p>
{sessions_table_html()}
<p>信息在可交易时点到来前已被境外市场与本品种夜盘吸收——与主设计结论互证，
也把"日盘无剩余预测力"的原因定位得更准：不只是境外吸收，本品种夜盘
自身就是吸收者之一。</p>

<h3>17.3 同期吸收复核：对全部修复稳健</h3>
<p><b>措辞修订（评审第 6 点）</b>：控制变量是美股 ETF 与美元指数 ETF 代理，
不是同窗口的 24 小时境外期货与 USDCNH——ETF 收盘后 Brent/WTI 期货仍在交易，
"增量"可能部分捕捉的是 ETF 盘后的期货变化。因此本节结论的正确表述是
"<b>控制可得代理集后仍有增量同期关联</b>"，初版"Polymarket 含有国际期货
之外的独立信息"的表述撤回，待接入真实境外期货与 USDCNH 数据后复核。</p>
{absorption_table_html()}
<p>在此限定下，mideast×SC 仍是全研究最强的证据链：控制同窗 USO 后基线
t = 3.79、v3 信号 <b>t = 4.94</b>（n = 114，第一部分为 3.68 @ n = 68），
对时点化、留一主题法与事件批次聚类等修订全部稳健。反例照旧如实报告：
metal×AU 控制 GLD 后 v3（2.04）弱于基线（6.44），投影可能平滑掉了 GLD
之外的边际信息，待更长样本判别。</p>
<p><b>与第一部分结论二的显式调和（审计补）</b>：扩展样本复核下
oil_price×SC·gap 控制 USO 后 t = 3.68（基线与 v3 一致）——与第一部分
75 天样本的"不再显著"（4.89→1.88）相反，价格类在扩展样本的 gap 窗<b>重新
显著</b>。因此第一部分"价格类无增量"（H1 依据）限定于原 75 天样本与
夜盘窗；§16.1 以 H1 为实证依据禁价格阈值族入影响层的资格规则，在扩展
样本下应重审。第一部分 §11.2 与结论二已同步加注。metal×AG·night 的
边际显著（第一部分 t=2.66）在本复核中消失（t=0.72），两部分的完整对照
以本节为准。</p>

<h3>17.4 H4：tension-波动的负相关不是活动度代理</h3>
<p>评审第 10 点给出了直接的判别设计：若 tension 只是"没人交易"的代理，
加入活动度控制后负号应消失。实测（活跃市场数 + 主题成交额 + 滞后波动）：</p>
{h4_table_html()}
<p>SC 的 t 从 -4.40 到<b>加控制后仍为 -4.40</b>，两个活动度变量自身均不显著
——最简单的注意力代理解释被排除；CU 在控制规格下也为负（-2.84）；AG 的
正号在控制后消失。表中 CF/CU/I/M 四行的基线列为"—"：基线规格（无控制的
单变量 tension 回归）只在 v1.1 的三个原品种（SC/AU/AG）上估计过，扩展
品种直接进入含控制的规格，非样本不足（审计补注）。
机制仍属探索（下一个判别：把失衡按"来自高成交 vs 无成交
市场"拆开），但这个负相关已经历一轮否证尝试后仍然存在。</p>

{SUPP_ABSORB_SUB}

{SUPP_TENSION_EP}

<h2 id="s18">18　第二部分结论（v3.1）</h2>
<div class="finding"><span class="no">一</span><b>样本扩展后第一部分的头部结论
全部保持并增强。</b>吸收结构在 125 个重叠交易日上复现；mideast×SC 控制 USO 代理
后的增量从 t=3.68（n=68）升到 t=4.94（n=114），且对全部六项评审修复稳健。
限定：控制集是 ETF 代理，非 24 小时境外期货与 USDCNH。</div>
<div class="finding"><span class="no">二</span><b>影响系数在当前样本下不可
识别——初版相反结论是伪重复的统计错觉，已撤回。</b>191 个市场事件只对应
72 个独立收益日，识别变差的 92% 是同收益日复制；episode 口径下全部组合
MDE 超限（最近差 38%）。统计功效门槛不再只是形式要求，而是实际拦截了一次已发表错误
的机制。</div>
<div class="finding"><span class="no">三</span><b>豆粕 M 的正向样本外增量是唯一
候选发现</b>：+4.9%（CW t 1.86、块自助 p 0.001、前后半段同号、对修复稳健），
但 BH-FDR q = 0.12 且缺豆粕期货 / USDCNH 控制。升级为结论的路径明确：
更长样本 + 真实境外控制。</div>
<div class="finding"><span class="no">四</span><b>v3 测量层的降噪收益是方向性
的</b>（6/8 品种优于基线，合并 p = 0.147）；其确定的价值在统一方差尺度、
功效核算输入与防前视协议本身，不在样本外增量的统计显著性。</div>
<div class="finding"><span class="no">五</span><b>tension-波动负相关经受了
活动度控制的否证尝试</b>（SC t = -4.40 不变），是值得专项深挖的新测量对象；
时段拆分确认信息在夜盘与境外市场被吸收，日盘无剩余。</div>
<div class="finding"><span class="no">六</span><b>方法学结论：这一轮修订本身
就是框架的论证。</b>数据隔离协议挡不住时间前视，市场计数骗过了 family 级
聚类——只有把"样本单位是什么"与"参数在哪个时点可知"作为一等公民对待，
预测市场研究的正结果才值得相信。修订前后的全部对照保留在产物与本节中。</div>
<h3>18.1 局限与下一步</h3>
<ul>
<li>境外控制为 ETF 代理（无 Brent/WTI、COMEX、CBOT 期货与 USDCNH 分钟数据，
库内 CNH ETF 已退市）；接入真实境外期货是解除吸收与 M 结论限定的唯一途径。</li>
<li>episode 按下一可交易日归并；更细的"同一现实事件跨多日轮次"人工标注会
进一步收缩有效样本，当前口径仍可能偏乐观。</li>
<li>主题聚合权重沿用全窗口流动性（静态，继承 v1.1；第三部分已改时点化
累计权重——在 v3 流程中加入同款变体，重跑吸收复核与 M 的样本外结果，列为下一步）。</li>
<li>下一步：(a) 爬虫日常增量，事后功效表按月复核；(b) 境外期货与
USDCNH 数据源接入；(c) tension 机制专项（按成交结构拆分失衡来源，§17.6
的条件试验是其起点）；(d) 样本外条件变体——"凌晨信号 × 夜盘未兑现"的条件
组合（第一部分 §12.9 W11 的样本外版本）在事前功效门槛框架内预注册评估一次；
(e) mideast×IM 线索按可比控制集复核。</li>
</ul>
"""


def build_appendix_c() -> str:
    """附录 C：v3 术语与产物索引。"""
    return """
<h2 id="appC">总附录 C　第二部分术语与产物索引</h2>
<div class="tablewrap"><table>
<tr><th>术语</th><th>含义</th></tr>
<tr><td>状态空间 / Kalman 滤波</td><td>把观测拆成潜在状态 + 噪声的递推估计；
本文状态为 logit 信念，观测为桶价 logit</td></tr>
<tr><td>时点化（point-in-time，防止时间前视）</td><td>任一时点的信号构造只使用该时点
已存在的数据；参数在训练窗估计后冻结</td></tr>
<tr><td>episode</td><td>影响回归的样本单位：(品种, 下一可交易日)。同一收益日
结算的全部市场并为一个观测</td></tr>
<tr><td>伪重复</td><td>把共享同一收益观测的多个市场当作独立样本，虚增
识别变差与 t 值</td></tr>
<tr><td>事前 / 事后功效门槛</td><td>前者只用训练截止前数据、决定样本外模型
结构；后者用全样本、仅作研究规划</td></tr>
<tr><td>LOFO（leave-one-family-out）</td><td>某主题正交化所用公共因子剔除该
主题自身的族，避免机械自减</td></tr>
<tr><td>标准化创新 ν̃</td><td>一步预测误差除以其条件标准差，跨市场可比的
信息增量单位</td></tr>
<tr><td>期限乘子 m(τ)</td><td>状态方差随剩余期限的比例修正（临近截止两天
方差 +32%）</td></tr>
<tr><td>PAVA / 等张回归</td><td>在单调约束下的加权最小二乘投影，有精确解</td></tr>
<tr><td>tension</td><td>族内价格到单调可行集的加权均方距离，定价失衡强度</td></tr>
<tr><td>hazard / 累计强度 Λ</td><td>−ln(1−F)；日期阶梯族的跨期限可比坐标</td></tr>
<tr><td>surprise s_i</td><td>事件实现结果减结算前 24h 滤波概率</td></tr>
<tr><td>MDE</td><td>给定显著性与功效下可检测的最小真实效应；超过预注册
经济上限即降级</td></tr>
<tr><td>仅保留方向（sign-only）</td><td>功效不足时的回退：只用方向先验 × 标准化信号</td></tr>
<tr><td>Clark-West</td><td>嵌套模型 MSPE 比较的修正检验（扣除大模型参数
估计噪声）</td></tr>
<tr><td>wild bootstrap / 块自助</td><td>对小样本回归系数 / 时间序列均值的
重采样推断</td></tr>
<tr><td>BH-FDR q 值</td><td>跨品种多重检验下控制错误发现率的校正显著性</td></tr>
</table></div>
<div class="tablewrap"><table>
<tr><th>产物</th><th>内容</th></tr>
<tr><td><code>alpha_data/polymarket/v3/</code></td><td>框架实现（逐笔成交 / 潜在状态 /
families / coherence / geometry / story / gates / impact 八个模块）</td></tr>
<tr><td><code>scripts/v3_build_tape.py</code></td><td>统一逐笔成交记录构建 + 登记表</td></tr>
<tr><td><code>scripts/v3_measurement.py</code></td><td>第一至第三测量层
（PIT + LOFO，约 2 分钟）</td></tr>
<tr><td><code>scripts/v3_baseline_extended.py</code></td><td>v1.1 基线信号在
扩展样本上的重建（约 22 分钟）</td></tr>
<tr><td><code>scripts/v3_layer5.py</code></td><td>事件批次双重功效门槛 + 第四、第五层
评估 + 多重检验</td></tr>
<tr><td><code>data/cn_futures/analysis/v3/</code></td><td>信号面板、两套功效表、
事件批次表、样本外与稳健性、吸收复核、H4 等 20 个产物表</td></tr>
<tr><td><code>docs/v3_framework_results.md</code></td><td>本部分的独立
Markdown 版结果文档（v3.1）</td></tr>
<tr><td><code>tests/test_v3_framework.py</code></td><td>24 项离线单元测试
（族解析 / PAVA / hazard / 滤波 / 无前视 / episode 聚类 / wild bootstrap /
MDE / 对齐 / 时间换算）</td></tr>
</table></div>
<h3>C.2 第一、二部分产物字段词典（审计补）</h3>
""" + _field_dict_html()


def _field_dict_html() -> str:
    """第一、二部分核心产物的字段级词典。"""
    tables = [
        ("theme_signals.parquet（第一部分主信号表）", [
            ("theme / product / trade_date", "主题、品种、交易日"),
            ("s_night / s_gap / s_day", "三窗口 logit 信念创新（方向统一、"
             "√usdc 加权聚合；品种间同值复制）"),
            ("n_active", "该日参与聚合的活跃市场数"),
            ("usdc_night / _gap / _day", "各窗口市场成交额合计（美元）"),
        ]),
        ("market_signals.parquet（市场级窗口信号）", [
            ("condition_id", "市场链上标识（聚合前粒度）"),
            ("s_* / usdc_* / flow_* / n_*", "逐市场三窗口信号、成交额、"
             "带方向资金流与笔数"),
            ("p_day_open / p_age_day_open", "09:00 端点价与其距最后成交的"
             "分钟数（>120 记缺失）"),
        ]),
        ("corr_table.parquet（第一部分检验总表）", [
            ("kind", "contemporaneous（同期吸收）/ predictive（预测）"),
            ("signal / ret", "信号列与收益列配对"),
            ("pearson / spearman / beta / t_hac / p_hac", "相关、回归系数与 "
             "Newey-West 推断"),
            ("bp_per_sd", "信号 1σ 对应收益（bp）"),
            ("q_bh / small_sample", "预测族 BH-FDR q 值；n<40 小样本标记"),
        ]),
        ("v3_theme_signals.parquet（第二部分测量层）", [
            ("s_*", "滤波 + 一致性投影后的窗口信念创新（coherent）"),
            ("s_*_rawz", "未投影的标准化创新（对照）"),
            ("tension", "族内逻辑一致性张力（投影修复幅度）"),
            ("hz_* / px_*", "结算 hazard 变化；价格阈值族隐含分布特征"),
            ("f_pm_*", "LOFO 平台公共因子（逐窗口）"),
        ]),
        ("v3_episodes.parquet（episode 层）", [
            ("theme / product / ev_date", "样本单位：（主题×品种×下一"
             "可交易日）"),
            ("surprise", "结算 surprise（事前概率 vs 实现结果）"),
            ("n_markets", "该 episode 折叠的市场事件数（伪重复核算分母）"),
        ]),
        ("v3_power_table*.parquet（功效表）", [
            ("mde", "最小可检测效应（80% 功效）"),
            ("passed", "是否通过统计功效门槛；事前版本用历史方差，事后版本用"
                       "实现方差"),
        ]),
    ]
    blocks = []
    for name, fields in tables:
        rows = "".join(f"<tr><td><code>{f}</code></td><td>{d}</td></tr>"
                       for f, d in fields)
        blocks.append(f"<p><b>{name}</b></p>"
                      f'<div class="tablewrap"><table class="wraptext">'
                      f"<tr><th>字段</th><th>含义</th></tr>{rows}</table></div>")
    return "\n".join(blocks)
