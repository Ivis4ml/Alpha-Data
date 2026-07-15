"""生成论文级自包含 HTML 报告（docs/cn_futures_polymarket_report.html）。

从 docs/figures/ 内嵌全部图表（base64），从 analysis/deep/ 读取消融与
国际基准控制表动态渲染。文本目标：无背景读者可独立读懂——动机、制度背景、
数据、方法逐步推导（公式）、实证设计、结果、消融、结论与推广。

用法::

    .venv/bin/python scripts/build_thesis_html.py [--out docs/cn_futures_polymarket_report.html]
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pub_style import tex_svg  # noqa: E402

FIG = ROOT / "docs" / "figures"
DEEP = ROOT / "data" / "cn_futures" / "analysis" / "deep"


def tex(formula: str) -> str:
    """行间 TeX 公式（离线渲染为内联 SVG，随主题变色）。"""
    return f'<div class="texblock">{tex_svg(formula, fontsize=13)}</div>'


def texi(formula: str) -> str:
    """行内 TeX 公式。"""
    return tex_svg(formula, fontsize=11, display=False)


def b64(rel: str) -> str:
    return base64.b64encode((FIG / rel).read_bytes()).decode()


def fig_tag(rel: str, caption: str) -> str:
    return (f'<figure><div class="figcard"><img src="data:image/png;base64,{b64(rel)}" '
            f'alt=""></div><figcaption>{caption}</figcaption></figure>')


def ablation_table_html() -> str:
    path = DEEP / "ablation.parquet"
    if not path.exists():
        return "<p>（消融表尚未生成）</p>"
    tab = pd.read_parquet(path)
    piv_r = tab.pivot_table(index="variant", columns=["theme", "window"],
                            values="pearson")
    piv_n = tab.pivot_table(index="variant", columns=["theme", "window"],
                            values="n")
    order = ["基线", "桶宽 5min", "桶宽 30min", "p_age 60min", "p_age 240min",
             "p_age 无上限", "截断 [0.01,0.99]", "截断 [0.05,0.95]", "等权聚合",
             "关闭时点化准入", "剔除近结算"]
    piv_r = piv_r.reindex([v for v in order if v in piv_r.index])
    cols = [("oil_price", "gap"), ("oil_price", "night"),
            ("mideast_conflict", "gap"), ("mideast_conflict", "night")]
    head = ("<tr><th>变体</th><th>oil×SC gap</th><th>oil×SC night</th>"
            "<th>mideast×SC gap</th><th>mideast×SC night</th></tr>")
    rows = []
    for v in piv_r.index:
        tds = []
        for c in cols:
            r = piv_r.loc[v, c] if c in piv_r.columns else float("nan")
            n = piv_n.loc[v, c] if c in piv_n.columns else float("nan")
            tds.append(f"<td>{r:+.2f} <span class='n'>(n={int(n)})</span></td>"
                       if pd.notna(r) else "<td>—</td>")
        bold = ' class="base"' if v == "基线" else ""
        rows.append(f"<tr{bold}><td>{v}</td>{''.join(tds)}</tr>")
    return ('<div class="tablewrap"><table>' + head + "".join(rows)
            + "</table></div>")


def intl_table_html() -> str:
    tab = pd.read_parquet(DEEP / "intl_control.parquet")
    head = ("<tr><th>检验</th><th>主题 × 品种 · 窗口</th><th>基准</th><th>n</th>"
            "<th>t(信号) 无控制</th><th>t(信号) 控制后</th><th>t(基准)</th>"
            "<th>R² 无控制</th><th>R² 控制后</th></tr>")
    rows = []
    for r in tab.itertuples(index=False):
        if pd.isna(r.t_raw):
            continue
        die = abs(r.t_ctl) < 2 <= abs(r.t_raw)
        survive = abs(r.t_ctl) >= 2
        cls = ' class="die"' if die else (' class="live"' if survive else "")
        rows.append(
            f"<tr{cls}><td>{r.test}</td><td>{r.theme} × {r.product} · {r.window}"
            f"</td><td>{r.bench}</td><td>{r.n}</td><td>{r.t_raw:+.2f}</td>"
            f"<td><b>{r.t_ctl:+.2f}</b></td><td>{r.t_bench:+.2f}</td>"
            f"<td>{r.r2_raw:.2f}</td><td>{r.r2_ctl:.2f}</td></tr>")
    return ('<div class="tablewrap"><table>' + head + "".join(rows)
            + "</table></div>")


STYLE = """
:root {
  --paper: #FAF9F5; --ink: #22262C; --ink-soft: #5C6470; --line: #E4E1D8;
  --card: #FFFFFF; --accent: #A8430D; --accent2: #1B5CC4; --chip: #F1EEE6;
  --step: #FDF6EE;
}
@media (prefers-color-scheme: dark) { :root {
  --paper: #15181D; --ink: #E6E4DD; --ink-soft: #9BA3AE; --line: #2C313A;
  --card: #1C2027; --accent: #E08A4C; --accent2: #6EA3F0; --chip: #242931;
  --step: #241F19;
}}
:root[data-theme="dark"] {
  --paper: #15181D; --ink: #E6E4DD; --ink-soft: #9BA3AE; --line: #2C313A;
  --card: #1C2027; --accent: #E08A4C; --accent2: #6EA3F0; --chip: #242931;
  --step: #241F19;
}
:root[data-theme="light"] {
  --paper: #FAF9F5; --ink: #22262C; --ink-soft: #5C6470; --line: #E4E1D8;
  --card: #FFFFFF; --accent: #A8430D; --accent2: #1B5CC4; --chip: #F1EEE6;
  --step: #FDF6EE;
}
body { background: var(--paper); color: var(--ink);
  font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  line-height: 1.9; margin: 0; font-size: 16px; }
main { max-width: 78ch; margin: 0 auto; padding: 3rem 1.5rem 5rem; }
h1, h2, h3, h4 { font-family: "Songti SC", "Noto Serif CJK SC", Georgia, serif;
  text-wrap: balance; }
h1 { font-size: 1.9rem; line-height: 1.35; margin: .5rem 0 .8rem; }
h2 { font-size: 1.45rem; margin: 3.4rem 0 1rem; border-bottom: 2px solid var(--line);
  padding-bottom: .35rem; counter-increment: sec; }
h3 { font-size: 1.1rem; margin: 2.2rem 0 .6rem; color: var(--accent); }
h4 { font-size: 1rem; margin: 1.6rem 0 .4rem; }
.eyebrow { font-size: .78rem; letter-spacing: .14em; color: var(--accent);
  text-transform: uppercase; font-weight: 600; }
.abstract { background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; padding: 1.2rem 1.5rem; margin: 1.6rem 0;
  font-size: .96rem; }
.abstract b.hd { font-family: "Songti SC", Georgia, serif; }
.toc { background: var(--chip); border-radius: 8px; padding: 1rem 1.6rem;
  font-size: .9rem; column-count: 2; column-gap: 2rem; }
.toc a { color: var(--ink); text-decoration: none; }
.toc a:hover { color: var(--accent); }
figure { margin: 1.6rem 0; }
.figcard { background: #FFFFFF; border: 1px solid var(--line); border-radius: 6px;
  padding: .7rem; }
.figcard img { max-width: 100%; display: block; }
figcaption { font-size: .85rem; color: var(--ink-soft); margin-top: .55rem;
  line-height: 1.65; }
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px;
  margin: 1rem 0; }
table { border-collapse: collapse; width: 100%; font-size: .86rem;
  background: var(--card); }
th, td { padding: .45rem .75rem; text-align: right; border-bottom: 1px solid
  var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; white-space: normal; }
th { font-size: .75rem; letter-spacing: .05em; color: var(--ink-soft); }
td { font-variant-numeric: tabular-nums; }
td .n { color: var(--ink-soft); font-size: .78em; }
tr.base td { font-weight: 700; background: var(--chip); }
tr.die td { color: var(--ink-soft); }
tr.die td b { color: var(--accent2); }
tr.live td b { color: var(--accent); }
tr:last-child td { border-bottom: none; }
.step { background: var(--step); border: 1px solid var(--line); border-radius: 8px;
  padding: 1rem 1.3rem; margin: 1.2rem 0; }
.step .tag { display: inline-block; background: var(--accent); color: #fff;
  border-radius: 999px; padding: .05rem .7rem; font-size: .75rem; font-weight: 600;
  margin-bottom: .4rem; }
.why { border-left: 3px solid var(--accent2); padding: .15rem 0 .15rem .9rem;
  margin: .7rem 0; color: var(--ink-soft); font-size: .93rem; }
.example { background: var(--card); border: 1px dashed var(--accent);
  border-radius: 8px; padding: 1rem 1.3rem; margin: 1.2rem 0; font-size: .94rem; }
.example b.hd { color: var(--accent); }
.formula { background: var(--chip); border-radius: 6px; padding: .7rem 1rem;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .88rem;
  overflow-x: auto; margin: .7rem 0; line-height: 1.7; }
.texblock { background: var(--chip); border-radius: 6px; padding: .8rem 1rem;
  margin: .8rem 0; text-align: center; overflow-x: auto; color: var(--ink); }
.texblock svg { vertical-align: middle; max-width: 100%; }
.texi svg { vertical-align: -0.35em; }
.texnote { font-size: .85rem; color: var(--ink-soft); text-align: center;
  margin-top: -.4rem; margin-bottom: .8rem; }
code { background: var(--chip); border-radius: 4px; padding: .08rem .4rem;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .85em; }
.callout { border-left: 3px solid var(--accent); background: var(--card);
  padding: .9rem 1.2rem; border-radius: 0 6px 6px 0; margin: 1.3rem 0;
  font-size: .95rem; }
.callout.blue { border-left-color: var(--accent2); }
.finding { background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; padding: .9rem 1.2rem; margin: .8rem 0; }
.finding .no { color: var(--accent); font-weight: 700; margin-right: .5rem; }
ul, ol { padding-left: 1.4rem; } li { margin: .45rem 0; }
.refs { font-size: .88rem; } .refs li { margin: .6rem 0; }
.footer { margin-top: 3.5rem; padding-top: 1.2rem; border-top: 1px solid
  var(--line); font-size: .85rem; color: var(--ink-soft); }
"""


def build() -> str:
    ablation_html = ablation_table_html()
    intl_html = intl_table_html()

    return f"""<title>Polymarket 事件概率与中国商品期货：传导、吸收与升水回归</title>
<style>{STYLE}</style>
<main>
<div class="eyebrow">Alpha-Data 研究报告 · feat/cn-futures-polymarket · 2026-07-14</div>
<h1>Polymarket 事件概率与中国商品期货：<br>跨市场传导、时段吸收与升水回归</h1>

<div class="abstract"><b class="hd">摘要</b>　本文研究去中心化预测市场 Polymarket 的
事件概率变化与中国商品期货收益之间的关系。我们把 388 个事件市场的逐笔成交转换为
按国内期货交易时段切分的 logit 概率变化信号（防前视），与 88 个品种的主力连续
分钟数据在约 75 个重叠交易日（2026-01-05 至 04-28，美伊冲突主导期）上对齐。
五个主要发现：（一）信息吸收强烈且集中在"美国活跃时段对应的国内闭市段"——
油价类市场信号与 INE 原油（SC）开盘跳空的相关达 0.75-0.86；但安慰剂检验显示
原始吸收含有大量宏观共同因子成分（同一信号对机制无关的锰硅 +0.67、对股指
−0.57），因此（二）才是承重检验：控制同窗口国际基准（USO/GLD/SLV）后，
价格类市场的解释力全部消失（H1 偏零检验通过），<b>唯有中东冲突事件概率对
SC 保留独立增量（t=3.68）</b>，与 SC 可交割中东油种的供给风险敞口一致；
（三）"吸收后反转"只在油价主题成立（β₀=+507bp/σ，β₁₋₃≈−230..−330），
分解显示反转的约八成来自 SC−USO 价差分量——是<b>升水回归</b>而非全球过度
反应；（四）"新市场创建"是最强事件类型（2 小时符号化 CAR +41bp）；
（五）方向之外，<b>信号强度显著预测波动</b>：|s| 在控制昨日已实现波动率后
仍预测 SC 日盘 RV（t=4.4，R²=0.64）。全部设计经 23 个智能体对抗审查修复
17 项缺陷；消融（11 变体）与循环置换检验（p&lt;0.0005）支持核心结果非
设定依赖、非统计巧合。样本短且由单一事件主导，所有结论为探索性。</div>

<div class="toc"><b>目录</b><br>
<a href="#s1">1 引言</a><br><a href="#s2">2 制度背景</a><br>
<a href="#s3">3 数据</a><br><a href="#s4">4 信号构造（五步推导）</a><br>
<a href="#s5">5 计量工具</a><br><a href="#s6">6 吸收：设计与结果</a><br>
<a href="#s7">7 预测性与"吸收后反转"</a><br><a href="#s8">8 因子结构与叠加</a><br>
<a href="#s9">9 双向传导与分钟级剖面</a><br><a href="#s10">10 事件研究</a><br>
<a href="#s11">11 国际基准控制（核心检验）</a><br><a href="#s12">12 消融与扩充稳健性实验</a><br>
<a href="#s13">13 结论、局限与推广</a><br><a href="#refs">参考文献</a><br>
<a href="#appA">附录 A 窗口边界</a><br><a href="#appB">附录 B 术语表</a></div>

<h2 id="s1">1　引言</h2>
<p><b>动机。</b>Polymarket 是一个用真金白银给"事件会不会发生"定价的市场：
"美军是否在 4 月 30 日前进入伊朗"这样的合约以 0 到 1 美元之间的价格成交，
价格即市场隐含概率。当中东冲突升级时，这些概率在链上被全天候、逐秒地更新；
而中国的原油期货（上海国际能源交易中心 SC 合约）每天只交易日盘与夜盘两段。
一个自然的问题是：<b>预测市场聚合的事件信息，是否、何时、以何种方式进入国内
商品期货的价格？</b></p>
<p><b>为什么答案不显然。</b>国内期货与 Polymarket 之间隔着国际市场：WTI、Brent、
COMEX 全天候交易，任何"Polymarket 领先国内期货"的发现都可能只是"国际期货领先
国内期货"的转述（研究规范 §0 约束 1）。因此本文的核心检验不是"有没有相关"，
而是<b>控制国际基准之后还剩下什么</b>（第 11 节）。</p>
<p><b>贡献。</b>（1）一套经对抗审查的数据管线：国内期货主力连续分钟库（含夜盘
归属、换月处理）与 Polymarket 逐笔到国内时段信号的转换（防前视、去 bounce、
时点化准入）；（2）时段级传导结构的完整刻画（哪个闭市段吸收哪类信息）；
（3）"吸收后反转"的精确判据与机制判别（升水回归 vs 过度反应）；（4）三类
Polymarket 事件的定义及其事后响应度量。</p>

<h2 id="s2">2　制度背景</h2>
<h3>2.1 Polymarket 如何定价一个事件</h3>
<p>Polymarket 上每个二元市场发行 YES/NO 两种代币，结算时正确一方每份兑付
1 USDC。中央限价订单簿（CLOB）撮合，链上成交事件（OrderFilled）公开可查。
YES 价格 p ∈ (0,1) 即市场隐含概率。本文数据（daily_aligned 层）已把每笔成交
归一为"YES 视角"：<code>p_event</code> 为事件概率，<code>D ∈ {{+1,−1}}</code>
为 taker（主动成交方）方向，<code>usdc_amount</code> 为成交金额。</p>
<h3>2.2 国内商品期货的交易时段</h3>
<p>以 SC（INE 原油）为例：日盘 09:00-15:00（午间休市），夜盘 21:00-次日 02:30。
一个"交易日"由前一自然日晚间的夜盘和当日日盘构成——周一交易日的夜盘发生在
上周五晚上。56 个品种有夜盘（收盘 23:00 / 01:00 / 02:30 三档），广期所与
中金所品种无夜盘。</p>
<h3>2.3 时差：美国的白天是中国的夜里</h3>
<p>北京时间 = UTC+8（无夏令时）；美东与北京相差 13 小时（冬令时）或
12 小时（夏令时，2026-03-08 起）。这个错位决定了信息传导的结构：</p>
{fig_tag("f5_activity_clock.png",
         "登记市场的 Polymarket 成交额按北京时间小时分布。高峰在北京 21 点后至凌晨"
         "（美国白天），恰与国内夜盘（褐带）重叠；国内日盘（绿带）对应美国深夜，"
         "Polymarket 活动最少。")}
<p>据此把一个国内交易日切成四段（附录 A 给出精确边界）：<b>傍晚闭市</b>
（前日 15:00-21:00 = 美东约 02:00-08:00）、<b>夜盘</b>（21:00-02:30 = 美东约
08:00-13:30，含美股开盘）、<b>凌晨闭市</b>（02:30-09:00 = 美东约 13:30-20:00，
<b>美国下午的主活跃段</b>）、<b>日盘</b>（09:00-15:00 = 美国深夜）。</p>

<h2 id="s3">3　数据</h2>
<h3>3.1 Polymarket 逐笔（信号侧）</h3>
<p>HuggingFace <code>TimeSeventeen/Polymarket-v1</code> 的 daily_aligned 层：
已去中继腿的逐笔成交，覆盖 2022-11-21 至 2026-04-28。重叠窗口内共 703 万笔。
经事前注册的映射规则（八主题、slug 关键词、方向先验，v1.1）命中 388 个市场；
市场只有在窗口内累计成交额首次达到 $10 万之后才进入面板（<b>时点化准入</b>，
记 admit_ts——用全窗口流动性筛选会让 1 月的样本构成依赖 4 月才实现的成交，
属于用未来信息选样）。</p>
<h3>3.2 国内期货分钟数据（收益侧）</h3>
<p>聚宽风格主力连续（<code>XX9999.交易所</code>）分钟 bar，88 品种、125 个
交易日、350 万根。三个必须处理的源数据事实：（i）<b>夜盘 bar 存放在其开始时刻
次一自然日的文件里</b>——周六文件是周五夜盘，周一文件从不含夜盘，凭周一文件
抽查会误判"没有夜盘数据"；重建时按"夜盘归属下一交易日"从时间戳重新归属，
并以"周一交易日必须包含上周五 21:01 至周六 02:30 的 bar"作为测试断言。
（ii）主力合约切换发生在 21:01（交易日边界），故交易日内合约唯一；主力连续为
数据商预拼，换月日的跨合约收益（隔夜、收对收）是换月跳空而非可实现收益，
一律置缺失并打 <code>roll</code> 标记。（iii）成交量/额为逐分钟增量，
持仓量为水平值，时间戳为北京时间 bar 收盘戳。</p>
<h3>3.3 国际基准（控制变量）</h3>
<p>仓库美股分钟库的 ETF：USO（WTI 原油）对 SC、GLD（黄金）对 AU、SLV（白银）
对 AG。分钟 bar 含盘前盘后（美东 04:01-20:00），换算北京时间后恰好覆盖国内
闭市窗口，可以构造与信号窗口<b>完全同界</b>的基准收益。局限：USO 是 ETF
（含展期成本），非 Brent/WTI 期货本身；作日频/窗口相关控制足够。</p>

<h2 id="s4">4　信号构造：从一笔成交到一个信号（五步推导）</h2>

<div class="step"><span class="tag">第 1 步 · 15 分钟桶内去 bounce 聚合</span><br>
把逐笔按 15 分钟分桶（桶为左闭右开 [b−15′, b)，<b>标签取右端</b>——标签 T 的桶
只含严格早于 T 的成交，这是全文防前视的基石）。桶内按 taker 方向分成两堆，
各取成交额加权中位数，再取两向平均：
{tex(r"\bar p_b \;=\; \frac{{1}}{{2}}\left[\,\mathrm{{wmed}}(p \mid D{{=}}{{+}}1) \;+\; \mathrm{{wmed}}(p \mid D{{=}}{{-}}1)\,\right]")}
<div class="texnote">wmed = 成交额加权中位数；只有单向成交时取该向。</div>
<div class="why"><b>为什么</b>：主动买单贴卖一价（偏高）、主动卖单贴买一价（偏低），
逐笔价在买卖间来回跳并不代表概率变化（bid-ask bounce）。两向分别聚合再平均
≈ 中间价。加权中位数抗单笔异常。按量加权的思想来自 Roan 原文第四章的 VWAP
论证——真实价格要按成交量加权，不能只看最优报价。</div>
<b>微观例子</b>：桶内两笔——买 0.62（$100）、卖 0.60（$100）→ p̄ = 0.61，
而非"涨到 0.62 又跌回 0.60"。</div>

<div class="step"><span class="tag">第 2 步 · logit 变换</span>
{tex(r"\ell(p) \;=\; \ln\frac{{p}}{{1-p}}, \qquad p \in [0.02,\; 0.98]")}
<div class="texnote">p 超出区间先截断，再取 logit。</div>
<div class="why"><b>为什么不用概率差</b>：0.05→0.15 意味着事件相对可能性翻了三倍，
0.50→0.60 只是温和修正，但两者概率差同为 0.10。Roan 原文第二章用 Bregman/KL
散度论证同一件事：欧氏距离不适合概率空间，尾部变化携带更多信息。logit 恰好
实现这个几何：ℓ(0.05→0.15)=1.21 是 ℓ(0.50→0.60)=0.41 的三倍。<b>截断</b>因为
logit 在 0/1 发散（对应原文 §3.4 的边界梯度爆炸）：临近结算的市场价格贴近 0/1，
不截断则个别市场信号无穷大。消融（第 12 节）显示结论对截断位置不敏感。</div></div>

<div class="step"><span class="tag">第 3 步 · 窗口切分与端点差</span><br>
每个时段窗口 w = [t₀, t₁) 的信号是两端 logit 之差：
{tex(r"s_w \;=\; \ell(\bar p(t_1^-)) \;-\; \ell(\bar p(t_0^-)), \qquad w=[t_0,\; t_1)")}
端点价取该时刻前最后一个桶（LOCF）。防前视三细节：桶标签严格早于语义（第 1 步）；
恰在 t₁ 时刻的成交归下一窗口；端点距最后一笔成交超过 120 分钟（p_age）视为
过时，该窗口记缺失——LOCF 不允许无限延伸。市场结算之后的窗口记缺失
（结算后价格恒为 0/1，不是信号）。四段窗口信号可加：无缺失时
s_gap_pm + s_night + s_gap_am + s_day 恰等于两日收盘间的总 logit 变化。</div>

<div class="step"><span class="tag">第 4 步 · 方向统一</span><br>
每个市场事前注册方向 orientation ∈ {{+1, −1}}：升级类（us-strikes-iran）+1，
降级类（ceasefire）−1——于是"停火概率上升"贡献负信号，与"袭击概率上升"的
正信号同向可加。注意陷阱：<code>ceasefire-end</code>（停火结束）是升级事件，
v1.0 曾被 ceasefire 子串误伤反号，对抗审查修正（v1.1）。</div>

<div class="step"><span class="tag">第 5 步 · 主题聚合</span><br>
主题信号 = Σ orientation×√usdc×s / Σ √usdc（分母取权重绝对值），等权为
稳健性变体。准入前（admit_ts 之前）的市场信号不参与。</div>

<div class="example"><b class="hd">完整实例：2026-03-06 周末（全样本最大信号日）</b><br><br>
周五 15:00 SC 收盘后，周末美国时段中东局势升级。
<code>will-crude-oil-cl-hit-high-90-by-end-of-march</code>（3 月底前 WTI 上
$90）在该闭市窗口成交 3,632 笔、$149 万，价格 <b>0.65 → 0.997</b>：<br>
　段首 ℓ(0.65) = +0.62；段末 ℓ(0.98 截断) = +3.89；该市场 innovation = <b>+3.27</b><br>
油价主题加权后周一 s_gap = <b>+2.85</b>（全样本 σ=0.59，约 +4.8σ）。
周一 SC 从 666 涨到 771.8 元/桶（收对收 <b>+1492bp</b>）——吸收；
随后三个交易日累计 <b>−663bp</b>——反转。这两个词的统计定义见第 7 节。</div>

<h2 id="s5">5　计量工具（为什么用这三样）</h2>
<p><b>Newey-West（HAC）标准误</b>：h 天前向收益的相邻观测重叠 h−1 天，误差
自相关，普通标准误低估不确定性；HAC（滞后 ≥ h）修正之。小样本（n&lt;40）下
HAC 仍偏激进，表格作小样本标记。<b>Benjamini-Hochberg FDR</b>：68 项预测性
检验同时做，5% 显著水平下期望 3-4 个假阳性；把 p 值（用 HAC p，而非 iid
Pearson p——后者对重叠窗口反保守，v1.0 的错误之一）排序控制错误发现率，
报告 q 值。<b>局部投影</b>（Jordà 2005）：对每个视界单独回归，见第 7 节。</p>

<h2 id="s6">6　吸收：设计与结果</h2>
<h3>6.1 设计</h3>
<p>"吸收"回答：信号动的窗口内，期货是否同方向动。对每个（主题×品种），
把逐日窗口信号与<b>恰好同窗口</b>的期货收益配对回归：</p>
{tex(r"r_w(t) \;=\; \alpha \;+\; \beta_{{\mathrm{{abs}}}}\, s_w(t) \;+\; \varepsilon_t")}
<div class="texnote">Newey-West 标准误；换月日剔除。吸收 ≝ β<sub>abs</sub> &gt; 0 且显著——同期关系，不构成预测。</div>
<p>四段窗口各配各的收益：傍晚段配"前收盘→夜盘开"跳空、夜盘段配夜盘内收益、
凌晨段配"夜盘收→日盘开"跳空、日盘段配日盘内收益。</p>
<h3>6.2 结果：吸收集中在美国活跃时段对应的闭市段</h3>
{fig_tag("deep/a_fine_absorption.png",
         "细分时段同期吸收（Pearson r 与样本量）。读法：mideast×SC 最强的格子在"
         "凌晨闭市段（+0.70）——美东 13:30-20:00 的信息在 SC 次日 09:00 开盘定价；"
         "oil×SC 傍晚段 +0.86 为全表最强。us_china_trade×M 整行偏负，与注册先验"
         "反号（见第 13 节）。")}
{fig_tag("deep/a_weekday_vs_monday.png",
         "平日 vs 周一：周一交易日的闭市窗口含整个周末，周末积累的信号一并在"
         "周一实现，闭市段吸收普遍更强。")}
<h3>6.3 散点：不是少数极端日撑起来的</h3>
{fig_tag("deep/b_scatter_grid.png",
         "每点一个交易日，红线为 OLS。oil×SC 面板右上离群点即 2026-03-09；"
         "整体沿拟合线分布，相关并非单点驱动。")}
{fig_tag("deep/e_rolling_corr.png",
         "30 交易日滚动相关：吸收强度随事件热度时变，2-3 月冲突高峰最强。"
         "这提示结论的条件性——见第 13 节推广。")}

<h2 id="s7">7　预测性与"吸收后反转"</h2>
<h3>7.1 预测性检验（FDR 控制）</h3>
<p>开盘前信号（s_night+s_gap）对当日日盘、全日信号对未来 1/3/5 日累计收益，
68 项检验，BH-FDR q&lt;0.1 的 10 项<b>全部为负系数</b>（信号升→其后回落），
无一正向。且信号累积窗从 3 小时拉到 120 小时（下图），预测相关都在 ±0.17
内無稳定形态——<b>吸收发生在开盘瞬间，之后没有剩余可预测性</b>。</p>
{fig_tag("deep/e_horizon_curve.png",
         "窗口敏感性：截止 09:00 往回累积 K 小时的信号对当日日盘收益的相关，"
         "K 从 3h 到 120h。全程弱且无形态。")}
<h3>7.2 局部投影与"吸收后反转"的精确判据</h3>
{tex(r"r_{{t\to t+h}} \;=\; \alpha_h \;+\; \beta_h\, s_t \;+\; \varepsilon_{{t,h}}, \qquad h = 0,1,\dots,8")}
{tex(r"\beta_0 > 0 \;\;\wedge\;\; \beta_h < 0 \quad (h \geq 1)")}
<div class="texnote">同时满足上式即判定为"吸收后反转"。</div>
<div class="texnote">r<sub>t→t+h</sub> 为第 t+1 至 t+h 交易日累计对数收益（h=0 取当日）；每个 h 单独回归，HAC 滞后 ≥ h。</div>
<p>对每个视界单独回归（局部投影）的好处：不必假设统一的 AR 动力学，
β_h 路径直接可读——"信号高一个标准差的那天之后，价格平均往哪儿走"。</p>
{fig_tag("deep/h_irf.png",
         "β_h 路径（bp/1σ，90% CI）。oil×SC：β₀=+507 → β₁₋₃≈−230..−330，满足"
         "反转判据；mideast×SC 全程为正（动量/持续吸收）；trade×M 无同期吸收、"
         "单调走负（先验可能有误，不称为反转）。三个主题三种形态——把全部负"
         "系数笼统称作反转是不精确的。")}
<h3>7.3 用真实交易日看反转</h3>
<div class="tablewrap"><table>
<tr><th>交易日</th><th>油价主题 s_all</th><th>SC 当日</th><th>其后 3 日累计</th><th>回吐</th></tr>
<tr><td>2026-03-09（周末升级后的周一）</td><td>+2.83（+4.8σ）</td><td>+1492bp</td><td>−663bp</td><td>44%</td></tr>
<tr><td>2026-03-10（次日降温）</td><td>−1.34</td><td>−1470bp</td><td>+1194bp</td><td>81%</td></tr>
<tr><td>2026-04-08（停火进展）</td><td>−1.34</td><td>−1339bp</td><td>+579bp</td><td>43%</td></tr>
</table></div>
<p>反转是什么机制——过度反应还是升水回归？第 11 节 T3 给出判别。</p>

<h2 id="s8">8　因子结构与叠加</h2>
<h3>8.1 市场之间的相关结构（"同质化"有多严重）</h3>
<p>做法：25 个存续 ≥25 天的头部市场，逐日总 innovation（方向统一）构成
75×25 面板 → 两两相关 → 以 1−r 为距离、平均连接法层次聚类。PC1 占比 =
相关阵最大特征值 / 特征值和。</p>
{fig_tag("deep/c_market_clustermap.png",
         "市场级相关矩阵（聚类排序，左侧色块=主题）。对角块：us-strikes-iran / "
         "khamenei-out / regime-fall 各自的不同截止日版本抱团——Roan 原文第一章的"
         "蕴含约束 P(3月底前)≤P(4月底前) 决定同族市场必然联动，它们是一个事件的"
         "期限结构而非独立样本。跨主题相关明显更低。")}
<p><b>PC1 占比：市场级 24%、主题级 21%</b>——族内同质化真实存在，但整个信号
面板远非单一宏观因子（单因子情形 PC1 常见 60% 以上）。</p>
{fig_tag("deep/c_theme_corr.png", "主题级日频信号相关矩阵。")}
<h3>8.2 叠加：多主题一起放进回归，谁有独立贡献</h3>
<p>做法：SC 闭市收益同时对油价、中东、俄乌三主题回归；每主题的边际 R² =
联合 R² − 去掉它后的 R²。</p>
{fig_tag("deep/d_stacking.png",
         "联合 R²=0.59；油价主题边际 0.25（大头，但它本质是国际油价镜像），"
         "中东主题在控制油价市场后仍有边际 0.06（t=3.2），俄乌无增量。"
         "这一初步迹象在第 11 节被更严格的 USO 控制证实。")}

<h2 id="s9">9　双向传导与分钟级剖面</h2>
<p>反过来问：期货动了，Polymarket 跟不跟？转移表把"先行窗口→后续窗口"排成
矩阵，前两列是 PM→期货，后三列是期货→PM。</p>
{fig_tag("deep/f_reverse_direction.png",
         "全表最强的格子是反向：AU 期货日盘收益 → 当晚 PM 金价市场夜盘 innovation "
         "r=+0.52（p=0.001）。价格阶梯类市场（will-gold-hit-X）本质是给已发生的"
         "价格记账，是跟随者；事件类主题双向皆弱。")}
{fig_tag("f4_night_leadlag_sc.png",
         "SC 夜盘内 5 分钟步长交叉相关：峰值在滞后 ≤0，正滞后（信号领先）侧"
         "最大仅 +0.05——PM 信号领先期货不超过一个 5 分钟桶。负滞后侧偏高部分是"
         "机械原因：桶内加权中位数天然比最新成交慢约半桶。")}

<h2 id="s10">10　事件研究</h2>
<p>三类事件的精确定义（15 分钟桶 b 上，限 SC 相关主题）：</p>
{tex(r"\mathrm{{E1}}:\;\; |\Delta\ell_b| > 3 \times 1.4826 \cdot \mathrm{{MAD}}_{{48}}(\Delta\ell) \;\;\wedge\;\; \mathrm{{usdc}}_b \geq \mathrm{{10k}}")}
{tex(r"\mathrm{{E2}}:\;\; \mathrm{{usdc}}_b \;>\; 10 \times \mathrm{{med}}_{{48}}(\mathrm{{usdc}}) \;\;\wedge\;\; n_b \geq 20")}
<div class="texnote"><b>E3</b>：登记市场的第一笔成交（无公式，事件时刻 = first_ts）。</div>
<div class="texnote">MAD₄₈ = 过去 48 桶中位数绝对偏差；1.4826·MAD 是稳健 σ 估计。</div>
<p>对落在 SC 夜盘内的事件，取事件后 0-120 分钟 SC 累计收益，按
orientation×sign(Δℓ) 符号化后平均；置信带 = 500 次事件重抽自助法。</p>
{fig_tag("deep/g_event_study.png",
         "E3 新市场创建最强（n=241；120 分钟 +41bp，CI [28,56]）——新市场上线"
         "这件事本身携带方向信息：上什么行权价、开盘定多少概率，反映做市者对"
         "局势的判断。E1 价格跳仅 +3.6bp（信息几分钟内吸收完的旁证）；E2 纯放量"
         "不带方向，不显著。")}

<h2 id="s11">11　国际基准控制（本文的核心检验）</h2>
<h3>11.1 设计</h3>
<p>把与信号窗口<b>完全同界</b>的基准收益加入吸收回归：</p>
{tex(r"r_w(t) \;=\; \alpha \;+\; \beta_s\, s_w(t) \;+\; \beta_b\, b_w(t) \;+\; \varepsilon_t")}
<div class="texnote">b<sub>w</sub> = 同窗口国际基准（USO/GLD/SLV）对数收益，端点 LOCF 与信号同口径。问题：β<sub>s</sub> 控制 b<sub>w</sub> 后是否存活。</div>
<h3>11.2 结果</h3>
{intl_html}
<p>表格读法（蓝色 t = 控制后死亡，橙色 = 幸存）：</p>
<ul>
<li><b>H1 偏零检验通过</b>：metal_price×AU 夜盘 t 2.24→0.72，GLD 解释 R²=0.98
——金价类 PM 市场在 GLD 之外没有任何信息。研究规范把它设计成"若显著则先查
管线错误"的健全性检验，结果符合先验，也反证管线对齐正确。</li>
<li><b>价格类市场全灭</b>：oil_price×SC 夜盘 2.68→−1.07、gap 4.89→1.88——
"will-oil-hit-X"类市场就是 USO 的镜像，无增量。</li>
<li><b>中东事件概率幸存且增强：mideast×SC gap t 2.53→3.68</b>（R² 0.20→0.62，
USO 自身 t=5.64 同在）。控制了国际油价当期变动后，中东冲突概率仍解释 SC 开盘
跳空。经济解释：INE 原油的可交割油种是中东油（阿曼、巴士拉轻质等），中东供给
风险对 SC 交割篮子的冲击本就大于对美国轻质油 WTI——事件概率携带的是"油种
错位"的风险溢价信息。</li>
</ul>
{fig_tag("deep/i1_intl_control.png",
         "控制前（蓝）后（橙）的信号系数 t 值。多数配对控制后死亡；"
         "mideast×SC·gap 是唯一控制后反而增强的。")}
<h3>11.3 反转机制判别：升水回归，不是全球过度反应</h3>
<p>把 SC 收对收拆成两个可加分量，对每个分量分别做局部投影：</p>
{tex(r"r^{{\mathrm{{SC}}}}_t \;=\; b^{{\mathrm{{USO}}}}_t \;+\; q_t, \qquad q_t \;\equiv\; r^{{\mathrm{{SC}}}}_t - b^{{\mathrm{{USO}}}}_t")}
<div class="texnote">b = 国际（USO）分量，q = 价差 / 升水分量。</div>
{fig_tag("deep/i2_irf_decomposition.png",
         "oil_price 信号下三个分量的 β_h。当日吸收 +507 中 USO 分量占 +424（84%），"
         "但 h=1 的反转 −226 中价差分量贡献 −184（81%）且 USO 分量仅 −41——"
         "SC 事件日对国际锚的升水扩大，随后几天向锚收敛。反转是套利摩擦下的"
         "升水回归，不是全球市场的过度反应。")}
{fig_tag("deep/i3_premium_timeline.png",
         "事件高峰期（2026-02-16 至 03-31）SC、USO 与价差的累计走势：03-09 前"
         "价差（红线）随冲突升级扩大至约 +18%，其后一个月缓慢收敛——升水回归的"
         "直观呈现。")}
<div class="callout"><b>合并解读</b>：Polymarket 信号对 SC 的"当日强吸收"主要是
国际油价传导的转述（84%）；"其后反转"主要是 SC 升水的均值回归（81%）；
真正带独立信息的是<b>中东事件概率对开盘跳空的增量</b>（t=3.68）——它反映
SC 可交割中东油种的特有风险敞口，是本研究最值得继续追的线索。</div>

<h2 id="s12">12　消融与扩充稳健性实验</h2>
<h3>12.1 设计选择消融（OFAT）</h3>
<p>以基线（15 分钟桶、p_age 120 分钟、截断 [0.02,0.98]、√usdc 权重、时点化
准入开、保留近结算）为中心，<b>一次只改一个因子</b>，看两个头部吸收结论的
变化。若结论只在特定设定下成立即不可信。</p>
{ablation_html}
{fig_tag("deep/j_ablation.png",
         "消融点图（虚线=基线）。核心吸收对桶宽、p_age、截断、权重与准入选择"
         "均稳健；唯一敏感项是 p_age 无上限——无限 LOCF 把过时价格掺入信号，"
         "把 mideast 夜盘从 0.38 稀释到 0.20，反证时效过滤的必要性。")}

<h3>12.2 安慰剂检验与循环置换（结果是否品种特异、是否统计巧合）</h3>
<p><b>安慰剂设计</b>：把 SC 的信号原样作用在机制上无关的品种（JD 鸡蛋、C 玉米、
SM 锰硅、IF 股指）上，若"吸收"是油价特异的传导，安慰剂应接近零。
<b>结果并不干净</b>：oil 信号对锰硅 +0.67、对股指 −0.57，mideast 对玉米 +0.36。
这说明样本期的原始吸收含有大量<b>宏观共同因子</b>（风险开关：冲突升级 →
商品普涨、股指下跌）成分。诚实的推论是：未控共同因子的吸收系数不能解读为
品种特异的传导，<b>第 11 节的国际基准控制才是承重检验</b>——mideast×SC 的
增量（t=3.68）正是在剔除共同因子（USO）后幸存的部分。</p>
<p><b>循环置换检验</b>：把信号序列整体循环移位（保留各自的自相关结构、只破坏
两序列的日历同步）2,000 次，真实 |r| 在零分布中的位置给出经验 p 值：
oil×SC 真实 0.73 对零分布 99 分位 0.29（p&lt;0.0005）；mideast×SC 真实 0.44 对
0.39（p&lt;0.0005）。同步性本身不是统计巧合。</p>
{fig_tag("deep/k1_placebo_permutation.png",
         "(a) 真实配对（SC，橙）与四个安慰剂品种的闭市 gap 吸收：安慰剂不为零，"
         "揭示宏观共同因子；(b) oil×SC 的循环置换零分布与真实值（红线）。")}

<h3>12.3 信号分位数组合与正负不对称</h3>
<p><b>分位数组合</b>（回归之外的无参数检验）：按 s_gap 五分位分组，各组闭市
收益均值单调递增——关系不是由函数形式假设造出来的。<b>不对称</b>：mideast
主题降级日（s&lt;0）的单位定价强于升级日（663 vs 171 bp/单位 logit，
t=6.9 vs 5.8）——坏消息涨得多、好消息跌得更多在本样本呈现为"降温更被认真
定价"；oil 主题降级日样本不足（n=17）不可判。</p>
{fig_tag("deep/k2_quantile_asymmetry.png",
         "(a) s_gap 五分位组的 SC 闭市收益均值 ± 标准误：单调；"
         "(b) 升级日 vs 降级日分别估计的吸收 β 的 HAC t。")}

<h3>12.4 三变量分钟剖面：PM 对国际市场也无领先</h3>
<p>规范 §5.2 的三变量设计：美国活跃时段（北京 21:00-05:00）内，主题 5 分钟
innovation 对 SC 与对 USO 的交叉相关双剖面。若 Polymarket 真的先于价格市场
反映信息，对 USO 也应有正滞后领先。结果：对 USO 的正滞后相关 ≈ 0（±0.02 内），
对 SC 最大 +0.05——<b>PM、USO、SC 三者在 5 分钟粒度内同步</b>，信息链上
Polymarket 不领先国际市场，国内市场在开市时段也无显著滞后。</p>
{fig_tag("deep/k4_pm_uso_leadlag.png",
         "(a) oil_price、(b) mideast_conflict 主题 innovation 对 USO（橙）与 "
         "SC（蓝）的 ±60 分钟交叉相关：正滞后侧均无领先。")}

<h3>12.5 波动率预测：方向不可测，风险可测</h3>
<p>HAR-lite 回归：SC 日盘已实现波动率（5 分钟收益平方和的平方根）对隔夜信号
强度 |s_gap| 与昨日 RV：</p>
{tex(r"\mathrm{{RV}}^{{\mathrm{{day}}}}_t \;=\; \alpha \;+\; \gamma\, |s^{{\mathrm{{gap}}}}_t| \;+\; \rho\, \mathrm{{RV}}^{{\mathrm{{day}}}}_{{t-1}} \;+\; \varepsilon_t")}
<p>oil：γ 的 t=4.4（R²=0.64）；mideast：t=3.1（R²=0.42）。隔夜事件概率的
波动幅度在控制波动率惯性后仍预测当日风险——即便方向没有剩余可预测性
（§7.1），<b>风险维度是可预测的</b>，对保证金、期权与仓位管理有直接含义。</p>
{fig_tag("deep/k5_volatility.png",
         "|s_gap|（oil_price）与 SC 日盘已实现波动率：正相关，"
         "回归控制昨日 RV 后仍显著（t=4.4）。")}

<h2 id="s13">13　结论、局限与推广</h2>
<h3>13.1 结论（按证据强度排序）</h3>
<div class="finding"><span class="no">一</span>信息传导真实且时段结构清晰：
Polymarket 在国内闭市期间积累的信息在开盘跳空与夜盘中被系统性定价，最强通道
对应美国主活跃时段（凌晨闭市段 +0.70、傍晚段 +0.86）；置换检验排除统计巧合
（p&lt;0.0005）。但安慰剂显示原始吸收大半是宏观共同因子（对锰硅 +0.67、
股指 −0.57），品种特异的部分须由结论二的控制检验认定。</div>
<div class="finding"><span class="no">二</span>控制国际基准后，价格类市场无增量
（H1 通过），<b>中东事件概率对 SC 开盘跳空保留独立增量（t=3.68）</b>——
与 INE 可交割中东油种的供给风险敞口一致。这是本研究唯一在最严格检验下
幸存的"Polymarket 独有信息"证据。</div>
<div class="finding"><span class="no">三</span>"吸收后反转"（β₀&gt;0 且 β_h&lt;0）
只在油价主题成立，且分解显示反转的约八成来自 SC−USO 价差——是升水回归
而非过度反应。任何"做反转"的策略实质是在做 SC 升水的均值回归。</div>
<div class="finding"><span class="no">四</span>事件层面，"新市场创建"是最强
事件类型（120 分钟 +41bp）；价格跳本身几分钟内吸收完毕，无分钟级可利用领先。</div>
<div class="finding"><span class="no">五</span>反向传导不对称：价格阶梯类市场
跟随期货（AU→PM +0.52），事件类市场双向皆弱；三变量分钟剖面显示 PM 对 USO
也无领先——信息链上 Polymarket 与国际市场同步，而非领先者。</div>
<div class="finding"><span class="no">六</span>方向之外风险可测：|s_gap| 在控制
昨日 RV 后仍预测 SC 日盘已实现波动率（t=4.4，R²=0.64）——事件概率的"动静大小"
是干净的波动率信号，与方向预测力的缺失并行不悖。</div>
<h3>13.2 局限</h3>
<ul>
<li>约 75 个重叠交易日、单一美伊冲突主导；滚动相关显示吸收强度在事件热度
消退后减弱。所有结论是"在此情景内"的条件陈述。</li>
<li>USO 是 ETF 而非期货本身；日盘时段无美股 bar，日盘窗口的控制缺失。</li>
<li>主力连续为数据商预拼；方向先验为判断性注册（两处违背如实报告：
mideast×AU 避险失效、trade×M 反号）。</li>
<li>多处小样本（n=24-43）HAC 显著性需打折；聚合权重含轻微样本内信息
（等权变体一致）。</li>
</ul>
<h3>13.3 推广（什么条件下结论可外推，下一步需要什么）</h3>
<ul>
<li><b>外推条件</b>：结论二（中东增量）依赖"事件直接冲击品种的交割篮子/供给
结构而国际基准不完全反映"这一机制——推广到其他品种需先论证同类错位存在
（例：对华关税 vs 大连豆粕的进口结构）。结论一（时段传导结构）由时差决定，
可直接外推到任何品种。</li>
<li><b>时间外推</b>：把期货历史向前扩展（聚宽同源数据可回溯多年）覆盖
2022-2025 的俄乌开战、2024 大选等多个独立事件，检验中东增量是否在其他
episode 复现——这是把"探索性"升级为"结论"的必要条件。</li>
<li><b>数据升级</b>：SII 独立采集数据集交叉验证（规范 §1B）；Brent 期货分钟数据
替代 USO；L2 盘口数据用 mid 价消除微观结构噪声。</li>
<li><b>策略含义（如果结论二稳健）</b>：可检验的交易假设是"中东事件概率隔夜
大幅变动 × SC 开盘跳空未充分反映时做跳空方向"，但当前样本不足以支持任何
实盘决策。</li>
</ul>

<h2 id="refs">参考文献</h2>
<ol class="refs">
<li>Roan (@RohOnChain). <i>The Math Needed for Trading on Polymarket (Complete
Roadmap)</i>. https://x.com/RohOnChain/status/2017314080395296995 。中文编译版
（MrRyanChi / ChainCatcher）见仓库 docs/。本文借用其 §2 KL 几何（logit 坐标）、
§3.4 边界发散（截断）、§1 蕴含约束（聚类解读）、§4.2 VWAP（去 bounce 聚合）、
§5.1 事件分类（E3）；未使用其站内套利机器（Bregman 投影 / Frank-Wolfe / 整数规划）。</li>
<li>Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local
Projections. <i>American Economic Review</i>, 95(1).</li>
<li>Newey, W. &amp; West, K. (1987). A Simple, Positive Semi-definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
<i>Econometrica</i>, 55(3).</li>
<li>Benjamini, Y. &amp; Hochberg, Y. (1995). Controlling the False Discovery
Rate. <i>JRSS-B</i>, 57(1).</li>
<li>研究规范 v3.2《Polymarket 隐含概率对国内商品期货收益的增量预测力研究》；
仓库文档 <code>docs/DATA_GUIDE.md</code>、<code>docs/mapping_taxonomy.md</code>（v1.1）。</li>
</ol>

<h2 id="appA">附录 A　窗口边界（以 SC 为例，夜盘收盘 02:30）</h2>
<div class="tablewrap"><table>
<tr><th>窗口</th><th>北京时间（交易日 t）</th><th>对应美东</th><th>信号列</th><th>匹配收益</th></tr>
<tr><td>傍晚闭市 gap_pm</td><td>t−1 日 15:00 → 21:00</td><td>约 02:00-08:00</td><td>s_gap_pm</td><td>r_gap_pm = ln(夜盘开/前日收)</td></tr>
<tr><td>夜盘 night</td><td>t−1 日 21:00 → t 日 02:30</td><td>约 08:00-13:30</td><td>s_night</td><td>r_night = ln(夜盘收/夜盘开)</td></tr>
<tr><td>凌晨闭市 gap_am</td><td>t 日 02:30 → 09:00</td><td>约 13:30-20:00</td><td>s_gap_am</td><td>r_gap_am = ln(日盘开/夜盘收)</td></tr>
<tr><td>日盘 day</td><td>t 日 09:00 → 15:00</td><td>约 20:00-02:00</td><td>s_day</td><td>r_day = ln(日盘收/日盘开)</td></tr>
</table></div>
<p>周一交易日的 t−1 是上周五：夜盘发生在周五晚，gap 覆盖整个周末。换月日
（主力合约切换）的跨合约收益（r_gap_pm、收对收）置缺失。无夜盘品种：
night/gap_am 无定义，闭市整段计入 gap_pm。</p>

<h2 id="appB">附录 B　术语表</h2>
<div class="tablewrap"><table>
<tr><th>术语</th><th>含义</th></tr>
<tr><td>logit / ℓ(p)</td><td>ln(p/(1−p))，把概率映到 (−∞,+∞) 的对数赔率</td></tr>
<tr><td>innovation</td><td>窗口两端 LOCF 聚合价的 logit 之差，即该窗口的信息增量</td></tr>
<tr><td>LOCF</td><td>last observation carried forward，取时点前最后一个观测</td></tr>
<tr><td>p_age</td><td>端点时刻距最后一笔成交的分钟数；超过阈值视为价格过时</td></tr>
<tr><td>bid-ask bounce</td><td>成交价在买一/卖一间跳动造成的伪波动</td></tr>
<tr><td>orientation / σ</td><td>市场方向先验：事件概率上升利多(+1)或利空(−1)目标品种</td></tr>
<tr><td>时点化准入 admit_ts</td><td>市场累计成交额首次达标的时刻；此前信号不入面板</td></tr>
<tr><td>吸收 β_abs</td><td>同窗口回归系数显著为正</td></tr>
<tr><td>局部投影 / β_h</td><td>对每个视界 h 单独回归未来累计收益于当期信号</td></tr>
<tr><td>吸收后反转</td><td>β₀&gt;0 且 β_h&lt;0 (h≥1)</td></tr>
<tr><td>HAC / Newey-West</td><td>对自相关与异方差稳健的标准误</td></tr>
<tr><td>BH-FDR / q 值</td><td>多重检验下控制错误发现率后的校正显著性</td></tr>
<tr><td>升水回归</td><td>SC 相对国际锚（USO）的价差扩大后向均值收敛</td></tr>
<tr><td>换月 roll</td><td>主力合约切换；跨合约收益是拼接假象，剔除</td></tr>
</table></div>

<div class="footer">复现：<code>build_cn_futures_db.py → select_polymarket_markets.py →
analyze_cn_futures_polymarket.py → deep_analysis_cn_polymarket.py →
intl_benchmark_analysis.py → ablation_cn_polymarket.py → build_thesis_html.py</code>
· 55+ 项 pytest · 管线经 23 智能体对抗审查（17 项缺陷修复）· 分支
feat/cn-futures-polymarket</div>
</main>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="生成论文级 HTML 报告")
    parser.add_argument(
        "--out", default=str(ROOT / "docs" / "cn_futures_polymarket_report.html"))
    args = parser.parse_args()
    html = build()
    Path(args.out).write_text(html)
    print(f"written {args.out} ({len(html) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
