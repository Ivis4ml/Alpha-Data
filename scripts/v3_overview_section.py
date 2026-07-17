"""主报告总览篇（第一眼层）：框架、方法、结论总表与导航。

置于文档最前，供评审在数分钟内看到全貌；每条结论与方法都内链到
细节篇（现有第一、二、三部分）的对应节。数字与细节篇正文同源
（引用已定稿的构建产物数字，细节篇为唯一权威出处）。

由 ``build_thesis_html.py`` 导入调用；不独立运行。
"""

from __future__ import annotations

OVERVIEW_STYLE = """
.ov-grade{display:inline-block;min-width:2.2em;text-align:center;
  border-radius:4px;padding:0 .35em;font-weight:700;color:#fff}
.gA{background:#2e7d32}.gB{background:#e08a00}
.gC{background:#607080}.gD{background:#b03a2e}
table.wraptext th, table.wraptext td{white-space:normal}
.ovnav td:first-child{white-space:nowrap}
@media print{
  .part{page-break-before:always}
  h2{page-break-before:always}
  h1,.abstract{page-break-before:avoid}
  tr,figure,.finding{page-break-inside:avoid}
  .toc{display:none}
  .tablewrap{overflow:visible;border:none}
  .tablewrap table{font-size:.72em}
  .tablewrap th,.tablewrap td{white-space:normal}
  img{max-width:100% !important}
}
"""


def build_overview() -> str:
    """总览篇 HTML（§O.1-O.6）。"""
    return """
<div class="part ov-first" id="overview"><div class="kicker">总览篇 · 第一眼层</div>
<div class="pt">框架、方法、结论总表与勘误记录</div>
<p>本篇是写给评审的入口：研究问题、数据量级、三层方法框架、按证据强度
分级的<b>结论总表</b>、以及我们自己审计出并公开修正的<b>勘误记录</b>。
每一条都链接到细节篇（第一、二、三部分）的对应节——总览只陈述，
证据、协议与全部边界情形在链接处。</p></div>

<h2 id="ov1" style="page-break-before:avoid">O.1　研究问题与总答案</h2>
<p><b>问题</b>：Polymarket（链上真金定价的事件预测市场）的概率变化，对
中国商品期货收益有没有可检验的信息含量？如果有，在哪个时间尺度上、
以什么形态存在？</p>
<p><b>总答案（三句话）</b>：</p>
<ol>
<li><b>信息真实存在，但不是隔日方向 alpha。</b>两套信号对次日方向的
样本外增量整体不为正（<a href="#s17">§17.2</a>）；波动预测样本内显著、
样本外无增量（<a href="#s12">§12.5b</a>）。</li>
<li><b>信息的变现形态是"闭市吸收 + 分钟级缺口"。</b>国内闭市期间
积累的事件概率变化在开盘跳空中被系统性定价（<a href="#s6">§6</a>），
控制同窗国际基准后中东事件概率对 SC 仍保留独立增量
（t=3.68，扩展样本 4.94，<a href="#s11">§11</a>/<a href="#s17">§17.3</a>）
且冲突期外更强（<a href="#s17">§17.5</a>）；分钟粒度上"信念已动而价格
未动"的缺口结构（C8）跨品种 RankIC 全正、ICIR 0.5-0.6
（<a href="#ms5">第三部分 §5</a>），而同一结构在日频窗口完全消失
（<a href="#s12">§12.9</a>）——信息的时间尺度被两级检验精确定位在
分钟到开盘之间。</li>
<li><b>组合不放大信息。</b>"两个事件同时发生"经连续交互（t=−2.88，
信息冗余，<a href="#s8">§8.3</a>）、离散共现（<a href="#s12">§12.10</a>）
与分钟级 X 族（<a href="#ms5">第三部分 §5</a>）三级检验，一致指向
冗余而非超加性；13 个窗口级组合中仅铁矿石上两项通过全部纪律，
定位为候选发现（<a href="#s12">§12.9</a>）。</li>
</ol>

<h2 id="ov2">O.2　数据一览</h2>
<div class="tablewrap"><table>
<tr><th>数据层</th><th>量级</th><th>窗口</th><th>细节</th></tr>
<tr><td><b>Polymarket 全量链上成交 tape</b><br>（HF 公开集 + 自建爬虫扩展，
同构拼接）</td><td>855,614,453 笔 / 约 265 亿美元 / 约 69 万个市场</td>
<td>2022-11-21 至 2026-07-14</td>
<td><a href="#ms1">第三部分 §1</a>（逐行定义、真实示例、22 条 edge
cases）、<a href="#s14">§14</a>（拼接与结算补爬）</td></tr>
<tr><td><b>事前注册的中国相关主题子集</b></td>
<td>8 主题 / 492 市场（v1.1 为 388）；结算三元组 455/492</td>
<td>同上</td><td><a href="#s3">§3.1</a>、<a href="#s14">§14.2</a></td></tr>
<tr><td><b>国内期货分钟库</b>（聚宽风格主力连续）</td>
<td>88 品种 / 约 350 万根 1 分钟 bar；主评估品种 SC/AU/AG/CU/M（+I/IF/CF）</td>
<td>2026-01-05 至 07-13，与信号重叠 125 个交易日</td>
<td><a href="#s3">§3.2</a>、<a href="#s3">§3.4</a>（edge cases 与质检表）</td></tr>
<tr><td><b>国际基准</b>（承重控制变量）</td>
<td>USO/GLD/SLV 等 ETF 分钟 bar（含盘前盘后，恰覆盖国内闭市窗口）</td>
<td>同期</td><td><a href="#s3">§3.3</a>、<a href="#s11">§11.1</a></td></tr>
</table></div>

<h2 id="ov3">O.3　方法框架：三个粒度层层推进</h2>
<div class="tablewrap"><table class="wraptext">
<tr><th>层</th><th>回答的问题</th><th>核心方法</th><th>关键防御</th></tr>
<tr><td><b>第一部分<br>窗口级（v1.1）</b><br><a href="#s1">§1-13</a></td>
<td>传导结构是什么：信息何时、经何通道进入国内价格</td>
<td>逐笔 → 15 分钟桶去 bounce → logit 端点差 → 主题聚合 → 按交易时段
窗口（夜盘/缺口/日盘）与同界收益配对；HAC + BH-FDR + 置换 + 安慰剂 +
消融 + 国际基准控制 + 中介链/代理充分性</td>
<td>时点化准入、p_age 过时屏蔽、方向预注册、换月剔除
（<a href="#s3">§3.4</a> 清单）</td></tr>
<tr><td><b>第二部分<br>识别与功效（v3.1）</b><br><a href="#s14">§14-18</a></td>
<td>哪些效应在统计上可识别：样本单位、功效、样本外</td>
<td>Kalman 测量层（point-in-time 冻结参数）→ episode 聚类（样本单位 =
品种×下一可交易日）→ 双门控功效核算 → 展开窗 OOS + Clark-West +
块自助</td>
<td>一次真实的证伪：初版"β 可识别"被 episode 聚类揭示为 92% 伪重复，
正式撤回（<a href="#s16">§16</a>）</td></tr>
<tr><td><b>第三部分<br>分钟级（答辩篇）</b><br><a href="#part3">篇首</a></td>
<td>信息以什么形态渗入价格：40 个信号的显式检验</td>
<td>分钟信念创新 + 量价组合 + 离散统计 + 条件组合四族；滚动 z 归一化；
IC/RankIC/ICIR；双基线事件研究；时段分层；双标签稳健性</td>
<td>1,128 格检验总量核算、三类一致性纪律、冲突段内外同号检验
（<a href="#ms43">第三部分 §4</a>）</td></tr>
</table></div>
<p><b>贯穿全文的研究纪律</b>：（i）全链防前视（桶右端标签、展开窗归一、
参数时点冻结、admit 时点化）；（ii）多重检验总账——窗口级约 750 个检验
单元（<a href="#s12">§12.11</a>）+ 分钟级 1,128 格（<a href="#ms6">第三
部分 §6</a>），期望假阳性明示；（iii）声称门槛 = 跨设计一致性而非单格
p 值；（iv）对抗审查文化——四轮共 100+ 项发现，全部修复或声明，实质
勘误公开（O.5）。</p>

<h2 id="ov4">O.4　结论总表（按证据强度分级）</h2>
<p>分级标准：<span class="ov-grade gA">A</span> 稳健（跨检验层级、跨样本
一致）；<span class="ov-grade gB">B</span> 候选（单一品种或单一通道，
待续期样本）；<span class="ov-grade gC">C</span> 诚实负结果（有信息量的
"不存在"）；<span class="ov-grade gD">D</span> 已撤回 / 勘误（原结论被
自己的审计推翻）。</p>
<div class="tablewrap"><table class="wraptext">
<tr><th>级</th><th>结论</th><th>关键证据</th><th>详见</th></tr>
<tr><td><span class="ov-grade gA">A1</span></td>
<td><b>时段吸收结构</b>：闭市积累的事件信息在开盘跳空/夜盘被系统性定价，
最强通道对应美国活跃时段</td>
<td>oil×SC 傍晚段 r=0.86；置换 p&lt;0.0005；冲突期外同号更强
（0.68 vs 0.35）；RankIC 与 Pearson 一致</td>
<td><a href="#s6">§6</a>、<a href="#s7">§7.4</a>、<a href="#s17">§17.5</a></td></tr>
<tr><td><span class="ov-grade gA">A2</span></td>
<td><b>中东事件概率对 SC 的独立增量</b>（控制国际基准后幸存的
"Polymarket 独有信息"）</td>
<td>控 USO 后 t=3.68 → 扩展样本 4.94；12 资产全谱下 SC/IF 两侧幸存；
限定：控制集为 ETF 代理</td>
<td><a href="#s11">§11</a>、<a href="#s12">§12.8</a>、<a href="#s17">§17.3</a></td></tr>
<tr><td><span class="ov-grade gA">A3</span></td>
<td><b>反转 = 升水回归</b>：吸收后反转的约八成来自 SC−USO 价差分量，
不是全球过度反应</td>
<td>IRF 分解：β₀ +507 中 USO 占 84%，反转 −226 中价差占 81%</td>
<td><a href="#s11">§11.3</a></td></tr>
<tr><td><span class="ov-grade gA">A4</span></td>
<td><b>分钟级未兑现缺口（C8）是唯一跨品种稳健的可预测结构</b>，
且该结构在日频窗口消失——信息时间尺度被定位在分钟到开盘之间</td>
<td>C8 六视界 RankIC 全正（SC/AU/CU），ICIR 0.5-0.6，冲突段内外同号；
窗口级对应物 W9/W11 全品种无信号</td>
<td><a href="#ms5">第三部分 §5</a>、<a href="#s12">§12.9(iii)</a></td></tr>
<tr><td><span class="ov-grade gA">A5</span></td>
<td><b>反向传导不对称</b>：价格阶梯类 PM 市场跟随期货，事件类双向皆弱
——Polymarket 与国际市场同步而非领先</td>
<td>AU→PM r=+0.52；PM 对 USO 亦无领先</td>
<td><a href="#s9">§9</a>、<a href="#s12">§12.4</a></td></tr>
<tr><td><span class="ov-grade gB">B1</span></td>
<td>豆粕 M 的正 OOS 增量（"少数事件日定价滞后"候选）</td>
<td>+4.9%，CW t=1.86，块自助 p≈0.001；但 BH-FDR q=0.12、删 3 个
影响日转负、缺豆粕期货/USDCNH 控制</td>
<td><a href="#s17">§17.2</a></td></tr>
<tr><td><span class="ov-grade gB">B2</span></td>
<td>铁矿石组合 W8（贸易信念×棉花夜盘确认）与 W10（符号一致性）</td>
<td>t=5.6 / 4.0，族内 FDR 存活且两个归因对照单独均不显著；但单品种、
n≈60、3 个月估计</td>
<td><a href="#s12">§12.9</a></td></tr>
<tr><td><span class="ov-grade gB">B3</span></td>
<td>tension（族内定价失衡）：负波动关联 + 高张力下吸收变弱</td>
<td>SC t=−4.40 经活动度控制不变；AU 门控交互 t=−2.76（探索）</td>
<td><a href="#s17">§17.4 / §17.6</a></td></tr>
<tr><td><span class="ov-grade gC">C1</span></td>
<td>隔日方向无剩余预测力：吸收发生在开盘瞬间</td>
<td>68 项预测检验 FDR 后全负；OOS 增量整体不为正；时段拆分显示夜盘
自身是吸收者</td>
<td><a href="#s7">§7</a>、<a href="#s17">§17.2</a></td></tr>
<tr><td><span class="ov-grade gC">C2</span></td>
<td>波动通道样本外无增量（样本内 t=4.4 不外推）</td>
<td>展开窗 OOS：ΔR²≈0、CW t≈0</td>
<td><a href="#s12">§12.5b</a></td></tr>
<tr><td><span class="ov-grade gC">C3</span></td>
<td>组合不放大信息：交互为负（冗余）、共现无增强、确认型组合一致为负</td>
<td>中东×油价交互 t=−2.88；离散共现 V1-V4 无超额；X 族分钟级为负</td>
<td><a href="#s8">§8.3</a>、<a href="#s12">§12.10</a>、
<a href="#ms5">第三部分 §5</a></td></tr>
<tr><td><span class="ov-grade gC">C4</span></td>
<td>W12 量能交互是"线性假象"教科书案例：q 值极小但秩相关为零、
跨品种符号翻转——按一致性纪律登记为反例</td>
<td>AU t=+5.9 而 RankIC≈0.00；CU t=−8.4</td>
<td><a href="#s12">§12.9(i)</a></td></tr>
<tr><td><span class="ov-grade gD">D1</span></td>
<td>「新市场创建是最强事件类型（+41bp）」<b>已撤回</b></td>
<td>视界标注错误（25 根 1min bar 标为 120 分钟）+ E3 符号未实现 +
无基线；修正版 CAR 区间含零且翻负</td>
<td><a href="#s10">§10b</a></td></tr>
<tr><td><span class="ov-grade gD">D2</span></td>
<td>「影响系数 β 可识别（t&gt;4）」<b>已撤回</b></td>
<td>episode 聚类：识别变差的 92% 是伪重复（191 市场事件 = 72 个独立
收益日）</td>
<td><a href="#s16">§16</a></td></tr>
<tr><td><span class="ov-grade gD">D3</span></td>
<td>S5 事件策略存在前视（事件前一分钟进场），修正后 Sharpe 降至约 1.0
（CI 含 0）</td>
<td>持有期实现 25 bar ≠ 声明 120 分钟，一并修正重跑</td>
<td><a href="#s12">§12.6b</a></td></tr>
<tr><td><span class="ov-grade gD">D4</span></td>
<td>「价格类全灭 / 唯有中东幸存」收紧为限定表述</td>
<td>metal×AG、fed×AU 边际幸存被自表格证伪；扩展样本上 oil×SC·gap
复活（t=3.68）</td>
<td><a href="#s11">§11.2</a>、<a href="#s17">§17.3</a></td></tr>
</table></div>

<h2 id="ov5">O.5　审计与勘误记录（可信度的来源）</h2>
<p>本报告的每一版都经过系统性对抗审查；实质勘误不掩盖、不静默修改，
在正文原位以「勘误」标注并保留修正前后对照：</p>
<div class="tablewrap"><table class="wraptext">
<tr><th>轮次</th><th>范围</th><th>结果</th></tr>
<tr><td>23 智能体对抗审查（v1.1）</td><td>第一部分管线</td>
<td>17 项缺陷修复（含 ceasefire-end 方向反号、iid p 值误用）</td></tr>
<tr><td>外部方法论评审（v3.1）</td><td>第二部分</td>
<td>episode 伪重复揭示 → D2 撤回；point-in-time 参数冻结；ex-ante
门控（<a href="#s16">§16</a>）</td></tr>
<tr><td>5 视角对抗审查（答辩篇）</td><td>第三部分</td>
<td>71 项发现修复（基线分母、跨段口径、权重时点化）</td></tr>
<tr><td>四视角答辩标准审计（本轮）</td><td>第一、二部分反哺</td>
<td>43 项发现：D1/D3/D4 勘误、统计基本功补齐（§3.4/§4.6/§7.4/§12.11）、
组合族首次实现（§8.3/§12.9-12.10）、冲突期分解（§17.5）</td></tr>
</table></div>

<h2 id="ov6">O.6　细节篇导航</h2>
<div class="tablewrap"><table class="ovnav">
<tr><th>想核对什么</th><th>去哪里</th></tr>
<tr><td>数据逐行定义、真实成交示例、edge cases</td>
<td><a href="#ms1">第三部分 §1</a>（链上数据）、<a href="#s3">§3.4</a>
（期货与映射）、<a href="#s14">§14</a>（tape 拼接）</td></tr>
<tr><td>信号的显式公式与归一化</td>
<td><a href="#s4">§4</a>（窗口级五步）、<a href="#ms2">第三部分 §2-3</a>
（40 个分钟级信号）、<a href="#s12">§12.9</a>（13 个组合）</td></tr>
<tr><td>统计基本功：频率 / 分布 / IC / ICIR</td>
<td><a href="#s4">§4.6</a>、<a href="#s7">§7.4</a>、
<a href="#ms43">第三部分 §4</a>、<a href="#s15">§15.4</a></td></tr>
<tr><td>多重检验总账与解读纪律</td>
<td><a href="#s12">§12.11</a>、<a href="#ms6">第三部分 §6</a></td></tr>
<tr><td>识别、功效与样本外协议</td>
<td><a href="#s15">§15</a>-<a href="#s17">§17</a></td></tr>
<tr><td>可交易性（成本 / 换手 / 容量 / 杠杆）</td>
<td><a href="#s12">§12.6 / §12.6b</a></td></tr>
<tr><td>写给期货研究员的 Polymarket 入门与 FAQ</td>
<td><a href="#mappA">第三部分附录 A</a>、<a href="#mappB">附录 B（FAQ
24 问）</a></td></tr>
<tr><td>复现命令链与产物索引</td>
<td>页脚、<a href="#appC">附录 C</a>、<a href="#mappE">第三部分附录 E</a></td></tr>
</table></div>

<div class="part"><div class="kicker">细节篇</div>
<div class="pt">证据、协议与全部边界情形</div>
<p>以下为完整研究记录（第一部分：窗口级传导；第二部分：识别与统计功效；
第三部分：分钟级信号检验），供逐条核对总览篇的每一句话。各部分自成
体系，可按 O.6 的导航直达。</p></div>
"""
