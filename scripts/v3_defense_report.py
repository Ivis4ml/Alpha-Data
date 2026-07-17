"""生成答辩文档 docs/v3_signal_defense.html（自包含：图 base64、公式 SVG）。

内容按答辩要求组织：数据问答（定义 / 时间戳 / 延迟 / 主动方向 / 统计量 /
期货分钟与时段处理）、30 个信号的公式与合理性、频率与取值分布、
数值信号 IC / ICIR、离散信号事件研究（对基线）、2022 至今逐月频率。

依赖 ``scripts/v3_defense_build.py`` 的产物。

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
    PRODUCT_THEMES,
)

D = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
OUT = ROOT / "docs" / "v3_signal_defense.html"
P = pub_style.PALETTE


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


#: 信号公式与一句话依据（渲染层；口径与 v3_defense_build.add_signals 一致）。
SIGNAL_DEFS: list[tuple[str, str, str, str]] = [
    ("N1", r"N1_t = \sum_m \frac{w_m\,[\ell(p_{m,t}) - \ell(p_{m,t^-})]}"
           r"{\sum_m |w_m|},\;\; w_m = \sigma_m\sqrt{\mathrm{USDC}_m}",
     "主题 1 分钟 logit 创新（orientation 加权）", "事件信念的最小时间单位增量"),
    ("N2", r"N2_t = \sum_{u=t-14}^{t} N1_u",
     "15 分钟累积创新", "与既有窗口研究可比的中尺度"),
    ("N3", r"N3_t = z\left(\sum_{u=t-14}^{t}\sum_m \sigma_m D\,\mathrm{USDC}"
           r"\right)", "15 分钟签名主动流 z", "价格外的用钱投票强度"),
    ("N4", r"N4_t = z(N2_t)", "创新 z 分数（4800 分钟窗）",
     "答辩要求的时序归一化基准形式"),
    ("N5", r"N5_t = \frac{\sum \sigma_m D\,\mathrm{USDC}}"
           r"{\sum \mathrm{USDC}}\in[-1,1]", "15 分钟流不平衡比",
     "自带归一的方向占比，对量纲稳健"),
    ("N6", r"N6_t = z\left(\log(1+n^{15\mathrm{min}}_t)\right)",
     "成交强度 z", "关注度/信息到达率的代理"),
    ("N7", r"N7_t = z\left(\sum_{\theta}\sum_{u=t-14}^{t} N1^{\theta}_u\right)",
     "多主题复合创新 z", "品种映射的全部主题合并，降低单主题噪声"),
    ("N8", r"N8_t = z\left(\overline{n^{\mathrm{mkts}}}^{15}_t\right)",
     "活跃市场数 z", "族内广度：多少个市场同时在动"),
    ("N9", r"N9_t = z\left(\mathrm{std}_{60}(N1)\right)",
     "信念波动 z", "事件不确定性的强度而非方向"),
    ("N10", r"N10_t = z\left(\sum_{u=t-14}^{t}|N1_u|\right)",
     "绝对创新和 z", "无方向的信息流量"),
    ("C1", r"C1_t = N4_t \times z(\mathrm{mom}^{15}_t)",
     "事件×动量确认", "两源同向时更可信"),
    ("C2", r"C2_t = N4_t \times \mathbf{1}\{|z(\mathrm{mom}^{15}_t)|<0.5\}",
     "事件动而价未动", "未兑现信息假说：期货尚未反应"),
    ("C3", r"C3_t = N4_t \,/\, (1 + z(\mathrm{RV}^{15}_t)_+)",
     "波动折减事件强度", "高波动时单位创新的信息含量更低"),
    ("C4", r"C4_t = N4_t \times z(\mathrm{Vol}^{15}_t)",
     "事件×放量确认", "期货放量佐证信息到达"),
    ("C5", r"C5_t = N3_t \times \mathbf{1}\{N3_t\cdot z(\mathrm{mom}^{15}_t)>0\}",
     "签名流与价格同向门", "只保留方向一致的流"),
    ("C6", r"C6_t = N4_t \times z(\mathrm{RV}^{15}_t)",
     "事件×高波动状态", "波动状态下事件冲击被放大"),
    ("C7", r"C7_t = N4_t - \hat\beta_t\, z(\mathrm{mom}^{15}_t),\;\;"
           r"\hat\beta_t = \frac{\mathrm{E}_{4800}[N4\cdot m]}"
           r"{\mathrm{E}_{4800}[m^2]}",
     "对期货动量的滚动残差", "剥离已被价格解释的创新（防同期共动）"),
    ("C8", r"C8_t = z\left(\sum_{120} N1\right) - z\left(\sum_{120} r_1\right)",
     "120 分钟未兑现缺口", "信念累积与价格累积之差"),
    ("C9", r"C9_t = N5_t \times z(\mathrm{Amt}^{15}_t)",
     "不平衡×成交额", "在厚市场里的方向性下注"),
    ("C10", r"C10_t = N9_t - z(\mathrm{RV}^{15}_t)",
     "信念波动-实现波动差", "事件风险相对已实现风险的溢出"),
    ("K1", r"K1_t = \sum_{60}(\mathbf{1}\{E^{up}\} - \mathbf{1}\{E^{dn}\})",
     "60 分钟净事件数", "离散事件的最直接计数聚合"),
    ("K2", r"K2_t = \frac{c_t - \mu_{hod}(c)}{\sigma_{hod}(c)},\;\;"
           r"c_t=\sum_{60}\mathbf{1}\{E\}",
     "分时基准化频率 z", "剔除时段固有活跃差（答辩要求的时序分桶）"),
    ("K3", r"K3_t = \sum_{\theta\in\Theta(j)}\sum_{60}"
           r"\mathbf{1}\{|N1^{\theta}|>\kappa\}",
     "跨主题截面事件计数", "答辩要求的截面统计（如 fed 类跨品种族）"),
    ("K4", r"K4_t = \sum_{s\leq t} e^{-(t-s)/30}\,\mathrm{sgn}(E_s)",
     "指数衰减签名强度", "近期事件权重更高的记忆核"),
    ("K5", r"K5_t = \max_{30}\,\mathrm{runlen}(\mathrm{sgn}(E))",
     "同向连发长度", "事件串（升级进行时）识别"),
    ("K6", r"K6_t = e^{-\Delta t_{last}/60}\cdot \mathrm{sgn}(E_{last})",
     "近因得分", "把「距上次信号的时间」转成数值"),
    ("K7", r"K7_t = z\left(\sum_{60}|N1|\cdot\mathbf{1}\{|N1|>\kappa\}\right)",
     "尾部幅度和 z", "只统计超阈值的大跳（答辩要求的比例跳动）"),
    ("K8", r"K8_t = \frac{\sum_{120}\mathbf{1}\{E^{up}\}-\sum_{120}"
           r"\mathbf{1}\{E^{dn}\}}{\sum_{120}\mathbf{1}\{E\}}",
     "方向一致率", "事件流的方向纯度 [-1,1]"),
    ("K9", r"K9_t = \left(1+\mathrm{med}_5(\Delta t_{inter})\right)^{-1}",
     "到达率热度", "事件间隔的倒数：越密越热"),
    ("K10", r"K10_t = \mathbf{1}\{\geq 2\;\mathrm{themes}\;\mathrm{in}\;15m\}"
            r"\cdot\mathrm{sgn}(N2_t)",
     "跨主题共振", "多主题同时触发的置信增强"),
]


def qa_section(meta: dict) -> str:
    lock_rows = "".join(
        f"<tr><td>{p}</td><td>{m['n_minutes']:,}</td><td>{m['n_days']}</td>"
        f"<td>{m['locked_share']:.2%}</td>"
        f"<td>{m['pm_active_minute_share']:.1%}</td></tr>"
        for p, m in meta.items()
    )
    return f"""
<h2 id="q1">1　数据：逐问回答</h2>
<h3>1.1 Polymarket 侧</h3>
<div class="qa"><b>每一行是一个交易吗？</b>　每行是一条链上 <code>OrderFilled</code>
成交腿（maker-taker 撮合对），交易所自身作对手方的<b>中继腿已剔除</b>（保留会使
成交量恰好翻倍）。一个吃单吃掉多档挂单会产生多行；如需订单级可按
<code>(tx_hash, taker)</code> 聚合。行键 <code>chainId_block_logIndex</code>
全局唯一、天然幂等。</div>
<div class="qa"><b>时间戳精确到多少？</b>　链上出块时间，<b>秒级</b>；Polygon
出块间隔约 2.1 秒，即同一秒内可有多笔、时间分辨率下限约 2 秒。分钟信号取
桶 [T-60s, T) 标签 T，只含严格早于 T 的成交。</div>
<div class="qa"><b>爬取的时间延迟是多少？</b>　三段：交易上链约 2 秒（出块）；
本文批量抓取上界取 <code>finalized</code> 标签（Polygon milestone 最终性约
1-2 分钟）或链头 −256 块（约 9 分钟）的保守回退——这是<b>研究口径</b>的
选择，防区块重组而非技术上限；若实盘增量可直接跟链头，端到端延迟约
2-5 秒 + 调度间隔。历史回填不存在延迟概念。</div>
<div class="qa"><b>如何定义主动买入 / 卖出？定义严谨吗？</b>　taker 方向来自
交易所撮合事件的<b>字段本身</b>（maker_asset / taker_asset 谁付抵押品），
不是 Lee-Ready 类的报价推断。方向变量 D = +1 定义为 taker 买 YES 或卖 NO
（把事件概率推高），D = −1 反之；该真值表经 2024-11 全月与公开数据集逐行
核对。严谨性强于股票市场常用的主动性推断。</div>
<div class="qa"><b>数据统计量？</b>　全 tape：<b>855,614,453 笔成交、
$265 亿名义、约 69 万个市场</b>（HF 段 6.02 亿笔至 2026-04-28 + 自爬段
2.54 亿笔至 07-14，两段同构拼接、迁移期同质性实测 v2 占比 &lt;0.04%）。
登记主题子集与逐月频率见第 5 节。</div>
<h3>1.2 期货 1 分钟数据</h3>
<div class="qa"><b>怎么获取的？</b>　聚宽风格主力连续（XX9999）分钟 CSV 重建：
88 品种、2026-01-05..07-13、350 万根 bar；夜盘 bar 原始存放在开始时刻次一
自然日的文件中，重建时按「夜盘归属下一交易日」从时间戳重归属并有针对性
测试。bar 时间戳为<b>收盘戳</b>。</div>
<div class="qa"><b>与 Polymarket 事件的时间先后？</b>　PM 分钟桶 [T-60s, T)
标签 T 与期货 bar 收盘戳 T 对齐：信号只含严格早于 T 的成交，前向收益从 T
起算——同一墙钟，无前视。链上时间与交易所时间均为 UTC 派生，PM 侧额外
延迟仅上链约 2 秒。</div>
<div class="qa"><b>国内不交易时段怎么处理？</b>　本答辩包只在<b>交易分钟</b>上
评估（信号出现后 1-15 分钟需要可交易价格）；闭市时段积累的 PM 信息由第一 /
第二部分报告的窗口设计（night / gap / day）处理，两套互补。</div>
<div class="qa"><b>午休、小节与日夜盘之间怎么处理？</b>　按<b>连续分钟段</b>
规则：相邻 bar 时间差 &gt; 1 分钟即断段，前向收益一律不跨段（置 NaN）——
午间休市、10:15 小节、日夜盘边界被同一条规则自动处理，无需手工日历。</div>
<div class="qa"><b>涨跌停如何处理？</b>　主力连续数据不带涨跌停标记；用
high == low 的「锁死分钟」占比作代理监控（下表，2-4%，多为清淡分钟而非
涨跌停），事件研究不剔除但如实报告；日级收益天然被涨跌停截尾，属于保守
方向的度量误差。</div>
{wrap('<tr><th>品种</th><th>交易分钟数</th><th>交易日</th>'
      '<th>high==low 分钟占比</th><th>PM 主题活跃分钟占比</th></tr>',
      [lock_rows])}
"""


def defs_section() -> str:
    blocks = []
    fam_note = {
        "N": "数值信号（信念创新与流）",
        "C": "量价组合信号（PM × 期货，全部成分先做 4800 分钟滚动 z）",
        "K": "离散事件的复杂统计指标（先定义原始事件，再做计数 / 频率 / "
             "衰减 / 截面聚合，全部转为有序数值）",
    }
    cur = ""
    for name, formula, title, why in SIGNAL_DEFS:
        fam = name[0]
        if fam != cur:
            cur = fam
            blocks.append(f"<h3>{fam} 族　{fam_note[fam]}</h3>")
            if fam == "K":
                blocks.append(
                    "<p>原始离散事件（构件）：利多事件 E<sup>up</sup>: "
                    "N1<sub>t</sub> &gt; κ<sub>t</sub>；利空 E<sup>dn</sup> 对称；"
                    "美元量爆发 E<sup>big</sup>: 分钟成交额 &gt; 滚动 99 分位。"
                    "阈值 κ 用稳健尺度，显式公式：</p>"
                    + tex(r"\kappa_t = \max\left(0.05,\; 3 \times 1.4826 \times "
                          r"\mathrm{MAD}_{4800}(|N1|_{\ne 0})\right)"))
        blocks.append(
            f"<div class='sigdef'><b>{name}　{title}</b>{tex(formula)}"
            f"<div class='why'>{why}</div></div>")
    header = (
        "<h2 id='q2'>2　30 个信号：公式与构造依据</h2>"
        "<p>通用记号：ℓ(p)=ln(p/(1−p))，p 截断 [0.02, 0.98]；σ<sub>m</sub> 为"
        "注册方向先验（利多 +1）；z(·) 为 4800 交易分钟（约 10 个交易日）滚动"
        "z 分数（答辩要求的时序归一化）；mom/RV/Vol/Amt 上标 15 为段内 15 分钟"
        "滚动统计；κ 为稳健跳跃阈值。所有信号均为实数、有大小与方向语义，"
        "缺测按「无信息 = 0」处理。</p>")
    return header + "".join(blocks)


def stats_section() -> str:
    st = pd.read_parquet(D / "signal_stats.parquet")
    sub = st[st["product"] == "SC"].set_index("signal").reindex(ALL_SIGNALS)
    head = ("<tr><th>信号</th><th>非零分钟占比</th><th>日均非零数</th>"
            "<th>均值</th><th>标准差</th><th>偏度</th><th>q1</th><th>中位</th>"
            "<th>q99</th></tr>")
    rows = [
        f"<tr><td>{s}</td><td>{r['nonzero_rate']:.1%}</td>"
        f"<td>{r['per_day']:.0f}</td><td>{r['mean']:+.3g}</td>"
        f"<td>{r['std']:.3g}</td><td>{r['skew']:+.2f}</td>"
        f"<td>{r['q1']:+.3g}</td><td>{r['q50']:+.3g}</td><td>{r['q99']:+.3g}</td></tr>"
        for s, r in sub.iterrows() if pd.notna(r["n_minutes"])
    ]
    figs = []
    panel = pd.read_parquet(D / "panel_SC.parquet",
                            columns=ALL_SIGNALS)
    for fam, sigs in (("N", ALL_SIGNALS[:10]), ("C", ALL_SIGNALS[10:20]),
                      ("K", ALL_SIGNALS[20:])):
        fig, axes = plt.subplots(2, 5, figsize=(10, 3.6))
        for ax, s in zip(axes.ravel(), sigs, strict=True):
            v = panel[s].replace([np.inf, -np.inf], np.nan).dropna()
            v = v[v != 0]
            if len(v) > 100:
                lo, hi = np.percentile(v, [0.5, 99.5])
                ax.hist(v.clip(lo, hi), bins=40, color=P["blue"])
            ax.set_yscale("log")
            ax.set_title(s, fontsize=8)
            ax.tick_params(labelsize=6)
        fig.tight_layout()
        figs.append(figure(
            fig, f"{fam} 族信号的非零取值分布（SC，截尾 [0.5%, 99.5%]，"
                 "对数频数轴）。"))
    return ("<h2 id='q3'>3　信号频率、取值统计与分布</h2>"
            "<p>下表为 SC（原油）上的统计（全部 5 品种 × 30 信号见 "
            "<code>signal_stats.parquet</code>）：</p>"
            + wrap(head, rows) + "".join(figs))


def ic_section() -> str:
    ic = pd.read_parquet(D / "ic_table.parquet")
    figs, tables = [], []
    for prod in PRODUCT_THEMES:
        sub = ic[ic["product"] == prod]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="signal", columns="horizon",
                              values="rank_ic").reindex(ALL_SIGNALS)
        piv_ir = sub.pivot_table(index="signal", columns="horizon",
                                 values="icir").reindex(ALL_SIGNALS)
        if prod == "SC":
            fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.6))
            for ax, (mat, ttl) in zip(
                axes,
                ((piv, "RankIC"), (piv_ir, "ICIR（逐日 IC 均值/标准差）")),
                strict=True,
            ):
                im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto",
                               cmap="RdBu_r", vmin=-0.06 if ttl == "RankIC"
                               else -0.6,
                               vmax=0.06 if ttl == "RankIC" else 0.6)
                ax.set_xticks(range(len(mat.columns)),
                              [f"{h}m" for h in mat.columns], fontsize=7)
                ax.set_yticks(range(len(mat.index)), mat.index, fontsize=6.5)
                ax.set_title(f"SC · {ttl}", fontsize=9)
                fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            figs.append(figure(
                fig, "SC 的 30 信号 × 6 视界 RankIC 与 ICIR 热图。"))
        best = (sub.assign(a=lambda x: x["rank_ic"].abs())
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
        tables.append(f"<h4>{prod}（|RankIC| 前 8）</h4>" + wrap(head, rows))
    note = (
        "<div class='callout'>多重检验注记：本表共约 30 信号 × 6 视界 × 5 品种"
        " ≈ 900 个相关系数，5% 名义水平下期望约 45 个偶然「显著」。解读只看"
        "（i）同信号跨视界 / 跨品种的符号一致性，（ii）|ICIR| 持续超过 0.15 "
        "的少数格子；单格 |IC| 不构成声称。</div>")
    return ("<h2 id='q4'>4　数值信号 IC / ICIR</h2>"
            "<p>对每个信号与视界 k ∈ {1,2,3,5,10,15} 分钟：全样本 Pearson IC 与"
            " RankIC；ICIR = 逐交易日 IC 序列的 mean/std（日内至少 10 个有效"
            "分钟才计入）。信号取值为 0（无信息）的分钟不参与。</p>"
            + "".join(figs) + "".join(tables) + note)


def event_section() -> str:
    ev = pd.read_parquet(D / "event_study.parquet")
    meta = json.loads((D / "meta.json").read_text())
    tables = []
    for prod in PRODUCT_THEMES:
        sub = ev[ev["product"] == prod]
        if sub.empty:
            continue
        base = meta[prod]["baseline_p_move"]
        head = ("<tr><th>事件</th><th>n</th><th>月均次数</th>"
                + "".join(f"<th>{k}m 签名收益 bp / P(|Δp|&gt;0.1%) / 提升</th>"
                          for k in (1, 5, 15))
                + "<th>直到下一信号 bp（中位间隔）</th></tr>")
        rows = []
        for r in sub.itertuples(index=False):
            cells = ""
            for k in (1, 5, 15):
                rb = getattr(r, f"ret_bp_{k}", np.nan)
                pm_ = getattr(r, f"p_move_{k}", np.nan)
                lf = getattr(r, f"lift_{k}", np.nan)
                cells += (f"<td>{rb:+.1f} / {pm_:.0%} / "
                          f"<b>{lf:.2f}x</b></td>" if np.isfinite(rb)
                          else "<td>—</td>")
            nx = getattr(r, "ret_bp_next", np.nan)
            sp = getattr(r, "med_span_min", np.nan)
            nxs = (f"{nx:+.1f}（{sp:.0f}min）" if np.isfinite(nx) else "—")
            rows.append(
                f"<tr><td>{r.event}</td><td>{r.n}</td>"
                f"<td>{r.per_month:.0f}</td>{cells}<td>{nxs}</td></tr>")
        base_str = "，".join(f"{k}m: {base[str(k)]:.0%}"
                             for k in (1, 5, 15))
        tables.append(
            f"<h4>{prod}</h4>" + wrap(head, rows)
            + f"<div class='texnote'>基线 P(|Δp|&gt;0.1%)（全部交易分钟）："
              f"{base_str}；「提升」= 事件后比例 / 基线。</div>")
    hist_html = ""
    hp = D / "history_monthly.parquet"
    if hp.exists():
        h = pd.read_parquet(hp)
        piv = (h.pivot_table(index="month", columns="theme",
                             values="n_events", aggfunc="sum")
               .fillna(0.0))
        fig, ax = plt.subplots(figsize=(9.0, 3.2))
        x = range(len(piv.index))
        bottom = np.zeros(len(piv))
        colors = [P[c] for c in
                  ("blue", "aqua", "yellow", "green", "violet", "red",
                   "magenta", "orange")]
        for i, c in enumerate(piv.columns):
            ax.bar(x, piv[c].to_numpy(), bottom=bottom,
                   label=c, color=colors[i % len(colors)], width=0.85)
            bottom += piv[c].to_numpy()
        step = max(len(piv) // 12, 1)
        ax.set_xticks(list(x)[::step], list(piv.index)[::step],
                      rotation=45, fontsize=6.5)
        ax.legend(fontsize=6, ncol=4, frameon=False)
        ax.set_ylabel("|Δlogit|>0.10 分钟事件数 / 月")
        pub_style.soft_grid(ax)
        fig.tight_layout()
        mon = h.groupby("theme").agg(
            月均事件=("n_events", lambda s: s.sum() / h["month"].nunique()),
            市场数=("n_markets", "max"),
        ).round(1)
        hist_html = (
            "<h3>5.1 全历史逐月频率（2022-11 至 2026-07）</h3>"
            + figure(fig, "规则匹配的全历史主题市场（不限 2026 窗口），"
                          "每月 |Δlogit|>0.10 的分钟级事件数（统一阈值）。"
                          "事件频率高度事件驱动：2026 年 2-3 月美伊冲突为峰。")
            + wrap("<tr><th>主题</th><th>月均事件数</th><th>历史市场数峰值</th></tr>",
                   [f"<tr><td>{i}</td><td>{r.月均事件:.0f}</td>"
                    f"<td>{int(r.市场数)}</td></tr>"
                    for i, r in mon.iterrows()]))
    return ("<h2 id='q5'>5　离散信号：频率与事件研究</h2>"
            "<p>事件定义见第 2 节 K 族构件；「签名收益」按事件方向符号化"
            "（E_big / E_burst 无先验方向，按当刻 N1 符号）；全部只在交易分钟"
            "上评估，前向不跨连续段。</p>"
            + "".join(tables) + hist_html)


STYLE_EXTRA = """
.qa { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: .7rem 1rem; margin: .6rem 0; font-size: .93rem; }
.qa b { color: var(--accent); }
.sigdef { background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; padding: .6rem 1rem; margin: .55rem 0; font-size: .92rem; }
.sigdef .why { color: var(--ink-soft); font-size: .85rem; }
"""


def build() -> str:
    from build_thesis_html import STYLE
    meta = json.loads((D / "meta.json").read_text())
    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Polymarket × 中国期货：分钟级信号库与 IC 检验（答辩数据包）</title>
<style>{STYLE}{STYLE_EXTRA}</style>
<main>
<div class="eyebrow">Alpha-Data 答辩数据包 · 2026-07-16 · 与主报告
（cn_futures_polymarket_report.html v2.1）互补</div>
<h1>Polymarket × 中国商品期货：<br>分钟级信号定义、统计与 IC 检验</h1>
<div class="abstract"><b class="hd">说明</b>　本文档按答辩要求组织：第 1 节逐问
回答数据定义（tape 行含义、时间戳精度、爬取延迟、主动方向的严谨定义、总量
统计；期货分钟数据来源、时段与涨跌停处理）；第 2 节给出 <b>30 个信号</b>的
显式公式与构造依据（10 数值 + 10 量价组合 + 10 离散复杂统计，全部转为有序
数值、全部时序归一化）；第 3-5 节给出每个信号的频率、取值分布、数值信号的
IC / RankIC / ICIR（1-15 分钟视界）与离散信号的事件研究（对无条件基线的提升
倍数、直到下一信号的收益）以及 2022-11 以来的逐月频率。评估品种：SC、AU、
AG、CU、M；样本 2026-01-05 至 07-13 的全部交易分钟。方向性结论请以主报告
第二部分（v3.1）的识别与功效框架为准——本包是其分钟级测量下沉，多重检验
注记见第 4 节。</div>
<div class="toc"><b>目录</b><br>
<a href="#q1">1 数据：逐问回答</a><br>
<a href="#q2">2 30 个信号：公式与依据</a><br>
<a href="#q3">3 频率、统计与分布</a><br>
<a href="#q4">4 IC / ICIR</a><br>
<a href="#q5">5 事件研究与全历史频率</a></div>
{qa_section(meta)}
{defs_section()}
{stats_section()}
{ic_section()}
{event_section()}
<div class="footer">复现：<code>v3_defense_build.py → v3_defense_report.py</code>
· 面板 / IC / 事件研究 / 逐月频率产物在
<code>data/cn_futures/analysis/v3/defense/</code> · 分支
feat/cn-futures-polymarket</div>
</main>
"""


def main() -> int:
    pub_style.setup(cn_font=True)
    html = build()
    OUT.write_text(html)
    print(f"written {OUT} ({len(html) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
