"""出版级图表样式（ICML / Nature 风格）与 TeX 公式离线渲染。

约定：刻度朝外、细边框（0.6pt）、无网格或极淡网格、小号无衬线字体、
多面板加 (a)(b)(c) 面板标号、300 dpi、图内不放长标题（叙述交给图注）。
色板沿用经过验证的分类色序（见 deep_analysis_cn_polymarket.C）。

``tex_svg`` 用 matplotlib mathtext 把 TeX 公式渲染成内联 SVG（fill 替换为
currentColor，随页面深浅主题变色），供 build_thesis_html 嵌入。
"""

from __future__ import annotations

import io
import itertools
import re

import matplotlib
import matplotlib.pyplot as plt

#: 与 dataviz 校验色板一致。
PALETTE = {
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4", "orange": "#eb6834",
    "ink": "#1a1a19", "ink2": "#52514e", "grid": "#e3e1da",
}

_SVG_ID_COUNTER = itertools.count(1)


def setup(cn_font: bool = True) -> None:
    """全局出版样式 rcParams。"""
    fams = (["Helvetica Neue", "Arial", "DejaVu Sans"] if not cn_font else
            ["Hiragino Sans GB", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"])
    plt.rcParams.update({
        "font.sans-serif": fams,
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "errorbar.capsize": 1.8,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink2"],
        "ytick.color": PALETTE["ink2"],
        "axes.edgecolor": PALETTE["ink2"],
        "axes.titlecolor": PALETTE["ink"],
    })


def panel(ax: plt.Axes, letter: str) -> None:
    """Nature 式面板标号（左上角外侧加粗小写字母）。"""
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right", color=PALETTE["ink"])


def soft_grid(ax: plt.Axes, axis: str = "y") -> None:
    """极淡的参考网格（仅在需要读数的图上使用）。"""
    ax.grid(True, axis=axis, alpha=0.4, lw=0.4, color=PALETTE["grid"])
    ax.set_axisbelow(True)


def tex_svg(formula: str, fontsize: int = 13, display: bool = True) -> str:
    """把 TeX 公式渲染为内联 SVG 字符串（mathtext，无外部依赖）。

    fill 一律替换为 ``currentColor``，由页面 CSS 决定颜色（深浅主题皆可）；
    SVG 以 em 高度缩放，基线对齐由调用方的容器样式控制。
    """
    prev = matplotlib.rcParams["mathtext.fontset"]
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    try:
        fig = plt.figure(figsize=(0.01, 0.01))
        t = fig.text(0, 0, f"${formula}$", fontsize=fontsize)
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.02,
                    transparent=True)
        plt.close(fig)
        _ = t
    finally:
        matplotlib.rcParams["mathtext.fontset"] = prev
    svg = buf.getvalue().decode()
    svg = svg[svg.index("<svg"):]
    # Matplotlib 每次都会生成 figure_1、patch_1、字体字形等相同 id。多个公式
    # 内联到同一 HTML 后，SVG 的 <use href="#…"> 会按整页全局 id 解析，可能
    # 错引前一个公式。为每次渲染加唯一前缀，并同步改写所有内部引用。
    prefix = f"tex{next(_SVG_ID_COUNTER)}-"
    svg_ids = set(re.findall(r'id="([^"]+)"', svg))
    for svg_id in svg_ids:
        new_id = f"{prefix}{svg_id}"
        svg = svg.replace(f'id="{svg_id}"', f'id="{new_id}"')
        svg = svg.replace(f'href="#{svg_id}"', f'href="#{new_id}"')
        svg = svg.replace(f'url(#{svg_id})', f'url(#{new_id})')
    # 颜色交给 CSS（glyph path 无自带 fill，根元素 currentColor 向下继承，
    # 随页面深浅主题变色）；高度改用 em 使其随正文字号缩放。
    svg = svg.replace("<svg ", '<svg fill="currentColor" ', 1)
    svg = re.sub(r'fill:\s*#[0-9a-fA-F]{6}', "fill: currentColor", svg)
    m = re.search(r'height="([\d.]+)pt"', svg)
    if m:
        em = float(m.group(1)) / fontsize * 1.0
        svg = re.sub(r'width="[\d.]+pt"', "", svg, count=1)
        svg = re.sub(r'height="[\d.]+pt"', f'height="{em:.2f}em"', svg, count=1)
    cls = "texd" if display else "texi"
    return f'<span class="{cls}">{svg}</span>'
