#!/usr/bin/env python3
"""
NPM系列文章合集 PDF 生成脚本
Markdown → HTML (markdown-it-py) → PDF (weasyprint)
"""

import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from weasyprint import HTML

# ── 路径 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = ROOT / "6_Articles_CN"
PICS_DIR = ROOT / "7_Articles_Pictures_CN"
FIGS_CN = ROOT / "3_Figures_CN"
OUTPUT_PDF = ROOT / "AI并不神秘_系列合集_NeuralCAE.pdf"

# ── 文章顺序 ──────────────────────────────────────────
ARTICLES = [
    "01_AI_First_Principles.md",
    "02_Neural_Percolation_Model.md",
    "03_Percolation_Meets_NN.md",
    "04_Panorama_Map.md",
    "05_Demystify_to_Debug.md",
    "06_Where_Is_The_Road.md",
    "07_EVM_Roadmap.md",
    "08_Cognitive_Percolation.md",
    "09_Beyond_Reasoning.md",
    "10_Creator_Interview.md",
    "11_Gratitude.md",
]

# ── 图片映射：占位符文本 → 图片文件路径 ──────────────
# 格式：(文章编号, 占位符关键词) → 相对于ROOT的路径
IMAGE_MAP = {
    # 第1-3篇：用论文图（3_Figures_CN）
    ("03", "图1：PNM"): FIGS_CN / "NPM_Fig1_PNM_CN.png",
    ("03", "图2：PNM–NPM变量"): FIGS_CN / "NPM_Fig2_Analogy_CN.png",
    ("03", "图3：统一主方程"): FIGS_CN / "NPM_Fig3_MasterEq_CN.png",
    ("03", "图4：NPM核心方程"): FIGS_CN / "NPM_Fig4_EqChain_CN.png",
    ("03", "图5：多阈值涌现"): FIGS_CN / "NPM_Fig5_CapField_CN.png",

    # 第四篇
    ("04", "渊源对照图"): PICS_DIR / "第四篇_渊源全景图_最终版_v2_260328" / "第四篇_渊源对照图.png",
    ("04", "名词焦虑对照表"): PICS_DIR / "第四篇_渊源全景图_最终版_v2_260328" / "第四篇_名词焦虑对照表.png",
    ("04", "学术全景图"): PICS_DIR / "第四篇_渊源全景图_最终版_v2_260328" / "npm_学术全景图.png",
    ("04", "工程全景图"): PICS_DIR / "第四篇_渊源全景图_最终版_v2_260328" / "npm_工程全景图.png",
    ("04", "统一全景图"): PICS_DIR / "第四篇_渊源全景图_最终版_v2_260328" / "npm_统一全景图_v2.png",

    # 第五篇
    ("05", "九个问题全景图"): PICS_DIR / "第五篇_AI袪魅到找茬_2603291512" / "第五篇_九个问题全景图.png",
    ("05", "三层祛魅逻辑图"): PICS_DIR / "第五篇_AI袪魅到找茬_2603291512" / "第五篇_三层祛魅逻辑图.png",

    # 第六篇
    ("06", "语言AI vs 物理AI"): PICS_DIR / "第六篇_路在何方_260329" / "第六篇_图1_两条腿对比.png",
    ("06", "数据破局"): PICS_DIR / "第六篇_路在何方_260329" / "第六篇_图2_数据破局.png",
    ("06", "V-model"): PICS_DIR / "第六篇_路在何方_260329" / "第六篇_图4_Vmodel.png",
    ("06", "三级台阶"): PICS_DIR / "第六篇_路在何方_260329" / "第六篇_图3_三级台阶.png",

    # 第七篇
    ("07", "V-model映射"): PICS_DIR / "第七篇_工程验证模型" / "第七篇_图2_Vmodel映射.png",
    ("07", "EVM技术路线图"): PICS_DIR / "第七篇_工程验证模型" / "第七篇_图1_EVM技术路线图.png",

    # 第八篇
    ("08", "图1"): PICS_DIR / "第八篇_思维涌现纪实——系列的创作过程_v3" / "01_sculpture.jpg",
    ("08", "图2"): PICS_DIR / "第八篇_思维涌现纪实——系列的创作过程_v3" / "02_moments_poem.png",
    ("08", "图3"): PICS_DIR / "第八篇_思维涌现纪实——系列的创作过程_v3" / "03_openclaw_chat.png",
    ("08", "图4"): PICS_DIR / "第八篇_思维涌现纪实——系列的创作过程_v3" / "04_moments_amateur.png",
    ("08", "图5"): PICS_DIR / "第八篇_思维涌现纪实——系列的创作过程_v3" / "05_openpnm.png",
    ("08", "图6"): PICS_DIR / "第八篇_思维涌现纪实——系列的创作过程_v3" / "06_zenodo.png",
    ("08", "图7"): PICS_DIR / "第八篇_思维涌现纪实——系列的创作过程_v3" / "07_moments_ai_hype.png",
    ("08", "图8"): PICS_DIR / "第八篇_思维涌现纪实——系列的创作过程_v3" / "08_chain_diagram.png",

    # 第九篇
    ("09", "图1"): PICS_DIR / "第九篇_推理之外的洞见" / "图1_变与不变双柱对照.png",
    ("09", "图2"): PICS_DIR / "第九篇_推理之外的洞见" / "图2_稳定剂证据链.png",
    ("09", "图3"): PICS_DIR / "第九篇_推理之外的洞见" / "图3_陷阱与正道.png",
    ("09", "图4"): PICS_DIR / "第九篇_推理之外的洞见" / "图4_AI各领域现状预估表.png",
    ("09", "图5"): PICS_DIR / "第九篇_推理之外的洞见" / "图5_四个变化的企业影响.png",

    # 第十篇
    ("10", "A.png"): PICS_DIR / "第十篇_创作团队" / "A.png",
    ("10", "B1.png"): PICS_DIR / "第十篇_创作团队" / "B1.png",
    ("10", "B2.png"): PICS_DIR / "第十篇_创作团队" / "B2.png",
    ("10", "C.png"): PICS_DIR / "第十篇_创作团队" / "C.png",
    ("10", "总结表格"): PICS_DIR / "第十篇_创作团队" / "第十篇_总结表格.png",
}


def resolve_image(article_num: str, placeholder_text: str) -> str | None:
    """根据文章编号和占位符文本，查找匹配的图片路径"""
    for (art, key), path in IMAGE_MAP.items():
        if art == article_num and key in placeholder_text:
            if path.exists():
                return path.as_uri()
            else:
                print(f"  ⚠ 图片不存在: {path}")
                return None
    return None


def process_article(md_file: Path, article_num: str) -> str:
    """读取md文件，替换图片占位符为<img>标签，返回HTML片段"""
    text = md_file.read_text(encoding="utf-8")

    # 替换 【图X：...】 和 【插入图X：...】 占位符
    def replace_placeholder(match):
        full = match.group(0)
        img_uri = resolve_image(article_num, full)
        if img_uri:
            # 提取图注文本
            caption = full.strip("【】")
            return f'<figure><img src="{img_uri}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
        return f'<p class="img-placeholder">{full}</p>'

    # 匹配 【图N：...】【插入图N：...】 但排除 【图神经网络...】等非占位符
    text = re.sub(r'【(?:插入)?图\d[^】]*】', replace_placeholder, text)

    # Markdown → HTML
    md = MarkdownIt("commonmark", {"typographer": True})
    md.enable("table")
    html = md.render(text)
    return html


# ── CSS 样式 ──────────────────────────────────────────
CSS = """
@page {
    size: A4;
    margin: 2.2cm 2cm 2.2cm 2cm;
    @bottom-center {
        content: "— " counter(page) " —";
        font-size: 9pt;
        color: #666;
    }
}
@page :first {
    @bottom-center { content: none; }
}

body {
    font-family: "Noto Serif CJK SC", "Noto Sans CJK SC", serif;
    font-size: 10.5pt;
    line-height: 1.75;
    color: #222;
}

/* 封面 */
.cover {
    page-break-after: always;
    text-align: center;
    padding-top: 18%;
}
.cover h1 {
    font-size: 26pt;
    font-weight: bold;
    margin-bottom: 0.3em;
    color: #1a1a1a;
}
.cover .subtitle {
    font-size: 13pt;
    color: #555;
    margin-bottom: 2em;
}
.cover .author {
    font-size: 12pt;
    color: #333;
    margin-top: 3em;
}
.cover .info {
    font-size: 9.5pt;
    color: #777;
    margin-top: 1.5em;
    line-height: 1.6;
}

/* 目录 */
.toc {
    page-break-after: always;
}
.toc h2 {
    text-align: center;
    margin-bottom: 1.5em;
}
.toc ul {
    list-style: none;
    padding: 0;
}
.toc li {
    margin: 0.5em 0;
    font-size: 11pt;
}
.toc .num {
    display: inline-block;
    width: 4em;
    color: #888;
}

/* 文章分隔 */
.article {
    page-break-before: always;
}
.article:first-of-type {
    page-break-before: auto;
}

/* 标题 */
h1 {
    font-size: 18pt;
    font-weight: bold;
    margin-top: 0.5em;
    margin-bottom: 0.3em;
    color: #1a1a1a;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.2em;
}
h2 {
    font-size: 14pt;
    font-weight: bold;
    margin-top: 1.2em;
    color: #2a2a2a;
}
h3 {
    font-size: 12pt;
    font-weight: bold;
    margin-top: 1em;
    color: #333;
}

/* 表格 */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 9.5pt;
}
th, td {
    border: 1px solid #ccc;
    padding: 5px 8px;
    text-align: left;
}
th {
    background: #f5f5f5;
    font-weight: bold;
}

/* 图片 */
figure {
    text-align: center;
    margin: 1.2em 0;
    page-break-inside: avoid;
}
figure img {
    max-width: 92%;
    max-height: 45vh;
}
figcaption {
    font-size: 9pt;
    color: #666;
    margin-top: 0.3em;
}
.img-placeholder {
    text-align: center;
    color: #999;
    font-style: italic;
    font-size: 9.5pt;
    margin: 1em 0;
}

/* 代码 */
code {
    font-family: "Noto Sans Mono CJK SC", "Courier New", monospace;
    font-size: 9pt;
    background: #f4f4f4;
    padding: 1px 4px;
    border-radius: 3px;
}
pre {
    background: #f4f4f4;
    padding: 10px 14px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 8.5pt;
    line-height: 1.5;
    page-break-inside: avoid;
}
pre code {
    background: none;
    padding: 0;
}

/* 引用 */
blockquote {
    border-left: 3px solid #ccc;
    margin: 1em 0;
    padding: 0.3em 1em;
    color: #555;
    font-style: italic;
}

/* 脚注 */
.article-footer {
    margin-top: 2em;
    padding-top: 0.5em;
    border-top: 1px solid #eee;
    font-size: 9pt;
    color: #888;
}
"""

# ── 文章标题映射 ──────────────────────────────────────
TITLES = {
    "01": "第一篇　AI到底是什么——一个CAE工程师的第一性原理推论",
    "02": "第二篇　神经网络在做什么——神经渗流模型NPM",
    "03": "第三篇　AI并不神秘——当渗流遇见神经网络",
    "04": "第四篇　神经网络与CAE仿真——渊源全景图",
    "05": "第五篇　当前的AI并不完美——从祛魅到找茬",
    "06": "第六篇　路在何方——工程界的世界模型",
    "07": "第七篇　物理AI的技术路线图——工程验证模型",
    "08": "第八篇　认知渗流",
    "09": "第九篇　推理之外的洞见",
    "10": "第十篇　工作解读——创作者专访",
    "11": "第十一篇　感恩同行",
}


def build_html() -> str:
    """构建完整HTML文档"""
    parts = []

    # 封面
    parts.append("""
    <div class="cover">
        <h1>AI并不神秘</h1>
        <div class="subtitle">一个CAE工程师的视角</div>
        <div class="subtitle" style="font-size:11pt; margin-top:1em;">
            系列文章合集（共十一篇）
        </div>
        <div class="author">
            丁铁新 · 神经CAE（NeuralCAE）· 独立研究者
        </div>
        <div class="info">
            首创概念：NPM / EVM / 三层同构 / 拜AI为师<br>
            NPM论文 DOI: 10.5281/zenodo.19209722<br>
            GitHub: github.com/tiexinding/neural-percolation-model<br>
            知乎专栏 · 微信公众号「NeuralCAE」<br><br>
            2026年4月
        </div>
    </div>
    """)

    # 目录
    toc_items = ""
    for num, title in TITLES.items():
        toc_items += f'<li><span class="num">{num}.</span>{title}</li>\n'
    parts.append(f"""
    <div class="toc">
        <h2>目　录</h2>
        <ul>{toc_items}</ul>
    </div>
    """)

    # 逐篇处理
    for fname in ARTICLES:
        md_path = ARTICLES_DIR / fname
        if not md_path.exists():
            print(f"⚠ 文件不存在: {md_path}")
            continue

        num = fname[:2]
        title = TITLES.get(num, fname)
        print(f"处理: {title}")

        article_html = process_article(md_path, num)

        parts.append(f"""
        <div class="article">
            {article_html}
        </div>
        """)

    # 组装
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <style>{CSS}</style>
</head>
<body>
    {"".join(parts)}
</body>
</html>"""
    return full_html


def main():
    print("=" * 50)
    print("NPM系列文章合集 PDF 生成")
    print("=" * 50)

    html_content = build_html()

    # 保存中间HTML（调试用）
    html_path = ROOT / "AI并不神秘_系列合集_NeuralCAE.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"\nHTML已生成: {html_path}")

    # 生成PDF
    print(f"正在生成PDF...")
    HTML(string=html_content, base_url=str(ROOT)).write_pdf(str(OUTPUT_PDF))
    print(f"PDF已生成: {OUTPUT_PDF}")
    print(f"文件大小: {OUTPUT_PDF.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
