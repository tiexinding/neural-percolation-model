# 神经渗流模型（NPM）

**基于孔网络模型比拟的神经网络信息传播与能力涌现工程框架**

含配套系列文章「AI并不神秘」十一篇，提出NPM / EVM / 三层同构 / 拜AI为师四个首创概念

作者：丁铁新 · 神经CAE（NeuralCAE） · 独立研究者

---

## 概述

NPM将多孔介质孔网络模型（PNM）的物理原理映射到神经网络信息传播。从单一守恒原理（稳态节点平衡）出发，推导出统一主方程：

**a_out = W_eff · a_in**

其中 W_eff 取三种形式，覆盖所有主流架构：前馈网络（固定权重）、GNN（图拉普拉斯，与PNM精确对应，误差=0）、Transformer（softmax注意力=流量守恒）。

## 四个首创概念

| 概念 | 性质 | 首次提出 | 核心内容 |
|------|------|---------|---------|
| **NPM（神经渗流模型）** | 数学框架 | 第二篇 | GNN消息传递=PNM节点守恒方程，四组同构，验证误差=0 |
| **EVM（工程验证模型）** | 应用路线图 | 第七篇 | 物理AI五阶段验证路线图，V-model映射 |
| **三层同构** | 理论发现 | 第八篇 | 自然界渗流 / AI能力涌现 / 人类认知涌现，服从同一数学机制 |
| **拜AI为师** | 方法论 | 第九篇 | 放低姿态用AI学AI，判断力才是智能的根本 |

## 系列文章：AI并不神秘——一个CAE工程师的视角

十一篇文章从第一性原理出发，用CAE工程师的语言系统解析AI。文章全文见 `6_Articles_CN/` 目录，同步发表于知乎专栏和微信公众号「NeuralCAE」。

| 篇目 | 标题 | 核心内容 |
|------|------|---------|
| 第一篇 | AI到底是什么——一个CAE工程师的第一性原理推论 | AI是自然界整体交互的多模态投影 |
| 第二篇 | 神经网络在做什么——神经渗流模型NPM | 投影通过渗流连通形成，NPM方程 |
| 第三篇 | AI并不神秘——当渗流遇见神经网络 | GNN=PNM数学验证，误差=0 |
| 第四篇 | 神经网络与CAE仿真——渊源全景图 | 四组同构，共享数学基底 |
| 第五篇 | 当前的AI并不完美——从祛魅到找茬 | MeshGraphNet实操，九个缺口全部指向验证 |
| 第六篇 | 路在何方——工程界的世界模型 | 数据+算力+验证三条腿，能show→能用→能信 |
| 第七篇 | 物理AI的技术路线图——工程验证模型 | EVM五阶段路线图，V-model七层映射 |
| 第八篇 | 认知渗流 | 创作过程纪实，三层同构发现 |
| 第九篇 | 推理之外的洞见 | AI时代的变与不变，拜AI为师 |
| 第十篇 | 工作解读——创作者专访 | 五个声音背对背回答，人机协作实录 |
| 第十一篇 | 感恩同行 | 致导师、领路人、家人、开源社区 |

## 核心贡献

1. **统一主方程** — 从一个守恒原理推导三种架构实例
2. **多阈值场视角** — 涌现是连续能力谱系，不是单一相变
3. **参与率（PR）法** — 从协方差谱定量估算数据分布广度σ
4. **17类LLM现象统一解释** — 含3个可证伪预测
5. **工程框架定位** — 类比达西定律与雷诺数的传统

## 文件结构

```
neural-percolation-model/
├── 1_Paper_CN/          # 中文论文（xelatex + xeCJK编译）
├── 2_Paper_EN/          # 英文论文（pdflatex编译，arXiv提交用）
├── 3_Figures_CN/        # 中文图表PNG
├── 4_Figures_EN/        # 英文图表PNG
├── 5_codes/             # Python代码
│   ├── npm_core.py          # NPM核心方程
│   ├── sigma_estimation.py  # PR法估算σ
│   ├── npm_validation.py    # A层数值检查
│   ├── npm_scaling_check.py # B层vs Kaplan/Chinchilla相关性
│   ├── npm_dimensionless.py # Cn无量纲数实证分析（14个模型）
│   ├── npm_pore_analysis.py # MIS孔隙形态分析
│   └── npm_param_fit.py     # B层参数拟合
├── 6_Articles_CN/       # 中文系列文章
│   ├── 01_AI_First_Principles.md
│   ├── 02_Neural_Percolation_Model.md
│   ├── 03_Percolation_Meets_NN.md
│   ├── 04_Panorama_Map.md
│   ├── 05_Demystify_to_Debug.md
│   ├── 06_Where_Is_The_Road.md
│   ├── 07_EVM_Roadmap.md
│   ├── 08_Cognitive_Percolation.md
│   ├── 09_Beyond_Reasoning.md
│   ├── 10_Creator_Interview.md
│   └── 11_Gratitude.md
├── README.md            # English
└── README_CN.md         # 中文（本文件）
```

## 编译与运行

**英文论文（arXiv提交）：**
```bash
cd 2_Paper_EN/ && pdflatex npm_arxiv_paper.tex && pdflatex npm_arxiv_paper.tex
```

**中文论文：**
```bash
cd 1_Paper_CN/ && xelatex npm_paper_CN.tex && xelatex npm_paper_CN.tex
```

**Python代码：**
```bash
cd 5_codes/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python npm_core.py
```

## MIS孔隙形态分析（核心实验）

借鉴PoreSpy的MIS方法，将PNM孔隙分析工具映射到NN权重空间。三个重要发现：

1. **NN权重分布与PNM孔径形状相似但族不同** — 消除尺度差异后r=0.98，NN比天然岩石更极端
2. **训练 = 渗流通道优化** — GPT-2训练使大孔率从0%→41%，选择性放大主通道
3. **Pythia训练三阶段相变** — 蓄压期→突破期→重组期，CV从0.82暴增至1.51，对应渗流巨连通分量形成

## 当前状态

| 组件 | 状态 |
|------|------|
| 统一主方程（A层） | ✓ 数值验证通过，误差=0 |
| GNN–PNM等价 | ✓ 精确数学等价 |
| B层参数拟合 | ✓ α=0.57, β=0.068, δ=2.76, k=68.5 (Spearman=0.70) |
| MIS孔隙分析 | ✓ GPT-2 & Pythia-70m 6项实验完成 |
| 训练三阶段相变 | ✓ 已识别 |
| 系列文章（中文） | ✓ 十一篇全部发表 |
| 能力涌现顺序 | △ 已预测，待BIG-Bench验证 |

## 合作邀请

本项目开放合作：拥有大模型checkpoint的研究者、理论研究者、工程师、实验者均欢迎。

联系方式：GitHub Issues 或 Pull Request

## 引用

```bibtex
@article{ding2026npm,
  title={Neural Percolation Model (NPM): An Engineering Analogy Framework
         for Neural Network Information Propagation and Capability Emergence},
  author={Ding, Tiexin},
  journal={arXiv preprint},
  year={2026}
}
```

## 致谢

作者使用Claude（Anthropic）作为写作辅助工具。所有理论内容、概念框架、方程和结论均为作者独立的知识贡献。

## 相关链接

- Zenodo存档：https://zenodo.org/records/19209722
- arXiv论文：[提交后补充]
- 知乎专栏：AI并不神秘——一个CAE工程师的视角
- 微信公众号：NeuralCAE
- GitHub：https://github.com/tiexinding/neural-percolation-model

---

**神经CAE · NeuralCAE**
20年仿真实战 · 首创NPM / EVM / 三层同构 / 拜AI为师
