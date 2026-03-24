# 神经渗流模型（NPM）

**基于孔网络模型比拟的神经网络信息传播与能力涌现工程框架**

作者：丁铁新 · 神经CAE · 独立研究者

---

## 概述

NPM将多孔介质孔网络模型（PNM）的物理原理映射到神经网络信息传播。从单一守恒原理（稳态节点平衡）出发，推导出统一主方程：

**a_out = W_eff · a_in**

其中 W_eff 取三种形式，覆盖所有主流架构：前馈网络（固定权重）、GNN（图拉普拉斯，与PNM精确对应，误差=0）、Transformer（softmax注意力=流量守恒）。

## 核心贡献

1. **统一主方程** — 从一个守恒原理推导三种架构实例
2. **多阈值场视角** — 涌现是连续能力谱系，不是单一相变
3. **参与率（PR）法** — 从协方差谱定量估算数据分布广度σ
4. **17类LLM现象统一解释** — 含3个可证伪预测
5. **工程框架定位** — 类比达西定律与雷诺数的传统

## 文件结构

```
NPM_v13_complete/
├── 1_Paper_CN/          # 中文论文（xelatex + xeCJK编译）
│   ├── npm_paper_CN.tex
│   ├── npm_paper_CN.pdf
│   └── NPM_Fig*_CN.png
├── 2_Paper_EN/          # 英文论文（pdflatex编译，arXiv提交用）
│   ├── npm_arxiv_paper.tex
│   ├── npm_arxiv_paper.pdf
│   └── NPM_Fig*_EN.png
├── 3_Figures_CN/        # 中文图表PNG
├── 4_Figures_EN/        # 英文图表PNG
├── 5_codes/             # Python代码
│   ├── .python-version      # Python版本（3.10+）
│   ├── requirements.txt     # 依赖清单
│   ├── npm_core.py          # NPM核心方程
│   ├── sigma_estimation.py  # PR法估算σ
│   ├── npm_validation.py    # A层数值检查
│   ├── npm_scaling_check.py # B层vs Kaplan/Chinchilla相关性
│   ├── npm_dimensionless.py # Cn无量纲数实证分析（14个模型）
│   ├── npm_pore_analysis.py # MIS孔隙形态分析（新增）
│   └── npm_param_fit.py     # B层参数拟合（新增）
├── README.md            # 英文说明
└── README_CN.md         # 本文件（中文说明）
```

## 编译方法

**英文版（arXiv提交）：**
```bash
cd 2_Paper_EN/
pdflatex npm_arxiv_paper.tex
pdflatex npm_arxiv_paper.tex  # 运行两次以解析交叉引用
```

**中文版：**
```bash
cd 1_Paper_CN/
xelatex npm_paper_CN.tex
xelatex npm_paper_CN.tex
```
需要：Noto CJK字体 + texlive-lang-chinese

## Python 环境配置

需要 Python 3.10 及以上版本。

```bash
cd 5_codes/
python -m venv .venv
source .venv/bin/activate   # Windows下: .venv\Scripts\activate
pip install -r requirements.txt
```

## 快速开始

```python
from npm_core import compute_density, compute_ceff, compute_critical_density

# 示例：100B token，中等分布广度
# 参数已拟合（详见 npm_param_fit.py）
rho = compute_density(N_B=100, sigma=0.5)
rho_c = compute_critical_density(sigma=0.5)
c_eff = compute_ceff(rho, rho_c)

print(f"数据密度: {rho:.1f}")
print(f"临界密度: {rho_c:.1f}")
print(f"泛化能力: {c_eff:.4f}")
```

## MIS孔隙形态分析实验（新增）

借鉴PoreSpy的MIS（最大内切球）方法，将PNM孔隙分析工具"翻译"到NN权重空间。在GPT-2 (124M) 和 Pythia-70m 上完成6项实验。

### 三个重要发现

**发现1：NN权重分布与PNM孔径"形状相似但族不同"**

|  | NN (GPT-2) | PNM (随机多孔介质) |
|--|-----------|-----------------|
| Weibull形状参数 k | 1.14（指数型，无峰） | 2.4~2.9（有峰，典型尺度） |
| 尾部幂律指数 α | 4.36（中等重尾） | 8~10（轻尾） |
| 变异系数 CV | 0.86 | 0.35~0.45 |
| Rank相关(vs PNM φ=0.7) | — | 0.983 |

消除尺度差异后，分布形状高度相似(r=0.98)。NN发展出比天然岩石更极端的异质结构——少数大权重承载主信息流，更像人工定向孔道材料而非天然砂岩。

**发现2：训练 = 渗流通道优化**

GPT-2训练使|w|均值扩大6.4倍，但标准差增长更快(7.3倍)。大孔率从0%→41%。训练选择性放大少数主通道、抑制其余——PNM中的"优势渗流路径"形成过程。

**发现3：Pythia训练轨迹揭示三阶段相变（核心发现）**

| 阶段 | 训练步数 | NN行为 | PNM类比 |
|------|---------|--------|---------|
| 蓄压期 | 0 – 1,000 | CV稳定在0.795，权重几乎不变 | 压力未达启动阈值 |
| 突破期 | 1,000 – 16,000 | \|w\|翻倍，大孔率峰值8.4% | 流体侵入最大孔喉 |
| 重组期 | 64,000 – 143,000 | \|w\|**反降**，但CV暴增(0.82→1.51) | 少数主通道被选中，其余关闭 |

**相变点：step 64,000 → 128,000** (ΔCV=+0.49)。这是渗流理论中"巨连通分量形成"在NN训练中的直接对应。

### 运行实验

```bash
cd 5_codes/
source .venv/bin/activate
pip install torch transformers porespy
python npm_pore_analysis.py
```

### 诚实声明

- 分布族差异(Weibull k=1.1 vs 2.5)说明类比是结构性的，非数学恒等
- 相变点精度受限于checkpoint间距
- 仅在GPT-2和Pythia-70m上验证，需更多模型和规模

## 当前状态

| 组件 | 状态 |
|------|------|
| 统一主方程（A层） | ✓ 数值验证通过，误差=0 |
| GNN–PNM等价 | ✓ 精确数学等价 |
| 注意力流量守恒 | ✓ 已验证 |
| σ单调性（PR法） | ✓ 合成数据上已验证 |
| MIS孔隙分析（权重分布） | ✓ 6项实验完成，GPT-2 & Pythia-70m |
| 训练相变（Pythia） | ✓ 三阶段模式已识别 |
| B层参数（α, β, δ, k） | ✓ 已拟合: α=0.57, β=0.068, δ=2.76, k=68.5 (Spearman=0.70, Kaplan斜率匹配) |
| 能力涌现顺序 | △ 已预测，待BIG-Bench验证 |

## 合作邀请

本项目开放合作：
- 拥有Pythia/OLMo checkpoint的研究者 — 进行定量验证
- 理论研究者 — 深化Transformer–PNM映射（浓度依赖扩散系数猜想）
- 工程师 — 将NPM应用于架构设计决策
- 实验者 — 在更大模型上验证三阶段训练相变

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
- 公众号文章：神经CAE
- 第一篇《AI本质论》：见公众号历史文章
