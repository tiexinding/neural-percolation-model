# Neural Percolation Model (NPM)

**An Engineering Analogy Framework for Neural Network Information Propagation and Capability Emergence**

With companion article series "AI Is Not Mysterious" (11 articles in Chinese), introducing four original concepts: NPM / EVM / Three-Layer Isomorphism / Learning from AI as a Teacher

Author: Tiexin Ding (丁铁新) · NeuralCAE · Independent Researcher, China

---

## Overview

NPM maps the physics of Pore Network Models (PNM) — widely used in reservoir engineering and materials science — onto neural network information propagation. Starting from a single conservation principle (steady-state nodal balance), the framework derives a unified master equation:

**a_out = W_eff · a_in**

where W_eff takes three forms covering all major architectures: feedforward networks (fixed weights), GNNs (graph Laplacian, exact PNM analog, verified with zero error), and Transformers (softmax attention as flow conservation).

## Four Original Concepts

| Concept | Nature | First Proposed | Core Content |
|---------|--------|---------------|-------------|
| **NPM (Neural Percolation Model)** | Mathematical framework | Article 2 | GNN message passing = PNM nodal conservation, 4 isomorphisms, verification error = 0 |
| **EVM (Engineering Verification Model)** | Application roadmap | Article 7 | 5-stage verification roadmap for Physical AI, V-model mapping |
| **Three-Layer Isomorphism** | Theoretical discovery | Article 8 | Natural percolation / AI capability emergence / human cognitive emergence follow the same mathematics |
| **Learning from AI as a Teacher** | Methodology | Article 9 | Lower your stance, learn AI by using AI; judgment is the essence of intelligence |

## Article Series: AI Is Not Mysterious — A CAE Engineer's Perspective

Eleven articles (in Chinese) systematically analyze AI from first principles using the language of CAE engineers. Full text available in `6_Articles_CN/` directory, also published on Zhihu and WeChat (NeuralCAE).

| # | Title | Core Content |
|---|-------|-------------|
| 1 | What Is AI — First Principles Reasoning | AI is a multimodal projection of nature's interaction processes |
| 2 | What Neural Networks Do — NPM | Projection forms through percolation connectivity |
| 3 | AI Is Not Mysterious — When Percolation Meets NN | GNN = PNM mathematical verification, error = 0 |
| 4 | Neural Networks and CAE — A Panoramic Map | Four isomorphisms, shared mathematical foundation |
| 5 | Current AI Is Not Perfect — From Demystification to Debugging | MeshGraphNet hands-on, nine gaps all pointing to verification |
| 6 | Where Is the Road — Engineering World Models | Data + compute + verification: three legs |
| 7 | Physical AI Roadmap — Engineering Verification Model | EVM 5-stage roadmap, V-model mapping |
| 8 | Cognitive Percolation | Creative process documentary, three-layer isomorphism discovery |
| 9 | Insights Beyond Reasoning | Change vs. permanence in the AI era, learning from AI |
| 10 | Creator Interview | Five voices answer independently, human-AI collaboration in practice |
| 11 | Gratitude | To mentors, family, open-source community, and readers |

## Key Contributions

1. **Unified master equation** — three architecture instantiations from one conservation principle
2. **Multi-threshold field view** — emergence as a continuous capability spectrum, not a single phase transition
3. **Participation Ratio (PR) method** — quantifying data distribution breadth σ from covariance spectra
4. **17 LLM phenomena explained** — under one physical mechanism, with 3 falsifiable predictions
5. **Engineering framework positioning** — in the tradition of Darcy's Law and the Reynolds number

## Repository Structure

```
neural-percolation-model/
├── 1_Paper_CN/          # Chinese paper (xelatex + xeCJK)
├── 2_Paper_EN/          # English paper (pdflatex, arXiv-ready)
├── 3_Figures_CN/        # Chinese figure PNGs
├── 4_Figures_EN/        # English figure PNGs
├── 5_codes/             # Python implementation
│   ├── npm_core.py          # Core NPM equations
│   ├── sigma_estimation.py  # PR method for σ estimation
│   ├── npm_validation.py    # A-layer numerical checks
│   ├── npm_scaling_check.py # B-layer vs Kaplan/Chinchilla
│   ├── npm_dimensionless.py # Cn empirical analysis (14 models)
│   ├── npm_pore_analysis.py # MIS-based pore analysis
│   └── npm_param_fit.py     # B-layer parameter fitting
├── 6_Articles_CN/       # Chinese article series (9 articles)
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
├── README.md            # This file (English)
└── README_CN.md         # Chinese README
```

## Compilation & Running

**English paper (arXiv submission):**
```bash
cd 2_Paper_EN/ && pdflatex npm_arxiv_paper.tex && pdflatex npm_arxiv_paper.tex
```

**Chinese paper:**
```bash
cd 1_Paper_CN/ && xelatex npm_paper_CN.tex && xelatex npm_paper_CN.tex
```

**Python code:**
```bash
cd 5_codes/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python npm_core.py
```

## MIS Pore Analysis (Core Experiments)

Inspired by PoreSpy's MIS method, we mapped PNM pore analysis tools onto NN weight space. Three key findings:

1. **NN weight distributions share shape with PNM pore-size distributions but belong to different families** — rank correlation r=0.98 after scale removal; NNs develop more extreme heterogeneity than natural rock
2. **Training = percolation channel optimization** — GPT-2 training increases large-pore fraction from 0% to 41%, selectively amplifying dominant pathways
3. **Three-phase training transition (Pythia-70m)** — Pressurization → Breakthrough → Reorganization; CV surges from 0.82 to 1.51, corresponding to giant connected component formation in percolation theory

## Current Status

| Component | Status |
|-----------|--------|
| Unified master equation (A-layer) | ✓ Numerically verified, error = 0 |
| GNN–PNM equivalence | ✓ Exact mathematical equivalence |
| B-layer parameters | ✓ Fitted: α=0.57, β=0.068, δ=2.76, k=68.5 (Spearman=0.70) |
| MIS pore analysis | ✓ 6 experiments on GPT-2 & Pythia-70m |
| Training phase transition | ✓ Three-phase pattern identified |
| Article series (Chinese) | ✓ All 11 articles published |
| Capability emergence ordering | △ Predicted, awaiting BIG-Bench validation |

## Collaboration

Open invitation for collaboration: researchers with large model checkpoints, theorists, engineers, and experimentalists are all welcome.

Contact: GitHub Issues or Pull Request.

## Citation

```bibtex
@article{ding2026npm,
  title={Neural Percolation Model (NPM): An Engineering Analogy Framework
         for Neural Network Information Propagation and Capability Emergence},
  author={Ding, Tiexin},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgments

The author used Claude (Anthropic) as a writing assistance tool. All theoretical content, conceptual framework, equations, and conclusions are the sole intellectual contribution of the author.

## Links

- Zenodo Archive: https://zenodo.org/records/19209722
- arXiv Paper: [to be added after acceptance]
- Zhihu Column: AI并不神秘——一个CAE工程师的视角
- WeChat: NeuralCAE
- GitHub: https://github.com/tiexinding/neural-percolation-model

---

**NeuralCAE · Tiexin Ding**
20 years of simulation practice · Original concepts: NPM / EVM / Three-Layer Isomorphism / Learning from AI
