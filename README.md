# Neural Percolation Model (NPM)

**An Engineering Analogy Framework for Neural Network Information Propagation and Capability Emergence**

Author: Ding Tiexin (丁铁新) · NeuralCAE · Independent Researcher, China

---

## Overview

NPM maps the physics of Pore Network Models (PNM) — widely used in reservoir engineering and materials science — onto neural network information propagation. Starting from a single conservation principle (steady-state nodal balance), the framework derives a unified master equation:

**a_out = W_eff · a_in**

where W_eff takes three forms covering all major architectures: feedforward networks, GNNs (exact PNM analog, verified with zero error), and Transformers (softmax attention as flow conservation).

## Key Contributions

1. **Unified master equation** — three architecture instantiations from one conservation principle
2. **Multi-threshold field view** — emergence as a continuous capability spectrum, not a single phase transition
3. **Participation Ratio (PR) method** — quantifying data distribution breadth σ from covariance spectra
4. **17 LLM phenomena explained** — under one physical mechanism, with 3 falsifiable predictions
5. **Engineering framework positioning** — in the tradition of Darcy's Law and the Reynolds number

## Repository Structure

```
NPM_v13_complete/
├── 1_Paper_CN/          # Chinese paper (xelatex + xeCJK)
│   ├── npm_paper_CN.tex
│   ├── npm_paper_CN.pdf
│   └── NPM_Fig*_CN.png
├── 2_Paper_EN/          # English paper (pdflatex, arXiv-ready)
│   ├── npm_arxiv_paper.tex
│   ├── npm_arxiv_paper.pdf
│   └── NPM_Fig*_EN.png
├── 3_Figures_CN/        # Chinese figure PNGs
├── 4_Figures_EN/        # English figure PNGs (not present, see note)
├── 5_codes/             # Python implementation
│   ├── .python-version      # Python version (3.10+)
│   ├── requirements.txt     # Dependencies
│   ├── npm_core.py          # Core NPM equations
│   ├── sigma_estimation.py  # PR method for σ estimation
│   ├── npm_validation.py    # A-layer numerical checks
│   ├── npm_scaling_check.py # B-layer vs Kaplan/Chinchilla
│   ├── npm_dimensionless.py # Cn empirical analysis (14 models)
│   ├── npm_pore_analysis.py # MIS-based pore analysis (NEW)
│   └── npm_param_fit.py     # B-layer parameter fitting (NEW)
├── README.md            # This file (English)
└── README_CN.md         # Chinese README
```

## Compilation

**English version (arXiv submission):**
```bash
cd 2_Paper_EN/
pdflatex npm_arxiv_paper.tex
pdflatex npm_arxiv_paper.tex  # run twice for cross-references
```

**Chinese version:**
```bash
cd 1_Paper_CN/
xelatex npm_paper_CN.tex
xelatex npm_paper_CN.tex
```
Requires: Noto CJK fonts + texlive-lang-chinese

## Python Environment

Requires Python 3.10+.

```bash
cd 5_codes/
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start (Code)

```python
from npm_core import compute_density, compute_ceff, compute_critical_density

# Example: 100B tokens, medium domain breadth
# Parameters are fitted values (see npm_param_fit.py)
rho = compute_density(N_B=100, sigma=0.5)
rho_c = compute_critical_density(sigma=0.5)
c_eff = compute_ceff(rho, rho_c)

print(f"Data density: {rho:.1f}")
print(f"Critical density: {rho_c:.1f}")
print(f"Generalization capacity: {c_eff:.4f}")
```

## MIS Pore Analysis Experiments (NEW)

Inspired by PoreSpy's Maximum Inscribed Sphere (MIS) method, we mapped PNM pore analysis tools onto NN weight space. Six experiments were conducted on GPT-2 (124M) and Pythia-70m:

### Key Findings

**1. Weight distributions share shape with pore-size distributions but belong to different families**

|  | NN (GPT-2) | PNM (random porous media) |
|--|-----------|--------------------------|
| Weibull shape k | 1.14 (exponential-type) | 2.4–2.9 (peaked) |
| Tail exponent α | 4.36 (moderate heavy tail) | 8–10 (light tail) |
| CV | 0.86 | 0.35–0.45 |
| Rank correlation (vs PNM φ=0.7) | — | 0.983 |

After removing scale differences (rank normalization), distribution shapes are highly similar (r=0.98). The NN develops a more heterogeneous channel structure than natural rock — resembling engineered directional-pore ceramics rather than sandstone.

**2. Training = percolation channel optimization**

GPT-2 training increases |w| mean by 6.4×, but standard deviation grows even faster (7.3×). The "large-pore fraction" (|w|>0.1) goes from 0% to 41%. Training selectively amplifies a few dominant pathways and suppresses the rest — precisely the "preferential flow path" formation in PNM.

**3. Three-phase transition during training (Pythia-70m, 13 checkpoints)**

| Phase | Training steps | NN behavior | PNM analogy |
|-------|---------------|-------------|-------------|
| Pressurization | 0 – 1,000 | CV stable at 0.795, weights barely change | Pressure below entry threshold |
| Breakthrough | 1,000 – 16,000 | |w| doubles, large-pore fraction peaks at 8.4% | Fluid invades largest pore throats |
| Reorganization | 64,000 – 143,000 | |w| **decreases**, but CV surges 0.82→1.51 | Dominant channels selected, rest shut down |

**Phase transition point: step 64,000 → 128,000** (ΔCV = +0.49). This is the NN equivalent of "giant connected component formation" in percolation theory.

### Running the experiments

```bash
cd 5_codes/
source .venv/bin/activate
pip install torch transformers porespy
python npm_pore_analysis.py
```

### Honesty note

- Distribution-family differences (Weibull k=1.1 vs 2.5) mean the analogy is structural, not mathematical identity
- Phase transition granularity is limited by available checkpoint spacing
- Validated on GPT-2 and Pythia-70m only; more models and scales needed

## Current Status

| Component | Status |
|-----------|--------|
| Unified master equation (A-layer) | ✓ Numerically verified, error = 0 |
| GNN–PNM equivalence | ✓ Exact mathematical equivalence |
| Attention flow conservation | ✓ Verified |
| σ monotonicity (PR method) | ✓ Validated on synthetic data |
| MIS pore analysis (weight distributions) | ✓ 6 experiments completed on GPT-2 & Pythia-70m |
| Training phase transition (Pythia) | ✓ Three-phase pattern identified |
| B-layer parameters (α, β, δ, k) | ✓ Fitted: α=0.57, β=0.068, δ=2.76, k=68.5 (Spearman=0.70, Kaplan slope matched) |
| Capability emergence ordering | △ Predicted, not yet tested against BIG-Bench |

## Collaboration

This is an open invitation for collaboration:
- Researchers with access to Pythia/OLMo checkpoints for quantitative validation
- Theorists who can formalize the Transformer–PNM mapping (concentration-dependent diffusivity conjecture)
- Engineers interested in applying NPM to architecture design decisions
- Experimentalists who can validate the three-phase training transition on larger models

Contact: see GitHub Issues or submit a Pull Request.

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

## Zenodo Archive

Archived version with all experimental results: https://zenodo.org/records/19209722

## License

MIT License. See individual files for details.

## Acknowledgments

The author used Claude (Anthropic) as a writing assistance tool. All theoretical content, conceptual framework, equations, and conclusions are the sole intellectual contribution of the author.
