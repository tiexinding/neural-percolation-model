"""
Neural Percolation Model (NPM) — Core Implementation v0.2
神经渗流模型 · 核心代码

Authors: Ding Tiexin + Claude
Date: 2026-03-23
Version: 0.2.0

Three-sentence summary / 三句话：
  Data exists as scattered fragments.          数据是分散的碎片。
  Neural networks connect them via percolation. 神经网络把它们连通。
  When connectivity exceeds critical threshold, capability emerges.
                                                连通超过临界，能力涌现。

Engineering purpose / 工程用途：
  Given dataset statistics → recommended model architecture.
  给定数据集统计量 → 输出模型结构建议。

Analogy / 类比来源：
  Pore Network Model (PNM) in porous media simulation
  多孔介质孔网络模型（PNM）

  data fragments  ≈  pores      数据碎片 ≈ 孔隙
  NN weights      ≈  throats    网络权重 ≈ 孔喉
  percolation threshold ≈ emergence point  渗流阈值 ≈ 涌现临界
  diffusion       ≈  information propagation  扩散 ≈ 信息传播

Status / 当前状态：
  Critical exponents fitted against 14 public models + Kaplan scaling law.
  See npm_param_fit.py for fitting procedure.
  临界指数已用14个公开模型+Kaplan定律拟合校准。详见npm_param_fit.py。
"""

import math
from dataclasses import dataclass, field


# ── 临界指数（经验校准值，2026-03-25拟合）─────────────────────────────────
# Fitted against 14 public models (MMLU) + Kaplan scaling law (N^-0.076).
# See npm_param_fit.py for fitting procedure and honesty notes.
# 原始物理估计值: α=1.5, β=1.4, δ=1.0, k=10.0
ALPHA   = 0.57   # concept volume:   V(σ)   = exp(α·σ)
BETA    = 0.068  # capability:       C_eff  ∝ (ρ − ρ_c)^β  (3D perc β≈0.41)
DELTA   = 2.76   # threshold:        ρ_c    ∝ σ^δ
K_BASE  = 68.5   # ρ_c proportionality constant

# 归一化基准：N以十亿token为单位输入（ρ的实际范围取决于N和σ，无固定上限）
# Normalization: N input in billions of tokens
N_UNIT  = 1e9


@dataclass
class DataProfile:
    """
    数据集的三个本征量 / Three intrinsic quantities of a dataset.

    N     : 数据量，单位：token数
            Total tokens (raw count, e.g. 3e12 for 3T tokens)
    sigma : 分布广度，0~1
            Conceptual spread. Estimate from domain diversity:
              0.1~0.3 = narrow domain (code, law, medicine)
              0.4~0.6 = moderate (general web)
              0.7~1.0 = very wide (multilingual + multimodal + all domains)
    task_depth : 任务推理深度，≥1
            Reasoning chain length required:
              1   = factual retrieval (事实检索)
              2~3 = summarization / translation (摘要/翻译)
              4~5 = mathematical proof / complex reasoning (数学/复杂推理)
    """
    N:          float
    sigma:      float
    task_depth: float = 1.0

    def __post_init__(self):
        if self.N <= 0:
            raise ValueError("N must be positive")
        if not (0 < self.sigma <= 1):
            raise ValueError("sigma must be in (0, 1]")
        if self.task_depth < 1:
            raise ValueError("task_depth must be >= 1")


@dataclass
class ModelSpec:
    """NPM输出的模型规格 / Model specification output from NPM."""
    min_layers:          int
    min_params_B:        float    # 十亿参数 / billions of parameters
    min_layer_width:     float    # 层间交互强度下限（归一化）
    critical_density:    float    # ρ_c
    actual_density:      float    # ρ
    emergence_margin:    float    # (ρ − ρ_c) / ρ_c
    predicted_capability:float    # C_eff (归一化，越大越强)
    diffusion_coeff:     float    # D_eff（信息传播速度）
    warnings:            list = field(default_factory=list)

    @property
    def emerged(self) -> bool:
        return self.emergence_margin > 0


class NeuralPercolationModel:
    """
    神经渗流模型（NPM）

    Core equations / 核心方程：

      V(σ)   = exp(α · σ)                   概念空间体积
      ρ      = N_B / V(σ)                   数据密度  (N_B = N/1e9)
      ρ_c    = k · σ^δ                      临界密度
      C_eff  = (ρ − ρ_c)^β   if ρ > ρ_c   泛化能力
             = 0              otherwise
      D_eff  = C_eff · W²                   有效扩散系数（墨水扩散速度）

      L_min  = ceil(α · σ^α · τ · 10)      最小层数  (τ = task_depth)
      P_min  = N_B · σ^α                   最小参数量（十亿）
      W_c    = σ^(δ/2)                      层间交互强度下限
    """

    def __init__(self,
                 alpha:  float = ALPHA,
                 beta:   float = BETA,
                 delta:  float = DELTA,
                 k_base: float = K_BASE):
        self.alpha  = alpha
        self.beta   = beta
        self.delta  = delta
        self.k_base = k_base

    # ── 核心计算 ────────────────────────────────────────────────────────────

    def _concept_volume(self, sigma: float) -> float:
        """V(σ) = exp(α·σ)"""
        return math.exp(self.alpha * sigma)

    def _data_density(self, N: float, sigma: float) -> float:
        """ρ = N_billions / V(σ)"""
        N_B = N / N_UNIT
        return N_B / self._concept_volume(sigma)

    def _critical_density(self, sigma: float) -> float:
        """ρ_c = k · σ^δ"""
        return self.k_base * (sigma ** self.delta)

    def _capability(self, rho: float, rho_c: float) -> float:
        """C_eff = (ρ − ρ_c)^β  if ρ > ρ_c, else 0"""
        if rho <= rho_c:
            return 0.0
        return (rho - rho_c) ** self.beta

    def _diffusion(self, capability: float, layer_width: float) -> float:
        """D_eff = C_eff · W²   (墨水扩散速度)"""
        return capability * (layer_width ** 2)

    # ── 架构推导 ────────────────────────────────────────────────────────────

    def _min_layers(self, sigma: float, task_depth: float) -> int:
        """L_min ∝ σ^α · task_depth"""
        raw = self.alpha * (sigma ** self.alpha) * task_depth * 10
        return max(2, int(math.ceil(raw)))

    def _min_params_B(self, N: float, sigma: float) -> float:
        """P_min (十亿参数) = N_billions · σ^α"""
        N_B = N / N_UNIT
        return N_B * (sigma ** self.alpha)

    def _min_layer_width(self, sigma: float) -> float:
        """W_c = σ^(δ/2)"""
        return sigma ** (self.delta / 2)

    # ── 主接口 ──────────────────────────────────────────────────────────────

    def analyze(self, data: DataProfile) -> ModelSpec:
        """给定数据集 → 返回模型规格 / Dataset → ModelSpec"""
        rho    = self._data_density(data.N, data.sigma)
        rho_c  = self._critical_density(data.sigma)
        W      = self._min_layer_width(data.sigma)
        C_eff  = self._capability(rho, rho_c)
        D_eff  = self._diffusion(C_eff, W)
        L      = self._min_layers(data.sigma, data.task_depth)
        P_B    = self._min_params_B(data.N, data.sigma)
        margin = (rho - rho_c) / rho_c if rho_c > 0 else 0.0

        warns = []
        if rho < rho_c:
            gap = (rho_c - rho) / rho_c * 100
            warns.append(
                f"⚠  未达渗流临界：数据密度不足 {gap:.0f}%。"
                "建议：增大N 或 缩小σ（聚焦领域）"
            )
        elif margin < 0.2:
            warns.append("⚠  刚超临界，能力极弱。建议增加数据量或优化数据分布密度。")
        if data.sigma > 0.75 and P_B < 10:
            warns.append("⚠  宽域数据需要大参数量，当前参数量估算偏低，涌现能力受限。")

        return ModelSpec(
            min_layers=L,
            min_params_B=round(P_B, 2),
            min_layer_width=round(W, 4),
            critical_density=round(rho_c, 3),
            actual_density=round(rho, 3),
            emergence_margin=round(margin, 3),
            predicted_capability=round(C_eff, 4),
            diffusion_coeff=round(D_eff, 4),
            warnings=warns,
        )

    def explain(self, data: DataProfile) -> str:
        """自然语言报告 / Human-readable report with percolation metaphor"""
        s = self.analyze(data)
        N_B = data.N / N_UNIT

        def sigma_desc(σ):
            if σ < 0.35: return "窄域（领域专用）"
            if σ < 0.65: return "中域（跨领域）"
            return "宽域（通用/多语言）"

        def depth_desc(d):
            if d < 2:  return "浅层任务（事实检索）"
            if d < 4:  return "中层任务（摘要/翻译）"
            return "深层任务（复杂推理/数学）"

        def water_metaphor(margin):
            if margin <= 0:
                return ("数据如孤立水坑，互不相通。\n"
                        "  滴入墨水只在坑内扩散——信息无法跨域传播，模型无泛化能力。")
            if margin < 0.5:
                return ("水坑刚刚连通，形成细流。\n"
                        "  墨水能流动但路径细窄——模型有基础能力，跨域推理和细粒度感知还弱。")
            if margin < 2.0:
                return ("水系贯通，形成稳定流域。\n"
                        "  滴墨水，感知扩散至相连水域——模型泛化能力良好，跨域推理可靠。")
            return ("水系全域贯通，形成巨连通分量。\n"
                    "  一滴墨水，感应遍布全域——这就是涌现。模型具备强泛化与复杂推理能力。")

        lines = [
            "┌─ 神经渗流模型 NPM v0.2 · 分析报告 ─────────────────────┐",
            f"│  数据量          N  = {N_B:.1f} B tokens",
            f"│  分布广度        σ  = {data.sigma:.2f}   {sigma_desc(data.sigma)}",
            f"│  任务深度        τ  = {data.task_depth:.1f}   {depth_desc(data.task_depth)}",
            "├─ 渗流状态 ──────────────────────────────────────────────┤",
            f"│  数据密度        ρ  = {s.actual_density:.2f}",
            f"│  临界密度       ρ_c = {s.critical_density:.2f}",
            f"│  超临界余量         = {s.emergence_margin*100:+.1f}%"
            f"  {'✓ 已涌现' if s.emerged else '✗ 未达临界'}",
            f"│  泛化能力      C_eff= {s.predicted_capability:.4f}",
            f"│  信息传播速度  D_eff= {s.diffusion_coeff:.4f}",
            "├─ 架构推荐 ──────────────────────────────────────────────┤",
            f"│  最小层数           = {s.min_layers} 层",
            f"│  最小参数量         = {s.min_params_B:.1f} B  ({s.min_params_B*1000:.0f} M)",
            f"│  层间交互下限    W_c = {s.min_layer_width:.4f}",
            "├─ 直觉解释（水坑比喻）──────────────────────────────────┤",
            f"│  {water_metaphor(s.emergence_margin)}",
        ]
        if s.warnings:
            lines.append("├─ 警告 ─────────────────────────────────────────────────┤")
            for w in s.warnings:
                lines.append(f"│  {w}")
        lines.append("├─ 说明 ─────────────────────────────────────────────────┤")
        lines.append("│  临界指数已拟合（详见npm_param_fit.py），σ需自行估算")
        lines.append("│  σ 估算方法：单一领域≈0.2，全网络数据≈0.8，可自行校准")
        lines.append("└─────────────────────────────────────────────────────────┘")
        return "\n".join(lines)

    def compare(self, scenarios: list[tuple[str, DataProfile]]) -> str:
        """场景对比表 / Side-by-side comparison"""
        header = (f"{'场景':<18} {'N(B)':>7} {'σ':>5} {'ρ':>8} "
                  f"{'ρ_c':>6} {'余量':>8} {'层数':>5} {'C_eff':>8} {'状态':>5}")
        sep = "─" * 78
        lines = [sep, header, sep]
        for name, dp in scenarios:
            s = self.analyze(dp)
            N_B = dp.N / N_UNIT
            status = "✓ 涌现" if s.emerged else "✗ 未达"
            lines.append(
                f"{name:<18} {N_B:>7.0f} {dp.sigma:>5.2f} "
                f"{s.actual_density:>8.2f} {s.critical_density:>6.2f} "
                f"{s.emergence_margin*100:>+7.0f}% "
                f"{s.min_layers:>5} {s.predicted_capability:>8.4f} {status:>6}"
            )
        lines.append(sep)
        return "\n".join(lines)


# ── 便捷函数（供 main.py 等脚本直接调用）─────────────────────────────────────

_default = NeuralPercolationModel()


def compute_density(N_B: float, sigma: float, alpha: float = ALPHA) -> float:
    """ρ = N_B / exp(α·σ)，N_B单位：十亿token"""
    return N_B / math.exp(alpha * sigma)


def compute_critical_density(sigma: float, tau: float = 1.0,
                             k: float = K_BASE, delta: float = DELTA) -> float:
    """ρ_c = k · σ^δ · τ   (τ默认=1.0，即不含任务深度修正)"""
    return k * (sigma ** delta) * tau


def compute_ceff(rho: float, rho_c: float, beta: float = BETA) -> float:
    """C_eff = (ρ − ρ_c)^β  if ρ > ρ_c, else 0"""
    if rho <= rho_c:
        return 0.0
    return (rho - rho_c) ** beta


# ── 演示 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    npm = NeuralPercolationModel()

    print("\n" + "="*60)
    print("  神经渗流模型 NPM v0.2  ·  Neural Percolation Model")
    print("="*60)

    scenarios = [
        ("GPT-3级通用模型",   DataProfile(N=300e9, sigma=0.75, task_depth=3.0)),
        ("GPT-4级通用模型",   DataProfile(N=3e12,  sigma=0.90, task_depth=5.0)),
        ("代码专用（小）",     DataProfile(N=50e9,  sigma=0.25, task_depth=2.0)),
        ("医疗专用（窄域）",   DataProfile(N=10e9,  sigma=0.20, task_depth=3.0)),
        ("多语言翻译",         DataProfile(N=500e9, sigma=0.65, task_depth=1.5)),
        ("数学推理专项",       DataProfile(N=20e9,  sigma=0.35, task_depth=5.0)),
        ("未达临界（太小）",   DataProfile(N=1e9,   sigma=0.60, task_depth=3.0)),
    ]

    # 详细报告：GPT级 vs 专用小模型
    for name, dp in scenarios[:2]:
        print(f"\n【{name}】")
        print(npm.explain(dp))

    # 对比表：所有场景
    print("\n【场景对比表】")
    print(npm.compare(scenarios))

    # 演示"数据密度"比"数据量"更本质
    print("\n【关键演示：相同数据量，不同分布广度，效果大不同】")
    same_N = [
        ("同等数据·窄域σ=0.2", DataProfile(N=100e9, sigma=0.20, task_depth=2.0)),
        ("同等数据·中域σ=0.5", DataProfile(N=100e9, sigma=0.50, task_depth=2.0)),
        ("同等数据·宽域σ=0.8", DataProfile(N=100e9, sigma=0.80, task_depth=2.0)),
    ]
    print(npm.compare(same_N))
    print("→ 同样100B tokens，领域越宽，密度越低，需要的模型越大，能力反而越弱")
    print("→ 这解释了为什么通用模型比专用模型需要更多数据和更大参数量")
