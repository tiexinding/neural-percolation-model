"""
NPM B层参数拟合：用公开数据校准 α, β, δ, k
================================================
拟合策略：
  用14个模型的(N, σ, MMLU)数据 + Kaplan scaling law + Chinchilla配比
  联合优化 α, β, δ, k，同时满足多个约束。

约束来源：
  1. MMLU排序: Spearman(C_eff, MMLU) 尽量高
  2. Kaplan:   C_eff ∝ N^0.076 (σ=0.75时的log-log斜率)
  3. Chinchilla: P_min = N_B · σ^α 给出合理量级

关键发现（先说结论）：
  α同时影响V(σ)=exp(ασ)和P_min=N_B·σ^α，这两个约束对α的要求矛盾
  （Chinchilla要α≈10，密度公式要α≈1~2）。
  解决方案：放弃用α同时拟合P_min，聚焦于密度-能力关系的校准。

数据来源：Kaplan (2020), Hoffmann (2022), LLaMA/Pythia/BLOOM论文
作者：丁铁新 · 2026-03-25
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import spearmanr

# ══════════════════════════════════════════════════════
# 公开模型数据
# ══════════════════════════════════════════════════════
models = [
    # (名称, N_B十亿tokens, σ估计, 参数量B, MMLU近似)
    ("GPT-2 (1.5B)",          40,    0.70,   1.5,   35),
    ("Pythia-1B",             300,   0.70,   1.0,   27),
    ("Pythia-6.9B",           300,   0.70,   6.9,   35),
    ("Pythia-12B",            300,   0.70,  12.0,   38),
    ("LLaMA-7B",             1000,   0.75,   7.0,   35),
    ("LLaMA-13B",            1000,   0.75,  13.0,   47),
    ("LLaMA-33B",            1400,   0.75,  33.0,   58),
    ("LLaMA-65B",            1400,   0.75,  65.0,   64),
    ("GPT-3 (175B)",          300,   0.75, 175.0,   44),
    ("Chinchilla (70B)",     1400,   0.75,  70.0,   68),
    ("LLaMA-2-70B",          2000,   0.75,  70.0,   69),
    ("CodeLlama-7B",          500,   0.30,   7.0,   31),
    ("CodeLlama-34B",         500,   0.30,  34.0,   42),
    ("BLOOM-176B",            366,   0.85, 176.0,   39),
]

N_data = np.array([m[1] for m in models], dtype=float)
sigma_data = np.array([m[2] for m in models], dtype=float)
mmlu_data = np.array([m[4] for m in models], dtype=float)


def npm_ceff(N_B, sigma, alpha, beta, delta, k):
    """NPM泛化能力: C_eff = (ρ - ρ_c)^β"""
    rho   = N_B / np.exp(alpha * sigma)
    rho_c = k * sigma**delta
    if rho <= rho_c:
        return 0.0
    return (rho - rho_c)**beta


def npm_ceff_vec(N_arr, s_arr, alpha, beta, delta, k):
    return np.array([npm_ceff(n, s, alpha, beta, delta, k)
                     for n, s in zip(N_arr, s_arr)])


# ══════════════════════════════════════════════════════
# 原参数基线
# ══════════════════════════════════════════════════════
print("="*70)
print("原参数基线 (α=1.5, β=1.4, δ=1.0, k=10)")
print("="*70)
print()

ceffs_old = npm_ceff_vec(N_data, sigma_data, 1.5, 1.4, 1.0, 10.0)
r_old, p_old = spearmanr(ceffs_old, mmlu_data)
print(f"  Spearman(C_eff, MMLU) = {r_old:.4f}  (p = {p_old:.4f})")

# Kaplan斜率
sigma_web = 0.75
N_scan = np.array([10, 30, 100, 300, 1000, 3000, 10000], dtype=float)
c_scan_old = npm_ceff_vec(N_scan, np.full_like(N_scan, sigma_web), 1.5, 1.4, 1.0, 10.0)
mask = c_scan_old > 0
if mask.sum() >= 3:
    slope_old = np.polyfit(np.log10(N_scan[mask]), np.log10(c_scan_old[mask]), 1)[0]
    print(f"  Kaplan斜率: C_eff ∝ N^{slope_old:.3f}  (目标: N^0.076)")
else:
    slope_old = None
    print(f"  Kaplan斜率: 无法计算（C_eff全为0）")


# ══════════════════════════════════════════════════════
# 联合优化
# ══════════════════════════════════════════════════════
print("\n")
print("="*70)
print("联合优化: 最大化Spearman + Kaplan斜率约束")
print("="*70)
print()

target_slope = 0.076

def objective(params):
    alpha, beta, delta, k = params

    # C_eff vs MMLU
    ceffs = npm_ceff_vec(N_data, sigma_data, alpha, beta, delta, k)
    if np.all(ceffs == 0) or len(np.unique(ceffs)) < 3:
        return 10.0

    r_s, _ = spearmanr(ceffs, mmlu_data)

    # Kaplan斜率约束
    c_scan = npm_ceff_vec(N_scan, np.full_like(N_scan, sigma_web),
                          alpha, beta, delta, k)
    mask = c_scan > 0
    slope_err = 0
    if mask.sum() >= 3:
        slope = np.polyfit(np.log10(N_scan[mask]), np.log10(c_scan[mask]), 1)[0]
        slope_err = (slope - target_slope)**2 * 200  # 强约束
    else:
        slope_err = 5.0

    # 惩罚所有模型C_eff=0的情况（至少一半模型应该超过临界）
    active_frac = (ceffs > 0).mean()
    if active_frac < 0.5:
        return 10.0 - active_frac

    return -r_s + slope_err


# 用差分进化（全局优化，不依赖初始值）
bounds = [
    (0.3, 4.0),   # α
    (0.01, 0.5),  # β
    (0.3, 3.0),   # δ
    (0.5, 100),   # k
]

print("  差分进化全局优化...")
print(f"  参数范围: α∈[0.3,4], β∈[0.01,0.5], δ∈[0.3,3], k∈[0.5,100]")
print()

result = differential_evolution(objective, bounds, seed=42,
                                 maxiter=2000, tol=1e-10, popsize=30)

alpha_fit, beta_fit, delta_fit, k_fit = result.x

# 也用Nelder-Mead从多个初始点精细优化
for x0 in [[alpha_fit, beta_fit, delta_fit, k_fit],
            [1.5, 0.1, 1.0, 10],
            [2.0, 0.05, 0.5, 5],
            [1.0, 0.2, 1.5, 20]]:
    res2 = minimize(objective, x0, method='Nelder-Mead',
                    options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-10})
    if res2.fun < result.fun:
        result = res2
        alpha_fit, beta_fit, delta_fit, k_fit = result.x

print(f"  拟合结果:")
print(f"    α = {alpha_fit:.4f}  (原: 1.5000)")
print(f"    β = {beta_fit:.4f}  (原: 1.4000)")
print(f"    δ = {delta_fit:.4f}  (原: 1.0000)")
print(f"    k = {k_fit:.4f}  (原: 10.0000)")

# 验证
ceffs_new = npm_ceff_vec(N_data, sigma_data, alpha_fit, beta_fit, delta_fit, k_fit)
r_new, p_new = spearmanr(ceffs_new, mmlu_data)
print(f"\n  Spearman(C_eff, MMLU) = {r_new:.4f}  (原: {r_old:.4f}, 改善: {r_new-r_old:+.4f})")

# Kaplan验证
c_scan_new = npm_ceff_vec(N_scan, np.full_like(N_scan, sigma_web),
                           alpha_fit, beta_fit, delta_fit, k_fit)
mask = c_scan_new > 0
if mask.sum() >= 3:
    slope_new = np.polyfit(np.log10(N_scan[mask]), np.log10(c_scan_new[mask]), 1)[0]
    print(f"  Kaplan斜率: C_eff ∝ N^{slope_new:.4f}  (目标: N^0.076)")
else:
    slope_new = None
    print(f"  Kaplan斜率: 无法计算")


# ══════════════════════════════════════════════════════
# α对P_min的影响分析
# ══════════════════════════════════════════════════════
print("\n")
print("="*70)
print("α 对 P_min 的影响（诚实分析）")
print("="*70)
print()
print(f"  NPM公式: P_min = N_B · σ^α")
print(f"  拟合 α = {alpha_fit:.4f}")
print(f"  σ=0.75时: σ^α = {sigma_web**alpha_fit:.4f}")
print(f"  Chinchilla实际: P/N ≈ 0.05")
print()

if sigma_web**alpha_fit > 0.01:
    print(f"  P_min偏差: σ^α / 0.05 = {sigma_web**alpha_fit / 0.05:.1f}x")
    if abs(sigma_web**alpha_fit - 0.05) / 0.05 > 0.5:
        print(f"  ⚠ P_min公式与Chinchilla偏差超过50%")
        print(f"  原因: α同时出现在V(σ)=exp(ασ)和P_min=N_B·σ^α中")
        print(f"  密度公式V(σ)对α敏感度高 → α被密度-能力关系锁定")
        print(f"  P_min公式需要独立的指数，或引入额外校正系数")
        print(f"  建议: P_min = N_B · σ^α / C_p, 其中C_p为校正因子")
        C_p = sigma_web**alpha_fit / 0.05
        print(f"  校正因子 C_p = {C_p:.2f}")


# ══════════════════════════════════════════════════════
# 逐模型对比
# ══════════════════════════════════════════════════════
print("\n")
print("="*70)
print("逐模型对比")
print("="*70)
print()
print(f"  {'模型':<22} {'MMLU':>5} {'C_eff(新)':>10} {'C_eff(旧)':>10} "
      f"{'ρ(新)':>8} {'ρ_c(新)':>8} {'Cn(新)':>8}")
print("  " + "-"*76)

for i, (name, N_B, sigma, P_B, mmlu) in enumerate(models):
    c_new = ceffs_new[i]
    c_old = ceffs_old[i]
    rho_new = N_B / np.exp(alpha_fit * sigma)
    rho_c_new = k_fit * sigma**delta_fit
    cn_new = rho_new / rho_c_new if rho_c_new > 0 else 0
    print(f"  {name:<22} {mmlu:>5} {c_new:>10.4f} {c_old:>10.1f} "
          f"{rho_new:>8.2f} {rho_c_new:>8.2f} {cn_new:>8.2f}")


# ══════════════════════════════════════════════════════
# 反常现象检验
# ══════════════════════════════════════════════════════
print("\n")
print("="*70)
print("反常现象检验（拟合后参数）")
print("="*70)
print()

# BLOOM vs LLaMA
bloom_c = npm_ceff(366, 0.85, alpha_fit, beta_fit, delta_fit, k_fit)
llama7_c = npm_ceff(1000, 0.75, alpha_fit, beta_fit, delta_fit, k_fit)
print(f"  BLOOM-176B (MMLU=39):  C_eff = {bloom_c:.4f}")
print(f"  LLaMA-7B   (MMLU=35): C_eff = {llama7_c:.4f}")
print(f"  NPM{'正确预测' if bloom_c < llama7_c else '未能预测'}: BLOOM弱于LLaMA-7B")

# Chinchilla vs GPT-3
chin_c = npm_ceff(1400, 0.75, alpha_fit, beta_fit, delta_fit, k_fit)
gpt3_c = npm_ceff(300, 0.75, alpha_fit, beta_fit, delta_fit, k_fit)
print(f"\n  Chinchilla (MMLU=68): C_eff = {chin_c:.4f}")
print(f"  GPT-3      (MMLU=44): C_eff = {gpt3_c:.4f}")
print(f"  NPM{'正确预测' if chin_c > gpt3_c else '未能预测'}: Chinchilla强于GPT-3")


# ══════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════
print("\n")
print("="*70)
print("B层参数拟合总结")
print("="*70)

# 3D渗流理论参考值
print(f"""
  拟合结果                       3D渗流理论参考
  ─────────────                 ────────────────
  α = {alpha_fit:.4f}                    —（无直接对应）
  β = {beta_fit:.4f}                    β_perc ≈ 0.41 (3D)
  δ = {delta_fit:.4f}                    —
  k = {k_fit:.4f}                    —

  β拟合值 vs 3D渗流:
    β_fit = {beta_fit:.4f}, β_3D = 0.41
    {'接近3D渗流理论值' if abs(beta_fit - 0.41) < 0.15 else '偏离3D渗流理论值'}
    {'这支持NPM-PNM类比的物理基础' if abs(beta_fit - 0.41) < 0.15 else '偏离可能因为NN不是严格3D渗流系统'}

  拟合质量:
    Spearman(C_eff, MMLU) = {r_new:.4f}
    Kaplan斜率: {'N^' + f'{slope_new:.4f}' if slope_new else '无法计算'} (目标: N^0.076)

  诚实声明:
    - 14个数据点拟合4个参数，存在过拟合风险
    - σ值为人工估计（非客观测量），是最大不确定性来源
    - Pythia系列(同σ不同规模)显示C_eff对模型大小不敏感
      这暴露了NPM当前公式仅含N和σ，缺少P(参数量)的独立贡献
    - P_min公式中α与密度公式共用产生矛盾，需要解耦
    - 这是初步校准，待更多数据（如OLMo、Gemma）验证稳定性
""")
