"""
NPM 与已知 Scaling Law 的相关性检验
====================================
目的：检验NPM公式是否与已知的经验规律数值兼容。
方法：用公开数据点计算相关系数，不做理论推导。

数据来源：
  Kaplan et al. (2020)  — Loss ∝ N^(-0.076) ∝ P^(-0.076)
  Hoffmann et al. (2022, Chinchilla) — 最优配比 N_opt ∝ P_opt

NPM公式：
  ρ     = N_B / exp(α·σ)
  ρ_c   = k · σ^δ
  C_eff = (ρ − ρ_c)^β    if ρ > ρ_c
  P_min = N_B · σ^α

作者：丁铁新 · 2026-03-24
"""

import numpy as np

# NPM 参数（经验校准值，2026-03-25拟合，详见npm_param_fit.py）
ALPHA = 0.57
BETA  = 0.068
DELTA = 2.76
K     = 68.5

def npm_ceff(N_B, sigma):
    """NPM泛化能力"""
    rho   = N_B / np.exp(ALPHA * sigma)
    rho_c = K * sigma**DELTA
    if rho <= rho_c:
        return 0.0
    return (rho - rho_c)**BETA

def npm_pmin(N_B, sigma):
    """NPM最小参数量（十亿）"""
    return N_B * sigma**ALPHA


print("="*70)
print("检验1：NPM C_eff(N) vs Kaplan Scaling Law")
print("="*70)
print()
print("已知：Kaplan (2020) 报告 Loss ∝ N^(-0.076)")
print("问题：NPM 的 1/C_eff(N) 是否也近似幂律？指数是否接近？")
print()

# 假设通用web数据，σ ≈ 0.75（论文表述）
sigma_web = 0.75

# 扫描 N 从 1B 到 10T tokens
N_B_vals = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000], dtype=float)
ceff_vals = np.array([npm_ceff(N_B, sigma_web) for N_B in N_B_vals])

# 过滤掉 C_eff=0 的点（未涌现）
mask = ceff_vals > 0
N_fit = N_B_vals[mask]
C_fit = ceff_vals[mask]

print(f"  σ = {sigma_web}（通用web数据估计值）")
print(f"  ρ_c = {K * sigma_web**DELTA:.1f}")
print()
print(f"  {'N_B(十亿)':>12} {'ρ':>10} {'C_eff':>12} {'log(N_B)':>10} {'log(C_eff)':>12}")
print("  " + "-"*60)

for N_B, c in zip(N_B_vals, ceff_vals):
    rho = N_B / np.exp(ALPHA * sigma_web)
    if c > 0:
        print(f"  {N_B:>12.0f} {rho:>10.2f} {c:>12.2f} {np.log10(N_B):>10.3f} {np.log10(c):>12.3f}")
    else:
        print(f"  {N_B:>12.0f} {rho:>10.2f} {'(未涌现)':>12}")

# 拟合 log(C_eff) vs log(N_B) 的斜率
if len(N_fit) >= 3:
    log_N = np.log10(N_fit)
    log_C = np.log10(C_fit)
    slope, intercept = np.polyfit(log_N, log_C, 1)
    residuals = log_C - (slope * log_N + intercept)
    r_squared = 1 - np.sum(residuals**2) / np.sum((log_C - log_C.mean())**2)

    print(f"\n  log-log 线性拟合：")
    print(f"    C_eff ∝ N_B^{slope:.3f}")
    print(f"    R² = {r_squared:.4f}")
    print(f"\n  对比：")
    print(f"    NPM:   C_eff ∝ N^{slope:.3f}   (数值拟合)")
    print(f"    Kaplan: Loss ∝ N^(-0.076)  (经验)")
    print(f"\n  注意：C_eff是能力（越大越好），Loss是损失（越小越好）")
    print(f"  如果 Loss ≈ 1/C_eff，则需要 C_eff ∝ N^0.076")
    print(f"  实际 NPM 给出 C_eff ∝ N^{slope:.3f}，与Kaplan的0.076相差 {slope/0.076:.0f} 倍")
    print(f"\n  诚实评估：NPM的C_eff增长比Kaplan快得多（{slope:.3f} vs 0.076）")
    print(f"  这说明当前β={BETA}偏大，或ρ与Loss的关系不是简单倒数")
    print(f"  需要拟合β使 C_eff 增长率与实际 scaling law 匹配")


print("\n")
print("="*70)
print("检验2：NPM P_min(N) vs Chinchilla 最优配比")
print("="*70)
print()
print("已知：Chinchilla (2022) 报告最优模型参数量与数据量近似线性")
print("      N_opt ≈ 20 × P_opt （每参数约20个token）")
print("问题：NPM 的 P_min = N_B · σ^α 是否给出合理的量级？")
print()

# Chinchilla 论文的几个关键数据点（近似值）
# (参数量B, 最优token数B)
chinchilla_points = [
    (0.4,    8),      # 400M 模型
    (1.0,   20),      # 1B 模型
    (3.0,   60),      # 3B 模型
    (10.0,  200),     # 10B 模型
    (67.0, 1400),     # 67B (Chinchilla本身)
]

print(f"  σ = {sigma_web}  →  σ^α = {sigma_web**ALPHA:.4f}")
print(f"  NPM公式: P_min = N_B × {sigma_web**ALPHA:.4f}")
print()
print(f"  {'P_actual(B)':>12} {'N_actual(B)':>12} {'N/P_ratio':>10} "
      f"{'NPM_P_min(B)':>14} {'NPM/actual':>12}")
print("  " + "-"*65)

npm_pmin_vals = []
actual_P_vals = []

for P_actual, N_actual in chinchilla_points:
    p_npm = npm_pmin(N_actual, sigma_web)
    ratio = p_npm / P_actual if P_actual > 0 else float('inf')
    npm_pmin_vals.append(p_npm)
    actual_P_vals.append(P_actual)
    print(f"  {P_actual:>12.1f} {N_actual:>12.0f} {N_actual/P_actual:>10.0f} "
          f"{p_npm:>14.2f} {ratio:>12.2f}x")

# 相关性
log_actual = np.log10(actual_P_vals)
log_npm = np.log10(npm_pmin_vals)
corr = np.corrcoef(log_actual, log_npm)[0, 1]

print(f"\n  log-log 相关系数 r = {corr:.4f}")
print(f"  NPM的P_min与Chinchilla的P_actual趋势{'一致' if corr > 0.99 else '不一致'}（都随N线性增长）")

# 但看绝对值
ratios = np.array(npm_pmin_vals) / np.array(actual_P_vals)
print(f"\n  绝对值比较：")
print(f"    NPM/Chinchilla 比值范围 = {ratios.min():.1f}x ~ {ratios.max():.1f}x")
print(f"    NPM系统性{'高估' if ratios.mean() > 1 else '低估'}约 {ratios.mean():.1f}x")
print(f"\n  这意味着：")
print(f"    σ^α = {sigma_web**ALPHA:.4f} 作为 P/N 的比例系数")
print(f"    Chinchilla 实际比例 ≈ 1/20 = 0.05")
print(f"    两者量级{'接近' if 0.1 < sigma_web**ALPHA < 1.0 else '差异大'}，但NPM偏高{ratios.mean():.0f}倍")
print(f"    可通过调整α或引入校正系数改善")


print("\n")
print("="*70)
print("检验3：不同σ下C_eff对比 — 专用vs通用模型")
print("="*70)
print()
print("已知事实：相同参数量下，专用模型（窄域）性能通常优于通用模型（宽域）")
print("问题：NPM是否复现这一趋势？")
print()

N_fixed = 100.0  # 固定100B tokens
sigmas = [0.15, 0.25, 0.40, 0.55, 0.70, 0.85]
labels = ["极窄(医疗)", "窄域(代码)", "中窄(金融)", "中域(混合)", "宽域(web)", "极宽(多语言)"]

print(f"  固定 N = {N_fixed:.0f}B tokens")
print(f"\n  {'场景':14} {'σ':>6} {'ρ':>8} {'ρ_c':>6} {'C_eff':>10} {'L_min':>6} {'趋势':>6}")
print("  " + "-"*60)

prev_c = float('inf')
for label, sigma in zip(labels, sigmas):
    rho = N_fixed / np.exp(ALPHA * sigma)
    rho_c = K * sigma**DELTA
    c = npm_ceff(N_fixed, sigma)
    L = max(1, int(np.ceil(ALPHA * sigma**ALPHA * 2 * 10)))
    trend = "↓" if c < prev_c else "↑"
    print(f"  {label:14} {sigma:>6.2f} {rho:>8.1f} {rho_c:>6.1f} {c:>10.1f} {L:>6} {trend:>6}")
    prev_c = c

print(f"\n  ✓ NPM复现：σ越小（专用）→ ρ越高 → C_eff越大 → 专用模型确实更强")
print(f"  这与业界经验一致，但仅是定性趋势，不构成定量验证")


print("\n")
print("="*70)
print("总结：推理链现状")
print("="*70)
print("""
  PNM物理                NPM公式                  状态
  ─────────      ───────────────────      ──────────────────
  密度=量/体积   ρ = N_B/exp(ασ)    ──✓── 参数已拟合(α=0.57)
  渗流临界       C_eff = (ρ-ρ_c)^β  ──✓── Kaplan斜率匹配(β=0.068)
  达西定律       P_min = N_B·σ^α   ──△── Chinchilla趋势对,量级偏17x
  迂曲度         L_min ∝ σ^α·τ    ──?── 深层网络为何更强？

  参数已拟合: α=0.57, β=0.068, δ=2.76, k=68.5
  Spearman(C_eff, MMLU) = 0.70
  详见 npm_param_fit.py
""")
