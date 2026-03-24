"""
NPM 无量纲数分析：Cn 能否坍缩不同模型的数据？
================================================
雷诺数的力量：不管水、油、空气，只要 Re 相同，流态就相同。
问题：NPM 的 Cn = ρ/ρ_c 能否做到类似的事？

方法：收集公开模型的 (数据量, 参数量, 领域, 评测分数)，
      计算每个模型的 Cn，看能力分数是否随 Cn 坍缩到一条曲线。

数据来源：公开论文报告的模型规格和评测结果（近似值）

作者：丁铁新 · 2026-03-24
"""

import numpy as np

ALPHA = 0.57
K     = 68.5
DELTA = 2.76

def compute_Cn(N_B, sigma):
    """计算连通数 Cn = ρ/ρ_c"""
    rho   = N_B / np.exp(ALPHA * sigma)
    rho_c = K * sigma**DELTA
    return rho / rho_c if rho_c > 0 else float('inf')


# ══════════════════════════════════════════════════════
# 公开模型数据（近似值，从论文/技术报告收集）
# ══════════════════════════════════════════════════════
# 格式：(名称, N_B十亿tokens, σ估计, 参数量B, 能力评分0-100)
# σ估计规则：
#   纯代码/单领域 ≈ 0.25
#   混合领域     ≈ 0.50
#   通用web      ≈ 0.75
#   多语言+多模态 ≈ 0.90
# 能力评分：MMLU 5-shot 近似值（标准化到0-100）

models = [
    # 小模型 / 少数据
    ("GPT-2 (1.5B)",          40,    0.70,   1.5,   35),
    ("Pythia-1B",             300,   0.70,   1.0,   27),
    ("Pythia-6.9B",           300,   0.70,   6.9,   35),
    ("Pythia-12B",            300,   0.70,  12.0,   38),

    # 中等模型
    ("LLaMA-7B",             1000,   0.75,   7.0,   35),
    ("LLaMA-13B",            1000,   0.75,  13.0,   47),
    ("LLaMA-33B",            1400,   0.75,  33.0,   58),
    ("LLaMA-65B",            1400,   0.75,  65.0,   64),

    # 大模型
    ("GPT-3 (175B)",          300,   0.75, 175.0,   44),
    ("Chinchilla (70B)",     1400,   0.75,  70.0,   68),
    ("LLaMA-2-70B",          2000,   0.75,  70.0,   69),

    # 专用模型（代码）
    ("CodeLlama-7B",          500,   0.30,   7.0,   31),
    ("CodeLlama-34B",         500,   0.30,  34.0,   42),

    # 多语言
    ("BLOOM-176B",            366,   0.85, 176.0,   39),
]

print("="*78)
print("NPM 无量纲数 Cn 分析：能否坍缩不同模型？")
print("="*78)
print()
print("雷诺数类比：Re = ρvL/μ 把流速、管径、粘度合成一个数")
print("连通数类比：Cn = ρ/ρ_c 把数据量、分布广度合成一个数")
print()

# 计算每个模型的 Cn
print(f"  {'模型':<22} {'N_B':>6} {'σ':>5} {'ρ':>8} {'ρ_c':>6} "
      f"{'Cn':>8} {'MMLU':>6}")
print("  " + "-"*68)

cn_vals = []
mmlu_vals = []
names = []

for name, N_B, sigma, P_B, mmlu in models:
    rho   = N_B / np.exp(ALPHA * sigma)
    rho_c = K * sigma**DELTA
    Cn    = rho / rho_c
    cn_vals.append(Cn)
    mmlu_vals.append(mmlu)
    names.append(name)
    print(f"  {name:<22} {N_B:>6.0f} {sigma:>5.2f} {rho:>8.1f} {rho_c:>6.1f} "
          f"{Cn:>8.1f} {mmlu:>6}")

cn_vals = np.array(cn_vals)
mmlu_vals = np.array(mmlu_vals)


# ══════════════════════════════════════════════════════
# 检验：MMLU 分数是否随 Cn 单调增长？
# ══════════════════════════════════════════════════════
print()
print("="*78)
print("检验1：MMLU 是否随 Cn 单调增长？")
print("="*78)
print()

# 按 Cn 排序
order = np.argsort(cn_vals)
print(f"  按 Cn 排序：")
print(f"  {'模型':<22} {'Cn':>8} {'MMLU':>6} {'趋势':>6}")
print("  " + "-"*46)
prev_mmlu = 0
monotone_count = 0
total_pairs = 0
for idx in order:
    trend = "↑" if mmlu_vals[idx] > prev_mmlu else "↓"
    if mmlu_vals[idx] > prev_mmlu:
        monotone_count += 1
    total_pairs += 1
    print(f"  {names[idx]:<22} {cn_vals[idx]:>8.1f} {mmlu_vals[idx]:>6} {trend:>6}")
    prev_mmlu = mmlu_vals[idx]

# Spearman 秩相关
from scipy.stats import spearmanr, pearsonr
rho_spear, p_spear = spearmanr(cn_vals, mmlu_vals)
rho_pears, p_pears = pearsonr(cn_vals, mmlu_vals)

print(f"\n  Spearman 秩相关：r_s = {rho_spear:.3f},  p = {p_spear:.4f}")
print(f"  Pearson  线性相关：r   = {rho_pears:.3f},  p = {p_pears:.4f}")
print(f"  单调递增比例：{monotone_count}/{total_pairs}")


# ══════════════════════════════════════════════════════
# 检验2：Cn 是否比单独的 N_B 或 P_B 更好？
# ══════════════════════════════════════════════════════
print()
print("="*78)
print("检验2：Cn 是否比单独用 N 或 P 预测 MMLU 更好？")
print("="*78)
print()

N_vals = np.array([m[1] for m in models], dtype=float)
P_vals = np.array([m[3] for m in models], dtype=float)

r_N, p_N = spearmanr(N_vals, mmlu_vals)
r_P, p_P = spearmanr(P_vals, mmlu_vals)
r_Cn, p_Cn = spearmanr(cn_vals, mmlu_vals)

# 也试 log(N*P) 作为朴素基线
NP_vals = N_vals * P_vals
r_NP, p_NP = spearmanr(NP_vals, mmlu_vals)

print(f"  {'预测变量':<20} {'Spearman r_s':>14} {'p-value':>10} {'判定':>8}")
print("  " + "-"*55)
print(f"  {'N_B (数据量)':<20} {r_N:>14.3f} {p_N:>10.4f} {'':>8}")
print(f"  {'P_B (参数量)':<20} {r_P:>14.3f} {p_P:>10.4f} {'':>8}")
print(f"  {'N×P (朴素基线)':<20} {r_NP:>14.3f} {p_NP:>10.4f} {'':>8}")
print(f"  {'Cn (NPM连通数)':<20} {r_Cn:>14.3f} {p_Cn:>10.4f} {'':>8}")

best_var = max([(abs(r_N), 'N_B'), (abs(r_P), 'P_B'),
                (abs(r_NP), 'N×P'), (abs(r_Cn), 'Cn')], key=lambda x: x[0])
print(f"\n  最佳预测变量：{best_var[1]}（|r_s| = {best_var[0]:.3f}）")

if best_var[1] == 'Cn':
    print("  ✓ Cn 优于单独的 N 或 P，说明密度概念有额外预测力")
else:
    print(f"  ✗ Cn 不是最佳预测变量，{best_var[1]} 更好")
    print(f"    这意味着 σ 的引入可能没有帮助，或 σ 估计不准")


# ══════════════════════════════════════════════════════
# 检验3：同 Cn 不同来源的模型，能力是否接近？
# ══════════════════════════════════════════════════════
print()
print("="*78)
print("检验3：'雷诺数相似律' — 同Cn的模型能力是否接近？")
print("="*78)
print()
print("  雷诺数相似律：Re相同 → 流态相同，不管具体流体是什么")
print("  NPM相似律假说：Cn相近 → 能力相近，不管具体是什么模型")
print()

# 找 Cn 接近的模型对
print(f"  Cn相近的模型对：")
print(f"  {'模型A':<22} {'模型B':<22} {'Cn_A':>6} {'Cn_B':>6} {'ΔMMLU':>7}")
print("  " + "-"*68)

pairs_found = 0
for i in range(len(models)):
    for j in range(i+1, len(models)):
        if abs(cn_vals[i] - cn_vals[j]) / max(cn_vals[i], cn_vals[j]) < 0.3:  # Cn差<30%
            delta_mmlu = abs(mmlu_vals[i] - mmlu_vals[j])
            print(f"  {names[i]:<22} {names[j]:<22} {cn_vals[i]:>6.1f} {cn_vals[j]:>6.1f} {delta_mmlu:>7}")
            pairs_found += 1

if pairs_found == 0:
    print("  （未找到Cn差距<30%的模型对）")
    print("  这本身说明：不同σ的模型Cn差异很大，难以直接对比")


# ══════════════════════════════════════════════════════
# 检验4：Cn 能否解释反常现象？
# ══════════════════════════════════════════════════════
print()
print("="*78)
print("检验4：Cn 解释反常现象")
print("="*78)
print()
print("  反常1：BLOOM-176B (176B参数) MMLU仅39，不如LLaMA-7B (7B参数) 的35?")
bloom_cn = compute_Cn(366, 0.85)
llama7_cn = compute_Cn(1000, 0.75)
print(f"    BLOOM:    σ=0.85(多语言), N=366B  → Cn = {bloom_cn:.1f}")
print(f"    LLaMA-7B: σ=0.75(英语为主), N=1000B → Cn = {llama7_cn:.1f}")
print(f"    BLOOM的Cn({'<' if bloom_cn < llama7_cn else '>'})LLaMA")
print(f"    NPM解释：BLOOM的σ更大(多语言)导致密度更低，即使参数量大25倍")
print()

print("  反常2：Chinchilla-70B > GPT-3-175B？")
chin_cn = compute_Cn(1400, 0.75)
gpt3_cn = compute_Cn(300, 0.75)
print(f"    Chinchilla: N=1400B, P=70B  → Cn = {chin_cn:.1f}")
print(f"    GPT-3:      N=300B,  P=175B → Cn = {gpt3_cn:.1f}")
print(f"    Chinchilla的Cn >> GPT-3的Cn")
print(f"    NPM解释：数据密度决定能力，不是参数量。")
print(f"    这与Chinchilla论文的核心发现一致：GPT-3欠训练了")
print()

print("  反常3：CodeLlama-34B 在代码上很强但MMLU较低？")
code_cn = compute_Cn(500, 0.30)
ll33_cn = compute_Cn(1400, 0.75)
print(f"    CodeLlama: σ=0.30(代码), N=500B  → Cn = {code_cn:.1f}")
print(f"    LLaMA-33B: σ=0.75(通用), N=1400B → Cn = {ll33_cn:.1f}")
print(f"    CodeLlama在代码域的Cn远高于LLaMA在通用域的Cn")
print(f"    NPM解释：专用模型在自己领域的Cn极高（密度大），")
print(f"    但MMLU测的是通用能力（σ≈0.75），不是代码域的σ≈0.30")
print(f"    这指向一个重要问题：Cn是任务相关的，不同评测对应不同ρ_c")


# ══════════════════════════════════════════════════════
# 总结：Cn 作为无量纲数的现状
# ══════════════════════════════════════════════════════
print()
print()
print("="*78)
print("Cn 作为'神经网络雷诺数'的现状评估")
print("="*78)
print(f"""
  雷诺数的特征          Cn是否具备           现状
  ─────────────       ──────────────       ──────────
  无量纲               ✓ 是                 满足
  合并多个变量         ✓ 合并了N和σ         满足
  预测状态转变         △ 能解释涌现方向     未定量验证
  数据坍缩             ? 需要更多数据点     待验
  与系统细节无关       ✗ σ的估计很主观      最大弱点

  关键差距：
  雷诺数：Re = ρvL/μ 里每个量都能独立精确测量
  连通数：Cn = ρ/ρ_c 里 σ 的估计高度主观（0.25还是0.30？）

  如果 σ 不能客观测量，Cn 就无法成为真正的工程判据。
  这回到了 sigma_estimation.py 的 PR 法——
  它是目前唯一的客观估算手段，但绝对尺度未校准。

  最有价值的下一步：
  1. 用真实语料（The Pile的不同子集）计算σ，建立σ基准
  2. 用Pythia检查点（同数据不同规模）检验Cn-能力曲线
  3. 如果Cn能坍缩Pythia数据，就有了第一个实证支撑
""")
