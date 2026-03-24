"""
NPM 主方程数值演示
==================
演示统一主方程 a_out = W_eff · a_in 在三种架构下的具体形式，
并复现经典随机图渗流相变作为 NPM 涌现假说的参考。

说明：
  检查项1和2是GCN/注意力的定义展开，不是独立发现。
  检查项3是经典Erdős–Rényi随机图相变的复现。
  三者共同展示NPM比拟的数学基础。

作者：丁铁新 · 神经CAE · 2026-03-23
"""
import numpy as np
from scipy.linalg import norm
from scipy.special import softmax as sp_softmax

np.random.seed(2026)
PASS = "✓ PASS"
FAIL = "✗ FAIL"
results = {}

print("="*62)
print("NPM 主方程数值演示")
print("="*62)

# ══════════════════════════════════════════════
# 检查项1：GCN消息传递的两种等价写法
# 说明：这不是新发现，而是 Kipf & Welling (2017) GCN定义的
#       逐节点展开形式与矩阵形式的数值一致性确认。
# ══════════════════════════════════════════════
print("\n检查项1  GCN逐节点消息传递 ≡ 图拉普拉斯矩阵乘法")
print("  （GCN定义的两种等价写法，非独立发现）")
print("-"*50)

n, d_in, d_out = 6, 4, 3
A = np.array([[0,1,1,0,0,0],[1,0,1,1,0,0],[1,1,0,0,1,0],
              [0,1,0,0,1,1],[0,0,1,1,0,1],[0,0,0,1,1,0]], dtype=float)
x      = np.random.randn(n, d_in)
W_feat = np.random.randn(d_in, d_out) * 0.3

# --- 写法A：GCN逐节点消息传递（循环实现）---
A_hat = A + np.eye(n)                         # 加自环
deg   = A_hat.sum(axis=1)                     # 度数
out_msg = np.zeros((n, d_out))
for i in range(n):
    agg = np.zeros(d_in)
    for j in range(n):
        if A_hat[i, j] > 0:
            agg += A_hat[i, j] * x[j] / np.sqrt(deg[i] * deg[j])
    out_msg[i] = agg @ W_feat

# --- 写法B：图拉普拉斯矩阵形式（一步矩阵乘法）---
D_inv_sq = np.diag(1.0 / np.sqrt(deg))
L_norm   = D_inv_sq @ A_hat @ D_inv_sq
out_mat  = L_norm @ x @ W_feat

# --- 确认两种写法一致 ---
err1 = norm(out_msg - out_mat)
print(f"  写法A：逐节点循环聚合邻居")
print(f"  写法B：L_norm @ x @ W")
print(f"  差异 = {err1:.2e}（浮点精度）")

# L_norm 的数学性质
sym_err = norm(L_norm - L_norm.T)
eigvals = np.linalg.eigvalsh(np.eye(n) - L_norm)
spd = eigvals.min() >= -1e-10

print(f"  L_norm 对称性   = {sym_err:.2e}")
print(f"  I-L_norm 半正定 = {spd}")
print(f"  NPM比拟：L_norm 对应 PNM 中的导度矩阵 G")

p1 = err1 < 1e-12 and sym_err < 1e-12 and spd
print(f"  {PASS if p1 else FAIL}")
results['检查项1·GCN两种写法一致'] = p1

# ══════════════════════════════════════════════
# 检查项2：注意力矩阵乘法的两种等价写法 + 行和性质
# 说明：矩阵乘法与逐元素求和的等价是线性代数恒等式，
#       softmax行和=1是softmax的定义性质。
#       NPM的贡献是将行和=1解读为PNM流量守恒。
# ══════════════════════════════════════════════
print("\n检查项2  注意力矩阵乘法 ≡ 逐token加权求和 + 行和=1")
print("  （线性代数恒等式 + softmax定义性质，NPM解读为流量守恒）")
print("-"*50)

seq, d_m, d_k = 5, 8, 4
X  = np.random.randn(seq, d_m)
Wq = np.random.randn(d_m, d_k) * 0.3
Wk = np.random.randn(d_m, d_k) * 0.3
Wv = np.random.randn(d_m, d_k) * 0.3
Q  = X @ Wq
K  = X @ Wk
V  = X @ Wv

scores = Q @ K.T / np.sqrt(d_k)
W_dyn  = np.apply_along_axis(sp_softmax, 1, scores)

# --- 写法A：矩阵乘法 ---
out_matrix = W_dyn @ V

# --- 写法B：逐token加权求和 ---
out_graph = np.zeros_like(V)
for i in range(seq):
    for j in range(seq):
        out_graph[i] += W_dyn[i, j] * V[j]

# --- 确认一致 ---
err2 = norm(out_matrix - out_graph)
print(f"  写法A：W_dyn @ V        （矩阵乘法）")
print(f"  写法B：逐token Σ_j w_ij·v_j")
print(f"  差异 = {err2:.2e}（浮点精度）")

# 行和=1（softmax定义保证）
row_sums = W_dyn.sum(axis=1)
conservation_err = norm(row_sums - 1.0)
print(f"  W_dyn 行和 = {row_sums.round(6)}")
print(f"  行和偏差   = {conservation_err:.2e}（softmax定义保证=1）")
print(f"  NPM比拟：行和=1 对应 PNM 节点流量守恒")

p2 = err2 < 1e-12 and conservation_err < 1e-12
print(f"  {PASS if p2 else FAIL}")
results['检查项2·注意力两种写法一致+行和=1'] = p2

# ══════════════════════════════════════════════
# 检查项3：经典随机图渗流相变复现
# 说明：这是 Erdős–Rényi (1960) 的经典结论。
#       NPM的贡献是将此相变比拟为LLM能力涌现，
#       本检查项仅复现经典结果作为参考。
# ══════════════════════════════════════════════
print("\n检查项3  Erdős–Rényi随机图渗流相变（经典结论复现）")
print("  （NPM将此比拟为能力涌现，此处仅复现经典结果）")
print("-"*50)

def giant(p, n=80, trials=50):
    sizes = []
    for _ in range(trials):
        adj = np.triu(np.random.rand(n,n)<p,1); adj+=adj.T
        vis = np.zeros(n,bool); mc=0
        for s in range(n):
            if not vis[s]:
                q,sz=[s],0; vis[s]=True
                while q:
                    v=q.pop(); sz+=1
                    for nb in np.where(adj[v])[0]:
                        if not vis[nb]: vis[nb]=True; q.append(nb)
                mc=max(mc,sz)
        sizes.append(mc/n)
    return np.mean(sizes)

n_nodes = 80
p_c = 1.0/n_nodes

print(f"  G({n_nodes},p)，理论 p_c=1/n={p_c:.4f}")
print(f"\n  {'Cn':>5} {'Giant':>8} {'趋势':>10}")

prev_g = 0
cn_vals = [0.3, 0.6, 1.0, 1.5, 2.5, 4.0]
g_vals  = []

for cn in cn_vals:
    g = giant(cn*p_c, n=n_nodes)
    g_vals.append(g)
    trend = "上升↑" if g > prev_g + 0.02 else ("平稳→" if abs(g-prev_g)<0.05 else "")
    print(f"  {cn:>5.1f} {g:>8.3f} {trend}")
    prev_g = g

below_pc = giant(0.6*p_c, n=n_nodes)
above_pc = giant(2.0*p_c, n=n_nodes)
transition = above_pc - below_pc

print(f"\n  Cn=0.6时Giant = {below_pc:.3f}（临界以下）")
print(f"  Cn=2.0时Giant = {above_pc:.3f}（临界以上）")
print(f"  相变幅度 = {transition:.3f}（>0.3视为明显相变）")

print(f"\n  注：n={n_nodes}较小，有限尺寸效应显著。")
print(f"      理论临界在n→∞时精确成立。")

p3 = transition > 0.3
print(f"  {PASS if p3 else FAIL}")
results['检查项3·随机图渗流相变'] = p3

# ══════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════
print("\n" + "="*62)
print("总结")
print("="*62)
for name, passed in results.items():
    mark = "✓" if passed else "✗"
    print(f"  {mark}  {name}")

print(f"""
NPM 统一主方程：a_out = W_eff · a_in
─────────────────────────────────────────────────────
  W_eff 的三种形式（已有工作，非NPM原创）：

  前馈NN       W_eff = W（固定权重矩阵）
  GNN          W_eff = D^{{-1/2}}(A+I)D^{{-1/2}}（Kipf & Welling 2017）
  Transformer  W_eff = softmax(QK^T/√d)（Vaswani et al. 2017）

  NPM的贡献是将三者统一解读为PNM导度矩阵的不同实例，
  并引入渗流相变 Cn=ρ/ρ_c 作为涌现的判据。
  B层方程（ρ, ρ_c, C_eff）的参数尚待实验拟合。
─────────────────────────────────────────────────────
""")
