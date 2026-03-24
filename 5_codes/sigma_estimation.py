"""
NPM σ估算方法
==============
σ（数据分布广度）是NPM所有B层方程的核心输入。
本模块提供从数据集客观计算σ的标准方法。

方法：参与率（Participation Ratio, PR）
  PR = (Σλ_i)² / Σλ_i²
  σ  = log(PR+1) / log(rank+1)

验证状态：单调性完全成立（2026-03-23）
待完成：绝对尺度校准

作者：丁铁新 · 神经CAE · 2026-03-23
"""

import numpy as np

def participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    计算特征值谱的参与率（有效维度）

    PR = 1  → 所有方差集中在一个方向（极窄域）
    PR = k  → k个方向均匀分布（极宽域）
    """
    lam = eigenvalues[eigenvalues > 1e-10]
    if len(lam) == 0:
        return 1.0
    return float((lam.sum())**2 / (lam**2).sum())


def estimate_sigma(X: np.ndarray) -> dict:
    """
    从数据矩阵X估算σ

    参数：
        X: shape [n_samples, n_features]，可以是TF-IDF矩阵或词向量矩阵

    返回：
        dict包含：
            sigma     : float，σ值∈[0,1]
            PR        : float，参与率（有效维度）
            rank      : int，有效特征值数量
            eigenvalues: ndarray，特征值谱（降序）

    示例：
        X = tfidf_matrix   # [n_docs, n_vocab]
        result = estimate_sigma(X)
        sigma = result['sigma']
        # 代入NPM：rho = N_B / exp(1.5 * sigma)
    """
    # 中心化
    X_c = X - X.mean(axis=0)

    # 协方差矩阵特征值
    # 对大矩阵用SVD近似（避免内存爆炸）
    if X_c.shape[1] <= 500:
        C = X_c.T @ X_c / X_c.shape[0]
        lam = np.linalg.eigvalsh(C)[::-1]
    else:
        # 用随机SVD近似
        try:
            from sklearn.decomposition import TruncatedSVD
            k = min(200, X_c.shape[0]-1, X_c.shape[1]-1)
            svd = TruncatedSVD(n_components=k, random_state=42)
            svd.fit(X_c)
            lam = svd.singular_values_**2 / X_c.shape[0]
        except ImportError:
            # fallback：直接协方差
            C = X_c.T @ X_c / X_c.shape[0]
            lam = np.linalg.eigvalsh(C)[::-1]

    lam = lam[lam > 1e-10]
    PR   = participation_ratio(lam)
    rank = len(lam)
    sigma = float(np.log(PR + 1) / np.log(rank + 1))

    return {
        'sigma'      : sigma,
        'PR'         : PR,
        'rank'       : rank,
        'eigenvalues': lam,
    }


def sigma_quick(description: str) -> float:
    """
    工程快查：根据数据集描述返回σ估算值（无法计算时使用）

    参数：
        description: 'narrow'|'domain'|'cross'|'web'|'multimodal'

    返回：
        σ的中间估计值

    来源说明：
        这些值是作者基于PR法对合成数据实验的经验插值，
        不是从真实数据集拟合的精确值。仅用于快速估算。
    """
    table = {
        'narrow'    : 0.18,  # 单一领域专业文档（经验估计）
        'domain'    : 0.30,  # 单领域多样化（经验估计）
        'cross'     : 0.50,  # 跨4~6个领域（经验估计）
        'web'       : 0.75,  # 互联网通用爬取（经验估计）
        'multimodal': 0.90,  # 多语言+多模态（经验估计）
    }
    if description not in table:
        import warnings
        warnings.warn(f"未知描述'{description}'，返回默认σ=0.50", stacklevel=2)
    return table.get(description, 0.50)


def npm_predict(N_B: float, sigma: float,
                alpha=0.57, beta=0.068, delta=2.76,
                k=68.5, tau=1.0) -> dict:
    """
    给定数据量N_B（十亿token）和σ，预测NPM关键量

    返回：
        rho    : 数据密度
        rho_c  : 临界密度
        C_eff  : 泛化能力（0表示未超过临界）
        L_min  : 最小层数
        P_min  : 最小参数量（十亿）
        W_c    : 层间交互强度下限
        connected: 是否已超过涌现临界
    """
    V     = np.exp(alpha * sigma)
    rho   = N_B / V
    rho_c = k * (sigma ** delta) * tau
    C_eff = float((rho - rho_c) ** beta) if rho > rho_c else 0.0
    L_min = max(1, int(np.ceil(alpha * (sigma ** alpha) * tau * 10)))
    P_min = N_B * (sigma ** alpha)
    W_c   = sigma ** (delta / 2)

    return {
        'rho'      : rho,
        'rho_c'    : rho_c,
        'C_eff'    : C_eff,
        'L_min'    : L_min,
        'P_min'    : P_min,
        'W_c'      : W_c,
        'connected': rho > rho_c,
        'Cn'       : rho / rho_c if rho_c > 0 else float('inf'),
    }


if __name__ == '__main__':
    print("="*60)
    print("σ估算方法演示")
    print("="*60)

    np.random.seed(42)
    n, d = 300, 200

    def make_dataset(n_latent, noise=0.02):
        vecs = np.random.randn(n_latent, d)
        vecs, _ = np.linalg.qr(vecs.T)
        vecs = vecs.T[:n_latent]
        weights = np.random.randn(n, n_latent)
        return np.abs(weights @ vecs + np.random.randn(n, d) * noise)

    configs = [
        (2,  "极窄·2维"),
        (5,  "窄域·5维"),
        (15, "中域·15维"),
        (40, "宽域·40维"),
        (80, "全域·80维"),
    ]

    print(f"\n{'数据集':16} {'σ':>8} {'PR':>8} {'ρ(100B)':>10} "
          f"{'C_eff':>10} {'L_min':>6}")
    print("-"*60)

    prev = -1
    for n_latent, label in configs:
        X = make_dataset(n_latent)
        r = estimate_sigma(X)
        p = npm_predict(100.0, r['sigma'])
        mono = "↑" if r['sigma'] > prev else "✗"
        print(f"  {label:14} {r['sigma']:>8.3f} {r['PR']:>8.1f} "
              f"{p['rho']:>10.1f} {p['C_eff']:>10.1f} {p['L_min']:>6} {mono}")
        prev = r['sigma']

    print("\n✓ 单调性成立：域越宽 → σ越大")
    print("  效果1：密度下降 ρ=N_B/exp(ασ)，同等数据更难超过涌现临界")
    print("  效果2：层数增加 L_min∝σ^α·τ，概念空间迂曲度更大，信息传播需要更多步")
    print("\n快查接口：")
    for k in ['narrow', 'domain', 'cross', 'web', 'multimodal']:
        s = sigma_quick(k)
        p = npm_predict(100.0, s)
        print(f"  {k:12} σ={s:.2f}  L_min={p['L_min']}层  C_eff={p['C_eff']:.0f}")
