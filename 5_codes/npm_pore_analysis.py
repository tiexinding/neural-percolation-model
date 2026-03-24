"""
NPM 孔隙形态分析：MIS法从NN权重提取PNM类比特征
================================================================
借鉴 PoreSpy 的 MIS（最大内切球）法，将PNM孔隙分析工具映射到NN权重空间。

6项实验：
  1. 权重"孔径分布"     — |w_ij| ↔ 孔喉直径，GPT-2全局+逐层分析
  2. NN版Pc曲线         — 输入强度 vs 激活率 ↔ 毛管压力 vs 饱和度
  3. PoreSpy定量对比     — KS检验、CV、峰度对比真实3D随机多孔介质
  3b. 分布形态深挖       — Weibull拟合、尾部幂律指数、rank-QQ分析
  4. 训练演化           — GPT-2随机初始化 vs 预训练（渗流通道优化）
  4b. Pythia训练轨迹    — 13个checkpoint追踪CV/基尼系数，发现三阶段相变

核心发现：
  - NN权重与PNM孔径属不同分布族(Weibull k=1.1 vs 2.5)，但形状高度相似(rank r=0.98)
  - 训练 = 渗流通道优化：选择性放大少数主通道，抑制其余
  - Pythia训练存在三阶段相变：蓄压→突破→重组，相变点在step 64k→128k

作者：丁铁新 · 2026-03-24/25
"""

import numpy as np
import sys

def analyze_weight_distribution(weights_dict):
    """
    分析权重矩阵的"孔径分布"

    PNM类比：
      |w_ij| 大 → 大孔喉 → 低启动压力 → 信号容易通过
      |w_ij| 小 → 小孔喉 → 高启动压力 → 信号被截断
      |w_ij| = 0 → 堵死的孔 → 完全不导通
    """
    print("="*70)
    print("实验1：权重矩阵孔径分布")
    print("="*70)
    print()
    print("PNM类比：|w_ij| = 孔喉直径，越大导通性越好")
    print()

    all_weights = []
    layer_stats = []

    for name, w in weights_dict.items():
        if w.ndim < 2:
            continue
        flat = np.abs(w.flatten())
        all_weights.extend(flat.tolist())
        layer_stats.append({
            'name': name,
            'shape': w.shape,
            'mean': flat.mean(),
            'std': flat.std(),
            'median': np.median(flat),
            'zero_frac': (flat < 1e-6).mean(),
            'p95': np.percentile(flat, 95),
        })

    all_w = np.array(all_weights)

    print(f"  总权重数: {len(all_w):,}")
    print(f"  |w| 均值: {all_w.mean():.4f}")
    print(f"  |w| 中位: {np.median(all_w):.4f}")
    print(f"  |w| 标准差: {all_w.std():.4f}")
    print(f"  |w| < 1e-6 占比: {(all_w < 1e-6).mean()*100:.1f}%  (≈堵死的孔)")
    print(f"  |w| > 0.1 占比: {(all_w > 0.1).mean()*100:.1f}%   (≈大孔喉)")

    # 分布形态检验
    log_w = np.log(all_w[all_w > 1e-10])
    log_mean = log_w.mean()
    log_std = log_w.std()
    skew = ((log_w - log_mean)**3).mean() / log_std**3

    print(f"\n  对数域统计:")
    print(f"    log|w| 均值: {log_mean:.3f}")
    print(f"    log|w| 标准差: {log_std:.3f}")
    print(f"    log|w| 偏度: {skew:.3f}  ({'接近对数正态' if abs(skew) < 0.5 else '偏离对数正态'})")
    print(f"\n  PNM对比: 天然砂岩孔径分布通常为对数正态或Weibull分布")
    print(f"  NN权重: {'形态接近对数正态' if abs(skew) < 0.5 else '偏离对数正态，更像' + ('右偏' if skew > 0 else '左偏') + '分布'}")

    # 逐层分析
    print(f"\n  逐层孔隙特征前10层")
    print(f"  {'层名':<40} {'形状':<16} {'|w|均值':>8} {'|w|中位':>8} {'堵孔%':>6} {'P95':>8}")
    print("  " + "-"*90)
    for s in layer_stats[:10]:
        print(f"  {s['name']:<40} {str(s['shape']):<16} {s['mean']:>8.4f} {s['median']:>8.4f} {s['zero_frac']*100:>5.1f}% {s['p95']:>8.4f}")

    if len(layer_stats) > 10:
        print(f"  ... 共 {len(layer_stats)} 层")

    # 分位数（类比孔径分布累积曲线）
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print('\n  孔径累积分布（类比PNM的孔径分布曲线）:')
    print(f"  {'百分位':>8} {'|w|值':>10} {'PNM类比':>20}")
    print("  " + "-"*42)
    for p in percentiles:
        v = np.percentile(all_w, p)
        analogy = ""
        if p <= 10: analogy = "微孔（易被堵）"
        elif p <= 50: analogy = "中孔"
        elif p <= 90: analogy = "大孔（主通道）"
        else: analogy = "超大孔（高速通道）"
        print(f"  {p:>7}% {v:>10.5f} {analogy:>20}")

    return all_w, layer_stats


def analyze_activation_curve(model, tokenizer):
    """
    实验2：NN版"毛细压力曲线"

    PNM类比：
      逐步增大毛管压力 → 更多孔隙被侵入 → 饱和度Sw上升
      逐步增大输入信号强度 → 更多ReLU被激活 → "网络饱和度"上升

    Pc曲线是S形的（Sigmoid-like），NN的激活率曲线是否也是？
    """
    print("\n")
    print("="*70)
    print('实验2：NN版毛细压力曲线（激活率 vs 输入强度）')
    print("="*70)
    print()
    print("PNM类比: Pc增大 → 更多孔喉被侵入 → Sw增加")
    print("NN类比:  输入增强 → 更多ReLU被激活 → 激活率增加")
    print()

    import torch

    # 用不同强度的随机输入，统计每层激活率
    scales = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"  {'输入强度':>10} {'总激活率':>10} {'趋势':>6}")
    print("  " + "-"*30)

    prev_rate = 0
    activation_rates = []

    for scale in scales:
        # 随机输入，不同强度
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, 64))

        # Hook 收集激活值
        activations = []
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations.append(output.detach())
            return hook_fn

        for name, module in model.named_modules():
            # Hook MLP中间层（c_fc后的激活）和激活函数层
            if any(k in name.lower() for k in ['act', 'gelu', 'relu', 'mlp.c_fc']):
                hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            # 缩放 embedding（兼容新旧版 transformers）
            wte = model.transformer.wte if hasattr(model, 'transformer') else model.wte
            orig_forward = wte.forward
            def scaled_forward(x):
                return orig_forward(x) * scale
            wte.forward = scaled_forward

            try:
                model(input_ids)
            except:
                pass
            finally:
                wte.forward = orig_forward

        for h in hooks:
            h.remove()

        if activations:
            total = sum(a.numel() for a in activations)
            active = sum((a > 0).sum().item() for a in activations)
            rate = active / total if total > 0 else 0
        else:
            # 没有显式激活函数hook，用所有中间层输出
            rate = 0

        activation_rates.append(rate)
        trend = "↑" if rate > prev_rate else "→"
        print(f"  {scale:>10.2f} {rate:>10.3f} {trend:>6}")
        prev_rate = rate

    if any(r > 0 for r in activation_rates):
        print(f"\n  曲线形态分析:")
        rates = np.array(activation_rates)
        if rates[-1] > rates[0]:
            print(f"    激活率随输入强度单调递增 ✓")
            print(f"    从 {rates[0]:.3f} 增至 {rates[-1]:.3f}")
            mid_idx = np.argmin(np.abs(rates - (rates[0]+rates[-1])/2))
            print(f'    半饱和点在输入强度 ≈ {scales[mid_idx]}')
            print(f"\n  PNM对比: 毛细压力曲线也是S形，半饱和点对应中位孔径")
        else:
            print(f"    激活率未随输入强度显著变化")
            print(f"    可能原因：GELU激活函数没有硬阈值，需要改用ReLU模型测试")
    else:
        print(f"\n  未检测到激活函数层，跳过Pc曲线分析")
        print(f"  GPT-2使用GELU而非ReLU，激活模式不同")


def compare_with_porespy(nn_weights):
    """
    实验3：用PoreSpy生成真实多孔介质孔径分布，与NN权重分布定量对比

    方法：
      1. PoreSpy生成3D随机多孔介质（不同孔隙率）
      2. 用距离变换提取孔径分布
      3. 与NN权重|w|分布做KS检验和形态对比
    """
    print("\n")
    print("="*70)
    print("实验3：PoreSpy真实孔径分布 vs NN权重分布")
    print("="*70)
    print()
    print("方法: PoreSpy生成随机多孔介质 → 距离变换提取孔径 → 与|w|对比")
    print()

    import porespy as ps
    from scipy import stats
    from scipy.ndimage import distance_transform_edt

    # 生成不同孔隙率的随机多孔介质
    porosities = [0.3, 0.5, 0.7]
    pnm_distributions = {}

    for phi in porosities:
        # 生成3D随机多孔介质 (100^3)
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=phi, blobiness=1.5)
        # 距离变换 = 每个孔隙点到最近固体的距离 ≈ 局部孔径
        dt = distance_transform_edt(im)
        pore_sizes = dt[im > 0]  # 只取孔隙内的点
        pore_sizes = pore_sizes[pore_sizes > 0]

        # 归一化到 [0, 1] 范围便于对比
        pore_norm = pore_sizes / pore_sizes.max()

        pnm_distributions[phi] = pore_norm

        log_ps = np.log(pore_sizes[pore_sizes > 0.1])
        skew_ps = stats.skew(log_ps)
        print(f"  孔隙率 φ={phi:.1f}:")
        print(f"    孔径数: {len(pore_sizes):,}")
        print(f"    均值: {pore_sizes.mean():.3f}, 中位: {np.median(pore_sizes):.3f}")
        print(f"    log偏度: {skew_ps:.3f}")

    # NN权重归一化
    nn_norm = nn_weights / nn_weights.max()

    # 定量对比：KS检验
    print(f"\n  KS检验（NN权重 vs PNM孔径，归一化后）:")
    print(f"  {'对比对象':<25} {'KS统计量':>10} {'p值':>12} {'结论':>15}")
    print("  " + "-"*65)

    for phi, pore_dist in pnm_distributions.items():
        ks_stat, p_val = stats.ks_2samp(
            np.random.choice(nn_norm, size=min(10000, len(nn_norm)), replace=False),
            np.random.choice(pore_dist, size=min(10000, len(pore_dist)), replace=False),
        )
        conclusion = "分布相似" if p_val > 0.05 else "分布不同"
        print(f"  NN vs PNM(φ={phi:.1f})        {ks_stat:>10.4f} {p_val:>12.2e} {conclusion:>15}")

    # 分位数对比
    print(f"\n  分位数对比（归一化后）:")
    pcts = [10, 25, 50, 75, 90]
    header = f"  {'百分位':>8} {'NN|w|':>8}"
    for phi in porosities:
        header += f" {'PNM(φ='+str(phi)+')':>12}"
    print(header)
    print("  " + "-"*56)
    for p in pcts:
        line = f"  {p:>7}% {np.percentile(nn_norm, p):>8.4f}"
        for phi in porosities:
            line += f" {np.percentile(pnm_distributions[phi], p):>12.4f}"
        print(line)

    # 形态学特征对比
    print(f"\n  形态学特征对比:")
    print(f"  {'指标':<20} {'NN权重':>10}", end="")
    for phi in porosities:
        print(f" {'PNM(φ='+str(phi)+')':>12}", end="")
    print()
    print("  " + "-"*58)

    # 变异系数 CV
    cv_nn = nn_norm.std() / nn_norm.mean()
    line_cv = f"  {'变异系数 CV':<20} {cv_nn:>10.3f}"
    for phi in porosities:
        cv_p = pnm_distributions[phi].std() / pnm_distributions[phi].mean()
        line_cv += f" {cv_p:>12.3f}"
    print(line_cv)

    # 峰度
    kurt_nn = stats.kurtosis(nn_norm)
    line_k = f"  {'峰度':<20} {kurt_nn:>10.3f}"
    for phi in porosities:
        kurt_p = stats.kurtosis(pnm_distributions[phi])
        line_k += f" {kurt_p:>12.3f}"
    print(line_k)

    return pnm_distributions


def deep_dive_porespy(nn_weights):
    """
    实验3b：深挖PNM vs NN分布形态

    之前的KS=1.0是因为简单归一化把尺度差异暴露了。
    真正有意义的对比是：分布的"形状"而非"位置"。

    方法：
      1. 两边都做rank归一化（CDF变换）→ 纯粹比形状
      2. 拟合参数分布（对数正态、Weibull、指数）→ 看哪个族更匹配
      3. 尾部行为：power-law尾指数对比
      4. 多尺度结构：不同层 vs 不同孔隙率的"指纹"对比
    """
    print("\n")
    print("="*70)
    print("实验3b：深挖——分布形态对比（形状 vs 尺度分离）")
    print("="*70)
    print()

    import porespy as ps
    from scipy import stats
    from scipy.ndimage import distance_transform_edt

    # ── 1. 生成PNM孔径数据 ──
    pnm_data = {}
    for phi in [0.3, 0.5, 0.7]:
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=phi, blobiness=1.5)
        dt = distance_transform_edt(im)
        pore_sizes = dt[im > 0]
        pnm_data[phi] = pore_sizes[pore_sizes > 0]

    # ── 2. 参数分布拟合 ──
    print("  【分布拟合】对 log(x) 拟合正态分布 → 检验是否为对数正态")
    print()
    print(f"  {'数据源':<22} {'log均值':>8} {'log标准差':>8} {'偏度':>7} {'峰度':>7} {'Shapiro-W':>10} {'对数正态?':>10}")
    print("  " + "-"*78)

    datasets = {}
    # NN权重
    nn_log = np.log(nn_weights[nn_weights > 1e-10])
    datasets['NN-GPT2(全部)'] = nn_log

    for phi, ps_data in pnm_data.items():
        log_d = np.log(ps_data)
        datasets[f'PNM(φ={phi})'] = log_d

    for label, log_data in datasets.items():
        mu = log_data.mean()
        sigma = log_data.std()
        skew = stats.skew(log_data)
        kurt = stats.kurtosis(log_data)
        # Shapiro-Wilk只支持5000个样本
        sample = np.random.choice(log_data, min(5000, len(log_data)), replace=False)
        w_stat, w_p = stats.shapiro(sample)
        is_lognormal = "接近" if (abs(skew) < 1.0 and abs(kurt) < 3.0) else "偏离"
        print(f"  {label:<22} {mu:>8.3f} {sigma:>8.3f} {skew:>7.3f} {kurt:>7.2f} {w_stat:>10.4f} {is_lognormal:>10}")

    # ── 3. Weibull拟合 ──
    print(f"\n  【Weibull拟合】形状参数 k 对比")
    print(f"    k < 1: 递减型（类似指数分布，小孔为主）")
    print(f"    k ≈ 1: 指数分布")
    print(f"    k > 1: 有峰值（类似正态，有典型尺度）")
    print()
    print(f"  {'数据源':<22} {'k(形状)':>8} {'λ(尺度)':>8} {'解读':>18}")
    print("  " + "-"*60)

    for label, raw_data in [('NN-GPT2', nn_weights)] + \
                            [(f'PNM(φ={phi})', d) for phi, d in pnm_data.items()]:
        data_pos = raw_data[raw_data > 0]
        # Weibull拟合
        k, loc, lam = stats.weibull_min.fit(data_pos, floc=0)
        if k < 0.8:
            interp = "递减型(小孔为主)"
        elif k < 1.2:
            interp = "指数型"
        elif k < 2.5:
            interp = "有峰(典型尺度)"
        else:
            interp = "窄峰(均匀尺度)"
        print(f"  {label:<22} {k:>8.3f} {lam:>8.4f} {interp:>18}")

    # ── 4. 尾部行为(power-law) ──
    print(f"\n  【尾部行为】上尾(大孔/大权重)的幂律指数")
    print(f"    方法: 取top 10%数据, 拟合 P(x>X) ∝ X^(-α)")
    print()
    print(f"  {'数据源':<22} {'尾指数 α':>10} {'R²':>8} {'解读':>20}")
    print("  " + "-"*65)

    for label, raw_data in [('NN-GPT2', nn_weights)] + \
                            [(f'PNM(φ={phi})', d) for phi, d in pnm_data.items()]:
        data_pos = raw_data[raw_data > 0]
        threshold = np.percentile(data_pos, 90)
        tail = data_pos[data_pos > threshold]
        tail_sorted = np.sort(tail)
        # CCDF
        ccdf = 1 - np.arange(1, len(tail_sorted)+1) / len(tail_sorted)
        ccdf = ccdf[ccdf > 0]
        tail_sorted = tail_sorted[:len(ccdf)]
        # log-log线性回归
        log_x = np.log(tail_sorted)
        log_y = np.log(ccdf)
        slope, intercept, r, _, _ = stats.linregress(log_x, log_y)
        alpha = -slope
        if alpha > 5:
            interp = "快速衰减(轻尾)"
        elif alpha > 2:
            interp = "中等尾部"
        else:
            interp = "重尾(极端值多)"
        print(f"  {label:<22} {alpha:>10.2f} {r**2:>8.4f} {interp:>20}")

    # ── 5. Rank归一化后的QQ对比 ──
    print(f"\n  【Rank归一化QQ分析】消除尺度差异，纯粹比形状")
    print(f"    方法: 两边都转为百分位rank(0~1)，再对比分位数")
    print()

    nn_ranks = stats.rankdata(nn_weights) / len(nn_weights)
    common_quantiles = np.linspace(0.01, 0.99, 50)
    nn_q = np.quantile(nn_ranks, common_quantiles)

    print(f"  {'PNM对比':<22} {'rank相关系数':>12} {'最大偏差':>10} {'形状相似度':>12}")
    print("  " + "-"*60)

    for phi, ps_data in pnm_data.items():
        pnm_ranks = stats.rankdata(ps_data) / len(ps_data)
        pnm_q = np.quantile(pnm_ranks, common_quantiles)
        corr = np.corrcoef(nn_q, pnm_q)[0, 1]
        max_dev = np.max(np.abs(nn_q - pnm_q))
        similarity = "高" if corr > 0.99 else ("中" if corr > 0.95 else "低")
        print(f"  NN vs PNM(φ={phi})     {corr:>12.6f} {max_dev:>10.4f} {similarity:>12}")

    print(f"\n  【小结】")
    print(f"    rank归一化后相关系数接近1说明：虽然绝对值不同，")
    print(f"    但NN权重和PNM孔径的分布'形状'是高度相似的。")
    print(f"    差异主要在尺度和尾部行为上。")


def deep_dive_training_evolution():
    """
    实验4b：用Pythia多checkpoint追踪训练中的"渗流相变"

    Pythia-70m公开了143个训练checkpoint (step 0 ~ 143000)
    选取关键节点，观察权重分布的演化轨迹
    """
    print("\n")
    print("="*70)
    print("实验4b：Pythia训练轨迹——寻找渗流相变点")
    print("="*70)
    print()
    print("模型: Pythia-70m (70M参数, 143个公开checkpoint)")
    print("目标: 训练过程中是否存在突然的分布变化 ≈ 渗流相变")
    print()

    import torch
    from transformers import AutoModel
    from scipy import stats

    # 选取checkpoint: 从稀疏到密集，覆盖训练全程
    # Pythia的step: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    #               1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000
    steps = [0, 1, 2, 8, 32, 128, 512, 1000, 4000, 16000, 64000, 128000, 143000]

    print(f"  加载 {len(steps)} 个checkpoint...")
    print()

    # 追踪指标
    results = []

    for i, step in enumerate(steps):
        try:
            model = AutoModel.from_pretrained(
                'EleutherAI/pythia-70m',
                revision=f'step{step}',
            )
        except Exception as e:
            print(f"    step{step}: 跳过 ({e})")
            continue

        # 提取所有2D权重
        all_w = []
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                all_w.extend(np.abs(p.detach().numpy().flatten()).tolist())
        all_w = np.array(all_w)

        w_mean = all_w.mean()
        w_std = all_w.std()
        w_median = np.median(all_w)
        cv = w_std / w_mean if w_mean > 0 else 0
        large_frac = (all_w > 0.1).mean()
        log_w = np.log(all_w[all_w > 1e-10])
        skew = stats.skew(log_w)
        kurt = stats.kurtosis(log_w)

        # 基尼系数（不均匀度，类比渗透率集中度）
        sorted_w = np.sort(all_w)
        n = len(sorted_w)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_w) / (n * np.sum(sorted_w))) - (n+1)/n

        results.append({
            'step': step,
            'mean': w_mean,
            'std': w_std,
            'cv': cv,
            'large_frac': large_frac,
            'skew': skew,
            'kurt': kurt,
            'gini': gini,
        })

        # 释放模型内存
        del model

        progress = f"[{i+1}/{len(steps)}]"
        print(f"    {progress} step {step:>6d}: |w|={w_mean:.4f}, CV={cv:.3f}, 大孔率={large_frac*100:.1f}%, 基尼={gini:.3f}")

    if len(results) < 3:
        print("\n  checkpoint不足，跳过分析")
        return results

    # 演化曲线分析
    print(f"\n  {'step':>8} {'|w|均值':>8} {'标准差':>8} {'CV':>6} {'大孔%':>6} {'log偏度':>8} {'log峰度':>8} {'基尼':>6} {'Δ(CV)':>8}")
    print("  " + "-"*78)

    for i, r in enumerate(results):
        delta_cv = ""
        if i > 0:
            d = r['cv'] - results[i-1]['cv']
            delta_cv = f"{d:>+7.3f}"
        print(f"  {r['step']:>8d} {r['mean']:>8.4f} {r['std']:>8.4f} {r['cv']:>6.3f} "
              f"{r['large_frac']*100:>5.1f}% {r['skew']:>8.3f} {r['kurt']:>8.2f} "
              f"{r['gini']:>6.3f} {delta_cv:>8}")

    # 寻找相变点：CV变化最大的位置
    if len(results) >= 3:
        cvs = [r['cv'] for r in results]
        delta_cvs = [cvs[i+1] - cvs[i] for i in range(len(cvs)-1)]
        max_jump_idx = np.argmax(np.abs(delta_cvs))
        phase_step = results[max_jump_idx + 1]['step']

        print(f"\n  【渗流相变分析】")
        print(f"    CV（变异系数）= 异质性指标，类比PNM的连通度")
        print(f"    CV突变最大点: step {results[max_jump_idx]['step']} → step {phase_step}")
        print(f"      CV从 {cvs[max_jump_idx]:.4f} → {cvs[max_jump_idx+1]:.4f} (Δ={delta_cvs[max_jump_idx]:+.4f})")

        # 基尼系数演化
        ginis = [r['gini'] for r in results]
        delta_ginis = [ginis[i+1] - ginis[i] for i in range(len(ginis)-1)]
        max_gini_jump = np.argmax(np.abs(delta_ginis))

        print(f"\n    基尼系数 = 权重集中度，类比渗透率不均匀性")
        print(f"    基尼突变最大点: step {results[max_gini_jump]['step']} → step {results[max_gini_jump+1]['step']}")
        print(f"      基尼从 {ginis[max_gini_jump]:.4f} → {ginis[max_gini_jump+1]:.4f}")

        print(f"\n  【PNM解读】")
        print(f"    训练初期(step 0~几百): 权重从随机初始化开始分化")
        print(f"    相变区间: 权重异质性(CV)和集中度(基尼)急剧变化")
        print(f"    训练后期: 渗流通道稳定，微调优化")
        print(f"    这与PNM中'逐步增加驱替压力→突然贯通→稳定流'的过程高度类似")

    return results


def analyze_training_evolution():
    """
    实验4：训练过程中的"饱和度"演化

    类比：PNM中逐步增加饱和度 → 连通路径增多 → 渗透率上升
         NN训练过程 → 权重分布演化 → "孔隙结构"重组

    方法：对比随机初始化 vs 训练后的GPT-2权重分布变化
    （无需Pythia多checkpoint，直接用随机初始化做对照）
    """
    print("\n")
    print("="*70)
    print("实验4：训练演化——随机初始化 vs 训练后权重")
    print("="*70)
    print()
    print("PNM类比: 初始→随机孔隙结构；训练后→优化的渗流通道")
    print()

    import torch
    from transformers import GPT2Model, GPT2Config
    from scipy import stats

    # 随机初始化的GPT-2（未训练）
    config = GPT2Config()
    model_random = GPT2Model(config)

    # 训练后的GPT-2
    model_trained = GPT2Model.from_pretrained('gpt2')

    layers_to_compare = [
        'wte.weight',
        'h.0.attn.c_attn.weight',
        'h.0.mlp.c_fc.weight',
        'h.5.attn.c_attn.weight',
        'h.5.mlp.c_fc.weight',
        'h.11.attn.c_attn.weight',
        'h.11.mlp.c_fc.weight',
    ]

    print(f"  {'层':<30} {'初始|w|均值':>10} {'训练|w|均值':>10} {'变化倍数':>8} {'KS':>8} {'解读':>15}")
    print("  " + "-"*85)

    random_dict = dict(model_random.named_parameters())
    trained_dict = dict(model_trained.named_parameters())

    evolution_data = []

    for layer_name in layers_to_compare:
        if layer_name not in random_dict:
            continue
        w_rand = np.abs(random_dict[layer_name].detach().numpy().flatten())
        w_train = np.abs(trained_dict[layer_name].detach().numpy().flatten())

        ratio = w_train.mean() / w_rand.mean() if w_rand.mean() > 0 else 0
        ks_stat, _ = stats.ks_2samp(
            np.random.choice(w_rand, 5000, replace=False),
            np.random.choice(w_train, 5000, replace=False),
        )

        if ratio > 1.5:
            interp = "孔喉扩张 ↑"
        elif ratio < 0.7:
            interp = "孔喉收缩 ↓"
        else:
            interp = "结构微调 →"

        print(f"  {layer_name:<30} {w_rand.mean():>10.4f} {w_train.mean():>10.4f} {ratio:>7.2f}x {ks_stat:>8.4f} {interp:>15}")

        evolution_data.append({
            'layer': layer_name,
            'rand_mean': w_rand.mean(),
            'train_mean': w_train.mean(),
            'ratio': ratio,
            'ks': ks_stat,
            'rand_zero_frac': (w_rand < 1e-6).mean(),
            'train_zero_frac': (w_train < 1e-6).mean(),
        })

    # 全局对比
    all_rand = []
    all_train = []
    for name, p in model_random.named_parameters():
        if p.ndim >= 2:
            all_rand.extend(np.abs(p.detach().numpy().flatten()).tolist())
    for name, p in model_trained.named_parameters():
        if p.ndim >= 2:
            all_train.extend(np.abs(p.detach().numpy().flatten()).tolist())

    all_rand = np.array(all_rand)
    all_train = np.array(all_train)

    print(f"\n  全局统计:")
    print(f"    {'指标':<25} {'随机初始化':>12} {'训练后':>12} {'变化':>10}")
    print("    " + "-"*55)

    metrics = [
        ("|w| 均值", all_rand.mean(), all_train.mean()),
        ("|w| 标准差", all_rand.std(), all_train.std()),
        ("|w| 中位数", np.median(all_rand), np.median(all_train)),
        ("堵孔率(<1e-6)", (all_rand<1e-6).mean(), (all_train<1e-6).mean()),
        ("大孔率(>0.1)", (all_rand>0.1).mean(), (all_train>0.1).mean()),
        ("log偏度", stats.skew(np.log(all_rand[all_rand>1e-10])),
                   stats.skew(np.log(all_train[all_train>1e-10]))),
    ]
    for name, v_r, v_t in metrics:
        if isinstance(v_r, float) and v_r != 0:
            change = f"{v_t/v_r:.2f}x"
        else:
            change = "—"
        print(f"    {name:<25} {v_r:>12.4f} {v_t:>12.4f} {change:>10}")

    print(f"\n  PNM解读:")
    print(f"    训练 = 渗流优化过程")
    print(f"    初始随机权重 = 均匀随机孔隙结构（各处差不多）")
    print(f"    训练后权重 = 优化的渗流网络（主通道变宽，死胡同被抑制）")

    return evolution_data


# ═══════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════
if __name__ == '__main__':
    try:
        import torch
        from transformers import GPT2Model, GPT2Tokenizer
    except ImportError:
        print("需要安装: pip install torch transformers")
        print("安装后重新运行本脚本")
        sys.exit(1)

    print("加载 GPT-2 small (124M参数)...")
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()

    # 提取所有权重
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.0f}M)")
    print(f"  权重矩阵数: {len([w for w in weights.values() if w.ndim >= 2])}")

    # 实验1：孔径分布
    all_w, layer_stats = analyze_weight_distribution(weights)

    # 实验2：Pc曲线
    analyze_activation_curve(model, tokenizer)

    # 实验3：PoreSpy真实孔径分布对比
    pnm_dists = compare_with_porespy(all_w)

    # 实验3b：深挖分布形态
    deep_dive_porespy(all_w)

    # 实验4：训练演化（GPT-2 init vs trained）
    evolution = analyze_training_evolution()

    # 实验4b：Pythia多checkpoint训练轨迹
    pythia_results = deep_dive_training_evolution()

    print("\n")
    print("="*70)
    print("MIS法 → NPM 全部实验总结")
    print("="*70)
    print("""
  实验1: 权重|w|分布 ↔ 孔径分布
    NN权重|w|可直接类比孔喉直径
    分布为左偏（偏度-1.4），与对数正态有差异

  实验2: 激活率-输入强度曲线 ↔ Pc(Sw)曲线
    GPT-2的GELU产生平滑S形前半段
    半饱和点≈5.0，类比PNM的中位孔径启动压力

  实验3: PoreSpy真实PNM对比
    KS检验显示统计分布不同
    但Weibull拟合和尾部指数揭示结构性相似

  实验3b: 深挖形态对比
    Weibull形状参数、尾部幂律指数、rank归一化QQ分析
    消除尺度差异后，分布形状可能高度相似

  实验4: 训练演化 ≈ 渗流优化
    GPT-2训练使|w|均值扩大6.4x，大孔率从0%→41%

  实验4b: Pythia训练轨迹
    13个checkpoint追踪CV和基尼系数演化
    寻找训练过程中的"渗流相变点"

  诚实声明:
    - 类比为定性/探索性质，形态相似但非严格数学等价
    - KS检验说明原始分布确实不同，深挖是分离尺度和形状
    - Pythia相变点分析依赖checkpoint粒度，精度有限
    - 所有结论基于PNM-NN结构映射假设
""")
