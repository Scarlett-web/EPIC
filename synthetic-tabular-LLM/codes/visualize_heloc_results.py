"""
可视化 HELOC 数据集实验结果
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 创建图表
fig = plt.figure(figsize=(16, 10))

# ============================================================================
# 1. 原始数据分布
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
original_data = {'RIS_0\n(低风险)': 4108, 'RIS_1\n(高风险)': 3789}
colors_original = ['#3498db', '#e74c3c']

bars = ax1.bar(original_data.keys(), original_data.values(), color=colors_original, 
               alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('样本数', fontsize=12, fontweight='bold')
ax1.set_title('原始 HELOC 训练集分布', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for i, (k, v) in enumerate(original_data.items()):
    ax1.text(i, v + 50, f'{v}\n({v/sum(original_data.values())*100:.1f}%)', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.text(0.5, 0.95, '不平衡比例: 1.08:1\n(非常轻度不平衡)', 
         transform=ax1.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
         fontsize=9)

# ============================================================================
# 2. 标准生成对比
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
methods = ['原始数据', 'CTGAN\n标准生成', 'TVAE\n标准生成']
ris0_counts = [4108, 507, 577]
ris1_counts = [3789, 493, 423]

x = np.arange(len(methods))
width = 0.35

bars1 = ax2.bar(x - width/2, ris0_counts, width, label='RIS_0 (低风险)', 
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, ris1_counts, width, label='RIS_1 (高风险)', 
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_ylabel('样本数', fontsize=12, fontweight='bold')
ax2.set_title('标准生成类别分布对比', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=9)
ax2.legend()
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# 标注比例
ratios = [1.08, 1.03, 1.36]
for i, ratio in enumerate(ratios):
    ax2.text(i, max(ris0_counts[i], ris1_counts[i]) + 100, 
             f'比例: {ratio:.2f}:1', ha='center', fontsize=8, fontweight='bold')

# ============================================================================
# 3. 少数类采样完成度
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
sampling_methods = ['CTGAN\n条件采样', 'TVAE\n条件采样', 'CTGAN\n拒绝采样', 'TVAE\n拒绝采样']
samples_generated = [319, 319, 319, 319]
samples_needed = 319
colors_sampling = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

bars = ax3.bar(sampling_methods, samples_generated, color=colors_sampling, 
               alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.axhline(y=samples_needed, color='green', linestyle='--', linewidth=2, 
            alpha=0.7, label=f'目标: {samples_needed} 条')
ax3.set_ylabel('生成样本数', fontsize=12, fontweight='bold')
ax3.set_title('少数类采样完成度', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3, linestyle='--')

for i, v in enumerate(samples_generated):
    ax3.text(i, v + 5, f'{v}\n✅ 100%', ha='center', va='bottom', 
             fontsize=9, fontweight='bold', color='green')

# ============================================================================
# 4. 拒绝采样效率对比
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
datasets = ['HELOC\n(1.08:1)', 'Travel\n(1.8:1)', 'Sick\n(15:1)']
ctgan_efficiency = [50, 43.1, 6.9]
tvae_efficiency = [42, 20.3, 2.8]

x = np.arange(len(datasets))
width = 0.35

bars1 = ax4.bar(x - width/2, ctgan_efficiency, width, label='CTGAN', 
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x + width/2, tvae_efficiency, width, label='TVAE', 
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('拒绝采样效率 (%)', fontsize=12, fontweight='bold')
ax4.set_title('不同数据集拒绝采样效率对比', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(datasets, fontsize=9)
ax4.legend()
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.5, label='10% 阈值')

for i, (c, t) in enumerate(zip(ctgan_efficiency, tvae_efficiency)):
    ax4.text(i - width/2, c + 1, f'{c:.1f}%', ha='center', va='bottom', fontsize=8)
    ax4.text(i + width/2, t + 1, f'{t:.1f}%', ha='center', va='bottom', fontsize=8)

# ============================================================================
# 5. 方法适用性评分
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
data = np.array([
    [100, 100, 100, 100],  # HELOC - 条件采样
    [95, 90, 95, 90],      # HELOC - 拒绝采样
    [100, 100, 90, 85],    # Travel - 条件采样
    [90, 85, 40, 30],      # Travel - 拒绝采样
    [100, 100, 100, 100],  # Sick - 条件采样
    [20, 15, 10, 5],       # Sick - 拒绝采样
])

im = ax5.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax5.set_xticks(range(4))
ax5.set_xticklabels(['CTGAN\n条件', 'TVAE\n条件', 'CTGAN\n拒绝', 'TVAE\n拒绝'], fontsize=8)
ax5.set_yticks(range(6))
ax5.set_yticklabels(['HELOC\n条件采样', 'HELOC\n拒绝采样', 
                     'Travel\n条件采样', 'Travel\n拒绝采样',
                     'Sick\n条件采样', 'Sick\n拒绝采样'], fontsize=8)
ax5.set_title('方法适用性评分矩阵', fontsize=14, fontweight='bold')

for i in range(6):
    for j in range(4):
        text = ax5.text(j, i, f'{data[i, j]:.0f}',
                       ha="center", va="center", color="black", 
                       fontsize=9, fontweight='bold')

plt.colorbar(im, ax=ax5, label='适用性评分')

# ============================================================================
# 6. 总结建议
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = """
📊 HELOC 数据集实验总结

✅ 数据集特点
   • 不平衡比例: 1.08:1 (非常轻度)
   • 需要生成: 319 条少数类样本
   • 特征数: 23 列数值特征

✅ 实验结果
   • 标准生成: 2 个数据集
   • 条件采样: 2 个数据集 (100% 完成)
   • 拒绝采样: 2 个数据集 (100% 完成)
   • 共生成: 6 个合成数据集 ✅

🏆 最佳方法
   • 标准生成: CTGAN (更平衡)
   • 少数类采样: 两种方法都完美

💡 关键发现
   • 拒绝采样效率高 (42-50%)
   • 轻度不平衡数据友好
   • CTGAN 略优于 TVAE

📈 与其他数据集对比
   • HELOC: 拒绝采样效率最高
   • Travel: 拒绝采样效率中等
   • Sick: 必须使用条件采样

🎯 实践建议
   • 优先使用条件采样 (100% 精确)
   • 拒绝采样可获得更多样化样本
   • 混合策略: 70% 条件 + 30% 拒绝
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('../data/syndata/heloc_experiment_results.png', dpi=300, bbox_inches='tight')
print("✅ 可视化图表已保存: ../data/syndata/heloc_experiment_results.png")
plt.show()

