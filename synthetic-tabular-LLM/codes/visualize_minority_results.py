"""
可视化少数类采样结果对比
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# Travel 数据集结果
travel_results = {
    'CTGAN\n(条件采样)': {'f1': 0.6593, 'ba': 0.7372, 'samples': 453, 'purity': 100},
    'TVAE\n(条件采样)': {'f1': 0.6618, 'ba': 0.7388, 'samples': 453, 'purity': 100},
    'CTGAN\n(拒绝采样)': {'f1': 0.6838, 'ba': 0.7552, 'samples': 453, 'purity': 100},
    'TVAE\n(拒绝采样)': {'f1': 0.6593, 'ba': 0.7372, 'samples': 453, 'purity': 100},
}

# Sick 数据集结果（条件采样）
sick_results = {
    'CTGAN\n(条件采样)': {'samples': 2598, 'purity': 100, 'efficiency': 100},
    'TVAE\n(条件采样)': {'samples': 2598, 'purity': 100, 'efficiency': 100},
    'CTGAN\n(拒绝采样)': {'samples': 894, 'purity': 100, 'efficiency': 6.9},
    'TVAE\n(拒绝采样)': {'samples': 360, 'purity': 100, 'efficiency': 2.8},
}

# 拒绝采样效率对比
rejection_efficiency = {
    'Travel\nCTGAN': 43.1,
    'Travel\nTVAE': 20.3,
    'Sick\nCTGAN': 6.9,
    'Sick\nTVAE': 2.8,
}

# 创建图表
fig = plt.figure(figsize=(16, 10))

# 1. Travel 数据集性能对比
ax1 = plt.subplot(2, 3, 1)
methods = list(travel_results.keys())
f1_scores = [travel_results[m]['f1'] for m in methods]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

bars = ax1.bar(methods, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax1.set_title('Travel 数据集 - F1 Score 对比', fontsize=14, fontweight='bold')
ax1.set_ylim(0.6, 0.7)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 标注最高值
max_idx = np.argmax(f1_scores)
bars[max_idx].set_edgecolor('gold')
bars[max_idx].set_linewidth(3)
ax1.text(max_idx, f1_scores[max_idx] + 0.002, '🏆 最佳', ha='center', fontsize=10, fontweight='bold')

for i, v in enumerate(f1_scores):
    ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# 2. Travel 数据集 Balanced Accuracy 对比
ax2 = plt.subplot(2, 3, 2)
ba_scores = [travel_results[m]['ba'] for m in methods]

bars = ax2.bar(methods, ba_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Balanced Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Travel 数据集 - Balanced Accuracy 对比', fontsize=14, fontweight='bold')
ax2.set_ylim(0.7, 0.77)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

max_idx = np.argmax(ba_scores)
bars[max_idx].set_edgecolor('gold')
bars[max_idx].set_linewidth(3)

for i, v in enumerate(ba_scores):
    ax2.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# 3. 拒绝采样效率对比
ax3 = plt.subplot(2, 3, 3)
methods_eff = list(rejection_efficiency.keys())
efficiencies = list(rejection_efficiency.values())
colors_eff = ['#3498db', '#e74c3c', '#9b59b6', '#e67e22']

bars = ax3.bar(methods_eff, efficiencies, color=colors_eff, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('效率 (%)', fontsize=12, fontweight='bold')
ax3.set_title('拒绝采样效率对比', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.5, label='10% 阈值')
ax3.legend()

for i, v in enumerate(efficiencies):
    ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Sick 数据集生成样本数对比
ax4 = plt.subplot(2, 3, 4)
sick_methods = list(sick_results.keys())
sick_samples = [sick_results[m]['samples'] for m in sick_methods]

bars = ax4.bar(sick_methods, sick_samples, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('生成样本数', fontsize=12, fontweight='bold')
ax4.set_title('Sick 数据集 - 生成样本数对比', fontsize=14, fontweight='bold')
ax4.axhline(y=2598, color='green', linestyle='--', linewidth=2, alpha=0.5, label='目标: 2598 条')
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.legend()

for i, v in enumerate(sick_samples):
    ax4.text(i, v + 50, f'{v}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if v < 2598:
        ax4.text(i, v/2, f'⚠️ 不足\n{v/2598*100:.1f}%', ha='center', va='center', fontsize=8, color='red', fontweight='bold')

# 5. 方法适用性矩阵
ax5 = plt.subplot(2, 3, 5)
data = np.array([
    [100, 100, 100, 100],  # 条件采样 - 轻度不平衡
    [100, 100, 100, 100],  # 条件采样 - 严重不平衡
    [90, 85, 40, 30],      # 拒绝采样 - 轻度不平衡
    [20, 15, 10, 5],       # 拒绝采样 - 严重不平衡
])

im = ax5.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax5.set_xticks(range(4))
ax5.set_xticklabels(['CTGAN\n条件', 'TVAE\n条件', 'CTGAN\n拒绝', 'TVAE\n拒绝'], fontsize=9)
ax5.set_yticks(range(4))
ax5.set_yticklabels(['轻度不平衡\n(<3:1)', '中度不平衡\n(3:1~10:1)', '严重不平衡\n(>10:1) 轻度', '严重不平衡\n(>10:1) 重度'], fontsize=9)
ax5.set_title('方法适用性评分矩阵', fontsize=14, fontweight='bold')

for i in range(4):
    for j in range(4):
        text = ax5.text(j, i, f'{data[i, j]:.0f}',
                       ha="center", va="center", color="black", fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax5, label='适用性评分')

# 6. 总结建议
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = """
📊 实验总结与建议

✅ 条件采样（Conditional Sampling）
   • 100% 精确，适用于所有不平衡程度
   • 推荐作为默认方法
   • Travel: F1=0.6593~0.6618
   • Sick: 完美生成 2598 条样本

⚠️ 拒绝采样（Rejection Sampling）
   • 在轻度不平衡数据上表现优异
   • Travel: F1=0.6838 (最佳) 🏆
   • 但在严重不平衡数据上效率极低
   • Sick: 仅生成 34%~14% 目标样本

💡 实践建议
   1. 不平衡比例 < 3:1 → 拒绝采样
   2. 不平衡比例 3:1~10:1 → 条件采样
   3. 不平衡比例 > 10:1 → 必须条件采样

🏆 最佳方法
   • Travel (1.8:1): CTGAN 拒绝采样
   • Sick (15:1): CTGAN/TVAE 条件采样
"""

ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('../data/syndata/minority_sampling_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 可视化图表已保存: ../data/syndata/minority_sampling_comparison.png")
plt.show()

