"""
验证生成的平衡数据集
"""

import pandas as pd
from pathlib import Path

# 配置路径
BALANCED_DIR = Path('../data/balanced_datasets')

# 数据集配置
DATASETS = {
    'Sick': {
        'target_column': 'Class',
        'files': [
            'Sick_balanced_CTGAN_conditional.csv',
            'Sick_balanced_TVAE_conditional.csv',
            'Sick_balanced_CTGAN_rejection.csv',
            'Sick_balanced_TVAE_rejection.csv',
        ]
    },
    'Travel': {
        'target_column': 'Target',
        'files': [
            'Travel_balanced_CTGAN_conditional.csv',
            'Travel_balanced_TVAE_conditional.csv',
            'Travel_balanced_CTGAN_rejection.csv',
            'Travel_balanced_TVAE_rejection.csv',
        ]
    },
    'HELOC': {
        'target_column': 'RiskPerformance',
        'files': [
            'HELOC_balanced_CTGAN_conditional.csv',
            'HELOC_balanced_TVAE_conditional.csv',
            'HELOC_balanced_CTGAN_rejection.csv',
            'HELOC_balanced_TVAE_rejection.csv',
        ]
    }
}

def verify_dataset(dataset_name, config):
    """验证单个数据集的所有平衡文件"""
    print(f"\n{'='*60}")
    print(f"📊 验证 {dataset_name} 数据集")
    print(f"{'='*60}")
    
    dataset_dir = BALANCED_DIR / dataset_name
    target_column = config['target_column']
    
    results = []
    
    for filename in config['files']:
        filepath = dataset_dir / filename
        
        if not filepath.exists():
            print(f"\n❌ {filename}: 文件不存在")
            continue
        
        # 读取数据
        df = pd.read_csv(filepath)
        
        # 统计信息
        total_samples = len(df)
        class_counts = df[target_column].value_counts()
        
        # 计算不平衡比例
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        # 判断平衡性
        if imbalance_ratio < 1.1:
            balance_status = "✅ 完美平衡"
        elif imbalance_ratio < 1.5:
            balance_status = "✅ 轻度不平衡"
        elif imbalance_ratio < 3.0:
            balance_status = "⚠️ 中度不平衡"
        else:
            balance_status = "❌ 严重不平衡"
        
        print(f"\n📄 {filename}")
        print(f"   总样本数: {total_samples}")
        print(f"   类别分布:")
        for cls, count in class_counts.items():
            percentage = count / total_samples * 100
            print(f"     {cls}: {count} ({percentage:.1f}%)")
        print(f"   不平衡比例: {imbalance_ratio:.2f}:1")
        print(f"   平衡状态: {balance_status}")
        
        results.append({
            'dataset': dataset_name,
            'file': filename,
            'samples': total_samples,
            'imbalance_ratio': imbalance_ratio,
            'status': balance_status
        })
    
    return results

# 主程序
if __name__ == '__main__':
    print("="*60)
    print("🔍 验证所有平衡数据集")
    print("="*60)
    
    all_results = []
    
    for dataset_name, config in DATASETS.items():
        results = verify_dataset(dataset_name, config)
        all_results.extend(results)
    
    # 总结
    print(f"\n{'='*60}")
    print("📊 验证总结")
    print(f"{'='*60}\n")
    
    # 按数据集分组统计
    for dataset_name in DATASETS.keys():
        dataset_results = [r for r in all_results if r['dataset'] == dataset_name]
        print(f"{dataset_name}:")
        for result in dataset_results:
            print(f"  {result['status']} {result['file']}")
            print(f"     样本数: {result['samples']}, 不平衡比例: {result['imbalance_ratio']:.2f}:1")
        print()
    
    # 统计完美平衡的数据集数量
    perfect_balance = sum(1 for r in all_results if r['imbalance_ratio'] < 1.1)
    print(f"✅ 完美平衡数据集: {perfect_balance}/{len(all_results)}")
    print(f"📁 所有数据集位置: {BALANCED_DIR}")
    print(f"\n🎉 验证完成！")

