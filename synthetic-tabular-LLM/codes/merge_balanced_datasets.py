"""
合并原始训练数据和生成的少数类样本，创建平衡数据集

为 Sick, Travel, HELOC 三个数据集分别合并：
1. 原始训练数据 + CTGAN条件采样少数类 → 平衡数据集
2. 原始训练数据 + TVAE条件采样少数类 → 平衡数据集
3. 原始训练数据 + CTGAN拒绝采样少数类 → 平衡数据集
4. 原始训练数据 + TVAE拒绝采样少数类 → 平衡数据集

所有平衡数据集保存到新文件夹: synthetic-tabular-LLM/data/balanced_datasets/
"""

import pandas as pd
import os
from pathlib import Path

# 配置路径
BASE_DIR = Path('../data')
REALDATA_DIR = BASE_DIR / 'realdata'
SYNDATA_DIR = BASE_DIR / 'syndata'
OUTPUT_DIR = BASE_DIR / 'balanced_datasets'

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据集配置
DATASETS_CONFIG = {
    'Sick': {
        'X_train': REALDATA_DIR / 'Sick' / 'X_train.csv',
        'y_train': REALDATA_DIR / 'Sick' / 'y_train.csv',
        'target_column': 'Class',
        'minority_class': 'sick',
        'majority_class': 'negative',
        'ctgan_conditional': SYNDATA_DIR / 'Sick_CTGAN_minority_only.csv',
        'tvae_conditional': SYNDATA_DIR / 'Sick_TVAE_minority_only.csv',
        'ctgan_rejection': SYNDATA_DIR / 'Sick_CTGAN_rejection_sampling.csv',
        'tvae_rejection': SYNDATA_DIR / 'Sick_TVAE_rejection_sampling.csv',
    },
    'Travel': {
        'X_train': REALDATA_DIR / 'travel' / 'X_train.csv',
        'y_train': REALDATA_DIR / 'travel' / 'y_train.csv',
        'target_column': 'Target',
        'minority_class': 1,
        'majority_class': 0,
        'ctgan_conditional': SYNDATA_DIR / 'Travel_CTGAN_minority_only.csv',
        'tvae_conditional': SYNDATA_DIR / 'Travel_TVAE_minority_only.csv',
        'ctgan_rejection': SYNDATA_DIR / 'Travel_CTGAN_rejection_sampling.csv',
        'tvae_rejection': SYNDATA_DIR / 'Travel_TVAE_rejection_sampling.csv',
    },
    'HELOC': {
        'X_train': REALDATA_DIR / 'HELOC' / 'X_train.csv',
        'y_train': REALDATA_DIR / 'HELOC' / 'y_train.csv',
        'target_column': 'RiskPerformance',
        'minority_class': 'RIS_1',
        'majority_class': 'RIS_0',
        'ctgan_conditional': SYNDATA_DIR / 'HELOC_CTGAN_minority_only.csv',
        'tvae_conditional': SYNDATA_DIR / 'HELOC_TVAE_minority_only.csv',
        'ctgan_rejection': SYNDATA_DIR / 'HELOC_CTGAN_rejection_sampling.csv',
        'tvae_rejection': SYNDATA_DIR / 'HELOC_TVAE_rejection_sampling.csv',
    }
}

def load_original_data(dataset_name, config):
    """加载原始训练数据"""
    print(f"\n{'='*60}")
    print(f"📂 加载 {dataset_name} 原始训练数据...")
    
    X_train = pd.read_csv(config['X_train'])
    y_train = pd.read_csv(config['y_train'])
    
    # 移除可能存在的 index 列
    if 'index' in y_train.columns:
        y_train = y_train.drop('index', axis=1)
    
    # 合并 X 和 y
    target_column = config['target_column']
    train_data = pd.concat([X_train, y_train], axis=1)
    
    print(f"✅ 原始训练数据: {len(train_data)} 条")
    print(f"   类别分布:")
    print(train_data[target_column].value_counts())
    
    return train_data, target_column

def merge_and_balance(original_data, synthetic_data, target_column, minority_class, dataset_name, method_name):
    """合并原始数据和合成少数类数据，创建平衡数据集"""
    
    # 确保合成数据只包含少数类
    minority_syn = synthetic_data[synthetic_data[target_column] == minority_class].copy()
    
    print(f"\n   📊 {method_name}:")
    print(f"      - 合成少数类样本: {len(minority_syn)} 条")
    
    # 合并数据
    balanced_data = pd.concat([original_data, minority_syn], axis=0, ignore_index=True)
    
    # 检查平衡性
    class_counts = balanced_data[target_column].value_counts()
    print(f"      - 合并后总样本: {len(balanced_data)} 条")
    print(f"      - 类别分布:")
    for cls, count in class_counts.items():
        percentage = count / len(balanced_data) * 100
        print(f"        {cls}: {count} ({percentage:.1f}%)")
    
    # 计算不平衡比例
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    print(f"      - 不平衡比例: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio < 1.1:
        print(f"      ✅ 数据集接近完美平衡！")
    elif imbalance_ratio < 1.5:
        print(f"      ✅ 数据集轻度不平衡")
    else:
        print(f"      ⚠️ 数据集仍有不平衡")
    
    return balanced_data

def process_dataset(dataset_name, config):
    """处理单个数据集"""
    print(f"\n{'#'*60}")
    print(f"# 处理数据集: {dataset_name}")
    print(f"{'#'*60}")
    
    # 加载原始数据
    original_data, target_column = load_original_data(dataset_name, config)
    minority_class = config['minority_class']
    
    # 创建数据集专属文件夹
    dataset_output_dir = OUTPUT_DIR / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理四种合成方法
    methods = [
        ('CTGAN_conditional', config['ctgan_conditional'], 'CTGAN条件采样'),
        ('TVAE_conditional', config['tvae_conditional'], 'TVAE条件采样'),
        ('CTGAN_rejection', config['ctgan_rejection'], 'CTGAN拒绝采样'),
        ('TVAE_rejection', config['tvae_rejection'], 'TVAE拒绝采样'),
    ]
    
    results = []
    
    for method_key, syn_file, method_name in methods:
        if not syn_file.exists():
            print(f"\n   ⚠️ {method_name}: 文件不存在 - {syn_file}")
            continue
        
        # 加载合成数据
        synthetic_data = pd.read_csv(syn_file)
        
        # 合并并平衡
        balanced_data = merge_and_balance(
            original_data, synthetic_data, target_column, 
            minority_class, dataset_name, method_name
        )
        
        # 保存平衡数据集
        output_file = dataset_output_dir / f'{dataset_name}_balanced_{method_key}.csv'
        balanced_data.to_csv(output_file, index=False)
        print(f"      💾 已保存: {output_file.name}")
        
        results.append({
            'method': method_name,
            'file': output_file.name,
            'samples': len(balanced_data)
        })
    
    return results

# 主程序
if __name__ == '__main__':
    print("="*60)
    print("🚀 开始合并原始数据和合成少数类数据，创建平衡数据集")
    print("="*60)
    
    all_results = {}
    
    for dataset_name, config in DATASETS_CONFIG.items():
        results = process_dataset(dataset_name, config)
        all_results[dataset_name] = results
    
    # 最终总结
    print(f"\n{'='*60}")
    print("🎉 所有数据集处理完成！")
    print(f"{'='*60}")
    
    print(f"\n📁 所有平衡数据集已保存到: {OUTPUT_DIR}")
    print(f"\n📊 生成的平衡数据集总结:\n")
    
    for dataset_name, results in all_results.items():
        print(f"  {dataset_name}:")
        for result in results:
            print(f"    ✅ {result['file']} ({result['samples']} 条)")
    
    print(f"\n✨ 完成！共生成 {sum(len(r) for r in all_results.values())} 个平衡数据集！")

