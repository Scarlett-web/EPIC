"""
验证 HELOC 数据集生成结果
"""
import pandas as pd
import os

print("=" * 80)
print("🔍 HELOC 数据集生成结果验证")
print("=" * 80)

syndata_path = '../data/syndata'
target_column = 'RiskPerformance'

# 定义要检查的文件
files_to_check = [
    ('HELOC_CTGAN_standard.csv', 'CTGAN 标准生成（保持不平衡）'),
    ('HELOC_TVAE_standard.csv', 'TVAE 标准生成（保持不平衡）'),
    ('HELOC_CTGAN_minority_only.csv', 'CTGAN 条件采样（只生成少数类）'),
    ('HELOC_TVAE_minority_only.csv', 'TVAE 条件采样（只生成少数类）'),
    ('HELOC_CTGAN_rejection_sampling.csv', 'CTGAN 拒绝采样（只生成少数类）'),
    ('HELOC_TVAE_rejection_sampling.csv', 'TVAE 拒绝采样（只生成少数类）'),
]

# 加载原始训练数据
y_train = pd.read_csv('../data/realdata/HELOC/y_train.csv')
if 'index' in y_train.columns:
    y_train = y_train.drop('index', axis=1)

print("\n📊 原始训练集类别分布:")
original_counts = y_train[target_column].value_counts()
for cls, count in original_counts.items():
    print(f"   {cls}: {count} ({count/len(y_train)*100:.2f}%)")

minority_class = 'RIS_1'
majority_class = 'RIS_0'
samples_needed = original_counts[majority_class] - original_counts[minority_class]
print(f"\n🎯 需要生成 {samples_needed} 条少数类样本来平衡数据集")

print("\n" + "=" * 80)
print("📁 生成文件验证")
print("=" * 80)

results = []

for filename, description in files_to_check:
    filepath = os.path.join(syndata_path, filename)
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        
        print(f"\n✅ {description}")
        print(f"   文件: {filename}")
        print(f"   样本数: {len(df)} 条")
        print(f"   特征数: {df.shape[1]} 列")
        
        if target_column in df.columns:
            class_dist = df[target_column].value_counts()
            print(f"   类别分布:")
            for cls, count in class_dist.items():
                print(f"      {cls}: {count} ({count/len(df)*100:.2f}%)")
            
            # 检查是否只有少数类
            if 'minority_only' in filename or 'rejection' in filename:
                if len(class_dist) == 1 and minority_class in class_dist:
                    print(f"   ✅ 纯度检查: 100% 少数类 ({minority_class})")
                    purity = 100.0
                else:
                    minority_pct = (class_dist.get(minority_class, 0) / len(df)) * 100
                    print(f"   ⚠️  纯度检查: {minority_pct:.1f}% 少数类")
                    purity = minority_pct
                
                # 检查样本数是否足够
                if len(df) >= samples_needed:
                    print(f"   ✅ 样本数检查: 足够（{len(df)}/{samples_needed}）")
                    completeness = 100.0
                else:
                    completeness = (len(df) / samples_needed) * 100
                    print(f"   ⚠️  样本数检查: 不足（{len(df)}/{samples_needed}，{completeness:.1f}%）")
                
                results.append({
                    'method': description,
                    'samples': len(df),
                    'purity': purity,
                    'completeness': completeness
                })
            else:
                # 标准生成，检查类别分布
                results.append({
                    'method': description,
                    'samples': len(df),
                    'distribution': class_dist.to_dict()
                })
        else:
            print(f"   ⚠️  警告: 未找到目标列 '{target_column}'")
    else:
        print(f"\n❌ {description}")
        print(f"   文件: {filename}")
        print(f"   状态: 文件不存在")

# 总结
print("\n" + "=" * 80)
print("📊 结果总结")
print("=" * 80)

print("\n1️⃣ 标准生成（保持不平衡）:")
for result in results:
    if 'distribution' in result:
        print(f"   {result['method']}: {result['samples']} 条")
        if 'distribution' in result:
            for cls, count in result['distribution'].items():
                print(f"      {cls}: {count} ({count/result['samples']*100:.1f}%)")

print("\n2️⃣ 少数类采样（条件采样 + 拒绝采样）:")
minority_results = [r for r in results if 'purity' in r]
if minority_results:
    print(f"\n{'方法':<40} {'样本数':<10} {'纯度':<10} {'完成度':<10}")
    print("-" * 70)
    for result in minority_results:
        print(f"{result['method']:<40} {result['samples']:<10} {result['purity']:<10.1f}% {result['completeness']:<10.1f}%")

print("\n" + "=" * 80)
print("✅ 验证完成！")
print("=" * 80)

