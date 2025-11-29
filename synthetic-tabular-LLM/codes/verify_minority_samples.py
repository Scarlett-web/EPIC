"""验证生成的少数类样本文件"""
import pandas as pd
import os

base_dir = '../data/syndata'

print("=" * 70)
print("📊 Sick 数据集 - 少数类样本生成验证")
print("=" * 70)

sick_files = [
    'Sick_CTGAN_minority_only.csv',
    'Sick_TVAE_minority_only.csv',
    'Sick_CTGAN_rejection_sampling.csv',
    'Sick_TVAE_rejection_sampling.csv'
]

for f in sick_files:
    path = os.path.join(base_dir, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n✅ {f}:")
        print(f"   样本数: {len(df)} 条")
        print(f"   类别分布: {dict(df['Class'].value_counts())}")
    else:
        print(f"\n❌ {f}: 文件不存在")

print("\n" + "=" * 70)
print("📊 Travel 数据集 - 少数类样本生成验证")
print("=" * 70)

travel_files = [
    'Travel_CTGAN_minority_only.csv',
    'Travel_TVAE_minority_only.csv',
    'Travel_CTGAN_rejection_sampling.csv',
    'Travel_TVAE_rejection_sampling.csv'
]

for f in travel_files:
    path = os.path.join(base_dir, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n✅ {f}:")
        print(f"   样本数: {len(df)} 条")
        print(f"   类别分布: {dict(df['Target'].value_counts())}")
    else:
        print(f"\n❌ {f}: 文件不存在")

print("\n" + "=" * 70)
print("🎉 验证完成！")
print("=" * 70)

