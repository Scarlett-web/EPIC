"""测试条件采样功能"""
import pandas as pd
import sys
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition

print("=" * 60)
print("测试 CTGAN 条件采样功能")
print("=" * 60)

# 读取数据
print("\n1. 读取数据...")
DATA_DIR = '../data/realdata/Sick'
X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv', index_col=0)
y_train = pd.read_csv(f'{DATA_DIR}/y_train.csv', index_col=0)
train_data = pd.concat([X_train, y_train], axis=1)

print(f"   训练数据: {train_data.shape}")
print(f"   类别分布: {dict(y_train['Class'].value_counts())}")

# 检测元数据
print("\n2. 检测元数据...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)
print("   ✅ 完成")

# 训练 CTGAN（少量轮数测试）
print("\n3. 训练 CTGAN（10 轮测试）...")
ctgan = CTGANSynthesizer(metadata, epochs=10, verbose=True)
ctgan.fit(train_data)
print("   ✅ 训练完成")

# 测试条件采样
print("\n4. 测试条件采样...")
print("   生成 10 条 'sick' 样本...")
condition = Condition(num_rows=10, column_values={'Class': 'sick'})
syn_data = ctgan.sample_from_conditions(conditions=[condition])

print(f"\n   生成结果:")
print(f"   形状: {syn_data.shape}")
print(f"   类别分布: {dict(syn_data['Class'].value_counts())}")
print(f"\n   前 3 行:")
print(syn_data.head(3))

print("\n✅ 条件采样测试成功！")

