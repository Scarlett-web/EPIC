"""
HELOC 数据预处理脚本
作用：将 heloc_train.csv 和 heloc_test.csv 划分为 EPIC 标准格式
输出：X_train.csv, y_train.csv, X_test.csv, y_test.csv
"""
import pandas as pd
import os

print("="*60)
print("🔧 HELOC 数据预处理")
print("="*60)

# 数据路径
DATA_DIR = "../../data/realdata/HELOC"
train_file = os.path.join(DATA_DIR, "heloc_train.csv")
test_file = os.path.join(DATA_DIR, "heloc_test.csv")

# 读取数据
print(f"\n📂 读取训练集: {train_file}")
train_df = pd.read_csv(train_file)
print(f"   训练集形状: {train_df.shape}")

print(f"\n📂 读取测试集: {test_file}")
test_df = pd.read_csv(test_file)
print(f"   测试集形状: {test_df.shape}")

# 目标变量
TARGET = "RiskPerformance"

# 分离特征和目标
print(f"\n🎯 分离特征和目标变量 (目标: {TARGET})")

# 训练集
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[[TARGET]]

# 测试集
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[[TARGET]]

# 添加索引列
X_train.insert(0, 'index', range(len(X_train)))
y_train.insert(0, 'index', range(len(y_train)))
X_test.insert(0, 'index', range(len(X_test)))
y_test.insert(0, 'index', range(len(y_test)))

print(f"\n✅ 划分完成:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_test: {y_test.shape}")

# 保存文件
print(f"\n💾 保存文件到: {DATA_DIR}")

X_train.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)
X_test.to_csv(os.path.join(DATA_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(DATA_DIR, "y_test.csv"), index=False)

print("\n✅ 预处理完成！生成的文件:")
print(f"   ✓ X_train.csv ({X_train.shape[0]} 行 × {X_train.shape[1]} 列)")
print(f"   ✓ y_train.csv ({y_train.shape[0]} 行 × {y_train.shape[1]} 列)")
print(f"   ✓ X_test.csv ({X_test.shape[0]} 行 × {X_test.shape[1]} 列)")
print(f"   ✓ y_test.csv ({y_test.shape[0]} 行 × {y_test.shape[1]} 列)")

# 显示目标变量分布
print(f"\n📊 目标变量分布:")
print(f"\n训练集:")
print(y_train[TARGET].value_counts())
print(f"\n测试集:")
print(y_test[TARGET].value_counts())

print("\n" + "="*60)
print("🎉 预处理完成！现在可以运行 EPIC 生成脚本了")
print("="*60)

