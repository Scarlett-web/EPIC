"""
数据预处理脚本：将 Travel 数据集转换为标准格式
作用：将 travel_train.csv 和 travel_test.csv 转换为 X_train.csv, y_train.csv, X_test.csv, y_test.csv
"""
import pandas as pd
import os

# ==========================================
# 📂 配置路径
# ==========================================
DATA_DIR = "../../data/realdata/travel"
OUTPUT_DIR = "../../data/realdata/Travel"  # 注意大写 T，与 Classification.py 保持一致

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 [1/3] 正在读取 Travel 数据集...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "travel_train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "travel_test.csv"))

print(f"    训练集形状: {train_df.shape}")
print(f"    测试集形状: {test_df.shape}")

# ==========================================
# 🧹 [2/3] 数据清洗
# ==========================================
print("\n🧹 [2/3] 数据清洗...")

# 删除无用的索引列
if 'Unnamed: 0' in train_df.columns:
    train_df = train_df.drop(columns=['Unnamed: 0'])
    test_df = test_df.drop(columns=['Unnamed: 0'])
    print("    ✅ 已删除 'Unnamed: 0' 列")

# 检查目标变量
print(f"\n    目标变量分布 (训练集):")
print(train_df['TravelInsurance'].value_counts())
print(f"\n    类别不平衡比例: {train_df['TravelInsurance'].value_counts()[0] / train_df['TravelInsurance'].value_counts()[1]:.2f}:1")

# ==========================================
# 📊 [3/3] 分离特征和标签
# ==========================================
print("\n📊 [3/3] 分离特征和标签...")

# 训练集
X_train = train_df.drop(columns=['TravelInsurance'])
y_train = train_df[['TravelInsurance']]
y_train.columns = ['Target']  # 重命名为 Target，与 Classification.py 保持一致

# 测试集
X_test = test_df.drop(columns=['TravelInsurance'])
y_test = test_df[['TravelInsurance']]
y_test.columns = ['Target']

# ==========================================
# 💾 保存文件
# ==========================================
print("\n💾 正在保存文件...")

X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index_label='index')
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index_label='index')
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index_label='index')
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index_label='index')

print(f"✅ 数据预处理完成！文件已保存到: {OUTPUT_DIR}")
print("\n📁 生成的文件:")
print(f"    - X_train.csv: {X_train.shape}")
print(f"    - y_train.csv: {y_train.shape}")
print(f"    - X_test.csv: {X_test.shape}")
print(f"    - y_test.csv: {y_test.shape}")

print("\n🎯 下一步：运行 generate_samples_Travel.py 生成合成数据")

