"""
🚀 Travel 数据集深度学习基线 (CTGAN & TVAE) - 简化版
"""
import pandas as pd
import numpy as np
import time
import warnings
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# 配置
DATA_DIR = '../data/realdata/travel'
SAVE_DIR = '../data/syndata'
SAMPLES_TO_GENERATE = 1000
EPOCHS = 50  # 减少到 50 轮以加快速度
TARGET_COLUMN = 'Target'

print("=" * 60)
print("🚀 Travel 数据集深度学习基线 (CTGAN & TVAE)")
print("=" * 60)

# 1. 读取数据
print("\n[1/6] 正在读取数据...")
X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv', index_col=0)
y_train = pd.read_csv(f'{DATA_DIR}/y_train.csv', index_col=0)
X_test = pd.read_csv(f'{DATA_DIR}/X_test.csv', index_col=0)
y_test = pd.read_csv(f'{DATA_DIR}/y_test.csv', index_col=0).values.ravel()

print(f"   训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")
print(f"   类别分布: {dict(pd.Series(y_train[TARGET_COLUMN]).value_counts())}")

# 合并数据
train_data = pd.concat([X_train, y_train], axis=1)

# 标签编码
le_y = LabelEncoder()
y_test_enc = le_y.fit_transform(y_test)
minority_class = np.argmin(np.bincount(y_test_enc))

# 2. 检测元数据
print("\n[2/6] 正在检测数据结构...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)
print("   ✅ 元数据检测完成")

# 评估函数
def evaluate_synthetic_data(name, synthetic_data):
    print(f"\n   评估 {name} 生成的数据...")
    
    X_syn = synthetic_data.drop(TARGET_COLUMN, axis=1)
    y_syn = synthetic_data[TARGET_COLUMN]
    
    print(f"   生成数据: {X_syn.shape[0]} 条")
    print(f"   类别分布: {dict(y_syn.value_counts())}")
    
    # 数据编码
    full_X = pd.concat([X_train, X_test, X_syn], axis=0)
    encoders = {}
    
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or X_train[col].nunique() < 10:
            le = LabelEncoder()
            full_X[col] = full_X[col].astype(str).fillna('Missing')
            le.fit(full_X[col])
            X_syn[col] = le.transform(X_syn[col].astype(str).fillna('Missing'))
            encoders[col] = le
    
    y_syn_enc = le_y.transform(y_syn)
    
    # 混合训练
    X_train_enc = X_train.copy()
    for col, le in encoders.items():
        X_train_enc[col] = le.transform(X_train_enc[col].astype(str).fillna('Missing'))
    
    X_final = pd.concat([X_train_enc, X_syn], axis=0)
    y_final = np.concatenate([le_y.transform(y_train.values.ravel()), y_syn_enc])
    
    # 训练 XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)
    model.fit(X_final, y_final)
    
    # 预测
    X_test_encoded = X_test.copy()
    for col, le in encoders.items():
        X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str).fillna('Missing'))
    
    y_pred = model.predict(X_test_encoded)
    
    f1 = f1_score(y_test_enc, y_pred, pos_label=minority_class)
    ba = balanced_accuracy_score(y_test_enc, y_pred)
    
    print(f"   F1 Score: {f1:.4f}, Balanced Acc: {ba:.4f}")
    return f1, ba

# 3. 训练 CTGAN
print("\n" + "=" * 60)
print("[3/6] 开始训练 CTGAN")
print("=" * 60)
print(f"   训练轮数: {EPOCHS}, 目标生成: {SAMPLES_TO_GENERATE} 条")

start_time = time.time()
ctgan = CTGANSynthesizer(metadata, epochs=EPOCHS, verbose=True)
print("   开始训练...")
ctgan.fit(train_data)
train_time = time.time() - start_time

print(f"\n   ✅ CTGAN 训练完成！耗时: {train_time/60:.2f} 分钟")
print(f"   正在生成数据...")

syn_ctgan = ctgan.sample(num_rows=SAMPLES_TO_GENERATE)
syn_ctgan.to_csv(f"{SAVE_DIR}/Travel_CTGAN_samples.csv", index=False)
print(f"   ✅ 已保存: {SAVE_DIR}/Travel_CTGAN_samples.csv")

f1_ctgan, ba_ctgan = evaluate_synthetic_data("CTGAN", syn_ctgan)

# 4. 训练 TVAE
print("\n" + "=" * 60)
print("[4/6] 开始训练 TVAE")
print("=" * 60)

start_time = time.time()
tvae = TVAESynthesizer(metadata, epochs=EPOCHS, verbose=True)
print("   开始训练...")
tvae.fit(train_data)
train_time = time.time() - start_time

print(f"\n   ✅ TVAE 训练完成！耗时: {train_time/60:.2f} 分钟")
print(f"   正在生成数据...")

syn_tvae = tvae.sample(num_rows=SAMPLES_TO_GENERATE)
syn_tvae.to_csv(f"{SAVE_DIR}/Travel_TVAE_samples.csv", index=False)
print(f"   ✅ 已保存: {SAVE_DIR}/Travel_TVAE_samples.csv")

f1_tvae, ba_tvae = evaluate_synthetic_data("TVAE", syn_tvae)

# 5. 总结
print("\n" + "=" * 60)
print("✅ [5/6] 所有深度学习基线运行完毕！")
print("=" * 60)
print(f"\n📊 最终结果对比:")
print(f"   CTGAN - F1: {f1_ctgan:.4f}, Balanced Acc: {ba_ctgan:.4f}")
print(f"   TVAE  - F1: {f1_tvae:.4f}, Balanced Acc: {ba_tvae:.4f}")
print(f"\n📁 生成的文件:")
print(f"   - {SAVE_DIR}/Travel_CTGAN_samples.csv")
print(f"   - {SAVE_DIR}/Travel_TVAE_samples.csv")
print("\n🎉 完成！")
print("=" * 60)

