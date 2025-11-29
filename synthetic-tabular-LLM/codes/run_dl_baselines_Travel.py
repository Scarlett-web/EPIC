"""
🚀 Travel 数据集深度学习基线 (CTGAN & TVAE)
运行 CTGAN 和 TVAE 生成合成数据，并评估质量
"""
import pandas as pd
import numpy as np
import time
import warnings
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
DATA_DIR = '../data/realdata/travel'  # Travel 数据集路径
SAVE_DIR = '../data/syndata'
SAMPLES_TO_GENERATE = 1000  # 生成 1000 条数据
EPOCHS = 100                # 训练轮数 (可以改小，比如 50)
TARGET_COLUMN = 'Target'    # Travel 数据集的目标列
# ===========================================

import sys
sys.stdout.reconfigure(line_buffering=True)

print("="*60, flush=True)
print("🚀 Travel 数据集深度学习基线 (CTGAN & TVAE)", flush=True)
print("="*60, flush=True)

print("\n📂 [1/5] 正在读取原始数据...", flush=True)
try:
    X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv', index_col=0)
    y_train = pd.read_csv(f'{DATA_DIR}/y_train.csv', index_col=0)
    X_test = pd.read_csv(f'{DATA_DIR}/X_test.csv', index_col=0)
    y_test = pd.read_csv(f'{DATA_DIR}/y_test.csv', index_col=0).values.ravel()
    
    print(f"   ✅ 训练集: {X_train.shape[0]} 条")
    print(f"   ✅ 测试集: {X_test.shape[0]} 条")
    print(f"   ✅ 特征数: {X_train.shape[1]} 个")
    
    # 合并 X 和 y，因为生成模型需要学习整张表
    train_data = pd.concat([X_train, y_train], axis=1)
    
    # 查看类别分布
    print(f"\n   📊 训练集类别分布:")
    print(f"      {y_train[TARGET_COLUMN].value_counts().to_dict()}")
    
    # 标签编码 (XGBoost评估用)
    le_y = LabelEncoder()
    y_test_enc = le_y.fit_transform(y_test)
    minority_class = np.argmin(np.bincount(y_test_enc))
    print(f"   🎯 少数类标签: {minority_class}")
    
except FileNotFoundError as e:
    print(f"❌ 错误：找不到数据文件！")
    print(f"   请确保已运行 preprocess_travel_data.py")
    print(f"   错误详情: {e}")
    exit()

# 自动检测元数据 (告诉模型哪些列是分类，哪些是数值)
print("\n🔍 [2/5] 正在自动检测数据结构...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)
print("   ✅ 元数据检测完成")

# 定义一个通用的评估函数
def evaluate_synthetic_data(name, synthetic_data):
    print(f"\n📊 正在评估 {name} 生成的数据质量...")
    
    # 1. 准备合成数据的 X 和 y
    X_syn = synthetic_data.drop(TARGET_COLUMN, axis=1)
    y_syn = synthetic_data[TARGET_COLUMN]
    
    print(f"   生成数据形状: {X_syn.shape}")
    print(f"   生成数据类别分布: {y_syn.value_counts().to_dict()}")
    
    # 2. 数据预处理 (转数字)
    # 合并所有数据以统一编码
    full_X = pd.concat([X_train, X_test, X_syn], axis=0)
    
    # 训练编码器
    encoders = {}
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or X_train[col].nunique() < 10:
            le = LabelEncoder()
            full_X[col] = full_X[col].astype(str).fillna('Missing')
            le.fit(full_X[col])
            
            # 转换合成数据
            X_syn[col] = le.transform(X_syn[col].astype(str).fillna('Missing'))
            encoders[col] = le
    
    # 转换标签
    y_syn_enc = le_y.transform(y_syn)
    
    # 3. 混合数据训练 (Real + Synthetic)
    X_train_enc = X_train.copy()
    for col, le in encoders.items():
        X_train_enc[col] = le.transform(X_train_enc[col].astype(str).fillna('Missing'))
        
    X_final = pd.concat([X_train_enc, X_syn], axis=0)
    y_final = np.concatenate([le_y.transform(y_train.values.ravel()), y_syn_enc])
    
    print(f"   混合训练集大小: {X_final.shape[0]} 条")
    
    # 4. 训练 XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_final, y_final)
    
    # 5. 预测
    X_test_encoded = X_test.copy()
    for col, le in encoders.items():
        X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str).fillna('Missing'))
        
    y_pred = model.predict(X_test_encoded)
    
    f1 = f1_score(y_test_enc, y_pred, pos_label=minority_class)
    return f1

# ===========================================
# 🤖 模型 1: CTGAN
# ===========================================
print("\n" + "="*60)
print("🤖 [3/5] 开始训练 CTGAN (这可能需要几分钟)...")
print("="*60)
start_time = time.time()

ctgan = CTGANSynthesizer(metadata, epochs=EPOCHS, verbose=True)
ctgan.fit(train_data)

print(f"\n   ✅ CTGAN 训练完成！耗时: {time.time() - start_time:.1f} 秒")
print(f"   正在生成 {SAMPLES_TO_GENERATE} 条数据...")

syn_ctgan = ctgan.sample(num_rows=SAMPLES_TO_GENERATE)
syn_ctgan.to_csv(f"{SAVE_DIR}/Travel_CTGAN_samples.csv", index=False)
print(f"   ✅ 数据已保存到: {SAVE_DIR}/Travel_CTGAN_samples.csv")

f1_ctgan = evaluate_synthetic_data("CTGAN", syn_ctgan)
print(f"\n🏆 CTGAN (Real+Syn) F1 Score: {f1_ctgan:.4f}")

# ===========================================
# 🤖 模型 2: TVAE
# ===========================================
print("\n" + "="*60)
print("🤖 [4/5] 开始训练 TVAE (通常比 CTGAN 快)...")
print("="*60)
start_time = time.time()

tvae = TVAESynthesizer(metadata, epochs=EPOCHS, verbose=True)
tvae.fit(train_data)

print(f"\n   ✅ TVAE 训练完成！耗时: {time.time() - start_time:.1f} 秒")
print(f"   正在生成 {SAMPLES_TO_GENERATE} 条数据...")

syn_tvae = tvae.sample(num_rows=SAMPLES_TO_GENERATE)
syn_tvae.to_csv(f"{SAVE_DIR}/Travel_TVAE_samples.csv", index=False)
print(f"   ✅ 数据已保存到: {SAVE_DIR}/Travel_TVAE_samples.csv")

f1_tvae = evaluate_synthetic_data("TVAE", syn_tvae)
print(f"\n🏆 TVAE (Real+Syn) F1 Score: {f1_tvae:.4f}")

# ===========================================
# 📊 最终总结
# ===========================================
print("\n" + "="*60)
print("✅ [5/5] 所有深度学习基线运行完毕！")
print("="*60)
print(f"\n📊 最终结果对比:")
print(f"   CTGAN F1 Score: {f1_ctgan:.4f}")
print(f"   TVAE  F1 Score: {f1_tvae:.4f}")
print(f"\n📁 生成的文件:")
print(f"   - {SAVE_DIR}/Travel_CTGAN_samples.csv")
print(f"   - {SAVE_DIR}/Travel_TVAE_samples.csv")
print("\n🎉 完成！可以使用这些合成数据进行下游任务评估。")
print("="*60)

