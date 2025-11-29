"""
🎯 Travel 数据集深度学习基线 - 只生成少数类样本（条件采样）
使用 CTGAN 和 TVAE 的条件采样功能，只生成 'Target=1' 类别的样本
"""
import pandas as pd
import numpy as np
import time
import warnings
import sys

# 强制刷新输出
sys.stdout.reconfigure(line_buffering=True)

from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
DATA_DIR = '../data/realdata/travel'
SAVE_DIR = '../data/syndata'
TARGET_COLUMN = 'Target'
MINORITY_CLASS = 1  # 少数类标签（购买保险）
# ===========================================

print("=" * 70, flush=True)
print("🎯 Travel 数据集深度学习基线 - 只生成少数类样本（条件采样）", flush=True)
print("=" * 70, flush=True)

# 1. 读取数据
print("\n[1/7] 正在读取数据...", flush=True)
X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv', index_col=0)
y_train = pd.read_csv(f'{DATA_DIR}/y_train.csv', index_col=0)
X_test = pd.read_csv(f'{DATA_DIR}/X_test.csv', index_col=0)
y_test = pd.read_csv(f'{DATA_DIR}/y_test.csv', index_col=0).values.ravel()

print(f"   训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条", flush=True)

# 合并数据
train_data = pd.concat([X_train, y_train], axis=1)

# 查看原始类别分布
print(f"\n   📊 原始训练集类别分布:", flush=True)
class_counts = y_train[TARGET_COLUMN].value_counts()
for cls, count in class_counts.items():
    print(f"      Target={cls}: {count} 条 ({count/len(y_train)*100:.2f}%)", flush=True)

# 计算需要生成的少数类样本数
majority_count = class_counts.max()
minority_count = class_counts.min()
samples_needed = int(majority_count - minority_count)  # 转换为 Python int

print(f"\n   🎯 需要生成 {samples_needed} 条 'Target={MINORITY_CLASS}' 样本以平衡数据集", flush=True)

# 标签编码
le_y = LabelEncoder()
y_test_enc = le_y.fit_transform(y_test)
minority_class_idx = np.argmin(np.bincount(y_test_enc))

# 2. 检测元数据
print("\n[2/7] 正在检测数据结构...", flush=True)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)
print("   ✅ 元数据检测完成", flush=True)

# 评估函数
def evaluate_balanced_data(name, synthetic_minority_data):
    """评估使用合成少数类样本平衡后的数据"""
    print(f"\n   📊 评估 {name} 生成的平衡数据...", flush=True)
    
    # 1. 合并原始训练数据和合成少数类数据
    X_syn = synthetic_minority_data.drop(TARGET_COLUMN, axis=1)
    y_syn = synthetic_minority_data[TARGET_COLUMN]
    
    print(f"   生成的少数类样本: {len(X_syn)} 条", flush=True)
    print(f"   少数类分布: {dict(y_syn.value_counts())}", flush=True)
    
    # 2. 创建平衡的训练集
    X_balanced = pd.concat([X_train, X_syn], axis=0)
    y_balanced = pd.concat([y_train, y_syn], axis=0)
    
    print(f"\n   平衡后训练集大小: {len(X_balanced)} 条", flush=True)
    print(f"   平衡后类别分布:", flush=True)
    balanced_counts = y_balanced[TARGET_COLUMN].value_counts()
    for cls, count in balanced_counts.items():
        print(f"      Target={cls}: {count} 条 ({count/len(y_balanced)*100:.2f}%)", flush=True)
    
    # 3. 数据编码
    full_X = pd.concat([X_balanced, X_test], axis=0)
    encoders = {}
    
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or X_train[col].nunique() < 10:
            le = LabelEncoder()
            full_X[col] = full_X[col].astype(str).fillna('Missing')
            le.fit(full_X[col])
            encoders[col] = le
    
    # 编码训练数据
    X_balanced_enc = X_balanced.copy()
    for col, le in encoders.items():
        X_balanced_enc[col] = le.transform(X_balanced_enc[col].astype(str).fillna('Missing'))
    
    y_balanced_enc = le_y.transform(y_balanced.values.ravel())
    
    # 4. 训练 XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)
    model.fit(X_balanced_enc, y_balanced_enc)
    
    # 5. 预测
    X_test_encoded = X_test.copy()
    for col, le in encoders.items():
        X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str).fillna('Missing'))
    
    y_pred = model.predict(X_test_encoded)
    
    # 6. 评估指标
    f1 = f1_score(y_test_enc, y_pred, pos_label=minority_class_idx)
    ba = balanced_accuracy_score(y_test_enc, y_pred)
    
    print(f"\n   ✅ F1 Score: {f1:.4f}", flush=True)
    print(f"   ✅ Balanced Accuracy: {ba:.4f}", flush=True)
    
    # 详细分类报告
    print(f"\n   📋 分类报告:", flush=True)
    report = classification_report(y_test_enc, y_pred, target_names=[str(c) for c in le_y.classes_], digits=4)
    print(report, flush=True)
    
    return f1, ba

# ===========================================
# 🤖 方法 1: CTGAN 条件采样
# ===========================================
print("\n" + "=" * 70, flush=True)
print("🤖 [3/7] 开始训练 CTGAN（条件采样模式）", flush=True)
print("=" * 70, flush=True)
print(f"   训练轮数: 100", flush=True)
print(f"   目标: 只生成 {samples_needed} 条 'Target={MINORITY_CLASS}' 样本", flush=True)

start_time = time.time()
ctgan = CTGANSynthesizer(metadata, epochs=100, verbose=True)
print("   开始训练...", flush=True)
ctgan.fit(train_data)
train_time = time.time() - start_time

print(f"\n   ✅ CTGAN 训练完成！耗时: {train_time/60:.2f} 分钟", flush=True)
print(f"   正在使用条件采样生成 'Target={MINORITY_CLASS}' 类样本...", flush=True)

# 🔥 关键：使用条件采样只生成少数类
condition = Condition(num_rows=samples_needed, column_values={TARGET_COLUMN: MINORITY_CLASS})
syn_ctgan_minority = ctgan.sample_from_conditions(conditions=[condition])

print(f"   ✅ 生成完成！", flush=True)
print(f"   验证生成的样本类别:", flush=True)
print(f"   {dict(syn_ctgan_minority[TARGET_COLUMN].value_counts())}", flush=True)

# 保存
syn_ctgan_minority.to_csv(f"{SAVE_DIR}/Travel_CTGAN_minority_only.csv", index=False)
print(f"   ✅ 已保存: {SAVE_DIR}/Travel_CTGAN_minority_only.csv", flush=True)

f1_ctgan, ba_ctgan = evaluate_balanced_data("CTGAN (条件采样)", syn_ctgan_minority)

# ===========================================
# 🤖 方法 2: TVAE 条件采样
# ===========================================
print("\n" + "=" * 70, flush=True)
print("🤖 [4/7] 开始训练 TVAE（条件采样模式）", flush=True)
print("=" * 70, flush=True)

start_time = time.time()
tvae = TVAESynthesizer(metadata, epochs=100, verbose=True)
print("   开始训练...", flush=True)
tvae.fit(train_data)
train_time = time.time() - start_time

print(f"\n   ✅ TVAE 训练完成！耗时: {train_time/60:.2f} 分钟", flush=True)
print(f"   正在使用条件采样生成 'Target={MINORITY_CLASS}' 类样本...", flush=True)

# 🔥 关键：使用条件采样只生成少数类
syn_tvae_minority = tvae.sample_from_conditions(conditions=[condition])

print(f"   ✅ 生成完成！", flush=True)
print(f"   验证生成的样本类别:", flush=True)
print(f"   {dict(syn_tvae_minority[TARGET_COLUMN].value_counts())}", flush=True)

# 保存
syn_tvae_minority.to_csv(f"{SAVE_DIR}/Travel_TVAE_minority_only.csv", index=False)
print(f"   ✅ 已保存: {SAVE_DIR}/Travel_TVAE_minority_only.csv", flush=True)

f1_tvae, ba_tvae = evaluate_balanced_data("TVAE (条件采样)", syn_tvae_minority)

# ===========================================
# 🤖 方法 3: CTGAN 拒绝采样（Rejection Sampling）
# ===========================================
print("\n" + "=" * 70, flush=True)
print("🤖 [5/7] CTGAN 拒绝采样方法（生成后筛选）", flush=True)
print("=" * 70, flush=True)
print(f"   策略: 生成大量样本，然后只保留 'Target={MINORITY_CLASS}' 类", flush=True)

# 生成更多样本以确保有足够的少数类
oversample_factor = 5  # 生成 5 倍的样本
total_samples = samples_needed * oversample_factor

print(f"   生成 {total_samples} 条样本，然后筛选出 {samples_needed} 条 'Target={MINORITY_CLASS}' 样本...", flush=True)

syn_ctgan_all = ctgan.sample(num_rows=total_samples)
syn_ctgan_rejected = syn_ctgan_all[syn_ctgan_all[TARGET_COLUMN] == MINORITY_CLASS].head(samples_needed)

print(f"\n   ✅ 拒绝采样完成！", flush=True)
print(f"   原始生成: {len(syn_ctgan_all)} 条", flush=True)
print(f"   其中 'Target={MINORITY_CLASS}': {(syn_ctgan_all[TARGET_COLUMN] == MINORITY_CLASS).sum()} 条", flush=True)
print(f"   保留: {len(syn_ctgan_rejected)} 条", flush=True)

# 保存
syn_ctgan_rejected.to_csv(f"{SAVE_DIR}/Travel_CTGAN_rejection_sampling.csv", index=False)
print(f"   ✅ 已保存: {SAVE_DIR}/Travel_CTGAN_rejection_sampling.csv", flush=True)

f1_ctgan_rej, ba_ctgan_rej = evaluate_balanced_data("CTGAN (拒绝采样)", syn_ctgan_rejected)

# ===========================================
# 🤖 方法 4: TVAE 拒绝采样（Rejection Sampling）
# ===========================================
print("\n" + "=" * 70, flush=True)
print("🤖 [6/7] TVAE 拒绝采样方法（生成后筛选）", flush=True)
print("=" * 70, flush=True)

print(f"   生成 {total_samples} 条样本，然后筛选出 {samples_needed} 条 'Target={MINORITY_CLASS}' 样本...", flush=True)

syn_tvae_all = tvae.sample(num_rows=total_samples)
syn_tvae_rejected = syn_tvae_all[syn_tvae_all[TARGET_COLUMN] == MINORITY_CLASS].head(samples_needed)

print(f"\n   ✅ 拒绝采样完成！", flush=True)
print(f"   原始生成: {len(syn_tvae_all)} 条", flush=True)
print(f"   其中 'Target={MINORITY_CLASS}': {(syn_tvae_all[TARGET_COLUMN] == MINORITY_CLASS).sum()} 条", flush=True)
print(f"   保留: {len(syn_tvae_rejected)} 条", flush=True)

# 如果生成的少数类样本不够，继续生成
if len(syn_tvae_rejected) < samples_needed:
    print(f"   ⚠️ 警告：生成的 'Target={MINORITY_CLASS}' 样本不足，需要继续生成...", flush=True)
    additional_needed = samples_needed - len(syn_tvae_rejected)
    additional_samples = tvae.sample(num_rows=additional_needed * 20)  # 生成更多
    additional_minority = additional_samples[additional_samples[TARGET_COLUMN] == MINORITY_CLASS].head(additional_needed)
    syn_tvae_rejected = pd.concat([syn_tvae_rejected, additional_minority], axis=0)
    print(f"   ✅ 补充完成，最终保留: {len(syn_tvae_rejected)} 条", flush=True)

# 保存
syn_tvae_rejected.to_csv(f"{SAVE_DIR}/Travel_TVAE_rejection_sampling.csv", index=False)
print(f"   ✅ 已保存: {SAVE_DIR}/Travel_TVAE_rejection_sampling.csv", flush=True)

f1_tvae_rej, ba_tvae_rej = evaluate_balanced_data("TVAE (拒绝采样)", syn_tvae_rejected)

# ===========================================
# 📊 最终总结
# ===========================================
print("\n" + "=" * 70, flush=True)
print("✅ [7/7] 所有方法运行完毕！", flush=True)
print("=" * 70, flush=True)

print(f"\n📊 最终结果对比:", flush=True)
print(f"\n{'方法':<30} {'F1 Score':<12} {'Balanced Acc':<15}", flush=True)
print("-" * 70, flush=True)
print(f"{'CTGAN (条件采样)':<30} {f1_ctgan:<12.4f} {ba_ctgan:<15.4f}", flush=True)
print(f"{'TVAE (条件采样)':<30} {f1_tvae:<12.4f} {ba_tvae:<15.4f}", flush=True)
print(f"{'CTGAN (拒绝采样)':<30} {f1_ctgan_rej:<12.4f} {ba_ctgan_rej:<15.4f}", flush=True)
print(f"{'TVAE (拒绝采样)':<30} {f1_tvae_rej:<12.4f} {ba_tvae_rej:<15.4f}", flush=True)

print(f"\n📁 生成的文件:", flush=True)
print(f"   - {SAVE_DIR}/Travel_CTGAN_minority_only.csv", flush=True)
print(f"   - {SAVE_DIR}/Travel_TVAE_minority_only.csv", flush=True)
print(f"   - {SAVE_DIR}/Travel_CTGAN_rejection_sampling.csv", flush=True)
print(f"   - {SAVE_DIR}/Travel_TVAE_rejection_sampling.csv", flush=True)

print("\n🎉 完成！所有方法都只生成了少数类样本来平衡数据集。", flush=True)
print("=" * 70, flush=True)

