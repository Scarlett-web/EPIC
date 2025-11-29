"""
HELOC 数据集完整深度学习实验
包括：
1. 标准生成（保持原始不平衡分布）- CTGAN + TVAE
2. 条件采样（只生成少数类）- CTGAN + TVAE
3. 拒绝采样（只生成少数类）- CTGAN + TVAE
一共生成 6 个数据集
"""
import sys
import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

# 配置输出缓冲
sys.stdout.reconfigure(line_buffering=True)

# 数据集配置
DATASET_NAME = 'HELOC'
TARGET_COLUMN = 'RiskPerformance'
MINORITY_CLASS = 'RIS_1'  # 少数类
MAJORITY_CLASS = 'RIS_0'  # 多数类
N_SAMPLES_STANDARD = 1000  # 标准生成的样本数

print("=" * 80, flush=True)
print("🚀 HELOC 数据集完整深度学习实验", flush=True)
print("=" * 80, flush=True)

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n📂 加载数据...", flush=True)
X_train = pd.read_csv('../data/realdata/HELOC/X_train.csv')
y_train = pd.read_csv('../data/realdata/HELOC/y_train.csv')
X_test = pd.read_csv('../data/realdata/HELOC/X_test.csv')
y_test = pd.read_csv('../data/realdata/HELOC/y_test.csv')

# 删除 index 列（如果存在）
if 'index' in X_train.columns:
    X_train = X_train.drop('index', axis=1)
if 'index' in X_test.columns:
    X_test = X_test.drop('index', axis=1)
if 'index' in y_train.columns:
    y_train = y_train.drop('index', axis=1)
if 'index' in y_test.columns:
    y_test = y_test.drop('index', axis=1)

# 合并 X 和 y
train_data = pd.concat([X_train, y_train], axis=1)

print(f"✅ 训练集: {train_data.shape}", flush=True)
print(f"✅ 测试集: {X_test.shape}", flush=True)

# 类别分布
class_counts = y_train[TARGET_COLUMN].value_counts()
print(f"\n📊 原始类别分布:", flush=True)
for cls, count in class_counts.items():
    print(f"   {cls}: {count} ({count/len(y_train)*100:.2f}%)", flush=True)

minority_count = class_counts[MINORITY_CLASS]
majority_count = class_counts[MAJORITY_CLASS]
samples_needed = int(majority_count - minority_count)

print(f"\n🎯 需要生成 {samples_needed} 条少数类样本来平衡数据集", flush=True)

# ============================================================================
# 2. 准备元数据
# ============================================================================
print("\n🔧 准备元数据...", flush=True)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)
print("✅ 元数据准备完成", flush=True)

# ============================================================================
# 3. 评估函数
# ============================================================================
def evaluate_model(name, X_train_eval, y_train_eval, X_test_eval, y_test_eval):
    """评估模型性能"""
    print(f"\n{'='*60}", flush=True)
    print(f"📊 评估: {name}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # 编码分类特征
    label_encoders = {}
    X_train_encoded = X_train_eval.copy()
    X_test_encoded = X_test_eval.copy()
    
    for col in X_train_encoded.columns:
        if X_train_encoded[col].dtype == 'object':
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
            X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
            label_encoders[col] = le
    
    # 编码目标变量
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train_eval[TARGET_COLUMN])
    y_test_encoded = le_target.transform(y_test_eval[TARGET_COLUMN])
    
    # 训练 XGBoost
    start_time = time.time()
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train_encoded, y_train_encoded)
    train_time = (time.time() - start_time) / 60
    
    # 预测
    y_pred = model.predict(X_test_encoded)
    
    # 计算指标
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')
    ba = balanced_accuracy_score(y_test_encoded, y_pred)
    
    print(f"✅ F1 Score: {f1:.4f}", flush=True)
    print(f"✅ Balanced Accuracy: {ba:.4f}", flush=True)
    print(f"⏱️  训练时间: {train_time:.2f} 分钟", flush=True)
    
    print(f"\n分类报告:", flush=True)
    print(classification_report(y_test_encoded, y_pred, 
                                target_names=le_target.classes_), flush=True)
    
    return f1, ba, train_time

# ============================================================================
# 4. 实验 1: CTGAN 标准生成（保持不平衡）
# ============================================================================
print("\n" + "="*80, flush=True)
print("🔵 实验 1/6: CTGAN 标准生成（保持原始不平衡分布）", flush=True)
print("="*80, flush=True)

print("\n🏋️  训练 CTGAN...", flush=True)
ctgan = CTGANSynthesizer(metadata, epochs=300, verbose=True)
ctgan.fit(train_data)
print("✅ CTGAN 训练完成", flush=True)

print(f"\n🎲 生成 {N_SAMPLES_STANDARD} 条样本...", flush=True)
syn_ctgan_standard = ctgan.sample(num_rows=N_SAMPLES_STANDARD)
print(f"✅ 生成完成: {syn_ctgan_standard.shape}", flush=True)
print(f"   类别分布: {syn_ctgan_standard[TARGET_COLUMN].value_counts().to_dict()}", flush=True)

# 保存
output_path = f'../data/syndata/HELOC_CTGAN_standard.csv'
syn_ctgan_standard.to_csv(output_path, index=False)
print(f"💾 已保存: {output_path}", flush=True)

# 评估
X_syn = syn_ctgan_standard.drop(TARGET_COLUMN, axis=1)
y_syn = syn_ctgan_standard[[TARGET_COLUMN]]
X_combined = pd.concat([X_train, X_syn], axis=0)
y_combined = pd.concat([y_train, y_syn], axis=0)
f1_ctgan_std, ba_ctgan_std, time_ctgan_std = evaluate_model(
    "CTGAN 标准生成", X_combined, y_combined, X_test, y_test
)

# ============================================================================
# 5. 实验 2: TVAE 标准生成（保持不平衡）
# ============================================================================
print("\n" + "="*80, flush=True)
print("🟢 实验 2/6: TVAE 标准生成（保持原始不平衡分布）", flush=True)
print("="*80, flush=True)

print("\n🏋️  训练 TVAE...", flush=True)
tvae = TVAESynthesizer(metadata, epochs=300, verbose=True)
tvae.fit(train_data)
print("✅ TVAE 训练完成", flush=True)

print(f"\n🎲 生成 {N_SAMPLES_STANDARD} 条样本...", flush=True)
syn_tvae_standard = tvae.sample(num_rows=N_SAMPLES_STANDARD)
print(f"✅ 生成完成: {syn_tvae_standard.shape}", flush=True)
print(f"   类别分布: {syn_tvae_standard[TARGET_COLUMN].value_counts().to_dict()}", flush=True)

# 保存
output_path = f'../data/syndata/HELOC_TVAE_standard.csv'
syn_tvae_standard.to_csv(output_path, index=False)
print(f"💾 已保存: {output_path}", flush=True)

# 评估
X_syn = syn_tvae_standard.drop(TARGET_COLUMN, axis=1)
y_syn = syn_tvae_standard[[TARGET_COLUMN]]
X_combined = pd.concat([X_train, X_syn], axis=0)
y_combined = pd.concat([y_train, y_syn], axis=0)
f1_tvae_std, ba_tvae_std, time_tvae_std = evaluate_model(
    "TVAE 标准生成", X_combined, y_combined, X_test, y_test
)

# ============================================================================
# 6. 实验 3: CTGAN 条件采样（只生成少数类）
# ============================================================================
print("\n" + "="*80, flush=True)
print("🟣 实验 3/6: CTGAN 条件采样（只生成少数类）", flush=True)
print("="*80, flush=True)

print(f"\n🎯 使用条件采样生成 {samples_needed} 条 {MINORITY_CLASS} 样本...", flush=True)
condition = Condition(num_rows=samples_needed, column_values={TARGET_COLUMN: MINORITY_CLASS})
syn_ctgan_conditional = ctgan.sample_from_conditions(conditions=[condition])
print(f"✅ 生成完成: {syn_ctgan_conditional.shape}", flush=True)
print(f"   类别分布: {syn_ctgan_conditional[TARGET_COLUMN].value_counts().to_dict()}", flush=True)

# 保存
output_path = f'../data/syndata/HELOC_CTGAN_minority_only.csv'
syn_ctgan_conditional.to_csv(output_path, index=False)
print(f"💾 已保存: {output_path}", flush=True)

# 评估
X_syn = syn_ctgan_conditional.drop(TARGET_COLUMN, axis=1)
y_syn = syn_ctgan_conditional[[TARGET_COLUMN]]
X_balanced = pd.concat([X_train, X_syn], axis=0)
y_balanced = pd.concat([y_train, y_syn], axis=0)

print(f"\n📊 平衡后的类别分布:", flush=True)
for cls, count in y_balanced[TARGET_COLUMN].value_counts().items():
    print(f"   {cls}: {count} ({count/len(y_balanced)*100:.2f}%)", flush=True)

f1_ctgan_cond, ba_ctgan_cond, time_ctgan_cond = evaluate_model(
    "CTGAN 条件采样", X_balanced, y_balanced, X_test, y_test
)

# ============================================================================
# 7. 实验 4: TVAE 条件采样（只生成少数类）
# ============================================================================
print("\n" + "="*80, flush=True)
print("🟡 实验 4/6: TVAE 条件采样（只生成少数类）", flush=True)
print("="*80, flush=True)

print(f"\n🎯 使用条件采样生成 {samples_needed} 条 {MINORITY_CLASS} 样本...", flush=True)
condition = Condition(num_rows=samples_needed, column_values={TARGET_COLUMN: MINORITY_CLASS})
syn_tvae_conditional = tvae.sample_from_conditions(conditions=[condition])
print(f"✅ 生成完成: {syn_tvae_conditional.shape}", flush=True)
print(f"   类别分布: {syn_tvae_conditional[TARGET_COLUMN].value_counts().to_dict()}", flush=True)

# 保存
output_path = f'../data/syndata/HELOC_TVAE_minority_only.csv'
syn_tvae_conditional.to_csv(output_path, index=False)
print(f"💾 已保存: {output_path}", flush=True)

# 评估
X_syn = syn_tvae_conditional.drop(TARGET_COLUMN, axis=1)
y_syn = syn_tvae_conditional[[TARGET_COLUMN]]
X_balanced = pd.concat([X_train, X_syn], axis=0)
y_balanced = pd.concat([y_train, y_syn], axis=0)

print(f"\n📊 平衡后的类别分布:", flush=True)
for cls, count in y_balanced[TARGET_COLUMN].value_counts().items():
    print(f"   {cls}: {count} ({count/len(y_balanced)*100:.2f}%)", flush=True)

f1_tvae_cond, ba_tvae_cond, time_tvae_cond = evaluate_model(
    "TVAE 条件采样", X_balanced, y_balanced, X_test, y_test
)

# ============================================================================
# 8. 实验 5: CTGAN 拒绝采样（只生成少数类）
# ============================================================================
print("\n" + "="*80, flush=True)
print("🔴 实验 5/6: CTGAN 拒绝采样（只生成少数类）", flush=True)
print("="*80, flush=True)

oversample_factor = 5
total_samples = samples_needed * oversample_factor
print(f"\n🎲 生成 {total_samples} 条样本（{oversample_factor}x 过采样）...", flush=True)
syn_all = ctgan.sample(num_rows=total_samples)
print(f"✅ 生成完成", flush=True)

print(f"\n🔍 筛选少数类样本...", flush=True)
syn_minority = syn_all[syn_all[TARGET_COLUMN] == MINORITY_CLASS]
print(f"   筛选出 {len(syn_minority)} 条少数类样本（效率: {len(syn_minority)/total_samples*100:.1f}%）", flush=True)

if len(syn_minority) >= samples_needed:
    syn_ctgan_rejection = syn_minority.head(samples_needed)
    print(f"✅ 保留 {samples_needed} 条样本", flush=True)
else:
    print(f"⚠️  样本不足，只有 {len(syn_minority)} 条（需要 {samples_needed} 条）", flush=True)
    syn_ctgan_rejection = syn_minority

print(f"   最终样本数: {syn_ctgan_rejection.shape}", flush=True)
print(f"   类别分布: {syn_ctgan_rejection[TARGET_COLUMN].value_counts().to_dict()}", flush=True)

# 保存
output_path = f'../data/syndata/HELOC_CTGAN_rejection_sampling.csv'
syn_ctgan_rejection.to_csv(output_path, index=False)
print(f"💾 已保存: {output_path}", flush=True)

# 评估
X_syn = syn_ctgan_rejection.drop(TARGET_COLUMN, axis=1)
y_syn = syn_ctgan_rejection[[TARGET_COLUMN]]
X_balanced = pd.concat([X_train, X_syn], axis=0)
y_balanced = pd.concat([y_train, y_syn], axis=0)

print(f"\n📊 平衡后的类别分布:", flush=True)
for cls, count in y_balanced[TARGET_COLUMN].value_counts().items():
    print(f"   {cls}: {count} ({count/len(y_balanced)*100:.2f}%)", flush=True)

f1_ctgan_rej, ba_ctgan_rej, time_ctgan_rej = evaluate_model(
    "CTGAN 拒绝采样", X_balanced, y_balanced, X_test, y_test
)

# ============================================================================
# 9. 实验 6: TVAE 拒绝采样（只生成少数类）
# ============================================================================
print("\n" + "="*80, flush=True)
print("🟠 实验 6/6: TVAE 拒绝采样（只生成少数类）", flush=True)
print("="*80, flush=True)

print(f"\n🎲 生成 {total_samples} 条样本（{oversample_factor}x 过采样）...", flush=True)
syn_all = tvae.sample(num_rows=total_samples)
print(f"✅ 生成完成", flush=True)

print(f"\n🔍 筛选少数类样本...", flush=True)
syn_minority = syn_all[syn_all[TARGET_COLUMN] == MINORITY_CLASS]
print(f"   筛选出 {len(syn_minority)} 条少数类样本（效率: {len(syn_minority)/total_samples*100:.1f}%）", flush=True)

if len(syn_minority) >= samples_needed:
    syn_tvae_rejection = syn_minority.head(samples_needed)
    print(f"✅ 保留 {samples_needed} 条样本", flush=True)
else:
    print(f"⚠️  样本不足，只有 {len(syn_minority)} 条（需要 {samples_needed} 条）", flush=True)
    syn_tvae_rejection = syn_minority

print(f"   最终样本数: {syn_tvae_rejection.shape}", flush=True)
print(f"   类别分布: {syn_tvae_rejection[TARGET_COLUMN].value_counts().to_dict()}", flush=True)

# 保存
output_path = f'../data/syndata/HELOC_TVAE_rejection_sampling.csv'
syn_tvae_rejection.to_csv(output_path, index=False)
print(f"💾 已保存: {output_path}", flush=True)

# 评估
X_syn = syn_tvae_rejection.drop(TARGET_COLUMN, axis=1)
y_syn = syn_tvae_rejection[[TARGET_COLUMN]]
X_balanced = pd.concat([X_train, X_syn], axis=0)
y_balanced = pd.concat([y_train, y_syn], axis=0)

print(f"\n📊 平衡后的类别分布:", flush=True)
for cls, count in y_balanced[TARGET_COLUMN].value_counts().items():
    print(f"   {cls}: {count} ({count/len(y_balanced)*100:.2f}%)", flush=True)

f1_tvae_rej, ba_tvae_rej, time_tvae_rej = evaluate_model(
    "TVAE 拒绝采样", X_balanced, y_balanced, X_test, y_test
)

# ============================================================================
# 10. 最终总结
# ============================================================================
print("\n" + "="*80, flush=True)
print("🎉 实验完成！最终结果汇总", flush=True)
print("="*80, flush=True)

results = {
    'CTGAN 标准生成': {'f1': f1_ctgan_std, 'ba': ba_ctgan_std, 'time': time_ctgan_std},
    'TVAE 标准生成': {'f1': f1_tvae_std, 'ba': ba_tvae_std, 'time': time_tvae_std},
    'CTGAN 条件采样': {'f1': f1_ctgan_cond, 'ba': ba_ctgan_cond, 'time': time_ctgan_cond},
    'TVAE 条件采样': {'f1': f1_tvae_cond, 'ba': ba_tvae_cond, 'time': time_tvae_cond},
    'CTGAN 拒绝采样': {'f1': f1_ctgan_rej, 'ba': ba_ctgan_rej, 'time': time_ctgan_rej},
    'TVAE 拒绝采样': {'f1': f1_tvae_rej, 'ba': ba_tvae_rej, 'time': time_tvae_rej},
}

print("\n📊 性能对比表格:")
print(f"{'方法':<20} {'F1 Score':<12} {'Balanced Acc':<15} {'训练时间(分钟)':<15}")
print("-" * 65)
for method, metrics in results.items():
    print(f"{method:<20} {metrics['f1']:<12.4f} {metrics['ba']:<15.4f} {metrics['time']:<15.2f}")

# 找出最佳方法
best_f1_method = max(results.items(), key=lambda x: x[1]['f1'])
best_ba_method = max(results.items(), key=lambda x: x[1]['ba'])

print(f"\n🏆 最佳 F1 Score: {best_f1_method[0]} ({best_f1_method[1]['f1']:.4f})")
print(f"🏆 最佳 Balanced Accuracy: {best_ba_method[0]} ({best_ba_method[1]['ba']:.4f})")

print("\n✅ 所有 6 个数据集已生成并保存到 ../data/syndata/")
print("="*80, flush=True)

