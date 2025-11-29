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
DATA_DIR = '../data/realdata/Sick'
SAVE_DIR = '../data/syndata'
SAMPLES_TO_GENERATE = 1000  # 论文要求生成 1K
EPOCHS = 100                # 训练轮数 (电脑慢可以改小，比如 50)
# ===========================================

print("🚀 [1/5] 正在读取原始数据...")
try:
    X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv', index_col=0)
    y_train = pd.read_csv(f'{DATA_DIR}/y_train.csv', index_col=0)
    X_test = pd.read_csv(f'{DATA_DIR}/X_test.csv', index_col=0)
    y_test = pd.read_csv(f'{DATA_DIR}/y_test.csv', index_col=0).values.ravel()
    
    # 合并 X 和 y，因为生成模型需要学习整张表
    train_data = pd.concat([X_train, y_train], axis=1)
    
    # 标签编码 (XGBoost评估用)
    le_y = LabelEncoder()
    y_test_enc = le_y.fit_transform(y_test)
    minority_class = np.argmin(np.bincount(y_test_enc)) # 自动找少数类
    
except FileNotFoundError:
    print("❌ 错误：找不到数据文件！")
    exit()

# 自动检测元数据 (告诉模型哪些列是分类，哪些是数值)
print("    正在自动检测数据结构...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)

# 定义一个通用的评估函数
def evaluate_synthetic_data(name, synthetic_data):
    print(f"\n📊 正在评估 {name} 生成的数据质量...")
    
    # 1. 准备合成数据的 X 和 y
    X_syn = synthetic_data.drop('Class', axis=1)
    y_syn = synthetic_data['Class']
    
    # 2. 数据预处理 (和之前一样，转数字)
    # 合并所有数据以统一编码
    full_X = pd.concat([X_train, X_test, X_syn], axis=0)
    
    # 训练编码器
    encoders = {}
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or X_train[col].nunique() < 10:
            le = LabelEncoder()
            full_X[col] = full_X[col].astype(str).fillna('Missing')
            le.fit(full_X[col])
            
            # 转换合成数据和测试集
            X_syn[col] = le.transform(X_syn[col].astype(str).fillna('Missing'))
            X_test_enc = X_test.copy() # 临时副本
            X_test_enc[col] = le.transform(X_test[col].astype(str).fillna('Missing'))
            encoders[col] = le
    
    # 转换标签
    y_syn_enc = le_y.transform(y_syn)
    
    # 3. 混合数据训练 (Real + Synthetic)
    # 这里为了简化，我们只用合成数据训练，看看它自己能不能打
    # (如果想复现论文的 +Augment，可以把 X_train 和 X_syn 拼起来)
    X_train_enc = X_train.copy()
    for col, le in encoders.items():
        X_train_enc[col] = le.transform(X_train_enc[col].astype(str).fillna('Missing'))
        
    X_final = pd.concat([X_train_enc, X_syn], axis=0)
    y_final = np.concatenate([le_y.transform(y_train.values.ravel()), y_syn_enc])
    
    # 4. 训练 XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_final, y_final)
    
    # 5. 预测
    # 注意：需要重新对 X_test 进行编码匹配
    X_test_encoded = X_test.copy()
    for col, le in encoders.items():
        X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str).fillna('Missing'))
        
    y_pred = model.predict(X_test_encoded)
    
    return f1_score(y_test_enc, y_pred, pos_label=minority_class)

# ===========================================
# 🤖 模型 1: CTGAN
# ===========================================
print("\n" + "="*40)
print("🤖 开始训练 CTGAN (这可能需要几分钟)...")
print("="*40)
ctgan = CTGANSynthesizer(metadata, epochs=EPOCHS, verbose=True)
ctgan.fit(train_data)

print("    正在生成 1000 条数据...")
syn_ctgan = ctgan.sample(num_rows=SAMPLES_TO_GENERATE)
syn_ctgan.to_csv(f"{SAVE_DIR}/Sick_CTGAN_samples.csv", index=False)

f1_ctgan = evaluate_synthetic_data("CTGAN", syn_ctgan)
print(f"🏆 CTGAN (Real+Syn) F1 Score: {f1_ctgan:.4f}")

# ===========================================
# 🤖 模型 2: TVAE
# ===========================================
print("\n" + "="*40)
print("🤖 开始训练 TVAE (通常比 CTGAN 快)...")
print("="*40)
tvae = TVAESynthesizer(metadata, epochs=EPOCHS, verbose=True)
tvae.fit(train_data)

print("    正在生成 1000 条数据...")
syn_tvae = tvae.sample(num_rows=SAMPLES_TO_GENERATE)
syn_tvae.to_csv(f"{SAVE_DIR}/Sick_TVAE_samples.csv", index=False)

f1_tvae = evaluate_synthetic_data("TVAE", syn_tvae)
print(f"🏆 TVAE (Real+Syn) F1 Score: {f1_tvae:.4f}")

print("\n" + "="*40)
print("✅ 所有深度学习基线运行完毕！")