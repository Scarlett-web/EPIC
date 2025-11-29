import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
DATA_DIR = '../data/realdata/Sick'
# ===========================================

print("🚀 [1/5] 正在读取原始数据...")
try:
    X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv', index_col=0)
    y_train = pd.read_csv(f'{DATA_DIR}/y_train.csv', index_col=0).values.ravel()
    X_test = pd.read_csv(f'{DATA_DIR}/X_test.csv', index_col=0)
    y_test = pd.read_csv(f'{DATA_DIR}/y_test.csv', index_col=0).values.ravel()
except FileNotFoundError:
    print("❌ 错误：找不到数据文件！请检查 DATA_DIR 路径。")
    exit()

# 🔥【核心修复】自动把 'sick'/'negative' 变成数字 1/0
print(f"    原始标签示例: {y_train[:5]}")
le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train) # 自动转换
y_test = le_y.transform(y_test)       # 保持一致

# 自动找哪个是少数类
counts = np.bincount(y_train)
minority_class = np.argmin(counts) # 数量最少的那个就是少数类
minority_label_name = le_y.inverse_transform([minority_class])[0]

print(f"    标签已自动转换。少数类是: '{minority_label_name}' (编码为 {minority_class})")
print(f"    少数类样本数: {counts[minority_class]}")

print("\n🧹 [2/5] 特征预处理 (X)...")
categorical_cols_indices = []
full_X = pd.concat([X_train, X_test], axis=0)

for i, col in enumerate(X_train.columns):
    # 如果是字符串，就标记为分类变量
    if X_train[col].dtype == 'object' or X_train[col].nunique() < 10:
        categorical_cols_indices.append(i)
        le_x = LabelEncoder()
        full_X[col] = full_X[col].astype(str).fillna('Missing')
        le_x.fit(full_X[col])
        X_train[col] = le_x.transform(X_train[col].astype(str).fillna('Missing'))
        X_test[col] = le_x.transform(X_test[col].astype(str).fillna('Missing'))

print(f"    检测到 {len(categorical_cols_indices)} 个分类特征，已完成编码。")

print("\n🤖 [3/5] 正在运行 SMOTENC (生成合成数据)...")
# 动态计算 k，确保不报错
k = min(5, counts[minority_class] - 1)
smote_nc = SMOTENC(categorical_features=categorical_cols_indices, k_neighbors=k, random_state=42)

try:
    X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)
    print(f"    SMOTE 完成！数据量从 {len(X_train)} 增加到 {len(X_resampled)}")
except Exception as e:
    print(f"❌ SMOTE 运行失败: {e}")
    exit()

print("\n⚔️ [4/5] 训练分类器 (XGBoost)...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_resampled, y_resampled)

print("\n📊 [5/5] 最终评估...")
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, pos_label=minority_class) 

print("=" * 40)
print(f"🏆 SMOTE 基线 F1 Score: {f1:.4f}")
print("=" * 40)