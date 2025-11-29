"""
Travel 数据集评估脚本
作用：评估 EPIC 生成的合成数据在下游分类任务上的性能
"""
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

# 导入原有的工具函数
import sys
sys.path.append('..')
from Classification import compute_metric, categorical_variable_encode, get_data, init_models

# ==========================================
# 📊 配置参数
# ==========================================
DATA_NAME = 'travel'  # 改为小写，匹配实际文件夹和文件名
synSamplingIndex = 0  # 如果生成了多批数据，可以选择第几批
n = 1000  # 使用多少条合成数据

# 要测试的合成方法列表
SYNTHETIC_METHODS = [
    'None',  # 只用真实数据（基线）
    'Travel_DeepSeek_EPIC',  # EPIC 方法
    # 'Travel_DeepSeek_EPICNorg',  # EPIC + 真实数据混合（暂时注释，因为还没生成混合数据）
    # 'SMOTENC',  # 传统过采样方法
    # 'SMOTENCNorg',  # SMOTENC + 真实数据混合
]

# ==========================================
# 🎯 数据集配置
# ==========================================
DATA2TARGET = {
    'income':'income',
    'Diabetes':'readmitted',
    'HELOC':'RiskPerformance',
    'Sick':'Class',
    'Travel':'Target',
    'travel':'Target'  # 添加小写版本
}

DATA2NCLASS = {
    'income':2,
    'Diabetes':3,
    'HELOC':2,
    'Sick':2,
    'Travel':2,
    'travel':2  # 添加小写版本
}

ML_PARAMS = {
    'Travel':{
        'lr_max_iter':100,
        'dt_max_depth':6,
        'rf_max_depth':12,
        'rf_n_estimators':75,
    },
    'travel':{  # 添加小写版本
        'lr_max_iter':100,
        'dt_max_depth':6,
        'rf_max_depth':12,
        'rf_n_estimators':75,
    },
}

# ==========================================
# 🚀 开始评估
# ==========================================
df_all_result = pd.DataFrame()

print("="*60)
print(f"🎯 开始评估 {DATA_NAME} 数据集")
print("="*60)

for sM in SYNTHETIC_METHODS:
    print(f"\n📊 正在测试方法: {sM}")
    print("-"*60)
    
    configs={
        'data': DATA_NAME,
        'target': DATA2TARGET[DATA_NAME],
        'n_class': DATA2NCLASS[DATA_NAME],
        'is_regression': False,

        # hyperparams
        'lr_max_iter': ML_PARAMS[DATA_NAME]['lr_max_iter'],
        'dt_max_depth': ML_PARAMS[DATA_NAME]['dt_max_depth'],
        'rf_max_depth': ML_PARAMS[DATA_NAME]['rf_max_depth'],
        'rf_n_estimators': ML_PARAMS[DATA_NAME]['rf_n_estimators'],

        # xgboost
        'xg_max_depth': 4,
        'xg_lr': 0.03,

        # catboost
        'cat_max_depth': 6,
        'cat_lr': 0.003,

        # lightGBM
        'lgbm_max_depth': 3,
        'lgbm_lr': 0.1,

        # synthetic data
        'synModel': sM,
        'synSamples': n,
        'synSamplingIndex': synSamplingIndex,
        'cat_idx': [1, 2, 4, 5],  # Travel 的分类特征索引
    }
    
    models = init_models(configs, 42)
    syn_data_save_dir = f"../../data/syndata/{configs['synModel']}"
    real_data_save_dir = f"../../data/realdata/{configs['data']}"

    for k in tqdm(models.keys(), desc=f"Testing {sM}"):
        configs['model'] = k    
        for random_state in range(5):  # 5 次随机种子
            configs['random_state'] = random_state
            
            try:
                X_train, y_train, X_test, y_test, n_syn, n_org_train, n_org_test = get_data(
                    configs, syn_data_save_dir, real_data_save_dir
                )
            except Exception as e:
                print(f"⚠️ 数据加载失败: {e}")
                continue
                
            df_save = pd.DataFrame([configs])
            df_save['n_syn'] = n_syn
            df_save['n_org_train'] = n_org_train
            df_save['n_org_test'] = n_org_test                

            model = init_models(configs, random_state)[k]

            scaler = StandardScaler()
            scaler.fit(X_train)

            X_train_np = scaler.transform(X_train)
            X_test_np = scaler.transform(X_test)

            model.fit(X_train_np, y_train)
            pred_test = model.predict(X_test_np)
            pred_test_proba = model.predict_proba(X_test_np)

            df_metric = compute_metric(y_test, pred_test, pred_test_proba, configs['n_class'], regression=configs['is_regression'])

            df_save = pd.concat([df_save, df_metric], axis=1)
            df_all_result = pd.concat([df_all_result, df_save.copy()])

# ==========================================
# 📊 结果汇总
# ==========================================
print("\n" + "="*60)
print("📊 最终结果汇总")
print("="*60)

df = df_all_result[['data','model','synModel','F1','BalancedACC','AUC']]
df = (df.groupby(['data','model','synModel']).mean()*100).reset_index()

print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".2f"))

# 保存结果
output_file = f"../../results/Travel_EPIC_results.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_all_result.to_csv(output_file, index=False)
print(f"\n✅ 详细结果已保存到: {output_file}")

