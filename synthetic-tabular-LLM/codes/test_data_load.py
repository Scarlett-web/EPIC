"""测试数据加载"""
import pandas as pd
import sys

DATA_DIR = '../data/realdata/travel'

print("=" * 60)
print("测试 Travel 数据集加载")
print("=" * 60)

try:
    print("\n正在读取数据...")
    X_train = pd.read_csv(f'{DATA_DIR}/X_train.csv', index_col=0)
    y_train = pd.read_csv(f'{DATA_DIR}/y_train.csv', index_col=0)
    X_test = pd.read_csv(f'{DATA_DIR}/X_test.csv', index_col=0)
    y_test = pd.read_csv(f'{DATA_DIR}/y_test.csv', index_col=0)
    
    print(f"✅ X_train: {X_train.shape}")
    print(f"✅ y_train: {y_train.shape}")
    print(f"✅ X_test: {X_test.shape}")
    print(f"✅ y_test: {y_test.shape}")
    
    print(f"\n目标变量分布:")
    print(y_train['Target'].value_counts())
    
    print("\n✅ 数据加载成功！")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

