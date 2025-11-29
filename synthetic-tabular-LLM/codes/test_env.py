"""测试环境是否正确"""
import sys
print("Python version:", sys.version)
print("\n测试导入...")

try:
    import pandas as pd
    print("✅ pandas:", pd.__version__)
except Exception as e:
    print("❌ pandas:", e)

try:
    import numpy as np
    print("✅ numpy:", np.__version__)
except Exception as e:
    print("❌ numpy:", e)

try:
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    print("✅ SDV (CTGAN/TVAE) 导入成功")
except Exception as e:
    print("❌ SDV:", e)

try:
    from xgboost import XGBClassifier
    print("✅ XGBoost 导入成功")
except Exception as e:
    print("❌ XGBoost:", e)

print("\n✅ 环境测试完成！")

