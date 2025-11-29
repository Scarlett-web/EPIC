# 📊 平衡数据集 - 完整说明

## 🎯 数据集概述

本文件夹包含通过**原始训练数据 + 深度学习生成的少数类样本**合并而成的平衡数据集。

共包含 **3个数据集** × **4种方法** = **12个平衡数据集**

---

## 📁 文件结构

```
balanced_datasets/
├── Sick/
│   ├── Sick_balanced_CTGAN_conditional.csv      (5566条, 1.00:1) ✅ 完美平衡
│   ├── Sick_balanced_TVAE_conditional.csv       (5566条, 1.00:1) ✅ 完美平衡
│   ├── Sick_balanced_CTGAN_rejection.csv        (3862条, 2.58:1) ⚠️ 中度不平衡
│   └── Sick_balanced_TVAE_rejection.csv         (3328条, 5.11:1) ❌ 严重不平衡
│
├── Travel/
│   ├── Travel_balanced_CTGAN_conditional.csv    (2042条, 1.00:1) ✅ 完美平衡
│   ├── Travel_balanced_TVAE_conditional.csv     (2042条, 1.00:1) ✅ 完美平衡
│   ├── Travel_balanced_CTGAN_rejection.csv      (2042条, 1.00:1) ✅ 完美平衡
│   └── Travel_balanced_TVAE_rejection.csv       (2042条, 1.00:1) ✅ 完美平衡
│
└── HELOC/
    ├── HELOC_balanced_CTGAN_conditional.csv     (8216条, 1.00:1) ✅ 完美平衡
    ├── HELOC_balanced_TVAE_conditional.csv      (8216条, 1.00:1) ✅ 完美平衡
    ├── HELOC_balanced_CTGAN_rejection.csv       (8216条, 1.00:1) ✅ 完美平衡
    └── HELOC_balanced_TVAE_rejection.csv        (8216条, 1.00:1) ✅ 完美平衡
```

---

## 📊 数据集详细信息

### 1️⃣ Sick 数据集（疾病诊断）

**原始数据特征**:
- 训练样本: 2,968 条
- 原始分布: negative (93.8%) | sick (6.2%)
- 不平衡比例: 15:1（严重不平衡）
- 需要生成: 2,598 条少数类样本

**平衡数据集**:

| 文件名 | 样本数 | 类别分布 | 不平衡比例 | 状态 |
|--------|--------|---------|-----------|------|
| `Sick_balanced_CTGAN_conditional.csv` | 5,566 | 50.0% / 50.0% | 1.00:1 | ✅ 完美 |
| `Sick_balanced_TVAE_conditional.csv` | 5,566 | 50.0% / 50.0% | 1.00:1 | ✅ 完美 |
| `Sick_balanced_CTGAN_rejection.csv` | 3,862 | 72.1% / 27.9% | 2.58:1 | ⚠️ 中度 |
| `Sick_balanced_TVAE_rejection.csv` | 3,328 | 83.6% / 16.4% | 5.11:1 | ❌ 严重 |

**推荐使用**: ✅ **条件采样方法**（CTGAN或TVAE）

---

### 2️⃣ Travel 数据集（旅游保险购买预测）

**原始数据特征**:
- 训练样本: 1,589 条
- 原始分布: Target=0 (64.2%) | Target=1 (35.8%)
- 不平衡比例: 1.8:1（中等不平衡）
- 需要生成: 453 条少数类样本

**平衡数据集**:

| 文件名 | 样本数 | 类别分布 | 不平衡比例 | 状态 |
|--------|--------|---------|-----------|------|
| `Travel_balanced_CTGAN_conditional.csv` | 2,042 | 50.0% / 50.0% | 1.00:1 | ✅ 完美 |
| `Travel_balanced_TVAE_conditional.csv` | 2,042 | 50.0% / 50.0% | 1.00:1 | ✅ 完美 |
| `Travel_balanced_CTGAN_rejection.csv` | 2,042 | 50.0% / 50.0% | 1.00:1 | ✅ 完美 |
| `Travel_balanced_TVAE_rejection.csv` | 2,042 | 50.0% / 50.0% | 1.00:1 | ✅ 完美 |

**推荐使用**: ✅ **所有方法都表现优异**

---

### 3️⃣ HELOC 数据集（房屋净值信贷风险）

**原始数据特征**:
- 训练样本: 7,897 条
- 原始分布: RIS_0 (52.0%) | RIS_1 (48.0%)
- 不平衡比例: 1.08:1（轻度不平衡）
- 需要生成: 319 条少数类样本

**平衡数据集**:

| 文件名 | 样本数 | 类别分布 | 不平衡比例 | 状态 |
|--------|--------|---------|-----------|------|
| `HELOC_balanced_CTGAN_conditional.csv` | 8,216 | 50.0% / 50.0% | 1.00:1 | ✅ 完美 |
| `HELOC_balanced_TVAE_conditional.csv` | 8,216 | 50.0% / 50.0% | 1.00:1 | ✅ 完美 |
| `HELOC_balanced_CTGAN_rejection.csv` | 8,216 | 50.0% / 50.0% | 1.00:1 | ✅ 完美 |
| `HELOC_balanced_TVAE_rejection.csv` | 8,216 | 50.0% / 50.0% | 1.00:1 | ✅ 完美 |

**推荐使用**: ✅ **所有方法都表现优异**

---

## 🔬 生成方法说明

### 1. 条件采样（Conditional Sampling）

**原理**: 使用 SDV 的 `Condition` 对象，明确指定生成特定类别的样本

**优点**:
- ✅ 100% 精确生成目标类别
- ✅ 适用于所有不平衡程度
- ✅ 可控性强

**缺点**:
- ⚠️ 可能缺乏多样性

**推荐场景**: 严重不平衡数据集（如 Sick 15:1）

---

### 2. 拒绝采样（Rejection Sampling）

**原理**: 生成大批量样本，筛选出目标类别

**优点**:
- ✅ 样本多样性更好
- ✅ 更接近真实数据分布

**缺点**:
- ❌ 效率取决于不平衡程度
- ❌ 严重不平衡时效率极低

**推荐场景**: 轻度/中度不平衡数据集（如 Travel 1.8:1, HELOC 1.08:1）

---

## 📈 方法对比总结

| 数据集 | 不平衡比例 | 条件采样效果 | 拒绝采样效果 | 推荐方法 |
|--------|-----------|------------|------------|---------|
| **Sick** | 15:1 | ✅ 完美 (1.00:1) | ❌ 失败 (2.58-5.11:1) | 条件采样 |
| **Travel** | 1.8:1 | ✅ 完美 (1.00:1) | ✅ 完美 (1.00:1) | 两种都可 |
| **HELOC** | 1.08:1 | ✅ 完美 (1.00:1) | ✅ 完美 (1.00:1) | 两种都可 |

---

## 💡 使用建议

### 对于机器学习实验:

1. **首选条件采样数据集**:
   - 保证完美平衡
   - 适用于所有场景

2. **拒绝采样数据集**:
   - 仅在轻度/中度不平衡时使用
   - 可能提供更好的泛化能力

3. **对比实验**:
   - 使用多个平衡数据集训练模型
   - 对比不同方法的效果

---

## 🚀 快速开始

### 加载数据示例

```python
import pandas as pd

# 加载 Sick 条件采样平衡数据集
sick_balanced = pd.read_csv('balanced_datasets/Sick/Sick_balanced_CTGAN_conditional.csv')

# 分离特征和标签
X = sick_balanced.drop('Class', axis=1)
y = sick_balanced['Class']

# 训练模型
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
```

---

## 📊 统计总结

- **总数据集数量**: 12 个
- **完美平衡数据集**: 10 个 (83.3%)
- **中度不平衡数据集**: 1 个 (8.3%)
- **严重不平衡数据集**: 1 个 (8.3%)

---

## 🎯 关键发现

1. ✅ **条件采样是最可靠的方法**，在所有数据集上都达到完美平衡
2. ✅ **拒绝采样在轻度不平衡数据上表现优异**（Travel, HELOC）
3. ⚠️ **拒绝采样在严重不平衡数据上效率低下**（Sick）
4. ✅ **CTGAN 和 TVAE 在条件采样模式下表现相当**

---

## 📝 生成脚本

- **合并脚本**: `codes/merge_balanced_datasets.py`
- **验证脚本**: `codes/verify_balanced_datasets.py`

---

**生成时间**: 2025-11-29  
**生成方法**: CTGAN + TVAE (条件采样 + 拒绝采样)  
**数据来源**: 原始训练数据 + 深度学习生成的少数类样本

