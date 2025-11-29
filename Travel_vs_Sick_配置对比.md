# 📊 Travel vs Sick 数据集配置对比表

## 🎯 核心差异总览

| 配置项 | Sick 数据集 | Travel 数据集 |
|--------|-------------|---------------|
| **数据集名称** | `Sick` | `Travel` |
| **目标变量名** | `Class` | `Target` |
| **目标变量类型** | 字符串 (`negative`, `sick`) | 数值 (`0`, `1`) |
| **特征数量** | 27 个 | 9 个 |
| **分类特征数** | 21 个 | 4 个 |
| **样本数（训练）** | 2969 | 1589 |
| **样本数（测试）** | 740 | 399 |
| **类别不平衡比** | ~15:1 (严重) | ~1.8:1 (中等) |
| **数据文件格式** | 已分离 (X_train.csv, y_train.csv) | 未分离 (travel_train.csv) |

---

## 📝 代码修改对比

### 1️⃣ **数据预处理**

#### Sick（无需预处理）
```python
# 数据已经是标准格式
X_train = pd.read_csv('data/realdata/Sick/X_train.csv')
y_train = pd.read_csv('data/realdata/Sick/y_train.csv')
```

#### Travel（需要预处理）
```python
# 需要先运行 preprocess_travel_data.py
# 将 travel_train.csv 转换为 X_train.csv 和 y_train.csv
```

---

### 2️⃣ **generate_samples_*.py 配置**

| 参数 | Sick | Travel |
|------|------|--------|
| `DATA_NAME` | `"Sick"` | `"Travel"` |
| `TARGET` | `"Class"` | `"Target"` |
| `MODEL_NAME` | `"Sick_DeepSeek_EPIC"` | `"Travel_DeepSeek_EPIC"` |
| `DATA_DIR` | `"../../data/realdata/Sick"` | `"../../data/realdata/Travel"` |

#### Sick 分类特征
```python
CATEGORICAL_FEATURES = [
    'sex', 'on_thyroxine', 'query_on_thyroxine', 
    'on_antithyroid_medication', 'sick', 'pregnant', 
    'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 
    'hypopituitary', 'psych', 'TSH_measured', 'T3_measured', 
    'TT4_measured', 'T4U_measured', 'FTI_measured', 
    'referral_source', 'Class'
]
```

#### Travel 分类特征
```python
CATEGORICAL_FEATURES = [
    'Employment Type', 
    'GraduateOrNot', 
    'FrequentFlyer', 
    'EverTravelledAbroad', 
    'Target'
]
```

---

### 3️⃣ **Prompt 模板差异**

#### Sick Prompt
```python
initial_prompt = """
[DATA DESCRIPTION]
Class: hypothyroidism is a condition in which the thyroid gland is underperforming,
age: the age of an patient,
sex: the biological sex of an patient,
TSH: thyroid stimulating hormone,
T3: triiodothyronine hormone,
TT4: total levothyroxine hormone,
T4U: levothyroxine hormone uptake,
FTI: free levothyroxine hormone index,
referral_source: institution that supplied the thyroid disease record.
"""
```

#### Travel Prompt
```python
initial_prompt = """
[DATA DESCRIPTION]
Target: whether the customer purchased travel insurance (0 = No, 1 = Yes),
Age: age of the customer,
Employment Type: employment status of the customer,
GraduateOrNot: whether the customer is a graduate,
AnnualIncome: annual income of the customer,
FamilyMembers: number of family members,
ChronicDiseases: whether the customer has chronic diseases (0 = No, 1 = Yes),
FrequentFlyer: whether the customer is a frequent flyer,
EverTravelledAbroad: whether the customer has ever travelled abroad.
"""
```

---

### 4️⃣ **Classification.py 配置**

#### 已有配置（无需修改）
```python
DATA2TARGET = {
    'Sick': 'Class',
    'Travel': 'Target'  # ✅ 已存在
}

DATA2NCLASS = {
    'Sick': 2,
    'Travel': 2  # ✅ 已存在
}

ML_PARAMS = {
    'Sick': {
        'lr_max_iter': 200,
        'dt_max_depth': 10,
        'rf_max_depth': 12,
        'rf_n_estimators': 90,
    },
    'Travel': {  # ✅ 已存在
        'lr_max_iter': 100,
        'dt_max_depth': 6,
        'rf_max_depth': 12,
        'rf_n_estimators': 75,
    }
}
```

#### 分类特征索引
```python
# Sick
cat_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26]

# Travel
cat_idx = [1, 2, 4, 5]  # ✅ 已存在
```

#### 标签编码
```python
# Sick
target_encode = True  # 需要将 'negative'/'sick' 转为 0/1

# Travel
target_encode = False  # 已经是 0/1，无需转换
```

---

## 🔄 完整运行流程对比

### Sick 数据集
```bash
# 步骤 1: 无需预处理（数据已是标准格式）

# 步骤 2: 生成合成数据
cd synthetic-tabular-LLM/codes/SyntheticDataGeneration
python generate_samples_Sick.py

# 步骤 3: 评估性能
cd ../DownstreamTasks
python Classification.py  # 修改 DATA_NAME = 'Sick'
```

### Travel 数据集
```bash
# 步骤 1: 数据预处理 ⭐ 新增步骤
cd synthetic-tabular-LLM/codes/SyntheticDataGeneration
python preprocess_travel_data.py

# 步骤 2: 生成合成数据
python generate_samples_Travel.py

# 步骤 3: 评估性能
cd ../DownstreamTasks
python Classification_Travel.py
```

---

## 📁 文件结构对比

### Sick 数据集
```
data/
├── realdata/
│   └── Sick/
│       ├── X_train.csv  ✅ 已存在
│       ├── y_train.csv  ✅ 已存在
│       ├── X_test.csv   ✅ 已存在
│       └── y_test.csv   ✅ 已存在
└── syndata/
    └── Sick_DeepSeek_EPIC/
        ├── Sick_samples.csv  ✅ 已生成
        └── Sick_samples.txt  ✅ 已生成
```

### Travel 数据集
```
data/
├── realdata/
│   ├── travel/  ⭐ 原始数据
│   │   ├── travel_train.csv  ✅ 已存在
│   │   └── travel_test.csv   ✅ 已存在
│   └── Travel/  ⭐ 预处理后（需要生成）
│       ├── X_train.csv  ❌ 需要生成
│       ├── y_train.csv  ❌ 需要生成
│       ├── X_test.csv   ❌ 需要生成
│       └── y_test.csv   ❌ 需要生成
└── syndata/
    └── Travel_DeepSeek_EPIC/  ❌ 需要生成
        ├── Travel_samples.csv
        └── Travel_samples.txt
```

---

## ⚙️ 参数推荐对比

| 参数 | Sick | Travel | 说明 |
|------|------|--------|------|
| `N_SAMPLES_PER_CLASS` | 15 | 15 | Few-shot 样本数 |
| `N_BATCH` | 20 | 20 | 每次生成行数 |
| `N_TARGET_SAMPLES` | 1000 | 1000 | 目标生成总数 |
| `temperature` | 0.1 | 0.1 | LLM 温度参数 |
| `USE_RANDOM_WORD` | True | True | 随机映射开关 |

**调整建议**：
- Travel 数据集特征较少，可以尝试 `N_SAMPLES_PER_CLASS = 20`
- 如果生成速度慢，可以降低 `N_TARGET_SAMPLES = 500`

---

## 🎯 关键注意事项

### ✅ Travel 数据集特殊之处

1. **目标变量已是数值**
   - Sick: `'negative'` → 需要编码为 0
   - Travel: `0` → 无需编码 ✅

2. **特征名包含空格**
   - `'Employment Type'` ⚠️ 注意引号
   - `'GraduateOrNot'` ✅ 无空格

3. **类别不平衡程度较轻**
   - Sick: 15:1（严重不平衡）
   - Travel: 1.8:1（中等不平衡）
   - 可能需要生成更多少数类样本

---

## 🚀 快速迁移检查清单

从 Sick 迁移到 Travel，需要修改：

- [ ] ✅ 创建 `preprocess_travel_data.py`（已完成）
- [ ] ✅ 创建 `generate_samples_Travel.py`（已完成）
- [ ] ✅ 修改 `DATA_NAME = "Travel"`
- [ ] ✅ 修改 `TARGET = "Target"`
- [ ] ✅ 修改 `CATEGORICAL_FEATURES` 列表
- [ ] ✅ 修改 Prompt 中的数据描述
- [ ] ✅ 创建 `Classification_Travel.py`（已完成）
- [ ] ❌ **无需修改** `Classification.py`（已支持 Travel）

---

## 📊 预期性能对比

| 数据集 | 基线 F1 | EPIC F1 | 提升幅度 |
|--------|---------|---------|----------|
| **Sick** | 65-70% | 78-85% | +13-15% |
| **Travel** | 68-72% | 75-80% | +7-8% |

**原因**：
- Sick 不平衡更严重，EPIC 提升更明显
- Travel 基线已较好，提升空间相对较小

---

## 🎓 总结

### 主要差异
1. **数据格式**：Travel 需要预处理
2. **目标变量**：Travel 已是数值，无需编码
3. **特征数量**：Travel 更简单（9 vs 27）
4. **不平衡程度**：Travel 较轻（1.8:1 vs 15:1）

### 已完成工作
✅ 所有必要的脚本已创建  
✅ 无需修改任何现有代码  
✅ 完全兼容原有框架  

### 下一步
🚀 直接运行 `python run_travel_epic.py` 即可！

