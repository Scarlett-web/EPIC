# 📘 Data README — EPIC_Replication

本文件说明 EPIC 论文复现项目中的数据存放结构、来源及处理方式。  
所有数据文件均存放于项目目录下的 `data/` 文件夹中。

---

## 🗂️ 数据目录结构

data/
├── raw/            # 原始数据（从 Kaggle、UCI、FICO 下载）
├── clean/          # 清洗后的数据（执行 preprocess.py 生成）
├── split/          # 训练 / 测试集划分结果（执行 split_data.py 生成）
└── mappings/       # 类别值映射文件（由 preprocess.py 自动创建）

---

## 📚 数据集列表

| 数据集 | 领域 | 来源 | 任务类型 | 目标列 | 样本数（约） |
|---------|------|------|-----------|---------|---------------|
| **Sick** | 医疗 | UCI (Thyroid Disease Data Set) | 二分类 | `binaryClass` | 3700 |
| **Thyroid_Diff** | 医疗 | UCI | 多分类 | `Risk` | 360 |
| **HELOC** | 金融 | FICO 官方 | 二分类（信贷风险） | `RiskPerformance` | 10,000 |
| **Income (Adult)** | 社会 | UCI / Kaggle | 二分类（>50K 收入预测） | `income` | 48,000 |
| **Diabetes (Pima)** | 医疗 | Kaggle | 二分类（是否患病） | `Outcome` | 768 |
| **Travel Insurance** | 营销 | Kaggle | 二分类（是否购买保险） | `TravelInsurance` | 2,000 |

---

## 🧹 数据阶段说明

### 1️⃣ 原始数据（`data/raw/`）
- 下载自论文中指定来源（Kaggle / UCI / FICO）。
- 文件命名规则：  
  `dataset_name.csv`  
- 不允许修改；如重新下载，请更新本文件中的来源链接。

### 2️⃣ 清洗后数据（`data/clean/`）
- 由 `scripts/preprocess.py` 生成。
- 清洗规则：
  1. 删除重复样本；
  2. 保留缺失值（除 Sick 外特殊处理）；
  3. 生成映射文件（`data/mappings/*.json`）；
  4. 检查类别平衡。
- 命名规则：  
  `dataset_name_clean.csv`

### 3️⃣ 训练 / 测试集（`data/split/`）
- 由 `scripts/split_data.py` 自动划分；
- 固定随机种子（42）；
- 按 80% / 20% 比例分层抽样；
- 命名规则：  
  `dataset_name_train.csv`  
  `dataset_name_test.csv`

### 4️⃣ 映射文件（`data/mappings/`）
- 每个数据集一个 `.json` 文件；
- 记录清洗时的类别映射关系，例如：
  ```json
  {
    "sex": { "Male": "SEX_0", "Female": "SEX_1" },
    "income": { "<=50K": "INC_0", ">50K": "INC_1" }
  }

---

## ⚙️ 更新规则

| 阶段 | 更新人 | 文件夹 | 是否可修改 |
|------|----------|---------|--------------|
| 原始数据下载 | Data Steward | `data/raw/` | ❌ 禁止覆盖 |
| 数据清洗 | Data Steward | `data/clean/` | ✅ 可更新 |
| 数据划分 | 全组共享 | `data/split/` | ❌ 不得重新划分 |
| 映射文件 | 自动生成 | `data/mappings/` | ⚙️ 自动更新 |

---

## 🔗 数据来源链接

| 数据集 | 链接 |
|--------|------|
| Sick / Thyroid Disease | https://archive.ics.uci.edu/ml/datasets/thyroid+disease |
| Diabetes (Pima) | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| HELOC | https://community.fico.com/s/explainable-machine-learning-challenge |
| Adult Income | https://archive.ics.uci.edu/ml/datasets/adult |
| Travel Insurance | https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data |
| Thyroid Diff | https://www.kaggle.com/datasets/yasirhussein/thyroid-disease-data-set |

---

## ✅ 文件命名约定

| 类型 | 示例文件名 | 说明 |
|------|--------------|------|
| 原始数据 | `income.csv` | 下载自 Kaggle/UCI |
| 清洗后数据 | `income_clean.csv` | 经 preprocess 清洗 |
| 训练集 | `income_train.csv` | 80% 分层抽样 |
| 测试集 | `income_test.csv` | 20% 分层抽样 |
| 映射文件 | `income_map.json` | 类别值对应关系 |

---

## 🧾 数据版本记录

| 日期 | 操作 | 负责人 |
|------|-------|----------|
| 2025-10-09 | 下载全部 6 个原始数据集 | 樊怡璇 |
| 2025-10-10 | 执行清洗脚本并验证 | 樊怡璇 |
| 2025-10-11 | 完成划分并生成 manifest | 樊怡璇 |

---

## 📚 附录

- 若数据路径更改，请同步更新 `scripts/preprocess.py` 与 `split_data.py` 中的 `CLEAN` / `SPLIT` 常量。
- 若新增数据集，请补充其来源与任务说明。

---

✍️ *Last updated: 2025-10-11*