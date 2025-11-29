# 🚀 Travel 数据集 EPIC 运行快速指南

## 📋 前置检查

### ✅ 确认文件存在
- [ ] `synthetic-tabular-LLM/data/realdata/travel/travel_train.csv`
- [ ] `synthetic-tabular-LLM/data/realdata/travel/travel_test.csv`

### ✅ 确认环境配置
```bash
# 检查 Python 版本 (需要 3.8+)
python --version

# 检查依赖包
pip list | grep -E "langchain|openai|xgboost|pandas"
```

---

## 🎯 方法一：一键运行（推荐）

```bash
cd synthetic-tabular-LLM/codes
python run_travel_epic.py
```

**选择选项**：
- `1` - 只预处理数据
- `2` - 只生成合成数据
- `3` - 只评估性能
- `4` - 完整流程（预处理 + 生成 + 评估）⭐
- `5` - 跳过生成，只评估已有数据

---

## 🎯 方法二：分步运行

### **步骤 1：数据预处理** ⏱️ 约 5 秒

```bash
cd synthetic-tabular-LLM/codes/SyntheticDataGeneration
python preprocess_travel_data.py
```

**输出**：
```
✅ 数据预处理完成！文件已保存到: ../../data/realdata/Travel
📁 生成的文件:
    - X_train.csv: (1589, 9)
    - y_train.csv: (1589, 1)
    - X_test.csv: (399, 9)
    - y_test.csv: (399, 1)
```

---

### **步骤 2：生成合成数据** ⏱️ 约 15-30 分钟

```bash
python generate_samples_Travel.py
```

**运行过程**：
```
Loading data from ../../data/realdata/Travel...
Applying Unique Variable Mapping strategy...
Start generating 1000 samples...
Progress: 20 / 1000
Sleeping for 60s to avoid rate limit...
Progress: 40 / 1000
...
✅ Done! Synthetic data saved to: ../../data/syndata/Travel_DeepSeek_EPIC/Travel_samples.csv
```

**输出文件**：
- `data/syndata/Travel_DeepSeek_EPIC/Travel_samples.csv` - 合成数据
- `data/syndata/Travel_DeepSeek_EPIC/Travel_samples.txt` - Prompt 记录

---

### **步骤 3：评估性能** ⏱️ 约 2-5 分钟

```bash
cd ../DownstreamTasks
python Classification_Travel.py
```

**输出示例**：
```
📊 最终结果汇总
+------+---------------------------+------------------------+-------+--------------+-------+
| data | model                     | synModel               |    F1 | BalancedACC  |   AUC |
+------+---------------------------+------------------------+-------+--------------+-------+
| Travel | XGBoostClassifier_grid  | None                   | 68.50 |        65.20 | 72.30 |
| Travel | XGBoostClassifier_grid  | Travel_DeepSeek_EPIC   | 76.80 |        73.50 | 79.20 |
| Travel | CatBoostClassifier_grid | Travel_DeepSeek_EPIC   | 75.20 |        72.10 | 78.50 |
+------+---------------------------+------------------------+-------+--------------+-------+
```

---

## 📊 关键参数调整

### 修改生成数量

编辑 `generate_samples_Travel.py`：

```python
params = {
    "N_TARGET_SAMPLES": 1000,  # 改为 500 或 2000
}
```

### 修改 Few-shot 样本数

```python
params = {
    "N_SAMPLES_PER_CLASS": 15,  # 每类给 LLM 看多少个样本
}
```

### 修改生成速度

```python
params = {
    "N_BATCH": 20,  # 每次生成多少行（越大越快，但可能格式错误率高）
}
```

---

## ⚠️ 常见问题排查

### 1. 找不到数据文件

**错误**：
```
❌ 错误: 找不到数据文件。请先运行 preprocess_travel_data.py
```

**解决**：
```bash
cd synthetic-tabular-LLM/codes/SyntheticDataGeneration
python preprocess_travel_data.py
```

---

### 2. API 调用失败

**错误**：
```
API Error (Check Quota/Network): ...
```

**解决方案**：

#### 方案 A：检查 API Key
```python
# 在 generate_samples_Travel.py 中
my_deepseek_key = "sk-your-key-here"  # 确保是完整的 Key
```

#### 方案 B：检查网络
```bash
# 测试能否访问 API
curl https://api.siliconflow.cn/v1/models
```

#### 方案 C：检查配额
- 登录 [SiliconFlow 控制台](https://cloud.siliconflow.cn/)
- 查看剩余额度

---

### 3. 生成数据格式错误率高

**现象**：
```
Parsing error: ...
Progress: 50 / 1000  (错误率 40%)
```

**解决**：

1. **降低批次大小**：
```python
"N_BATCH": 10,  # 从 20 改为 10
```

2. **增加 Few-shot 样本**：
```python
"N_SAMPLES_PER_CLASS": 20,  # 从 15 改为 20
```

3. **调整 Temperature**：
```python
llm = ChatOpenAI(
    temperature=0.05,  # 从 0.1 改为 0.05（更保守）
)
```

---

### 4. 评估时找不到合成数据

**错误**：
```
⚠️ 数据加载失败: FileNotFoundError
```

**解决**：
确保已经运行步骤 2 生成了数据，或者修改评估脚本：

```python
SYNTHETIC_METHODS = [
    'None',  # 只用真实数据
    # 'Travel_DeepSeek_EPIC',  # 如果还没生成，先注释掉
]
```

---

## 📈 预期结果对比

| 方法 | F1-Score | Balanced Accuracy | 说明 |
|------|----------|-------------------|------|
| **None (仅真实数据)** | 68-72% | 65-70% | 基线 |
| **SMOTENC** | 70-75% | 68-73% | 传统过采样 |
| **CTGAN** | 68-72% | 65-70% | 深度学习生成 |
| **EPIC (LLM)** | 75-80% | 72-78% | 🏆 最佳 |

---

## 🔒 安全提示

**重要**：代码中包含明文 API Key！

### 推荐做法：

1. **创建 `.env` 文件**：
```bash
cd synthetic-tabular-LLM/codes/SyntheticDataGeneration
echo "DEEPSEEK_API_KEY=sk-your-key-here" > .env
```

2. **修改代码**：
```python
from dotenv import load_dotenv
load_dotenv()

my_deepseek_key = os.getenv("DEEPSEEK_API_KEY")
```

3. **添加到 .gitignore**：
```bash
echo ".env" >> .gitignore
```

---

## 📚 文件清单

### 新创建的文件：
- ✅ `codes/SyntheticDataGeneration/preprocess_travel_data.py` - 数据预处理
- ✅ `codes/SyntheticDataGeneration/generate_samples_Travel.py` - EPIC 生成
- ✅ `codes/SyntheticDataGeneration/README_Travel.md` - 详细文档
- ✅ `codes/DownstreamTasks/Classification_Travel.py` - 性能评估
- ✅ `codes/run_travel_epic.py` - 一键运行脚本
- ✅ `TRAVEL_EPIC_快速指南.md` - 本文档

### 需要修改的文件：
- ❌ **无需修改任何现有文件**（已完全兼容）

---

## 🎓 下一步

1. **运行完整流程**：
```bash
cd synthetic-tabular-LLM/codes
python run_travel_epic.py
# 选择选项 4
```

2. **对比不同方法**：
   - 运行 SMOTE 基线：`python run_smote_baseline.py`（需要修改数据集名称）
   - 运行 CTGAN/TVAE：`python run_dl_baselines.py`（需要修改数据集名称）

3. **调整参数优化**：
   - 尝试不同的 `N_SAMPLES_PER_CLASS`
   - 尝试不同的 `temperature`
   - 尝试不同的 Prompt 模板

---

## 📞 需要帮助？

如果遇到问题，请检查：
1. ✅ Python 版本是否 >= 3.8
2. ✅ 依赖包是否完整安装
3. ✅ API Key 是否有效
4. ✅ 数据文件路径是否正确

祝你实验顺利！🚀

