"""
HELOC 数据集 EPIC 生成脚本
作用：使用 DeepSeek-V3 生成 HELOC 数据集的合成样本
"""
import time
import openai
import os
import pandas as pd
import string
import random
import httpx

from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

# 强制清除系统代理设置
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# ==========================================
# 🔑 API 配置
# ==========================================
my_deepseek_key = "sk-erltuaebsxiimieebxdxlbeifvootbvnacyzmglozboutlyg"

# ==========================================
# 📊 参数配置
# ==========================================
params = {
    "openai_key": my_deepseek_key,
    "model": "deepseek-ai/DeepSeek-V3",  # 使用完整模型名称
    "DATA_NAME": "HELOC",  # 数据集名称（大写，匹配文件夹）
    "TARGET": "RiskPerformance",  # 目标变量名
    "N_CLASS": 2,  # 类别数
    "N_SAMPLES_PER_CLASS": 15,  # 每类给 15 个样本作为示例
    "N_SET": 4,
    "USE_RANDOM_WORD": True,  # 使用随机单词映射
    "N_BATCH": 20,  # 每次生成 20 行
    "MODEL_NAME": "HELOC_DeepSeek_EPIC",  # 模型名称
    "N_TARGET_SAMPLES": 1000,  # 目标生成 1000 条
}

params.update({
    "DATA_DIR": f"../../data/realdata/{params['DATA_NAME']}",
    "SAVE_DIR": f"../../data/syndata/{params['MODEL_NAME']}"
})

# ==========================================
# 🔌 初始化 API
# ==========================================
print("="*60)
print(f"🚀 开始生成 {params['DATA_NAME']} 数据集的合成样本")
print("="*60)

# 创建自定义 HTTP 客户端（禁用代理）
http_client = httpx.Client(timeout=60.0)

llm = ChatOpenAI(
    model=params['model'],
    openai_api_key=params['openai_key'],
    openai_api_base="https://api.siliconflow.cn/v1",
    temperature=0.1,
    http_client=http_client
)

# ==========================================
# 📝 Prompt 模板
# ==========================================
initial_prompt = """
[SYSTEM INSTRUCTION]
You are a strict tabular data generator. Generate EXACTLY {n_batch} rows of synthetic data in CSV format.

[DATA DESCRIPTION]
This is HELOC (Home Equity Line of Credit) risk assessment data with the following features:

RiskPerformance: credit risk level (RIS_0 = Good, RIS_1 = Bad),
ExternalRiskEstimate: external risk score (0-100),
MSinceOldestTradeOpen: months since oldest trade opened,
MSinceMostRecentTradeOpen: months since most recent trade opened,
AverageMInFile: average months in file,
NumSatisfactoryTrades: number of satisfactory trades,
NumTrades60Ever2DerogPubRec: number of trades 60+ days past due,
NumTrades90Ever2DerogPubRec: number of trades 90+ days past due,
PercentTradesNeverDelq: percent of trades never delinquent,
MSinceMostRecentDelq: months since most recent delinquency,
MaxDelq2PublicRecLast12M: maximum delinquency in last 12 months,
MaxDelqEver: maximum delinquency ever,
NumTotalTrades: total number of trades,
NumTradesOpeninLast12M: number of trades opened in last 12 months,
PercentInstallTrades: percent of installment trades,
MSinceMostRecentInqexcl7days: months since most recent inquiry (excluding 7 days),
NumInqLast6M: number of inquiries in last 6 months,
NumInqLast6Mexcl7days: number of inquiries in last 6 months (excluding 7 days),
NetFractionRevolvingBurden: net fraction of revolving burden,
NetFractionInstallBurden: net fraction of installment burden,
NumRevolvingTradesWBalance: number of revolving trades with balance,
NumInstallTradesWBalance: number of installment trades with balance,
NumBank2NatlTradesWHighUtilization: number of bank/national trades with high utilization,
PercentTradesWBalance: percent of trades with balance

[EXAMPLES]
{examples}

[TASK]
Generate EXACTLY {n_batch} new rows following the same pattern. Output ONLY the CSV data with header, no explanations.
"""

# ==========================================
# 📂 加载数据
# ==========================================
print(f"\n📂 Loading data from {params['DATA_DIR']}...")
X_train = pd.read_csv(os.path.join(params['DATA_DIR'], 'X_train.csv'), index_col='index')
y_train = pd.read_csv(os.path.join(params['DATA_DIR'], 'y_train.csv'), index_col='index')

# 合并特征和目标
train_data = pd.concat([y_train, X_train], axis=1)

print(f"   训练数据形状: {train_data.shape}")
print(f"   目标变量分布:")
print(train_data[params['TARGET']].value_counts())

# ==========================================
# 🔀 随机单词映射（Random Word Mapping）
# ==========================================
# HELOC 数据集只有目标变量是分类的，其他都是数值
CATEGORICAL_FEATURES = ['RiskPerformance']

mapping_dict = {}
reverse_mapping_dict = {}

if params['USE_RANDOM_WORD']: 
    print(f"\n🔀 Applying Unique Variable Mapping strategy...")
    
    for col in CATEGORICAL_FEATURES:
        if col in train_data.columns:
            unique_values = train_data[col].unique()
            random_codes = [f"{col[:3].upper()}_{i}" for i in range(len(unique_values))]
            
            mapping_dict[col] = dict(zip(unique_values, random_codes))
            reverse_mapping_dict[col] = dict(zip(random_codes, unique_values))
            
            train_data[col] = train_data[col].map(mapping_dict[col])
            
            print(f"   {col}: {mapping_dict[col]}")

# ==========================================
# 🎯 Few-shot 采样
# ==========================================
print(f"\n🎯 Sampling {params['N_SAMPLES_PER_CLASS']} examples per class...")

sampled_data = []
for class_label in train_data[params['TARGET']].unique():
    class_data = train_data[train_data[params['TARGET']] == class_label]
    sampled = class_data.sample(n=min(params['N_SAMPLES_PER_CLASS'], len(class_data)), random_state=42)
    sampled_data.append(sampled)

few_shot_examples = pd.concat(sampled_data, axis=0)
examples_csv = few_shot_examples.to_csv(index=False)

print(f"   Few-shot 样本数: {len(few_shot_examples)}")

# ==========================================
# 🔄 批量生成
# ==========================================
print(f"\n🔄 Start generating {params['N_TARGET_SAMPLES']} samples...")

prompt_template = PromptTemplate(
    input_variables=["examples", "n_batch"],
    template=initial_prompt
)

chain = prompt_template | llm | StrOutputParser()

all_generated_samples = []
n_generated = 0
n_errors = 0

while n_generated < params['N_TARGET_SAMPLES']:
    try:
        # 调用 LLM
        response = chain.invoke({
            "examples": examples_csv,
            "n_batch": params['N_BATCH']
        })

        # 解析 CSV
        from io import StringIO
        try:
            generated_df = pd.read_csv(StringIO(response))

            # 验证列名
            if set(generated_df.columns) == set(train_data.columns):
                all_generated_samples.append(generated_df)
                n_generated += len(generated_df)
                print(f"Progress: {n_generated} / {params['N_TARGET_SAMPLES']}")
            else:
                n_errors += 1
                print(f"⚠️  Column mismatch, skipping...")

        except Exception as e:
            n_errors += 1
            print(f"⚠️  Parsing error: {e}")

        # 避免 API 限流
        if n_generated < params['N_TARGET_SAMPLES']:
            print(f"Sleeping for 60s to avoid rate limit...")
            time.sleep(60)

    except Exception as e:
        print(f"❌ API Error: {e}")
        print(f"Retrying in 60s...")
        time.sleep(60)

# ==========================================
# 💾 合并并保存
# ==========================================
print(f"\n💾 Merging and saving results...")

final_df = pd.concat(all_generated_samples, axis=0, ignore_index=True)
final_df.insert(0, 'synindex', range(len(final_df)))

# ==========================================
# 🔄 反向映射（恢复原始值）
# ==========================================
if params['USE_RANDOM_WORD']:
    print(f"\n🔄 Reversing Unique Variable Mapping...")

    for col in CATEGORICAL_FEATURES:
        if col in final_df.columns and col in reverse_mapping_dict:
            final_df[col] = final_df[col].map(reverse_mapping_dict[col])

# ==========================================
# 💾 保存文件
# ==========================================
os.makedirs(params['SAVE_DIR'], exist_ok=True)

output_csv = os.path.join(params['SAVE_DIR'], f"{params['DATA_NAME']}_samples.csv")
output_txt = os.path.join(params['SAVE_DIR'], f"{params['DATA_NAME']}_samples.txt")

final_df.to_csv(output_csv, index=False)

# 保存 Prompt 模板
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write(initial_prompt)

print(f"\n✅ Done! Synthetic data saved to: {output_csv}")
print(f"📊 Total samples generated: {len(final_df)}")
print(f"❌ Parsing errors: {n_errors}")

print("\n" + "="*60)
print("🎉 生成完成！")
print("="*60)

