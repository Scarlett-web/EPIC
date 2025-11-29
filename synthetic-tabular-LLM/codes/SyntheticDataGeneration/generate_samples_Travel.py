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

from util import get_prompt_conclass, parse_prompt2df, parse_result, get_unique_features, make_final_prompt

# ==========================================
# 🛠️ 配置参数
# ==========================================
# 🔒 安全提示：建议将 API Key 存储在环境变量中
# 方法1: 使用环境变量（推荐）
# my_deepseek_key = os.getenv("DEEPSEEK_API_KEY", "your-default-key")
# 方法2: 直接填写（仅用于测试）
my_deepseek_key = "sk-erltuaebsxiimieebxdxlbeifvootbvnacyzmglozboutlyg"

params = {
    "openai_key": my_deepseek_key,
    "model": "deepseek-chat",
    "DATA_NAME": "travel",  # 改为小写，匹配实际文件夹名
    "TARGET": "Target",
    "N_CLASS": 2,
    "N_SAMPLES_PER_CLASS": 15,  # 每类给 15 个样本作为示例
    "N_SET": 4,
    "USE_RANDOM_WORD": True,
    "N_BATCH": 20,  # 每次生成 20 行
    "MODEL_NAME": "Travel_DeepSeek_EPIC",
    "N_TARGET_SAMPLES": 1000,  # 目标生成 1000 条
}

params.update({
    "DATA_DIR": f"../../data/realdata/{params['DATA_NAME']}",
    "SAVE_DIR": f"../../data/syndata/{params['MODEL_NAME']}"
})

# ==========================================
# 🔌 初始化 API
# ==========================================
openai.api_key = params['openai_key']
os.environ["OPENAI_API_KEY"] = params['openai_key']

http_client = httpx.Client(trust_env=False)

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    openai_api_key=params['openai_key'],
    openai_api_base="https://api.siliconflow.cn/v1",
    temperature=0.1,
    http_client=http_client
)

output_parser = StrOutputParser()

# ==========================================
# 📂 数据加载与预处理
# ==========================================
DATA_NAME = params['DATA_NAME']
TARGET = params['TARGET']
REAL_DATA_SAVE_DIR = params['DATA_DIR']
symModel = params['MODEL_NAME']
SYN_DATA_SAVE_DIR = params['SAVE_DIR']
os.makedirs(SYN_DATA_SAVE_DIR, exist_ok=True)

print(f"Loading data from {REAL_DATA_SAVE_DIR}...")

try:
    X_train = pd.read_csv(os.path.join(REAL_DATA_SAVE_DIR, 'X_train.csv'), index_col=0)
    y_train = pd.read_csv(os.path.join(REAL_DATA_SAVE_DIR, 'y_train.csv'), index_col=0)
except FileNotFoundError:
    print("❌ 错误: 找不到数据文件。请先运行 preprocess_travel_data.py")
    exit()

data = pd.concat((y_train, X_train), axis=1)

# Travel 数据集的分类变量定义
CATEGORICAL_FEATURES = ['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad', 'Target']
NAME_COLS = ','.join(data.columns) + '\n'    

unique_categorical_features = get_unique_features(data, CATEGORICAL_FEATURES)

cat_idx = []
for i, c in enumerate(X_train.columns):
    if c in CATEGORICAL_FEATURES:
        cat_idx.append(i)

# ==========================================
# 🔠 EPIC 核心: 随机单词映射
# ==========================================
if params['USE_RANDOM_WORD']:
    print("Applying Unique Variable Mapping strategy...")
    def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        first = ''.join(random.choice(string.ascii_uppercase) for _ in range(1))
        left = ''.join(random.choice(chars) for _ in range(size-1))
        return first + left
    
    def make_random_categorical_values(unique_categorical_features):
        mapper = {}
        mapper_r = {}
        new_unique_categorical_features = {}
        for c in unique_categorical_features:
            mapper[c] = {}
            mapper_r[c] = {}
            new_unique_categorical_features[c] = []
    
            for v in unique_categorical_features[c]:
                a = id_generator(3)
                new_unique_categorical_features[c].append(a)
    
                mapper[c][v] = a
                mapper_r[c][a] = v
        return mapper, mapper_r, new_unique_categorical_features
    
    mapper, mapper_r, unique_categorical_features = make_random_categorical_values(unique_categorical_features)
        
    for c in mapper:
        data[c] = data[c].map(lambda x: mapper[c][x] if x in mapper[c] else x)

# ==========================================
# 📝 Prompt 模板构建
# ==========================================
initial_prompt = """
[SYSTEM INSTRUCTION]
You are a strict tabular data generator. 
Your task is to generate new synthetic data samples that follow the EXACT format, distribution, and unique encoding patterns of the few-shot examples provided below.

RULES:
1. Output ONLY the CSV data rows.
2. Do NOT include any explanations, analysis, headers, or introductory text.
3. Do NOT use Markdown formatting (no ```csv ... ```).
4. Each line must be a valid comma-separated value string.
5. Generate exactly 20 lines of new data.

[DATA DESCRIPTION]
Target: whether the customer purchased travel insurance (0 = No, 1 = Yes),
Age: age of the customer,
Employment Type: employment status of the customer,
GraduateOrNot: whether the customer is a graduate,
AnnualIncome: annual income of the customer,
FamilyMembers: number of family members,
ChronicDiseases: whether the customer has chronic diseases (0 = No, 1 = Yes),
FrequentFlyer: whether the customer is a frequent flyer,
EverTravelledAbroad: whether the customer has ever travelled abroad.\n\n
"""

numbering = ['A', 'B', 'C', 'D']
N_CLASS = params['N_CLASS']
N_SAMPLES_PER_CLASS = params['N_SAMPLES_PER_CLASS']
N_SET = params['N_SET']
N_BATCH = params['N_BATCH']
N_SAMPLES_TOTAL = N_SAMPLES_PER_CLASS * N_SET * N_BATCH

prompt = get_prompt_conclass(initial_prompt, numbering, N_SAMPLES_PER_CLASS, N_CLASS, N_SET, NAME_COLS)

template1 = prompt
template1_prompt = PromptTemplate.from_template(template1)

llm1 = (
    template1_prompt
    | llm
    | output_parser
)

# ==========================================
# 🔄 开始生成循环
# ==========================================
input_df_all = pd.DataFrame()
synthetic_df_all = pd.DataFrame()
text_results = []
columns1 = data.columns
columns2 = list(data.columns)
err = []

print(f"Start generating {params['N_TARGET_SAMPLES']} samples...")

while len(synthetic_df_all) < params['N_TARGET_SAMPLES']:
    # 构建 Prompt Batch
    final_prompt, inputs_batch = make_final_prompt(unique_categorical_features, TARGET, data, template1_prompt,
                                                   N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS)

    try:
        # 调用 DeepSeek API
        inter_text = llm1.batch(inputs_batch)

        for i in range(len(inter_text)):
            try:
                text_results.append(final_prompt[i].text + inter_text[i])
                # 解析生成的文本为 DataFrame
                input_df = parse_prompt2df(final_prompt[i].text, split=NAME_COLS, inital_prompt=initial_prompt, col_name=columns1)
                result_df = parse_result(inter_text[i], NAME_COLS, columns2, CATEGORICAL_FEATURES, unique_categorical_features, filter_flag=False)

                input_df_all = pd.concat([input_df_all, input_df], axis=0)
                synthetic_df_all = pd.concat([synthetic_df_all, result_df], axis=0)
            except Exception as e:
                err.append(inter_text[i])
                print(f"Parsing error: {e}")

        print(f'Progress: {len(synthetic_df_all)} / {params["N_TARGET_SAMPLES"]}')
        print("Sleeping for 60s to avoid rate limit...")
        time.sleep(60)

    except Exception as e:
        print(f"API Error (Check Quota/Network): {e}")
        break


# ==========================================
# 💾 还原映射并保存
# ==========================================
synthetic_df_all_r = synthetic_df_all.copy()

if params['USE_RANDOM_WORD']:
    print("Reversing Unique Variable Mapping...")
    for c in mapper_r:
        if c in input_df_all.columns:
            input_df_all[c] = input_df_all[c].map(lambda x: mapper_r[c][x] if x in mapper_r[c] else x)
    for c in mapper_r:
        if c in synthetic_df_all_r.columns:
            synthetic_df_all_r[c] = synthetic_df_all_r[c].map(lambda x: mapper_r[c][x] if x in mapper_r[c] else x)

# 保存文件
file_name = os.path.join(SYN_DATA_SAVE_DIR, f'{DATA_NAME}_samples.csv')

# 保存 Prompt 模板以供检查
with open(file_name.replace('.csv', '.txt'), 'w', encoding='utf-8') as f:
    f.write(template1 + '\n===\n' + (final_prompt[0].text if len(final_prompt) > 0 else ""))

# 保存最终的合成数据
synthetic_df_all_r.to_csv(file_name, index_label='synindex')
print(f'✅ Done! Synthetic data saved to: {file_name}')
print(f'📊 Total samples generated: {len(synthetic_df_all_r)}')
print(f'❌ Parsing errors: {len(err)}')

