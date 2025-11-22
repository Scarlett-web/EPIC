# EPIC 论文复现项目 (CUFE 数据挖掘小组)
这是我们小组复现 NeurIPS 2024 论文 《EPIC: Effective Prompting for Imbalanced-Class Data Synthesis in Tabular Data Classification via Large Language Models》 的代码仓库。

本项目在原作者开源代码的基础上，进行了本地化适配与依赖修复，使其能够在国内网络环境下稳定运行。

🌟 主要改进与特性
相比原论文代码，本仓库做了以下优化，队友请重点关注：
模型迁移 (Cost-Effective)：
将原昂贵的 OpenAI API 替换为国产 DeepSeek API。
优势：无需魔法（VPN），直连访问，成本极低（几乎免费）。
环境修复 (Bug Fixes)：
修复了 LangChain 与 OpenAI 库新版本的 proxies 参数冲突问题。
锁定了稳定的依赖版本，避免环境报错。
鲁棒性增强 (Robust Parsing)：
重写了 util.py 中的解析逻辑。
增加了正则表达式清洗功能，解决了 DeepSeek 输出废话导致 Pandas 解析失败的问题。
🚀 快速开始 (队友必读)
请严格按照以下步骤配置环境，确保代码能跑通。
# 1. 克隆仓库
  git clone https://github.com/Scarlett-web/EPIC-Reproduction.git
  cd EPIC-Reproduction
2. 配置环境
建议使用 Conda 创建独立的 Python 3.8 环境，防止污染本地环境。
  #1. 创建环境
    conda create -n epic_env python=3.8 -y
  
  #2. 激活环境
    conda activate epic_env
  
  #3. 安装依赖 (一定要用这个命令，包含了我修复后的版本)
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   
3. 填写 API Key (关键)
出于安全考虑，代码中的 API Key 是空的。请联系 @Scarlett-web 获取 Key，或注册 DeepSeek 自行申请。
打开文件：synthetic-tabular-LLM/codes/SyntheticDataGeneration/generate_samples_Sick.py
  # 请将你的 Key 填入引号中
  my_deepseek_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
4. 运行数据生成
进入脚本目录并运行：
  # 进入代码目录
  cd synthetic-tabular-LLM/codes/SyntheticDataGeneration
  
  # 运行生成脚本
  python generate_samples_Sick.py
5. 预期结果
如果运行成功，你将看到进度条：
  Loading data from ../../data/realdata/Sick...
  Applying Unique Variable Mapping strategy...
  Start generating 1000 samples...
  Progress: 20 / 1000
  ...
  ✅ Done! Synthetic data saved to: ../../data/syndata/Sick_DeepSeek_EPIC/Sick_samples.csv

项目结构：
    EPIC-Reproduction/
  ├── synthetic-tabular-LLM/
  │   ├── codes/
  │   │   ├── SyntheticDataGeneration/  # 核心生成代码 (generate_samples_Sick.py)
  │   │   ├── DownstreamTasks/          # 下游分类任务代码
  │   │   └── ...
  │   ├── data/
  │   │   ├── realdata/                 # 原始数据集 (Sick)
  │   │   └── syndata/                  # 生成的合成数据 (结果在这里)
  │   └── util.py                       # 工具类 (已修复解析 Bug)
  ├── requirements.txt                  # 环境依赖列表
  └── README.md                         # 项目说明

  📋 下一步计划
  [x]跑通核心生成代码 (Data Synthesis)
  
  [x] 解决 API 连接与解析 Bug
  
  [ ] 运行下游分类任务 (Classification)，验证数据质量 (F1 Score)
  
  [ ] 尝试其他数据集 (可选)
  
  Troubleshooting:
  
  如果报错 402 Insufficient Balance：DeepSeek 余额不足，请充值（几块钱即可）。
  
  如果报错 FileNotFound：请检查你是否 cd 到了正确的子文件夹下运行代码。
