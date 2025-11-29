"""
测试 API 连接
"""
import os
import httpx
from langchain_openai import ChatOpenAI

# 清除代理
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

my_deepseek_key = "sk-erltuaebsxiimieebxdxlbeifvootbvnacyzmglozboutlyg"

print("🔌 测试 API 连接...")

try:
    http_client = httpx.Client(timeout=60.0)
    
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        openai_api_key=my_deepseek_key,
        openai_api_base="https://api.siliconflow.cn/v1",
        temperature=0.1,
        http_client=http_client
    )
    
    print("✅ LLM 初始化成功")
    
    # 测试简单调用
    print("\n🧪 测试简单调用...")
    response = llm.invoke("Say 'Hello, EPIC!'")
    print(f"✅ API 响应: {response.content}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

