import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 读取密钥
api_key = os.getenv("DEEPSEEK_API_KEY")

if api_key:
    print(f"✅ 读取成功！密钥前8位：{api_key[:8]}...")
else:
    print("❌ 读取失败！请检查 .env 文件是否存在、格式是否正确")
