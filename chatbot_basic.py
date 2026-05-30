import os
from openai import OpenAI
from dotenv import load_dotenv

# ===== 1. 加载密钥 =====
load_dotenv()

# ===== 2. 创建 DeepSeek 客户端 =====
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",   # DeepSeek 官方地址
)

MODEL = "deepseek-chat"  # DeepSeek-V3 模型

# ===== 3. 核心函数 =====
def ask_ai(question):
    """发送问题给 DeepSeek，返回回答"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个有用的编程助手，回答简洁清晰，代码示例要完整可运行。"},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=2048,        # DeepSeek 需要指定，否则可能截断
    )
    return response.choices[0].message.content


# ===== 4. 交互循环 =====
print("=" * 50)
print("🤖 DeepSeek 智能问答机器人")
print("输入 'q' 退出")
print("=" * 50)

while True:
    user_input = input("\n🧑 你：")
    
    if user_input.lower() in ['q', 'quit', '退出']:
        print("🤖 机器人：再见！")
        break
    
    if user_input.strip() == "":
        continue
    
    answer = ask_ai(user_input)
    print(f"\n🤖 机器人：{answer}")
