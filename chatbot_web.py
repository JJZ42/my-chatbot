import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ===== 1. 页面设置 =====
st.set_page_config(
    page_title="DeepSeek 问答助手",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 DeepSeek 智能问答机器人")
st.caption("DeepSeek-V3 驱动 · 计算机大一作品")

# ===== 2. 创建 DeepSeek 客户端 =====
@st.cache_resource  # 缓存客户端，避免重复创建
def get_client():
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

client = get_client()

# ===== 3. 侧边栏设置 =====
with st.sidebar:
    st.header("⚙️ 设置面板")
    
    # 模型选择（DeepSeek 目前主要有两个）
    model = st.selectbox(
        "选择模型",
        ["deepseek-chat", "deepseek-reasoner"],  # V3 和 R1
        help="deepseek-chat = 通用对话 | deepseek-reasoner = 推理增强"
    )
    
    # 人格选择
    persona = st.selectbox(
        "机器人人格",
        ["编程导师 👨‍💻", "英语老师 📚", "通用助手 🤖", "段子手 😂", "C语言专家 🔧"]
    )
    
    persona_dict = {
        "编程导师 👨‍💻": "你是一位耐心的编程老师。用通俗语言解释概念，必须给出可运行的代码示例，并指出新手常见错误。",
        "英语老师 📚": "你是中英双语老师。用中英双语回答，纠正用户的语法错误，给出更地道的表达方式。",
        "通用助手 🤖": "你是一位知识渊博的AI助手，回答准确、简洁、有条理。",
        "段子手 😂": "你是一位幽默的段子手。用轻松诙谐的方式回答问题，可以调侃但保持友善。",
        "C语言专家 🔧": "你是C语言深度专家。解释指针、内存管理、编译原理等底层概念时深入浅出，强调最佳实践和常见陷阱。",
    }
    system_prompt = persona_dict[persona]
    
    # 创意度
    temperature = st.slider(
        "创意程度", 
        0.0, 2.0, 0.7, 0.1,   # DeepSeek 支持 0-2
        help="0=严谨精确  1=平衡  2=天马行空"
    )
    
    # 最大长度
    max_tokens = st.slider("回答最大长度", 256, 4096, 2048, 256)
    
    # 清空对话按钮
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ===== 4. 对话历史管理 =====
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===== 5. 用户输入 & AI回复 =====
if user_input := st.chat_input("在这里输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 调用 DeepSeek
    with st.chat_message("assistant"):
        with st.spinner("DeepSeek 思考中..."):
            # 构建消息列表
            full_messages = [
                {"role": "system", "content": system_prompt}
            ] + st.session_state.messages
            
            response = client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content
            
            st.markdown(answer)
    
    # 保存回复
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # 侧边栏显示统计
    with st.sidebar:
        total_chars = sum(len(m["content"]) for m in st.session_state.messages)
        st.metric("对话轮次", len(st.session_state.messages) // 2)
        st.metric("总字符数", f"{total_chars:,}")
