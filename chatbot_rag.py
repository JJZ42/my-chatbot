# ============================================================
# 第1部分：导入库
# ============================================================
import os
import chromadb                      # 向量数据库，存知识 + 搜知识
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # 读取 .env 里的 DEEPSEEK_API_KEY


# ============================================================
# 第2部分：知识库相关函数
# ============================================================

def create_knowledge_base(texts):
    """
    把一堆文本变成「可搜索的知识库」
    
    参数：
        texts: 列表，每个元素是一段知识
        例如：["C语言是编译型的", "指针存储地址", "malloc在堆上分配内存"]
    
    返回：
        collection: 一个可以搜索的向量数据库对象
    """
    # 创建 chromadb 客户端（数据存在内存里，关程序就没了）
    chroma_client = chromadb.Client()
    
    # 删除旧的同名知识库（如果之前创建过）
    try:
        chroma_client.delete_collection("my_docs")
    except:
        pass  # 没找到就跳过，不影响
    
    # 创建新的知识库（collection = 集合 = 知识库）
    collection = chroma_client.create_collection(
        "my_docs",
        metadata={"hnsw:space": "cosine"}  # 用余弦相似度比较文本
    )
    
    # 把每条知识加进去
    for i, text in enumerate(texts):
        collection.add(
            documents=[text],   # 文本内容
            ids=[f"id_{i}"]     # 每条知识一个唯一ID
        )
    
    return collection


def search_docs(collection, query, top_n=3):
    """
    从知识库里搜索和问题最相关的内容
    
    参数：
        collection: 知识库对象
        query: 用户的问题
        top_n: 返回最相关的几条（默认3条）
    
    返回：
        列表，包含最相关的文本
        例如：["指针存储内存地址...", "malloc在堆上分配...", "int *p声明指针..."]
    """
    results = collection.query(
        query_texts=[query],   # 要搜索的问题
        n_results=top_n        # 返回几条结果
    )
    
    # results 结构：
    # {
    #   'ids': [['id_3', 'id_7', 'id_1']],
    #   'documents': [['文本3', '文本7', '文本1']],
    #   'distances': [[0.23, 0.45, 0.67]]   ← 越小越相关
    # }
    
    return results['documents'][0]  # 取第一个列表（因为query_texts只有一个问题）


# ============================================================
# 第3部分：DeepSeek 客户端
# ============================================================

# @st.cache_resource 的作用：只创建一次客户端，反复使用
# 不加这个的话，每次点按钮都会重新创建，浪费资源
@st.cache_resource
def get_client():
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

client = get_client()


# ============================================================
# 第4部分：Streamlit 网页界面
# ============================================================

# 浏览器标签页设置
st.set_page_config(
    page_title="DeepSeek知识库问答",
    page_icon="📚",
    layout="centered"
)

st.title("📚 DeepSeek 知识库问答机器人")
st.caption("上传你的专属知识 → DeepSeek 严格基于你的文档回答")


# ============================================================
# 第5部分：侧边栏 —— 知识库管理
# ============================================================

with st.sidebar:
    st.header("📄 知识库管理")
    
    # ---- 方式一：自己粘贴文本 ----
    st.subheader("方式一：粘贴文本")
    st.caption("每段知识用 --- 分隔")
    
    raw_text = st.text_area(
        "知识内容",
        height=200,
        placeholder="C语言是编译型语言...\n---\n指针存储内存地址...\n---\nmalloc在堆上分配内存..."
    )
    
    # 加载按钮
    if st.button("✅ 加载我的知识", use_container_width=True):
        if raw_text.strip():
            # 用 --- 分割成多段
            texts = [t.strip() for t in raw_text.split("---") if t.strip()]
            st.session_state.kb = create_knowledge_base(texts)
            st.success(f"✅ 已加载 {len(texts)} 条知识！")
        else:
            st.warning("请先粘贴知识内容")
    
    # ---- 分隔线 ----
    st.divider()
    
    # ---- 方式二：一键加载示例 ----
    st.subheader("方式二：加载示例知识库")
    
    # 两个按钮并排
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Python入门", use_container_width=True):
            sample_python = [
                "Python由Guido van Rossum于1991年首次发布。",
                "Python是解释型语言：代码逐行执行，不需要先编译。",
                "变量命名规则：只能包含字母、数字、下划线，不能以数字开头。",
                "列表list用方括号定义：fruits = ['apple', 'banana']，元素可增删改。",
                "元组tuple用圆括号定义：point = (3, 4)，一旦创建不可修改。",
                "字典dict用花括号：student = {'name': '张三', 'age': 20}。",
                "if语法：if 条件:\n    缩进代码块\nelif 条件:\n    缩进代码块\nelse:\n    缩进代码块",
                "for循环：for item in my_list:\n    print(item)",
                "函数定义：def greet(name):\n    return f'Hello, {name}'",
                "pip安装包：pip install 包名",
                "文件读取：with open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()",
                "异常处理：try:\n    可能出错的代码\nexcept Exception as e:\n    print(f'出错：{e}')",
            ]
            st.session_state.kb = create_knowledge_base(sample_python)
            st.success(f"✅ 已加载 {len(sample_python)} 条Python知识")
    
    with col2:
        if st.button("📥 C语言基础", use_container_width=True):
            sample_c = [
                "C语言由Dennis Ritchie于1972年在贝尔实验室开发。",
                "C语言是编译型语言：源代码需要经过编译、链接才能运行。",
                "C语言基本数据类型：int(整数), float(单精度浮点), double(双精度浮点), char(字符)。",
                "变量声明必须指定类型：int age = 20; float score = 95.5;",
                "printf用于输出：printf('Hello World\\n');",
                "scanf用于输入：scanf('%d', &num); 注意要加&取地址符。",
                "指针是C语言的核心：int *p = &num; 表示p存储num的地址，*p可以访问num的值。",
                "malloc动态分配内存：int *arr = (int*)malloc(5 * sizeof(int)); 用完后要free(arr)。",
                "数组定义：int scores[5] = {90, 85, 77, 92, 88};",
                "for循环语法：for(int i = 0; i < n; i++) { /* 循环体 */ }",
                "函数声明：返回类型 函数名(参数类型 参数名) { 函数体 }",
                "结构体：struct Student { char name[20]; int age; float score; };",
                "文件操作：FILE *fp = fopen('data.txt', 'r'); 用完要fclose(fp)。",
                "字符串是字符数组：char str[20] = 'hello'; 以\\0结尾。",
            ]
            st.session_state.kb = create_knowledge_base(sample_c)
            st.success(f"✅ 已加载 {len(sample_c)} 条C语言知识")
    
    # ---- 状态提示 ----
    if "kb" in st.session_state:
        st.info("📌 知识库已就绪，可以在右侧提问了")


# ============================================================
# 第6部分：主对话区域
# ============================================================

# 初始化对话历史（只有第一次运行时会创建）
if "messages_rag" not in st.session_state:
    st.session_state.messages_rag = []

# 显示所有历史对话（气泡样式）
for msg in st.session_state.messages_rag:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if user_input := st.chat_input("基于知识库提问（例如：list和tuple的区别？）"):
    
    # ---- 显示用户消息 ----
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages_rag.append({"role": "user", "content": user_input})
    
    # ---- AI 回复 ----
    with st.chat_message("assistant"):
        
        # 检查有没有加载知识库
        if "kb" not in st.session_state:
            st.warning("⚠️ 请先在左边侧边栏加载知识库！")
            answer = "请先加载知识库，我才能基于文档回答。"
        
        else:
            # ----- RAG 三步走 -----
            
            # 第①步：检索（Retrieval）
            with st.spinner("🔍 正在知识库中搜索相关内容..."):
                relevant_docs = search_docs(
                    st.session_state.kb, 
                    user_input, 
                    top_n=3
                )
            
            # 第②步：增强（Augmented）—— 拼接提示词
            context = "\n\n".join(relevant_docs)  # 把3条知识拼在一起
            
            prompt = f"""你是一个严格基于文档的问答助手。

## ⚠️ 核心规则（必须遵守）
- 只根据下面「参考资料」中的内容回答问题
- 如果参考资料中没有相关信息，直接回复："抱歉，知识库中没有找到相关内容"
- 引用原文时用引号标出
- 回答要条理清晰，分点列出

## 参考资料
{context}

## 用户问题
{user_input}

## 你的回答："""
            
            # 第③步：生成（Generation）—— 调用DeepSeek
            with st.spinner("💭 DeepSeek 正在基于文档生成回答..."):
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,      # 低温度 = 更严谨
                    max_tokens=2048,
                )
                answer = response.choices[0].message.content
            
            # 显示回答
            st.markdown(answer)
            
            # 显示参考来源（可折叠）
            with st.expander("📖 查看参考的知识片段"):
                for i, doc in enumerate(relevant_docs, 1):
                    st.caption(f"**片段 {i}：** {doc}")
    
    # 保存 AI 回答到历史
    st.session_state.messages_rag.append({"role": "assistant", "content": answer})
