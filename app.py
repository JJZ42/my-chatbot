import os
import io
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# PDF / Word 文本提取
from PyPDF2 import PdfReader
from docx import Document

load_dotenv()

st.set_page_config(
    page_title="DeepSeek 知识库助手",
    page_icon="📚",
    layout="centered"
)

# ===== 🔐 密码保护 =====
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "jjztaishuaile")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 私人AI助手")
    st.caption("请输入密码访问")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        if password == ACCESS_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("密码错误！")
    st.stop()

# ============================================
st.title("📚 DeepSeek 知识库问答机器人")
st.caption("上传专属知识 → DeepSeek 基于你的文档回答 · 私人版")

@st.cache_resource
def get_client():
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

client = get_client()

# ============================================
# 知识库函数（用 TF-IDF 代替 chromadb）
# ============================================

@st.cache_resource
def compute_tfidf(texts_tuple):
    """缓存 TF-IDF 计算结果。texts_tuple 必须是 tuple（hashable）。
    
    使用 char_wb + 1~3 字 n-gram，确保中文检索精度。
    旧版 TfidfVectorizer() 默认按空格分词，中文全挤成1个token → 全0分。
    """
    texts = list(texts_tuple)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3))
    doc_vectors = vectorizer.fit_transform(texts)
    return vectorizer, doc_vectors, texts


def search_docs(kb_data, query, top_n=3):
    """搜索最相关的内容"""
    vectorizer, doc_vectors, texts = kb_data
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [texts[i] for i in top_indices]

# ============================================
# 文件提取函数
# ============================================

def extract_text_from_pdf(file_bytes):
    """从 PDF 字节流提取文本"""
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes):
    """从 Word .docx 字节流提取文本"""
    doc = Document(io.BytesIO(file_bytes))
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    return "\n\n".join(text_parts)


def extract_text_from_file(uploaded_file):
    """根据文件类型自动选择提取方式"""
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:
        return None


def split_into_chunks(text, min_len=20):
    """将长文本按段落切分为知识片段
    
    - 先用双换行（段落间空行）切分
    - 过滤掉过短的片段
    - 每个片段作为一条知识
    """
    # 规范化换行：多个连续换行统一为双换行
    import re
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text)  # 单换行也升级为双换行
    # 按段落切分
    raw_chunks = text.split("\n\n")
    chunks = []
    for ch in raw_chunks:
        ch = ch.strip()
        # 跳过空段和过短段
        if len(ch) >= min_len:
            # 如果单段太长（>500字），按句号再切一次
            if len(ch) > 500:
                sub_chunks = [s.strip() + "。" for s in ch.split("。") if s.strip()]
                # 句子级别阈值（中文短句常见 4~8 字）
                sub_min = 3
                for sc in sub_chunks:
                    if len(sc) >= sub_min:
                        chunks.append(sc)
            else:
                chunks.append(ch)
    return chunks


# ============================================
# 侧边栏
# ============================================

with st.sidebar:
    st.header("📄 知识库管理")
    
    # ---- 初始化 ----
    if "knowledge_bases" not in st.session_state:
        st.session_state.knowledge_bases = {}  # {name: [texts]}
    if "active_kb" not in st.session_state:
        st.session_state.active_kb = None
    if "messages_rag" not in st.session_state:
        st.session_state.messages_rag = []
    
    # ---- 当前使用的知识库 ----
    if st.session_state.knowledge_bases:
        st.subheader("📌 当前知识库")
        kb_names = list(st.session_state.knowledge_bases.keys())
        if st.session_state.active_kb not in kb_names:
            st.session_state.active_kb = kb_names[0]
        
        selected = st.selectbox(
            "选择知识库",
            kb_names,
            index=kb_names.index(st.session_state.active_kb),
            key="kb_selector"
        )
        if selected != st.session_state.active_kb:
            st.session_state.active_kb = selected
            st.session_state.messages_rag = []
            st.rerun()
        
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.caption(f"共 {len(st.session_state.knowledge_bases[st.session_state.active_kb])} 条知识")
        with col_b:
            if st.button("🗑️", key="del_kb", help="删除当前知识库", use_container_width=True):
                del st.session_state.knowledge_bases[st.session_state.active_kb]
                st.session_state.active_kb = list(st.session_state.knowledge_bases.keys())[0] if st.session_state.knowledge_bases else None
                st.session_state.messages_rag = []
                st.rerun()
    else:
        st.info("👆 还没有加载知识库，请在下方添加")
    
    st.divider()
    
    st.subheader("方式一：粘贴文本")
    st.caption("每段知识用 --- 分隔")
    
    kb_name = st.text_input("知识库名称", placeholder="例如：我的学习笔记")
    raw_text = st.text_area(
        "知识内容",
        height=150,
        placeholder="C语言是编译型语言...\n---\n指针存储内存地址...\n---\nmalloc在堆上分配内存..."
    )
    
    if st.button("✅ 加载我的知识", use_container_width=True):
        if raw_text.strip():
            texts = [t.strip() for t in raw_text.split("---") if t.strip()]
            name = kb_name.strip() or "我的知识"
            st.session_state.knowledge_bases[name] = texts
            if not st.session_state.active_kb:
                st.session_state.active_kb = name
            st.success(f"✅ 已加载 {len(texts)} 条知识到「{name}」！")
        else:
            st.warning("请先粘贴知识内容")
    
    st.divider()
    st.subheader("方式二：加载示例")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Python入门", use_container_width=True):
            sample_python = [
                "Python 由 Guido van Rossum 于 1991 年发布，设计哲学是「可读性至上」。用缩进（4 个空格）代替大括号划分代码块。PEP 8 是官方风格指南：变量和函数用 snake_case，类名用 PascalCase，常量用 UPPER_CASE。Python 是解释型、动态类型、强类型语言——变量不需要声明类型，但不同类型不会自动转换（如 '5'+3 报 TypeError）。",
                "Python 中一切皆对象。每个对象有类型 type、值 value、身份 id。`a is b` 比较 id（是否同一对象），`a == b` 比较值。小整数（-5 到 256）和短字符串会被缓存复用。`del x` 只删除变量名引用，对象由垃圾回收器（引用计数+循环检测）自动回收。`sys.getrefcount(obj)` 查看引用计数。CPython 有 GIL（全局解释器锁），导致 CPU 密集型多线程无法利用多核，IO 密集型多线程仍有加速效果。",
                "Python 数字：int 无限精度（Python 3 统一了 int 和 long），float 是 64 位 IEEE 754 双精度（约 15 位有效数字），complex 如 `3+4j`。运算：`+ - *` 结果为 int，`/` 一定返回 float，`//` 地板除（向下取整，-5//2=-3），`%` 取余（结果符号同除数），`**` 幂运算。`divmod(a,b)` 同时返回商和余。`round(x, n)` 四舍五入到 n 位小数（注意银行家舍入）。`abs(x)` 绝对值，`pow(x,y,z)` 高效计算 (x**y) % z。`int('101', 2)` 二进制字符串转整数。",
                "float 精度问题：0.1 + 0.2 != 0.3（因为 0.1 在二进制中是无限循环小数）。比较浮点数用 `math.isclose(a, b, rel_tol=1e-9)` 或 `abs(a-b) < 1e-9`。float 特殊值：`float('inf')` 正无穷，`float('-inf')` 负无穷，`float('nan')` 非数字（nan != nan 为 True）。`math.isfinite(x)` 判断是否有限。decimal 模块提供高精度十进制运算（适合金融），fractions 模块提供有理数精确运算。",
                "Python 字符串是不可变 Unicode 序列。定义方式：单引号 'a'、双引号 \"a\"、三引号 '''a''' 可跨行。原始字符串 r'\\n' 不转义。字节字符串 b'hello' 是 bytes 类型（0-255 的整数序列）。f-string（Python 3.6+）是推荐格式化方式：`f'{name} 今年 {age} 岁，均分 {sum(s)/len(s):.2f}'`。`f'{x=}'` 输出调试格式。`f'{num:0>5d}'` 左侧补零到 5 位整数。`f'{num:,}'` 千位分隔符。",
                "字符串方法：`.upper()`, `.lower()`, `.title()`, `.capitalize()`, `.swapcase()` 大小写转换。`.strip()`, `.lstrip()`, `.rstrip()` 去空白（可指定字符如 `.strip('.,!')`）。`.split(sep, maxsplit)` 分割为列表，`.rsplit()` 从右分割。`sep.join(iterable)` 拼接（比循环 += 快得多）。`.replace(old, new, count)` 替换。`.find(sub)` 返回索引（-1 表示没找到），`.index(sub)` 没找到抛 ValueError。`.startswith(prefix)`, `.endswith(suffix)`（可传元组匹配多个）。`.count(sub)` 统计次数。`.isalpha()`, `.isdigit()`, `.isalnum()`, `.isspace()`, `.isupper()`, `.islower()`。`.encode('utf-8')` 字符串→bytes，`.decode('utf-8')` bytes→字符串。",
                "列表 list 是可变的序列，用 `[]` 定义。方法：`.append(x)` 末尾追加一个，`.extend(iterable)` 追加多个，`.insert(i, x)` 在位置 i 插入，`.remove(x)` 删除第一个值为 x 的元素（找不到报错），`.pop(i)` 删除并返回索引 i 的元素（默认最后），`.clear()` 清空，`.index(x, start, end)` 查找位置，`.count(x)` 统计出现次数，`.sort(key=..., reverse=True)` 原地排序（稳定排序），`.reverse()` 原地反转，`.copy()` 浅拷贝（等价于 `lst[:]`）。`len(lst)` 长度，`x in lst` 成员检查，`lst1 + lst2` 拼接返回新列表。",
                "列表切片 `lst[start:end:step]` 返回新列表（浅拷贝）。省略 start 默认 0，省略 end 默认末尾，省略 step 默认 1。`lst[::-1]` 反转，`lst[::2]` 隔一个取一个。切片可赋值：`lst[1:3] = [10,20,30]` 替换一段（长度可以不相等）。`del lst[1:3]` 删除一段。列表推导式：`[x**2 for x in range(10) if x%2==0]` 比 for+append 快且更 pythonic。嵌套推导式：`[y for row in matrix for y in row]` 展平二维列表。",
                "元组 tuple 是不可变序列，用 `()` 定义。单元素元组必须加逗号：`(1,)`。元组不可变意味着不能增删改元素，但元素如果是可变对象则该对象内容可改。元组可哈希（可作为字典键和集合元素）。元组拆包：`a, b, *rest = (1, 2, 3, 4)` 得到 a=1, b=2, rest=[3,4]。`namedtuple`（collections 模块）创建具名元组：`Point = namedtuple('Point', ['x', 'y'])` 后可用 `p.x` 访问。",
                "字典 dict 是键值对映射，Python 3.7+ 保证插入顺序。键必须可哈希，值任意。常用操作：`d[key]`（键不存在报 KeyError），`d.get(key, default)` 安全取值，`d.setdefault(key, default)` 有则返回无则设置，`d.pop(key, default)` 删除并返回值，`d.popitem()` 删除并返回最后一对（LIFO）。`d1 | d2` 合并（3.9+），`{**d1, **d2}` 合并（3.5+）。`d.keys()`、`d.values()`、`d.items()` 返回动态视图。字典推导式：`{x: x**2 for x in range(5)}`。defaultdict 自动为不存在的键创建默认值，Counter 统计可哈希对象频次。",
                "集合 set 是无序不重复的可变集合，`{}` 或 `set()` 定义（空集合只能用 `set()`）。集合运算：`a|b` 并集，`a&b` 交集，`a-b` 差集，`a^b` 对称差集。方法：`.add(x)`, `.remove(x)`, `.discard(x)`, `.pop()`, `.clear()`, `.update(iterable)`, `.isdisjoint(other)`, `.issubset(other)`, `.issuperset(other)`。frozenset 是不可变集合，可哈希。集合推导式：`{x for x in seq if cond}`。",
                "条件判断：`if cond:` → `elif cond:` → `else:`。条件可以是任意对象——0、''、[]、{}、None、False 视为假，其他视为真。`and` 和 `or` 短路求值：`a or b` 中 a 为真立即返回 a 不计算 b，`a and b` 中 a 为假立即返回 a。`not` 取反。三元表达式：`x if cond else y`。match-case（Python 3.10+）是模式匹配：可匹配类型、解构序列、带 if 守卫。`case _:` 是默认分支。",
                "for 循环遍历任何可迭代对象。`for i in range(n):` 循环 n 次。`range(start, stop, step)` 是惰性不可变序列。`enumerate(iterable, start=0)` 同时获取索引和值。`zip(*iterables, strict=False)`（Python 3.10+ strict 参数在长度不等时抛错）并行遍历。`reversed(seq)` 反向遍历。`sorted(iterable, key=..., reverse=...)` 返回排序后的新列表。for-else 语法：循环被 break 跳出时不执行 else，正常结束（包括空循环）时执行。",
                "while 循环：`while cond:`。`break` 跳出整个循环，`continue` 跳过本轮剩余部分。`while True:` 常见死循环，内部必须 break 退出。while-else：被 break 跳出时不执行 else，条件变为假时执行。`pass` 是空语句占位符，什么也不做。",
                "函数定义：`def func(param1, param2=default):`。参数顺序：必选参数 → *args（可变位置参数，打包为元组）→ 仅限关键字参数（放在 * 或 *args 之后）→ **kwargs（可变关键字参数，打包为字典）。仅限关键字参数强制调用者使用参数名。`/` 表示前面的参数只能按位置传递（Python 3.8+）。默认参数在函数定义时求值一次，所以默认值不要用可变对象，应用 None 然后在函数内判断。",
                "Python 传参方式是「传对象引用」：不可变对象的修改不影响外部，可变对象的原地修改影响外部。函数内部对参数的重新赋值（如 `lst = [1,2]`）不会影响外部。`return` 可返回多个值（打包为元组），无 return 或 `return` 时返回 None。函数是第一类对象：可以赋值给变量、作为参数传递、作为返回值。嵌套函数可以访问外部函数变量（闭包），`nonlocal x` 可修改外部函数变量。",
                "lambda 是匿名单表达式函数：`lambda x, y: x + y`。不能包含语句和赋值，只能包含一个表达式。常用于 `sorted(lst, key=lambda x: x.score)` 等需要简短短回调的场景。`map(func, iterable)` 对每个元素调用 func，返回 map 对象（可用 list() 转换）。`filter(func, iterable)` 保留 func 返回 True 的元素。`functools.reduce(func, iterable, initial)` 累积计算。列表推导式通常比 map/filter 更可读。",
                "装饰器本质是 `@decorator` 等价于 `func = decorator(func)`。`functools.wraps(func)` 保留原函数的 `__name__`、`__doc__` 等元信息。带参数的装饰器需要三层嵌套：外层接收装饰器参数，中层接收函数，内层返回包装函数。内置装饰器：`@staticmethod`、`@classmethod`（第一个参数是 cls）、`@property`（将方法转为属性）。`@functools.lru_cache(maxsize=128)` 缓存函数返回值加速重复调用。`@functools.singledispatch` 基于第一个参数类型进行函数重载。",
                "Python 变量作用域 LEGB：Local→Enclosing→Global→Built-in。函数内部读取外层变量不需要声明，但赋值操作会创建新的局部变量（除非用 `global` 或 `nonlocal` 声明）。`global x` 用于在函数内修改全局变量。`nonlocal x` 用于在嵌套函数内修改外层函数的变量。`globals()` 返回全局变量字典，`locals()` 返回局部变量字典。",
                "生成器用 `yield` 定义，惰性产生值，节省内存。每次 `next(gen)` 执行到下一个 yield 暂停并返回值。`gen.send(value)` 发送值给生成器（在 yield 表达式接收），`gen.throw(exc)` 抛出异常，`gen.close()` 终止。`yield from iterable` 将生成委托给子生成器。生成器表达式用圆括号：`(x**2 for x in range(10))`。生成器只能迭代一次，用完即空。",
                "迭代器协议：可迭代对象实现 `__iter__()` 返回迭代器；迭代器实现 `__iter__()` 和 `__next__()` 方法。`next(it)` 取下一个值，取完抛 StopIteration。for 循环底层就是 `iter()` + `next()` 直到 StopIteration。`iter(obj, sentinel)` 创建哨兵迭代器，每次调用 callable 直到返回值等于 sentinel。`itertools` 模块提供丰富的迭代器工具。",
                "类定义：`class Dog:`。`__init__(self, ...)` 是初始化方法（对象已由 `__new__` 创建）。实例属性用 `self.xxx`，类属性直接在 class 内定义（所有实例共享）。self 不是关键字只是约定，实例方法第一个参数必须是 self。`__new__(cls, ...)` 是真正的构造器（很少重写），用于控制对象创建（如单例模式）。`__del__(self)` 是析构器，在对象被垃圾回收时调用（不推荐依赖它做资源清理，用 with 语句更好）。",
                "Python 没有真正的私有成员。`_name` 单下划线前缀表示「受保护」（约定，外部仍可访问）。`__name` 双下划线前缀触发名称改写 name mangling，变成 `_ClassName__name`，用于避免子类意外覆盖。`__xxx__` 双下划线包围是魔术方法，不应自己定义这种名称。`@property` 装饰器将方法转为属性：定义 getter（@property）、setter（@x.setter）、deleter（@x.deleter）。",
                "继承：`class Child(Parent):`。`super().__init__(args)` 调用父类初始化。Python 支持多继承，方法解析顺序 MRO 使用 C3 线性化，`cls.__mro__` 查看。钻石继承问题：`class D(B,C)` 两个父类继承自 A，MRO 保证 A 只执行一次且顺序正确。抽象基类（ABC）：`from abc import ABC, abstractmethod` ——`class Shape(ABC):` 含 `@abstractmethod` 的方法子类必须实现。",
                "魔术方法一览：`__str__` 用户友好字符串（print/str 调用，应可读），`__repr__` 开发者表示（应尽量能 `eval(repr(obj))` 还原），`__eq__` 定义 == 行为（定义 __eq__ 会丢掉继承的 __hash__，需显式设置），`__hash__` 使对象可哈希，`__len__` 支持 len()，`__bool__` 支持 bool()，`__getitem__/__setitem__/__delitem__` 支持索引，`__iter__` 和 `__next__` 使对象可迭代，`__enter__/__exit__` 支持 with 语句，`__call__` 使实例可调用，`__add__/__sub__/__mul__` 等运算符重载，`__lt__/__le__/__gt__/__ge__` 比较。`functools.total_ordering` 装饰器补全缺少的比较方法。",
                "异常处理完整语法：`try:` → `except SomeError as e:` → `except (A, B):` → `except Exception as e:` → `else:`（无异常执行）→ `finally:`（总是执行，即使 try 中有 return/break/continue 也会在退出前执行）。不要裸写 `except:`（不指定异常类型），这会连 KeyboardInterrupt 和 SystemExit 一起捕获。`raise` 不带参数在 except 块中重新抛出当前异常（保留原始 traceback）。`raise NewError(...) from original_exc` 进行异常链式转换。",
                "常见内置异常层级：BaseException → SystemExit, KeyboardInterrupt, Exception。Exception 子类：ArithmeticError → ZeroDivisionError, OverflowError；LookupError → IndexError, KeyError；ValueError, TypeError, AttributeError, FileNotFoundError, PermissionError, IsADirectoryError, ImportError, ModuleNotFoundError, StopIteration, StopAsyncIteration, RuntimeError, NotImplementedError。自定义异常：`class MyError(Exception): pass`，可添加 `__init__` 接收额外参数如错误码。",
                "上下文管理器：实现 `__enter__()` 和 `__exit__(exc_type, exc_val, exc_tb)`。`__exit__` 返回 True 可抑制异常。`contextlib.contextmanager` 装饰器可将生成器函数转为上下文管理器（yield 前是 enter，yield 后是 exit）。`contextlib.closing(obj)` 将 `.close()` 方法转为上下文。`contextlib.suppress(SomeError)` 抑制指定异常。`contextlib.ExitStack` 动态管理多个上下文。",
                "文件操作最佳实践：`with open('file.txt', mode, encoding='utf-8') as f:`。模式：`r` 只读，`w` 写（清空），`a` 追加，`x` 排他创建，`+` 读写，`b` 二进制，`t` 文本（默认）。组合：`rb` 二进制读，`w+` 读写清空。读取：`f.read(size)`, `f.readline()`, `f.readlines()`, `for line in f:`。写入：`f.write(s)`, `f.writelines(lines)`（不会自动加换行符）。`f.seek(offset, whence)` 移动指针（SEEK_SET=0, SEEK_CUR=1, SEEK_END=2），`f.tell()` 返回当前位置。",
                "pathlib（Python 3.4+）是面向对象路径处理库：`Path('a') / 'b' / 'c.txt'` 拼接路径。`.read_text(encoding='utf-8')`, `.read_bytes()`, `.write_text(data)`, `.exists()`, `.is_file()`, `.is_dir()`, `.glob('*.py')`, `.rglob('**/*.py')` 递归搜索，`.mkdir(parents=True, exist_ok=True)`, `.unlink()` 删除文件，`.rmdir()` 删除空目录，`.rename(target)`, `.stat()` 获取文件信息，`.with_suffix('.new')` 改后缀，`.name` 文件名，`.stem` 无后缀名，`.suffix` 后缀名。",
                "CSV 处理：`import csv` → `csv.reader(f)` 返回各行列表，`csv.DictReader(f)` 返回字典列表（第一行作键名）。`csv.writer(f)` 写各行，`csv.DictWriter(f, fieldnames)` 写字典。JSON：`json.dumps(obj)` 序列化，`json.loads(s)` 反序列化，`json.dump(obj, f)` 写文件，`json.load(f)` 读文件。`indent=2` 格式化输出，`ensure_ascii=False` 保留中文。datetime 不能直接 JSON 化，需写自定义 encoder。pickle 模块用于 Python 对象序列化（不安全，不要 unpickle 不可信数据）。",
                "collections 模块：`namedtuple` 具名元组，`defaultdict` 有默认值的字典（`defaultdict(int)` 访问不存在的键返回 0），`Counter` 计数器（`c.most_common(n)` 频率最高的 n 个元素，`c1 + c2` 合并计数），`deque` 双端队列（`appendleft/popleft` O(1)），`OrderedDict` 有序字典（保持插入顺序），`ChainMap` 多字典逻辑合并（查找时依次搜索）。",
                "itertools 模块：`chain(*iters)` 串联，`cycle(it)` 无限循环，`repeat(x, n)` 重复，`count(start, step)` 无限计数，`islice(it, start, stop, step)` 切片，`takewhile(pred, it)/dropwhile(pred, it)` 按条件取/丢，`groupby(it, key)` 按 key 分组（需预先排序），`product(*iters)` 笛卡尔积，`permutations(it, r)` 排列，`combinations(it, r)` 组合（不放回），`combinations_with_replacement(it, r)` 组合（放回），`zip_longest(*iters, fillvalue=...)` 最长拉齐。",
                "functools 模块：`reduce(func, seq)` 累积计算，`partial(func, *args, **kwargs)` 部分应用参数，`lru_cache(maxsize=128)` 和 `cache`（无上限 LRU）记忆化缓存，`total_ordering` 补全比较方法（只需定义 __eq__ 和一个比较），`wraps(func)` 保留被装饰函数的元信息，`singledispatch` 基于第一个参数类型的函数重载，`cached_property` 缓存属性值（计算一次后不再计算）。",
                "datetime 模块：`datetime.now()` 当前时间，`datetime(year, month, day, hour, minute, second)` 构造，`strftime('%Y-%m-%d %H:%M:%S')` 格式化，`strptime(s, fmt)` 解析字符串。`timedelta(days=7, hours=2)` 时间差。`date.today()` 今天日期。`time.time()` 返回 Unix 时间戳（秒），`time.sleep(seconds)` 暂停，`time.perf_counter()` 高精度计时（用于性能测量，不受系统时间调整影响）。",
                "os/os.path 和 shutil 模块：`os.getcwd()` 当前工作目录，`os.chdir(path)` 切换目录，`os.listdir(path)` 列出目录内容，`os.mkdir(path)` 创建目录，`os.makedirs(path, exist_ok=True)` 创建多层目录，`os.remove(path)` 删除文件，`os.rmdir(path)` 删除空目录，`os.rename(src, dst)` 重命名/移动，`os.environ` 环境变量字典，`os.system(cmd)` 执行 shell 命令。`shutil.copy(src, dst)`, `shutil.copytree(src, dst)`, `shutil.rmtree(path)` 递归删除，`shutil.move(src, dst)` 移动。",
                "正则表达式 re 模块：`re.search(pattern, text)` 搜索第一个匹配，`re.match(pattern, text)` 从开头匹配，`re.findall(pattern, text)` 返回所有匹配字符串列表，`re.sub(pattern, repl, text)` 替换，`re.split(pattern, text)` 分割，`re.compile(pattern)` 预编译提高效率。常用模式：`.` 任意字符，`\\d` 数字，`\\w` 单词字符，`\\s` 空白，`*` 0或多，`+` 1或多，`?` 0或1，`{n}` 恰好n，`[]` 字符集，`[^]` 排除，`^` 行首，`$` 行尾，`( )` 分组。",
                "threading 模块：`t = Thread(target=func, args=(a,b))` 创建线程，`t.start()` 启动，`t.join(timeout)` 等待结束。GIL 导致 CPU 密集任务无法利用多核，对 IO 密集有效。`Lock()` 互斥锁：`with lock:` 上下文管理。`RLock()` 可重入锁。`Semaphore(n)` 信号量限制并发数。`Event()` 线程间通信。`Condition()` 条件变量。`local()` 线程本地存储。",
                "multiprocessing 模块绕过 GIL 利用多核：`p = Process(target=func, args=(a,b))`，`p.start()`, `p.join()`。`Pool(n)` 进程池：`pool.map(func, iterable)` 并行映射。`Queue()` 进程间通信，`Pipe()` 双向管道，`Manager()` 共享数据容器。`Value(typecode, value)` 和 `Array(typecode, sequence)` 共享内存。",
                "asyncio（Python 3.4+）是单线程异步 IO：`async def func(): await some_io()`。`asyncio.run(main())` 运行主协程。`asyncio.gather(*tasks)` 并发运行多个协程。`asyncio.wait_for(coro, timeout)` 超时控制。`asyncio.Queue` 协程间通信，`asyncio.Lock`/`asyncio.Semaphore` 协程同步原语。适合高并发网络请求，不适合 CPU 密集任务。",
                "pip 是 Python 包管理器：`pip install pkg`, `pip install -r requirements.txt`, `pip uninstall pkg`, `pip list`, `pip freeze > requirements.txt`。国内镜像加速：`-i https://pypi.tuna.tsinghua.edu.cn/simple`。venv 虚拟环境：`python -m venv myenv` 创建，激活后 pip install 的包只在该环境生效。",
                "pytest 是推荐测试框架：测试文件 `test_xxx.py`，测试函数 `def test_xxx():`，用 `assert` 判断。`pytest -v` 详细输出。`@pytest.fixture` 定义测试夹具。`@pytest.mark.parametrize` 参数化测试。`pytest.raises(SomeError)` 断言异常。",
                "调试技术：`print()` 最基础。`assert condition, message` 开发阶段验证。`logging` 模块分级输出：`logging.debug/info/warning/error/critical`。`breakpoint()`（Python 3.7+）进入 pdb 调试。`traceback.print_exc()` 打印异常堆栈。",
                "描述符协议：实现 `__get__(self, instance, owner)`, `__set__(self, instance, value)`, `__delete__(self, instance)` 的对象是描述符。property、classmethod、staticmethod 都是描述符的实现。数据描述符（有 __set__）优先级高于实例字典。",
                "元类 metaclass 是类的「类」：`class Meta(type): def __new__(meta, name, bases, namespace): ...`。使用：`class MyClass(metaclass=Meta):`。`type(name, bases, namespace)` 以编程方式创建类。大多数情况下不需要使用元类。",
                "Python 的类型注解（Type Hints）：`def greet(name: str) -> str:`。`typing` 模块：`List[int]`, `Optional[int]` = `int | None`（3.10+），`Union[int, str]`，`Callable[[int, int], str]`，`Any`，`Literal['a', 'b']`。`dataclasses.dataclass` 简化数据类定义。",
                "内存视图和 buffer 协议：`memoryview(obj)` 高效零拷贝切片访问。`ctypes` 调用 C 动态库。`struct` 模块打包/解包 C 结构体。`mmap` 内存映射文件。",
            ]
            st.session_state.knowledge_bases["Python入门"] = sample_python
            if not st.session_state.active_kb:
                st.session_state.active_kb = "Python入门"
            st.success(f"✅ 已加载 {len(sample_python)} 条Python知识")
    
    with col2:
        if st.button("📥 C语言基础", use_container_width=True):
            sample_c = [
                "C 语言由 Dennis Ritchie 于 1972 年在贝尔实验室开发，用于重写 Unix 操作系统。C 标准演进：K&R C → C89/C90（ANSI C）→ C99（引入变长数组、单行注释 //、for 内声明变量）→ C11（引入 _Generic、_Atomic、多线程）→ C17（修复 C11 缺陷）→ C23（最新标准）。每个 C 程序从 main 函数开始执行：`int main(void)` 或 `int main(int argc, char *argv[])`。argc 是命令行参数个数，argv 是参数字符串数组。return 0 表示正常退出。",
                "C 编译过程四阶段：①预处理（cpp）展开 `#include` 和 `#define`，生成 .i 文件（gcc -E）；②编译（cc1）将预处理后的代码转为汇编 .s 文件（gcc -S）；③汇编（as）将汇编转为目标文件 .o（gcc -c）；④链接（ld）合并多个 .o 和库文件生成可执行文件。GCC 常用选项：`-Wall -Wextra` 开启所有警告，`-Werror` 把警告当错误，`-g` 包含调试信息，`-O2` 优化级别，`-std=c11` 指定标准版本，`-lm` 链接数学库。",
                "C 语言基本数据类型及大小（64 位系统）：`char` 1 字节，`short` 2 字节，`int` 4 字节，`long` 8 字节（Windows 64 位下仍 4 字节），`long long` 8 字节，`float` 4 字节（约 7 位有效数字），`double` 8 字节（约 15 位），`long double` 16 字节。`sizeof(type)` 返回字节数（size_t 类型，用 %zu 打印）。精确宽度类型在 `<stdint.h>`：`int8_t`, `uint32_t`, `int64_t` 等。`<stdbool.h>`（C99）提供 bool 类型。",
                "有符号和无符号：`unsigned` 修饰整数类型使所有位用于表示非负数（范围扩大一倍）。有符号和无符号混用时，有符号自动转为无符号（可能导致意外结果：`-1 > 1U` 为 true）。size_t 是无符号类型，循环中用 size_t 倒序遍历数组时注意：`for (size_t i = n; i-- > 0;)` 而不是 `for (size_t i = n-1; i >= 0; i--)`（后者死循环！）。ptrdiff_t 是有符号类型，存储两个指针之差。",
                "浮点类型细节：float 和 double 基于 IEEE 754 标准。0.1 在二进制中无限循环，所以浮点数不是精确的。浮点字面量默认是 double，要 float 需加 f 后缀：`3.14f`。不可用 `==` 比较浮点数，应用 `fabs(a-b) < epsilon`。特殊值：`INFINITY`（无穷大）、`NAN`（非数字）。所有三角函数用弧度制，编译时需 `-lm`。",
                "变量声明规则：必须先声明后使用。变量名只能包含字母、数字和下划线，不能以数字开头，区分大小写。局部变量必须显式初始化，否则值是未定义的（垃圾值）。全局变量默认初始化为 0/NULL。C99 起允许 in for 初始化部分声明变量：`for (int i = 0; i < n; i++)`。",
                "存储类关键字：①`auto`（默认）几乎从不显式使用；②`register` 建议变量放寄存器（C11 后主要保留为语法，不能取地址）；③`static` 修饰局部变量：生命周期延长到程序结束（保留上次值），修饰全局变量/函数：限制为文件内部可见（内部链接）；④`extern` 声明引用其他文件的全局变量或函数。`const` 声明不可修改的变量：`const int MAX = 100;` 比 `#define MAX 100` 有类型检查、可被调试器看到。",
                "C 运算符优先级（从高到低）：① `() [] -> .`（后缀）② `! ~ ++ -- + - * & (type) sizeof`（一元）③ `* / %`（乘除取余）④ `+ -`（加减）⑤ `<< >>`（移位）⑥ `> >= < <=`（关系）⑦ `== !=`（相等）⑧ `&`（按位与）⑨ `^`（按位异或）⑩ `|`（按位或）⑪ `&&`（逻辑与）⑫ `||`（逻辑或）⑬ `?:`（条件）⑭ `=` 等（赋值）⑮ `,`（逗号）。不确定时用括号。",
                "常见运算符陷阱：① `=` 赋值 vs `==` 比较——`if (x = 5)` 永远为真！写 `if (5 == x)` 可防漏写等号；② 整数除法截断：`5/2 = 2`；③ `%` 取余结果符号与被除数相同：`-5 % 2 = -1`；④ `++i` vs `i++` 在复杂表达式中不要用多次自增自减；⑤ `&&` 和 `||` 短路求值——`p && *p` 在 p 为 NULL 时安全；⑥ `sizeof` 是运算符不是函数：`sizeof *p` 返回 int 大小而非指针大小。",
                "位运算详解：`&` 按位与（全 1 得 1），`|` 按位或（有 1 得 1），`^` 按位异或（不同得 1），`~` 按位取反，`<< n` 左移 n 位（低位补 0，等价乘以 2^n），`>> n` 右移 n 位（unsigned 逻辑右移高位补 0，signed 由实现定义）。奇技淫巧：`n & 1` 判断奇偶，`n & (1 << k)` 检查第 k 位，`x & (x-1)` 清除最低位的 1。",
                "printf 格式说明符完整表：`%d/%i` 有符号 int；`%u` 无符号；`%x/%X` 十六进制；`%o` 八进制；`%f` 浮点（默认6位小数）；`%e/%E` 科学计数法；`%c` 字符；`%s` 字符串；`%p` 指针地址；`%%` 百分号。修饰符：`%5d` 最小宽度5，`%-5d` 左对齐，`%05d` 0填充，`%.2f` 2位小数。长度修饰：`l`（long），`ll`（long long）。`%zu` 打印 size_t，`%td` 打印 ptrdiff_t。",
                "scanf 的陷阱和正确用法：`scanf('%d', &num)`——每个变量前必须加 `&`（数组名和指针变量除外）。`%s` 不检查缓冲区大小，是经典安全漏洞。解决：`scanf('%19s', str)` 限制字段宽度。scanf 读取数值后缓冲区残留换行符——之后 fgets 会立即读到空行。解决：`while (getchar() != '\\n');` 清空缓冲区。推荐：统一用 `fgets(str, sizeof(str), stdin)` 读整行，再用 `sscanf` 解析。",
                "文件操作完整流程：`FILE *fp = fopen('file.txt', 'r');` 打开（返回 NULL 表示失败）。`fclose(fp);` 关闭。`fread/fwrite` 读写二进制。`fprintf/fscanf` 格式化读写。`fgets/fputs` 行读写。`fgetc/fputc` 字符读写。`feof(fp)` 检测文件尾，`ferror(fp)` 检测错误。`fseek(fp, offset, whence)` 移动指针（SEEK_SET=0, SEEK_CUR=1, SEEK_END=2）。`ftell(fp)` 返回当前位置。",
                "一维数组：`int arr[5] = {1,2,3};` 未指定元素自动为 0。`int arr[] = {1,2,3};` 编译器推断长度。`sizeof(arr)/sizeof(arr[0])` 计算元素个数（仅对原数组有效！指针参数无效）。数组名在大多数表达式中退化为指向第一个元素的指针。`arr[i]` 等价于 `*(arr + i)`。例外：`sizeof(arr)` 不退化为指针，`&arr` 是 `type (*)[N]` 类型。",
                "多维数组：`int m[2][3];` 内存按行优先连续存储。初始化：`int m[2][3] = {{1,2,3}, {4,5,6}};` 或 `{1,2,3,4,5,6}`。作函数参数必须指定除第一维外的所有维度：`void func(int arr[][3], int rows)` 或 `void func(int (*arr)[3], int rows)`。变长数组 VLA（C99）：数组长度可以是运行时变量（C11 改为可选特性，不建议用于生产代码）。",
                "C 字符串是 char 数组，以空字符 `'\\0'` 结尾。`char s[] = \"hello\";` 在栈上分配可修改的数组（6字节含 \\0）。`char *s = \"hello\";` 指向只读数据段中的字符串字面量，修改内容导致未定义行为。`strlen(s)` 返回长度（不含 \\0），O(n)遍历——不要放在循环条件中！`sizeof(s)` 在数组上返回总字节数，在指针上返回指针大小。",
                "<string.h> 函数及安全问题：`strcpy(dest, src)` 不检查 dest 大小→缓冲区溢出。`strncpy(dest, src, n)` 最多复制 n 个字符，但如果 src 长度 ≥ n，dest 不会以 \\0 结尾！必须手动加 `dest[n-1] = '\\0';`。推荐用 `snprintf(dest, size, '%s%s', a, b)` 替代。`strcmp(a, b)` =0 相等。`strchr(s, c)` 查找字符。`strstr(haystack, needle)` 查找子串。`memcpy/memmove/memset/memcmp` 操作原始内存。",
                "字符串与数字转换：`atoi/atol/atof` 简单但不建议——无法区分 '0' 和转换失败。推荐 `strtol()` 系列：`long strtol(const char *s, char **endptr, int base);` 转换后 *endptr 指向第一个未转换字符，可检查是否完整转换。`snprintf(buf, sizeof(buf), '%d', num)` 数字→字符串。",
                "指针本质：存储另一个变量的内存地址。`int *p = &num;`（& 取地址，* 在声明中表示 p 是指针）。解引用：`*p` 访问 num 的值。多级指针：`int **pp = &p;` → `**pp` 最终访问 num。void 指针 `void *vp` 可指向任意类型，但不能直接解引用和做指针算术（使用前需强制转换）。函数指针：`int (*fp)(int, int) = &add;` 调用 `result = fp(3, 5);` 或 `(*fp)(3, 5);`。",
                "指针与 const 三种组合：① `const int *p` = `int const *p`——const 在 * 左边，修饰指向的值（不能通过 p 修改值，但 p 可指向别处）；② `int * const p`——const 在 * 右边，修饰指针本身（p 永远指向固定地址，但可修改值）；③ `const int * const p`——两边都 const，都不能改。",
                "指针算术：`p + n` 指向第 n 个元素（地址增加 `n * sizeof(*p)` 字节）。`p++` 移动到下一个元素。两指针相减得到元素个数（ptrdiff_t）。`p[i]` 等价于 `*(p + i)`。`NULL` 是空指针宏（`(void*)0`）。解引用 NULL 导致段错误。",
                "常见指针错误 Top 10：①野指针（未初始化就解引用）；②空指针解引用；③数组越界；④悬空指针（free 后仍使用）；⑤返回局部变量的地址；⑥忘记字符串需要 \\0 结尾；⑦内存泄漏（malloc 没 free）；⑧重复 free（double free）；⑨类型不匹配的指针转换；⑩指针赋值遗漏 &。",
                "动态内存分配四函数（`<stdlib.h>`）：①`malloc(size_t size)` 分配未初始化内存；②`calloc(n, size)` 分配 n 个元素并清零；③`realloc(ptr, new_size)` 调整大小，可能移动位置返回新地址——失败返回 NULL 但原内存仍有效（不要直接 `p = realloc(p, size)`，先赋临时变量检查）；④`free(ptr)` 释放。一次 malloc 必须对应一次 free。",
                "malloc 最佳实践：`int *arr = malloc(n * sizeof(*arr));` 用 `sizeof(*ptr)` 自动适配类型。始终检查返回值：`if (p == NULL) { exit(EXIT_FAILURE); }`。free(p) 后立即 `p = NULL;`。内存泄漏检测工具：Valgrind（`valgrind --leak-check=full ./program`），AddressSanitizer（`gcc -fsanitize=address -g program.c`）。",
                "结构体定义：`struct Point { int x; int y; };`。C99 指定初始化器：`struct Point p = {.y = 20, .x = 10};`。`typedef struct { int x; int y; } Point;` 简化声明。结构体指针用箭头运算符 `pp->x` 等价于 `(*pp).x`。结构体作为参数时整个拷贝（值传递），大结构体应传指针。结构体不能包含自身类型的成员，但可包含指向自身的指针（链表/树基础）。",
                "结构体对齐与填充：编译器在成员之间和末尾插入 padding 字节以保证对齐。`sizeof(struct)` 通常 > 成员大小之和。调整成员声明顺序（大类型在前）可减少填充。`_Alignas(n)`（C11）手动指定对齐；`_Alignof(type)` 获取类型对齐要求；`offsetof(type, member)`（`<stddef.h>`）获取成员偏移量。",
                "联合体 union：所有成员共享同一块内存，大小等于最大成员的大小。用途：节省内存、判断系统大小端、变体数据类型。枚举 enum：`enum Color { RED, GREEN, BLUE };` 默认 RED=0, GREEN=1。`typedef enum { ... } Color;` 使 Color 成为类型名。",
                "预处理器指令在编译前执行，不遵循 C 语法。`#include <file>` 搜索系统路径；`#include \"file\"` 先搜索当前目录。头文件保护：`#ifndef HEADER_H` / `#define HEADER_H` / `...` / `#endif`。函数宏：`#define MAX(a,b) ((a) > (b) ? (a) : (b))`——每个参数和整体必须加括号！宏参数不要带副作用。多语句宏用 `do { ... } while(0)` 包裹。字符串化 `#param`，连接 `a##b`。",
                "条件编译：`#ifdef DEBUG` / `#ifndef` / `#if` / `#elif` / `#else` / `#endif`。用途：调试打印、平台适配、功能开关。`gcc -DDEBUG` 命令行定义宏。预定义宏：`__FILE__` 文件名，`__LINE__` 行号，`__DATE__` 编译日期，`__func__`（C99）函数名，`__STDC_VERSION__` 标准版本。",
                "stdlib.h 核心函数：`strtol/strtod` 字符串转数字（推荐，有 endptr 检查转换完整性），`rand()` 伪随机数（用 `srand((unsigned)time(NULL))` 设种子），`qsort(base, n, size, cmp)` 快速排序（比较函数签名为 `int cmp(const void *a, const void *b)`），`bsearch` 二分查找，`system(cmd)` 执行 shell 命令，`exit(0)` 终止程序，`atexit(func)` 注册终止回调。",
                "math.h 数学函数（编译需 `-lm`）：`sqrt/cbrt/pow/exp/log/log10/log2`，`fabs/ceil/floor/round/trunc/fmod`，`sin/cos/tan/asin/acos/atan/atan2(y,x)`（弧度制！），`hypot(x,y)` 计算 sqrt(x²+y²)。",
                "ctype.h 字符分类：`isalpha/isdigit/isalnum/isspace/isupper/islower/isxdigit/ispunct/isprint/isgraph/iscntrl`，`toupper/tolower`。参数必须是 unsigned char 或 EOF（对负值行为未定义，建议 `isalpha((unsigned char)c)`）。",
                "time.h 时间处理：`time_t now = time(NULL);` 获取 Unix 时间戳。`struct tm *localtime(const time_t *t)` 转为本地时间（tm_year 年-1900, tm_mon 月-1）。`strftime(buf, sizeof(buf), '%Y-%m-%d %H:%M:%S', &tm)` 格式化。`clock()` 返回程序运行时间（除以 CLOCKS_PER_SEC 得秒）。`timespec_get()`（C11）纳秒级时间。",
                "C 语言未定义行为（UB）是标准没有规定结果的代码——可能崩溃、可能错误、可能看似正常运行（最危险）。必须避免：①数组越界；②有符号整数溢出；③解引用 NULL/野指针；④使用未初始化局部变量；⑤除以 0；⑥修改字符串字面量；⑦重复释放或释放非动态分配内存；⑧使用已释放内存（use-after-free）；⑨同一变量在序列点间多次修改（`i = i++`）；⑩违反 strict aliasing 规则；⑪返回局部变量地址；⑫多线程数据竞争。",
                "安全编码实践：①绝不用 `gets()`（已从 C11 移除），始终用 `fgets`；②用 `snprintf` 代替 `sprintf`；③检查所有 malloc/fopen 返回值是否为 NULL；④数组循环用 `< length` 不是 `<=`；⑤scanf 读取字符串限制宽度：`scanf('%19s', str)`；⑥所有外部输入视为不可信；⑦用 `-Wall -Wextra -Wpedantic -Werror` 编译；⑧开发阶段使用 `-fsanitize=address,undefined` 运行时检测。",
            ]
            st.session_state.knowledge_bases["C语言基础"] = sample_c
            if not st.session_state.active_kb:
                st.session_state.active_kb = "C语言基础"
            st.success(f"✅ 已加载 {len(sample_c)} 条C语言知识")
    
    st.divider()
    st.subheader("方式三：上传文件 📎")
    st.caption("支持 PDF / Word，拖拽或点击上传")
    
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    
    if uploaded_files:
        for uf in uploaded_files:
            # 提取文本
            with st.spinner(f"📄 正在解析 {uf.name}..."):
                raw_text = extract_text_from_file(uf)
            
            if raw_text is None:
                st.error(f"❌ {uf.name}：不支持的文件格式")
            elif not raw_text.strip():
                st.error(f"❌ {uf.name}：无法提取文字（可能是扫描件或图片PDF）")
            else:
                chunks = split_into_chunks(raw_text)
                if not chunks:
                    st.error(f"❌ {uf.name}：提取的文字太短")
                else:
                    # 文件名去掉后缀作为知识库名称
                    kb_name = os.path.splitext(uf.name)[0]
                    st.session_state.knowledge_bases[kb_name] = chunks
                    if not st.session_state.active_kb:
                        st.session_state.active_kb = kb_name
                    st.success(f"✅ {uf.name} → 「{kb_name}」({len(chunks)} 条知识)")
    
    st.divider()
    
    if st.button("🗑️ 清空全部知识库", use_container_width=True):
        st.session_state.knowledge_bases = {}
        st.session_state.active_kb = None
        st.session_state.messages_rag = []
        st.rerun()
    
    if st.button("💬 清空对话", use_container_width=True):
        st.session_state.messages_rag = []
        st.rerun()

# ============================================
# 主对话区域
# ============================================

# messages_rag 已在侧边栏初始化

for msg in st.session_state.messages_rag:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("基于知识库提问（例如：list和tuple的区别？）"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages_rag.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        if not st.session_state.active_kb or not st.session_state.knowledge_bases:
            st.warning("⚠️ 请先在左边加载知识库！")
            answer = "请先加载知识库，我才能基于文档回答。点击左边「📥 Python入门」或「📥 C语言基础」加载示例，或粘贴你自己的知识。"
        else:
            texts = st.session_state.knowledge_bases[st.session_state.active_kb]
            kb_data = compute_tfidf(tuple(texts))
            
            with st.spinner("🔍 检索知识库..."):
                relevant_docs = search_docs(kb_data, user_input, top_n=3)
            
            context = "\n\n".join(relevant_docs)
            prompt = f"""你是一个严格基于文档的问答助手。

## ⚠️ 核心规则
- 只根据下面「参考资料」中的内容回答问题
- 如果参考资料中没有相关信息，直接回复："抱歉，知识库中没有找到相关内容"
- 引用原文时用引号标出
- 回答要条理清晰，分点列出

## 参考资料
{context}

## 用户问题
{user_input}

## 你的回答："""
            
            # 🔥 对话记忆：把历史对话传给 DeepSeek，让 AI 理解上下文
            chat_messages = []
            for msg in st.session_state.messages_rag[:-1]:  # 已包含历史 user+assistant
                chat_messages.append({"role": msg["role"], "content": msg["content"]})
            chat_messages.append({"role": "user", "content": prompt})  # 当前 RAG 增强问题
            
            with st.spinner("💭 DeepSeek 生成回答..."):
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=chat_messages,
                    temperature=0.3,
                    max_tokens=2048,
                )
                answer = response.choices[0].message.content
            
            st.markdown(answer)
            
            with st.expander("📖 参考的知识片段"):
                for i, doc in enumerate(relevant_docs, 1):
                    st.caption(f"**片段 {i}：** {doc}")
    
    st.session_state.messages_rag.append({"role": "assistant", "content": answer})
