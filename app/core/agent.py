"""基于 Function Calling 的最小 Agent

核心思路：让 LLM 自主决定「是否需要检索知识库」，而不是固定流程。
这是从「RAG」升级到「RAG Agent」的关键一步。

流程：
  用户提问 → LLM 判断是否调用工具(search_knowledge_base)
            ├─ 调用   → 执行检索 → 结果喂回 → LLM 生成答案
            └─ 不调用（闲聊/通用问题）→ 直接回答
"""
import ast

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.core.memory_manager import get_or_create_session,add_message
from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, MAX_TOOL_ROUNDS, FINAL_K


@tool(response_format="content_and_artifact")
def search_knowledge_base(question: str):
    """在企业知识库中检索与用户问题相关的文档内容。

    当用户的问题涉及企业内部知识（制度、流程、政策、福利、考勤等）时，调用此工具。
    Args:
        question: 需要检索的问题。

    返回 (喂给模型的正文, 引用来源)：正文进对话，来源走 artifact 通道（模型看不到，
    但我们在 agent_ask 里能取出来返回给前端）。
    """
    from app.core.retriever import retrieve, expand_to_parents
    try:
        docs = retrieve(question, k=FINAL_K)
    except FileNotFoundError:
        return "知识库为空，暂无可用文档，请先上传文档。", []

    if not docs:
        return "知识库中没有找到相关内容。", []

    sources = [
        {"content": d.page_content[:200], "source": d.metadata.get("source", "未知")}
        for d in docs
    ]
    # 命中子块 → 回表换成整节父块，给模型完整上下文
    return expand_to_parents(docs), sources

# 只允许出现的语法零件：数字字面量、算术运算、一整条表达式。
# 不在这个清单里的（Call 函数调用、Attribute 取属性、Name 变量名…）一律拒绝。
# 注意：故意不放 ast.Pow —— 2**99999999 能过校验，但会把进程算到卡死。
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.USub, ast.UAdd, ast.Load,
)


def safe_eval(expression: str) -> float:
    """只允许数字字面量和算术运算；不碰 eval 的任意代码执行面

    顺序很关键：先把整棵树查一遍，确认全是白名单零件，才真正执行。
    """
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"表达式里不允许 {type(node).__name__}")
    # 执行时把内置函数清空 —— 白名单万一被绕过，__import__/open 这些也已经不存在
    return eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})


@tool
def caculator(expression: str) -> str:
    """计算数学表达式（只支持数字和 + - * / %，不支持任何函数或文字）。"""
    try:
        return str(safe_eval(expression))
    except Exception as e:
        return f"无法计算：{e}"

@tool
def get_current_time():
    """获取当前日期和时间。当用户问「现在几点」「今天几号」时调用。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool(response_format="content_and_artifact")
def query_csv(question: str):
    """对企业数据表（CSV）做统计查询：平均、合计、最高/最低、排序、筛选、一共有多少条。

    当用户问的是表格里的数值统计（而不是制度、流程、政策）时，调用此工具。
    Args:
        question: 需要统计的问题，例如「产品列表里哪个类别的平均单价最高」。

    返回 (一句人话答案, 生成的 SQL)：答案喂模型，SQL 走 artifact 通道给前端展示。
    """
    from app.core.router import run_sql_path
    result = run_sql_path(question, force=True)
    if result is None:
        return "没有找到可查询的数据表，或这个问题无法转成查询语句。", []
    # run_sql_path 返回的 sources 就是 [{'content': 生成的SQL, 'source': '表名.csv'}]
    return result['answer'], result['sources']

TOOL_MAP = {
    'search_knowledge_base': search_knowledge_base,
    'query_csv': query_csv,
    'caculator': caculator,
    'get_current_time': get_current_time,
}


def _build_agent_llm(bind_tools: bool = True):
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
    )
    if bind_tools:
        return llm.bind_tools([search_knowledge_base, query_csv, caculator, get_current_time])
    return llm



def _dedup_sources(sources: list) -> list:
    """同一份文档可能被检索两次（模型多轮调工具），按 (来源, 内容) 去重"""
    return list({(s["source"], s["content"]): s for s in sources}.values())


def agent_ask(question: str) -> dict:
    """Agent 问答入口：LLM 自主决策是否检索知识库。

    Returns:
        dict: {"answer": 最终答案, "used_tool": 调过哪些工具, "sources": 引用来源}
    """
    llm = _build_agent_llm()
    plain_llm = _build_agent_llm(bind_tools=False)
    
    messages = [HumanMessage(content=question)]

    # 第一轮：让 LLM 决定是否调用工具
    response = llm.invoke(messages)
    used_tool = []
    sources = []

    rounds = 0
    
    while response.tool_calls :
        rounds += 1
        messages.append(response)
        for call in response.tool_calls:
            tool = TOOL_MAP[call["name"]]
            # 传整个 tool_call（不是 args）→ 拿回 ToolMessage；
            # 声明了 content_and_artifact 的工具，artifact 里就是引用来源
            msg = tool.invoke(call)
            used_tool.append(call['name'])
            sources.extend(msg.artifact or [])
            messages.append(msg)
        if rounds >= MAX_TOOL_ROUNDS:
            response = plain_llm.invoke(messages)
            break
        # 第二轮：基于工具结果生成最终答案
        response = llm.invoke(messages)

    return {"answer": response.content, "used_tool": used_tool,
            "sources": _dedup_sources(sources)}

def agent_chat(session_id:str ,question:str):
    """带记忆的 Agent 对话"""
    llm = _build_agent_llm()
    plain_llm = _build_agent_llm(bind_tools=False)
    history = get_or_create_session(session_id)      # 拿到历史消息列表
    messages = history + [HumanMessage(content=question)]  # 历史 + 新问题

    response = llm.invoke(messages)
    used_tools = []
    sources = []
    rounds = 0
    while response.tool_calls:
        rounds += 1
        messages.append(response)
        for call in response.tool_calls:
            tool = TOOL_MAP[call["name"]]
            msg = tool.invoke(call)
            used_tools.append(call["name"])
            sources.extend(msg.artifact or [])
            messages.append(msg)
        if rounds >= MAX_TOOL_ROUNDS:
            # 到上限就换成不绑工具的模型收尾，逼它给最终答案；
            # 不加这个上限，模型一直要工具 → while 没有出口，接口永远不返回
            response = plain_llm.invoke(messages)
            break
        response = llm.invoke(messages)
    
    add_message(session_id, 'user',question)
    add_message(session_id, 'assistant',response.content)

    return {"answer": response.content, "used_tool": used_tools,
            "sources": _dedup_sources(sources)}