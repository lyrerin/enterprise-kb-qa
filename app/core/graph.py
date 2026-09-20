"""LangGraph 版问答流程

把原来 app/core/agent_graph.py 那个「脚本」改成「模块」，修掉它四个毛病：
  1. 原来 :45 有模块级 app.invoke(...)+print → 一 import 就跑一次查询；现在没有了
  2. 原来节点往 add_messages 里塞 str → 会 InvalidUpdateError；现在返回 AIMessage 对象
  3. 原来 answer_node 返回占位字符串 '（这里是直接回答的逻辑）'；现在是真回答
  4. 原来靠 graph.set_entry_point("start") 挂一个空白节点；现在用条件入口点

流程：
  用户问题 → should_retrieve 判断
              ├─ 涉及公司制度/流程/福利等 → retrieve 节点（完整 RAG：SQL 分叉 + 父块展开）
              └─ 闲聊/通用问题           → answer  节点（直接问 LLM，不查库）
"""
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.core.rag_chain import ask_with_sources
from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

# 判断「要不要查知识库」的关键词：命中就走检索。
# 这一层是规则版；以后要更稳可以换成「让 LLM 判一次」，一次只改这一处。
RETRIEVE_HINTS = (
    "公司","年假","报销","制度","考勤","福利","流程","绩效","晋升","培训",
    "补贴","工资","差旅","请假","加班","手册","政策","规定","预算",
)


class State(TypedDict):
    messages: Annotated[list,add_messages]   # 只放 Message 对象，不放 str
    sources: list                            # 把引用来源带出图，API 那一层要用


def should_retrieve(state: State):
    """判断是否需要检索知识库"""
    q = state["messages"][-1].content
    if any(k in q for k in RETRIEVE_HINTS):
        return "retrieve"
    return "answer"


def retrieve_node(state: State):
    """检索节点：走完整 RAG（含 SQL 分叉 + 父块展开）"""
    q = state["messages"][-1].content
    result = ask_with_sources(q)
    return {"messages": [AIMessage(content=result["answer"])],
            "sources": result["sources"]}


def answer_node(state: State):
    """不检索，直接让 LLM 回答（闲聊/通用问题）"""
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,temperature=0.3,
    )
    answer = llm.invoke(state["messages"]).content
    return {"messages": [AIMessage(content=answer)], "sources": []}


builder = StateGraph(State)
builder.add_node("retrieve",retrieve_node)
builder.add_node("answer",answer_node)
builder.set_conditional_entry_point(
    should_retrieve, {"retrieve":"retrieve","answer":"answer"}
)
builder.add_edge("retrieve",END)
builder.add_edge("answer",END)

graph = builder.compile()


def ask_graph(question: str) -> dict:
    """给 API 用的入口：把问题喂进图，取出答案和引用来源"""
    result = graph.invoke({"messages":[HumanMessage(content=question)],"sources":[]})
    return {
        "answer": result["messages"][-1].content,
        "sources": result.get("sources",[]),
    }


if __name__ == "__main__":
    # 只在「直接运行本文件」时 demo；被 import 时什么都不做
    out = ask_graph("公司年假政策")
    print(out["answer"])
