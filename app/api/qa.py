from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from app.models.schemas import QuestionRequest, AnswerResponse, SourceInfo
from app.core.rag_chain import ask_with_sources
from app.core.memory_manager import chat as memory_chat, clear_history, get_chat_history
from app.core.vectorstore import load_vectorstore
from typing import AsyncGenerator
from app.core.exception import KnowledgeError
import json
import asyncio

router = APIRouter()


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """单轮问答（返回完整答案）"""
    try:
        result = ask_with_sources(request.question)
      
        return AnswerResponse(
            question=request.question,
            answer=result["answer"],
            sources=[
                SourceInfo(content=s["content"], source=s["source"])
                for s in result["sources"]
            ],
            session_id=request.session_id,
        )
    except FileNotFoundError:
        raise KnowledgeError()
    


@router.post("/chat", response_model=AnswerResponse)
async def chat_with_memory(request: QuestionRequest):
    """多轮对话（带记忆）"""
    try:
        answer = memory_chat(request.session_id, request.question)
      
        # 多轮对话的引用来源简化处理
        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=[],  # 多轮对话暂不返回来源（可后续优化）
            session_id=request.session_id,
        )
    except FileNotFoundError:
        raise KnowledgeError()


@router.post("/agent")
async def agent_ask_question(request: QuestionRequest):
    """Agent 问答：LLM 自主决定是否检索知识库（Function Calling）

    与 /ask 的区别：这里不是固定流程，而是由 LLM 判断是否需要调用检索工具。
    """
    
    from app.core.agent import agent_ask
    result = agent_ask(request.question)
    return {
            "question": request.question,
            "answer": result["answer"],
            "used_tool": result["used_tool"],
            "sources": result["sources"],
            "session_id": request.session_id,
        }
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Agent 问答失败: {str(e)}" )


@router.post("/graph")
async def graph_ask_question(request: QuestionRequest):
    """LangGraph 版问答：先判要不要查知识库，再决定走检索还是直接回答

    与 /agent 的区别：/agent 由 LLM 用 Function Calling 自己决定调不调工具；
    这里是一张显式的图，路由规则看得见、可改、能加节点（改写重试等）。
    """
    from app.core.graph import ask_graph
    result = ask_graph(request.question)
    return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "session_id": request.session_id,
        }


@router.post("/ask/stream")
async def ask_stream(request: QuestionRequest):
    """流式问答（SSE）— 像 ChatGPT 一样逐字输出
  
    这是面试中的亮点功能！
    """
    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            from app.core.rag_chain import build_rag_chain
            from langchain_core.output_parsers import StrOutputParser
          
            # 构建链
            chain = build_rag_chain()
          
            # 流式生成
            async for chunk in chain.astream(request.question):
                yield {"event": "token", "data": chunk}
          
            yield {"event": "done", "data": ""}
          
        except Exception as e:
            yield {"event": "error", "data": str(e)}
  
    return EventSourceResponse(event_generator())


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """清除指定会话的历史"""
    clear_history(session_id)
    return {"message": f"会话 {session_id} 已清除"}


@router.get("/history/{session_id}")
async def view_chat_history(session_id: str):
    """查看会话历史（调试用）"""
    from langchain_core.messages import HumanMessage, AIMessage
  
    messages = get_chat_history(session_id)
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
  
    return {"session_id": session_id, "messages": result, "count": len(result)}

@router.post("/agent/chat")
async def agent_chat_with_memory(request: QuestionRequest):
    """带记忆的 Agent 对话"""
    try:
        from app.core.agent import agent_chat
        result = agent_chat(request.session_id, request.question)
        return {
            "question": request.question,
            "answer": result["answer"],
            "used_tool": result["used_tool"],
            "sources": result["sources"],
            "session_id": request.session_id,
        }
    except FileNotFoundError:
        raise KnowledgeError()