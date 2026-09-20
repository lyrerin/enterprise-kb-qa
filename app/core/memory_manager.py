from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from app.core.vectorstore import load_vectorstore
from app.config import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,DEEPSEEK_MODEL,FINAL_K,HISTORY_WINDOW_TURNS
from app.core.retriever import retrieve
import sqlite3
import time
import os

# _session_store: Dict[str, List[BaseMessage]] = {}

# def get_or_create_session(session_id: str):
#     if session_id not in _session_store:
#         _session_store[session_id] = []
#     return _session_store[session_id]

# def add_message(session_id: str, role: str, content: str):
#     """添加一条消息到会话历史"""
#     history = get_or_create_session(session_id)
#     if role == 'user':
#         history.append(HumanMessage(content=content))
#     elif role == 'assistant':
#         history.append(AIMessage(content=content))
    
# def clear_history(session_id:str):
#     if session_id in _session_store:
#         del _session_store[session_id]

# def get_chat_history(session_id: str):
#     return get_or_create_session(session_id)
# NEW: 从 SQLite 读写会话历史，而不是从内存中读写  
DB_PATH = os.path.join(os.path.dirname(__file__), 'chat_messages.db')

# 建表和建索引拆成两段：老库已经有 chat_messages 表（缺 user_id 列），
# 如果索引和建表写在同一段脚本里，CREATE INDEX 会抢在补列之前执行 → no such column: user_id
_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts REAL NOT NULL
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_session_ts ON chat_messages (session_id,ts);
CREATE INDEX IF NOT EXISTS idx_user_session ON chat_messages (user_id,session_id,ts);
"""

def _get_conn():
    """获取数据库连接，并确保表存在"""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)      # 1. 先确保表在（老库已存在则原样跳过，不改结构）
    _migrate(conn)                   # 2. 再补 user_id 列
    conn.executescript(_INDEXES)     # 3. 最后建索引：这时列一定齐了
    return conn

def _migrate(conn):
    """给旧的 chat_messages 表补 user_id 列（必须在建索引之前跑）"""
    cols = {row[1] for row in conn.execute('PRAGMA table_info(chat_messages)')}
    if 'user_id' not in cols:
        conn.execute('ALTER TABLE chat_messages ADD COLUMN user_id TEXT DEFAULT NULL')
        conn.commit()
def add_message(session_id:str,role:str,content:str,user_id:str | None = None):
    """添加一条消息到会话历史"""
    conn = _get_conn()
    conn.execute(
        'INSERT INTO chat_messages (session_id,user_id,role,content,ts) VALUES(?,?,?,?,?)',
        (session_id,user_id,role,content,time.time())
    )
    conn.commit()
    conn.close()

def get_chat_history(session_id:str,turn:int = HISTORY_WINDOW_TURNS,user_id: str | None = None):
    """获取会话历史,取最近的turn轮消息"""
    limit = turn*2
    conn = _get_conn()
    if user_id is None:
        sql = '''SELECT role,content FROM ( 
            SELECT role,content,ts FROM chat_messages
            WHERE session_id = ? ORDER BY ts DESC LIMIT ?
        ) ORDER BY ts ASC'''
        args = (session_id, limit)
    else:
        sql = '''SELECT role,content FROM ( 
            SELECT role,content,ts FROM chat_messages
            WHERE user_id = ? AND session_id = ? ORDER BY ts DESC LIMIT ?
        ) ORDER BY ts ASC'''
        args = (session_id,user_id,limit)
    rows = conn.execute(
        sql,
        args,
    ).fetchall()
    conn.close()

    messages = []
    for r,c in rows:
        if r =='user':
            messages.append(HumanMessage(content = c))
        elif r =='assistant':
            messages.append(AIMessage(content = c))
    return messages

def clear_history(session_id:str):
    """清除会话历史"""
    conn = _get_conn()
    conn.execute(
        'DELETE FROM chat_messages WHERE session_id = ?',
        (session_id,)
    )
    conn.commit()
    conn.close()

def get_or_create_session(session_id: str):
    return get_chat_history(session_id)


def build_conversational_rag_chain():
    """构建带对话记忆的 RAG 链
    
    关键区别：prompt 中加入了 MessagesPlaceholder，会插入历史对话
    """
    # vectorstore = load_vectorstore()
    # retriever = retrieve_with_rerank(question, recall_k=20,final_k=3)
    
    # 注意：模板中多了一个 {chat_history}
    # MessagesPlaceholder 会自动把历史消息填进去
    template = """你是一个专业的企业知识库助手，请根据以下文档内容回答用户问题。

## 对话历史
{chat_history}

## 参考文档
{context}

## 当前问题
{question}

## 规则
- 只使用提供的文档内容来回答
- 如果文档中没有相关信息，请明确说无法回答
- 如果当前问题是追问（如"那第二个呢？"），结合对话历史理解用户意图
- 回答简洁准确"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name = 'chat_history'),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(
        model_name = DEEPSEEK_MODEL,
        api_key = DEEPSEEK_API_KEY,
        base_url = DEEPSEEK_BASE_URL,
        temperature = 0.3,
    )
    def _retrieve_and_format(inputs:dict):
        question = inputs['question']
        docs = retrieve(question, k=FINAL_K)
        context = '\n\n'.join(
            f'[来源:{d.metadata.get('source')}]{d.page_content}'for d in docs
        )

        return{
            'context':context,
            'question':question,
            'chat_history':inputs.get('chat_history',[]),
        }
    
    chain =(
        RunnablePassthrough.assign(context_and_question = _retrieve_and_format)
        | (lambda x: {
            'context':x['context_and_question']['context'],
            'question': x['context_and_question']['question'],
            'chat_history': x['context_and_question']['chat_history'],
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def chat(session_id: str, question: str):
    """多轮对话接口"""
    history = get_or_create_session(session_id)
    chain = build_conversational_rag_chain()
    answer = chain.invoke({
        'question':question,
        'chat_history':history,
    })
    add_message(session_id, 'user',question)
    add_message(session_id,'assistant',answer)
    return answer

    # ====== 测试多轮对话 ======

# def demo_multi_turn():
#     """演示多轮对话：追问、代词指代"""
#     session = "test_session_001"
#     clear_history(session)
    
#     conversations = [
#         "公司的绩效考核多久进行一次？",       # 第一轮
#         "有哪些等级？",                      # 追问（省略主语）
#         "那个最高等级有什么好处？",           # 指代消解
#         "团建活动的预算呢？",                # 话题切换
#     ]
    
#     for q in conversations:
#         print(f"\n{'='*50}")
#         print(f"👤 用户: {q}")
#         answer = chat(session, q)
#         print(f"🤖 助手: {answer}")


# if __name__ == "__main__":
#     demo_multi_turn()
