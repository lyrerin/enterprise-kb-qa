import os 
import re
import sqlite3

import pandas as pd
from app.core.parsers import sniff_encoding
from langchain_openai import ChatOpenAI
from app.config import DEEPSEEK_MODEL,DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL

_FORBIDDEN = re.compile(r'\b(drop|delete|update|insert|alter|attach|pragma|create)\b|;',re.I)
_SQL_PROMPT =  """你是 SQLite 专家。根据下面的表结构和用户问题，写出一条查询 SQL。

{schema}

要求：
1. 只输出 SQL 本身，不要解释，不要 markdown 代码块
2. 只能写 SELECT，禁止增删改
3. 表名和列名用双引号包起来

用户问题：{question}
SQL："""

_ANSWER_PROMPT = """用户问了一个数据统计问题，下面是数据库查询出的真实结果。
请用一句简洁的中文直接回答用户，不要提 SQL，不要解释过程。

用户问题：{question}
查询结果的列名依次是：{columns}
查询结果：{rows}

回答："""
def llm(temperature:float = 0):
    """LLM 生成回答"""
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
        )
    return llm
def build_conn(csv_path:str):
    """CSV → 内存 SQLite（副本，不碰真实库），表名 = 文件名"""
    df = pd.read_csv(csv_path,encoding=sniff_encoding(csv_path))
    table = os.path.splitext(os.path.basename(csv_path))[0]
    conn = sqlite3.connect(':memory:')
    df.to_sql(table,conn,index = False)
    return conn,table

def get_schema(conn,table:str):

    cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    rows = conn.execute(f'SELECT * FROM "{table}" LIMIT 3').fetchall() 
    return (f'表 {table}(' + ", ".join(f"{c[1]} {c[2]}" for c in cols) + ")\n"
            f"样例数据：{rows}")

def _gen_sql(question:str,schema:str):
    """LLM 生成 SQL"""
    prompt = _SQL_PROMPT.format(schema=schema,question=question)
    return llm().invoke(prompt).content.strip()
    
    

def text_to_sql(question:str,conn,table:str):
    """LLM 生成 SQL → 白名单校验 → 只读执行"""
    sql = _gen_sql(question,get_schema(conn,table)).strip().rstrip(';')
    if not sql.lower().startswith('select'):
        raise ValueError("SQL 语句必须以 SELECT 开头")
    if _FORBIDDEN.search(sql):
        raise ValueError("SQL 语句包含禁止的关键词")

    cur = conn.execute(sql)
    columns = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return sql, columns, rows

def summarize(question: str, columns: list, rows: list) -> str:
    """把查询结果那串元组，转成一句人话"""
    if not rows:
        return "根据数据表查询，没有找到符合条件的记录。"
    prompt = _ANSWER_PROMPT.format(question=question, columns=columns, rows=rows)
    return llm(0.3).invoke(prompt).content.strip()
