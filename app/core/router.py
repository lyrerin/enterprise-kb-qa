import os

from app.core.sql_tool import build_conn, summarize, text_to_sql

AGG = ("平均", "总共", "一共", "多少条", "最高", "最低", "最大", "最小",
       "排序", "排名", "占比", "合计", "大于", "小于")

_CSV_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'uploads',
)


def list_tables() -> dict:
    """扫描 uploads 目录，返回 {表名: csv文件路径}"""
    tables = {}
    if not os.path.isdir(_CSV_DIR):
        return tables
    for name in os.listdir(_CSV_DIR):
        if name.lower().endswith('.csv'):
            tables[os.path.splitext(name)[0]] = os.path.join(_CSV_DIR, name)
    return tables


def match_table(question: str, tables: dict) -> str | None:
    """找出问题里提到的是哪张表"""
    for name in tables:
        core = name.replace('列表', '').replace('表', '')
        if name in question or (core and core in question):
            return name
    return None


def route(question: str, table_names: list) -> str:
    """规则版：命中聚合/筛选词 + 命中表名 → 走 SQL；否则走 RAG"""
    if any(w in question for w in AGG) and any(t in question for t in table_names):
        return 'sql'
    return 'rag'


def run_sql_path(question: str, force: bool = False) -> dict | None:
    """SQL 那条路。走不通就返回 None，让调用方回去走 RAG

    force=True：跳过「聚合词 + 表名」的规则闸。
    Agent 直接把这个函数当工具用时，模型已经自己判断过该走 SQL，
    再拿规则卡一道反而会让「库存小于30的有几个」这类不提表名的问题走不通。
    """
    tables = list_tables()
    if not tables:
        return None

    table = match_table(question, tables)
    if table is None and len(tables) == 1:
        table = next(iter(tables))          # 全库只有一张表 → 不用在问题里点名
    if table is None:
        return None
    if not force and route(question, [table]) != 'sql':
        return None

    try:
        conn, table_name = build_conn(tables[table])
        sql, columns, rows = text_to_sql(question, conn, table_name)
        answer = summarize(question, columns, rows)
    except Exception as e:
        print(f'[SQL 路径失败，回退 RAG] {e}')
        return None

    return {
        'question': question,
        'answer': answer,
        'sources': [{'content': sql, 'source': f'{table}.csv'}],
    }