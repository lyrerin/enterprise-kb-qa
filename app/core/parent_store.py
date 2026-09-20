import json
import os
import sqlite3

PARENT_DB = os.path.join(os.path.dirname(__file__), "parent.db")

def _conn():
    conn = sqlite3.connect(PARENT_DB)
    conn.execute('CREATE TABLE IF NOT EXISTS parent_chunks '
        '(parent_id TEXT PRIMARY KEY,content TEXT, metadata TEXT)')
    return conn

def save_parent(parent_id, content, metadata):
    conn = _conn()
    conn.execute('INSERT OR REPLACE INTO parent_chunks VALUES (?, ?, ?)'
        , (parent_id, content, json.dumps(metadata, ensure_ascii=False)))
    conn.commit()
    conn.close()

def get_parents(parent_ids: list[str]):
    """返回 [(parent_id, content), ...]，顺序按传进来的 parent_ids 排

    带上 parent_id 是必须的：库里少一个 id 时，光返回正文会让调用方按位置配来源串位。
    """
    if not parent_ids:
        return []
    conn =_conn()
    q = ','.join('?' * len(parent_ids))
    rows = conn.execute(f'SELECT parent_id,content FROM parent_chunks WHERE parent_id IN ({q})', parent_ids).fetchall()
    conn.close()
    order = {pid: i for i, pid in enumerate(parent_ids)}
    return sorted(rows, key=lambda r: order.get(r[0],1e9))

def clear_parents():
    """清空父块表——清知识库时要一起清，否则会捞出幽灵内容"""
    conn = _conn()
    conn.execute('DELETE FROM parent_chunks')
    conn.commit()
    conn.close()

