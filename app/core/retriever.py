from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from app.core.vectorstore import load_vectorstore
from app.core.reranker import Reranker
from rank_bm25 import BM25Okapi
import jieba
from app.config import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,DEEPSEEK_MODEL,RETRIEVAL_STRATEGY, RECALL_K, FINAL_K
import threading

def get_retriever(search_type: str = 'similarity', k: int = 3) -> ChromaRetriever:
    """获取检索器
    
    Args:
        search_type: 
            - "similarity": 纯相似度检索（默认）
            - "mmr": 最大边际相关（结果更多样，避免重复）
        k: 返回的文档数
    """
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(
        search_type = search_type,
        search_kwargs = {'k': k}
    )

def similarity_search_with_score(query:str , k :int = 5,score_threshole:float = 0.0):
    """带相似度分数的检索"""
    vectorstore = load_vectorstore()
    return vectorstore.similarity_search_with_relevance_scores(query, k=k)

def mmr_search(query:str ,k: int = 4,fetch_k: int = 10,lambda_mult:float = 0.5):
    """MMR 检索"""
    vectorstore = load_vectorstore()
    return vectorstore.max_marginal_relevance_search(query,k = k,fetch_k=fetch_k,lambda_mult=lambda_mult)

_bm25_lock = threading.Lock()
_bm25_cache = {"count": -1, "bm25": None, "docs": None, "metas": None}
        
def _get_bm25(vectorstore):
    collection = vectorstore._collection
    with _bm25_lock:
        count = collection.count()
        if _bm25_cache['count'] == count and _bm25_cache['bm25'] is not None:
            return _bm25_cache['bm25'], _bm25_cache['docs'], _bm25_cache['metas']
        if count == 0:
            _bm25_cache.update(count = count,bm25 = None,docs = None,metas = None)
            return None, [], []
        data = collection.get(include=['documents','metadatas'])
        docs = data['documents'] or []
        metas = data['metadatas'] or [{} for _ in docs]
        bm25 = BM25Okapi([list(jieba.cut(doc)) for doc in docs])
        _bm25_cache.update(count=count, bm25=bm25, docs=docs, metas=metas)
        return bm25, docs, metas

def hybrid_search(query: str, k: int = FINAL_K):
    """混合检索：向量 + BM25 双路召回，RRF 融合"""
    vectorstore = load_vectorstore()
    vector_docs = vectorstore.similarity_search(query, k=k * 2)   # ① 向量路

    bm25, all_docs, metas = _get_bm25(vectorstore)                # ② BM25 路（走缓存）
    if bm25 is None:                                             # ③ 空库兜底
        return vector_docs[:k]

    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:k * 2]

    rrf_scores = {}
    doc_obj_map = {}
    RRF_K = 60

    for rank, doc in enumerate(vector_docs, start=1):             # ④ 向量路计分
        rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + 1 / (RRF_K + rank)
        doc_obj_map[doc.page_content] = doc

    for rank, idx in enumerate(bm25_top_indices, start=1):        # ⑤ BM25 路计分
        content = all_docs[idx]
        rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (RRF_K + rank)
        if content not in doc_obj_map:
            doc_obj_map[content] = Document(                      # ⑥ 用 metas 补回来源
                page_content=content, metadata=metas[idx]
            )

    sorted_contents = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [doc_obj_map[c] for c in sorted_contents[:k]]
        
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()          # 首次调用才真正加载模型
    return _reranker

def retrieve_with_rerank(query:str,recall_k:int = RECALL_K,final_k:int = FINAL_K):
    

    vectorstore = load_vectorstore()

    candidates = vectorstore.similarity_search(query, k=recall_k)
    reranker = get_reranker()

    return reranker.rerank(query,candidates,top_n=final_k)

def rewrite_query(question:str):
    """改写问题"""
    vectorstore = load_vectorstore()
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        temperature=0.3,
    )
    prompt = (
        '把下面的口语化问题改写为正式用语，适合知识库检索的表述'
        '只输出改写后的问题，不要解释：\n' + question
    )
    return llm.invoke(prompt).content.strip()

def retrieve_with_rewrite(query:str,k):
    rewritten = rewrite_query(query)
    # vectorstore = load_vectorstore()
    return retrieve_with_rerank(rewritten, recall_k = RECALL_K,final_k = k)



STRATEGY = {
    'vector': lambda q,k,**kwargs: load_vectorstore().similarity_search(q, k=k, **kwargs),
    'rerank': lambda q,k,**kwargs: retrieve_with_rerank(q,recall_k = RECALL_K,final_k =k),
    'hybrid': lambda q,k,**kwargs: hybrid_search(q,k = k,**kwargs),
    'rewrite': lambda q,k,**kwargs: retrieve_with_rewrite(q,k,**kwargs),
}

def retrieve(query:str,strategy:str | None = None,k:int | None = None,**filters):
    name = (strategy or RETRIEVAL_STRATEGY).lower()
    if name not in STRATEGY:
        raise ValueError(f"未知检索策略: {name}，可选 {list(STRATEGY)}")
    return STRATEGY[name](query,FINAL_K if k is None else k,**filters)


def expand_to_parents(docs, max_chars: int = 4000):
    """命中子块 → 回表换成整节父块，拼成带来源标注的 context 字符串

    没有 parent_id 的块（CSV 行、整表等）直接用原文兜底，不会被丢掉。
    """
    from app.core.parent_store import get_parents

    pids = list(dict.fromkeys(
        d.metadata["parent_id"] for d in docs if d.metadata.get("parent_id")
    ))
    parent_of = dict(get_parents(pids))          # {parent_id: 父块正文}

    parts, seen, total = [], set(), 0
    for d in docs:                               # 按检索相关性顺序遍历子块
        pid = d.metadata.get("parent_id")
        if pid and pid in parent_of:
            if pid in seen:                      # 同一个父块命中多个子块 → 只放一次
                continue
            seen.add(pid)
            content = parent_of[pid]
        else:
            content = d.page_content             # 没有父块 → 用原文兜底

        if parts and total + len(content) > max_chars:   # 限总长，不限长 token 会爆
            break
        if not parts and len(content) > max_chars:
            content = content[:max_chars]                # 第一块就超长 → 截断，不能返回空
        total += len(content)
        parts.append(
            f"[文档{len(parts)+1}] 来源: {d.metadata.get('source','未知')}\n{content}"
        )

    return "\n\n".join(parts)

