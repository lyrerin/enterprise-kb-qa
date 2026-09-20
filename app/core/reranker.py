# 注意：HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 在 app/__init__.py 里统一设置
# （必须比 huggingface_hub 的 import 更早，否则不生效）。
from typing import List
from langchain_core.documents import Document

class Reranker:
    def __init__(self,model_name:str = 'BAAI/bge-reranker-base'):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(model_name, use_fp16 =True)
    def rerank(self,query:str,docs:List[Document],top_n:int = 10):
        if not docs:
            return docs
        
        pairs = [[query,doc.page_content] for doc in docs]

        # FlagEmbedding 不同版本 API 不同：旧版 compute_scores / 新版 compute_score
        scorer = getattr(self.model, "compute_scores", None) or self.model.compute_score
        scores = scorer(pairs)
        # 传单对时可能返回 float，统一转成 list
        if isinstance(scores, (int, float)):
            scores = [scores]

        ranked = sorted(zip(docs,scores),key=lambda x:x[1],reverse=True)
        return [doc for doc,_ in ranked[:top_n]]