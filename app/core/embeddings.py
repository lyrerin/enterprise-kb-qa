# 注意：HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 在 app/__init__.py 里统一设置
# （必须比 huggingface_hub 的 import 更早，否则不生效）。
import os
import threading

from app.config import EMBEDDING_MODEL
from langchain_huggingface import HuggingFaceEmbeddings

# 缓存：不加这个，每次 retrieve() 都会经 load_vectorstore() 重新 load 一遍模型
# （服务端日志里会看到一次次 "Loading weights 0/71"）。加锁是因为 uvicorn 是多线程，
# 两个请求同时进来会各造一个模型。
_embeddings = None
_embeddings_lock = threading.Lock()


def get_embeddings():
    """本地 HuggingFace 向量模型（dashscope 已注销，不再走在线 embedding API）"""
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:      # 双重检查：拿到锁后再确认一次，避免重复加载
                _embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu", "local_files_only": True},
                    encode_kwargs={"normalize_embeddings": True},
                )
    return _embeddings


# ====== 换/下载新模型时怎么弄 ======
# 1) 先设镜像（huggingface.co 不可达，hf-mirror.com 可达），再让库联网下载一次：
#      $env:HF_ENDPOINT = "https://hf-mirror.com"
#      $env:HF_HUB_OFFLINE = "0"
#      .\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"
# 2) 下完把 .env 里的 EMBEDDING_MODEL 改成新名字，并重新入库（向量维度和旧库不兼容）。
# 3) 也可以直接用 modelscope 下：modelscope.cn 同样可达。


# ====== 旧版（dashscope，已注销，留作对照）======
# from langchain_community.embeddings import DashScopeEmbeddings
# def get_embeddings():
#     return DashScopeEmbeddings(model="text-embedding-v2")
