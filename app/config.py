import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv("APP_NAME","企业知识库问答问答系统")
HOST = os.getenv("HOST","0.0.0.0")
PORT = int(os.getenv("PORT",8000))
LOG_LEVEL = os.getenv("LOG_LEVEL","INFO")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
# 向量模型走本地 HuggingFace（dashscope 已注销，DASHSCOPE_API_KEY 已移除）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

KNOWLEDGE_DIR = os.getenv(
    "KNOWLEDGE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_docs"),
)
CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    os.path.join(os.path.dirname(__file__), "chroma_db"),
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))


FINAL_K = int(os.getenv("FINAL_K", "3"))   # 最终返回的K个回答
RECALL_K = int(os.getenv("RECALL_K", "20"))   # 回调的K个文档
HISTORY_WINDOW_TURNS = int(os.getenv("HISTORY_WINDOW_TURNS", "6"))   # 历史窗口的轮数
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "rerank")   # 检索策略
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "2"))   # 最大工具调用轮数


