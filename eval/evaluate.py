"""RAG 评估：召回率 / 精确率 / 忠实度（最小可用版）

用法：python eval/evaluate.py
先保证：.env 里有 DEEPSEEK_API_KEY（第 6 项），且知识库已导入文档。
"""
import json
import statistics
import sys
from pathlib import Path

# ↓↓↓ 这两行不在 v1 归档原文里，是额外加的：直接 `python eval/evaluate.py` 跑时，
#     sys.path 里只有 eval/ 目录，import app 会失败，加上项目根目录两种跑法都能用。
#     不想要可以删掉这两行，改成 `python -m eval.evaluate` 跑。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ↑↑↑

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.core.retriever import retrieve_with_rerank      # 默认检索策略（可换）
from app.core.rag_chain import ask_with_sources          # answer + sources(模型实际用的上下文)
from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

EVAL_SET = Path(__file__).parent / "eval_set.json"


def _is_relevant(doc: Document, item: dict) -> bool:
    """判断一个检索结果是否'命中'该题的黄金答案"""
    src = doc.metadata.get("source", "")
    if src in item.get("expected_sources", []):
        return True
    text = doc.page_content
    return any(kw in text for kw in item.get("expected_keywords", []))


def evaluate_retrieval(retriever, items, k: int = 3) -> dict:
    """检索指标：命中率(召回率@题目粒度) + 精确率@k"""
    hits, precisions = [], []
    for item in items:
        docs = retriever(item["question"])[:k]
        relevant = [d for d in docs if _is_relevant(d, item)]
        hits.append(1.0 if relevant else 0.0)                       # 该题有没有检到黄金块
        precisions.append(len(relevant) / len(docs) if docs else 0.0)  # 检出的块里相关占比
    return {
        "召回率(hit_rate)": round(statistics.mean(hits), 4),
        "精确率(precision@k)": round(statistics.mean(precisions), 4),
    }


def _judge_llm():
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,          # 当裁判要稳定，用低温
    )


def evaluate_faithfulness(items) -> dict:
    """生成指标：忠实度 = 回答论断被上下文支持的比例（LLM 当裁判）"""
    judge = _judge_llm()
    scores = []
    for item in items:
        result = ask_with_sources(item["question"])        # 走完整 RAG 链路
        answer = result["answer"]
        context = "\n".join(s["content"] for s in result["sources"])
        prompt = (
            "你是严谨的审查员。逐句判断下面回答中的论断，有多少能被参考资料支持。\n"
            f"参考资料：\n{context}\n\n回答：{answer}\n\n"
            "只输出一个 0~1 之间的小数（受支持论断的比例），不要解释。"
        )
        try:
            score = float(judge.invoke(prompt).content.strip())
        except Exception:
            score = 0.0
        scores.append(score)
    return {"忠实度(faithfulness)": round(statistics.mean(scores), 4)}


def run():
    items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    # 检索策略可插拔：想对比哪一路就把 lambda 换成哪一路
    retriever = lambda q: retrieve_with_rerank(q, recall_k=20, final_k=3)
    # retriever = lambda q: retrieve_with_rewrite(q, k=3)
    # retriever = lambda q: hybrid_search(q, k=3)
    print("检索指标:", evaluate_retrieval(retriever, items))
    print("生成指标:", evaluate_faithfulness(items))


if __name__ == "__main__":
    run()
