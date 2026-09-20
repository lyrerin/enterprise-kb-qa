from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,DEEPSEEK_MODEL,FINAL_K
from app.core.retriever import retrieve, expand_to_parents

def build_rag_chain():
    """构建 RAG 链
    
    流程：
    用户问题 → 向量检索(找到相关文档) → 拼入提示词 → LLM生成答案 → 输出
    """
    
    # 1. 检索不再自己 load_vectorstore + as_retriever(k=3)（那样绕过了总闸，
    #    重排/混合/改写/父块全享受不到）。统一走 retrieve()，策略由 RETRIEVAL_STRATEGY 决定，
    #    再回表展开成整节父块，拼成带 [文档i] 来源标注的 context 字符串。
    
    # 2. 设计提示词模板
    #    这是 RAG 最核心的 prompt，质量直接决定答案好坏
    template = """你是一个专业的企业知识库助手。请根据以下文档内容回答用户问题。

## 规则
- 只使用下面提供的文档内容来回答
- 如果文档中没有相关信息，请明确说"根据现有资料，无法回答此问题"
- 回答时尽量引用原文的具体信息（如天数、金额、流程等）
- 回答要简洁、准确、结构化

## 参考文档
{context}

## 用户问题
{question}

## 回答"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 3. 初始化 LLM
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,  # 知识问答用低温，减少幻觉
    )
    
    # 4. 组装链
    #    RunnablePassthrough 用于把原始问题原样传递给 prompt 的 question 变量
    rag_chain = (
        {"context": RunnableLambda(lambda q: expand_to_parents(retrieve(q,k=FINAL_K))),
         "question":  RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def ask(question: str) -> str:
    """向知识库提问"""
    chain = build_rag_chain()
    answer = chain.invoke(question)
    return answer


# ====== 测试问答 ======

def demo_qa():
    """测试各种类型的问题"""
    
    questions = [
        # 精确事实类
        "员工入职满一年有多少天年假？",
        # 流程类
        "报销差旅费的流程是什么？",
        # 数字类
        "团建活动的预算是多少？",
        # 条件类
        "什么条件下可以申请晋升？",
        # 陷阱类（文档中没有的信息）
        "公司有没有住房补贴？",
        # 综合类
        "总结一下公司的员工福利有哪些？",
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"❓ {q}")
        print(f"{'='*60}")
        answer = ask(q)
        print(f"🤖 {answer}")
        print()






def ask_with_sources(question: str) -> dict:
    """提问并返回答案 + 引用来源"""
    from app.core.router import run_sql_path
    sql_result = run_sql_path(question)
    if sql_result is not None:
        return sql_result
    # vectorstore = load_vectorstore()
    # retriever = vectorstore.as_retriever(
    #     search_kwargs={"k": 3}  # 每次检索最相关的3个文本块
    # )
    retrieved_docs = retrieve(question,k=FINAL_K)
    # 命中子块 → 回表换成整节父块（没有父块的块用原文兜底），带 [文档i] 来源标注
    context = expand_to_parents(retrieved_docs)

    template = """你是一个专业的企业知识库助手。请根据以下文档内容回答用户问题。

## 规则
- 只使用下面提供的文档内容来回答
- 如果文档中没有相关信息，请明确说"根据现有资料，无法回答此问题"
- **重要：回答中使用 [文档1]、[文档2] 标注信息来源**

## 参考文档
{context}

## 用户问题
{question}

## 回答（请标注引用来源）"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return{
        'question':question,
        'answer':answer,
        'sources':[
            {'content':doc.page_content[:200],'source':doc.metadata.get('source')} 
            for doc in retrieved_docs
        ]
    }
def demo_rag_vs_no_rag():
    """对比：有知识库 vs 纯 LLM 的回答差异"""
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
    )

    questions = [
        "宏远科技公司员工入职满一年有多少天年假？",
        "宏远科技的绩效考核分为哪些等级？",
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"❓ {q}")
        print(f"{'='*60}")
        
        # 无 RAG：直接问 LLM
        no_rag_answer = llm.invoke(q).content
        print(f"\n🚫 无RAG（纯LLM）:\n{no_rag_answer}")
        
        # 有 RAG：通过知识库
        rag_answer = ask(q)
        print(f"\n✅ 有RAG（知识库增强）:\n{rag_answer}")
if __name__ == "__main__":
    print("=" * 60)
    print("📋 基础 RAG 问答测试")
    print("=" * 60)
    demo_qa()
    print("=" * 60)
    print("📋 RAG 对比测试：有知识库 vs 纯 LLM 的回答差异")
    print("=" * 60)
    demo_rag_vs_no_rag()