import os 
from typing import List
from collections import defaultdict
from langchain_core.documents import Document
from app.core.loader import load_document, load_documents_from_directory
from app.core.splitter import split_by_type, detect_doc_type
from app.core.embeddings import get_embeddings
from app.config import CHROMA_DIR
from langchain_community.vectorstores import Chroma

def ingest_file(file_path:str,clear_first: bool = False) -> int:
    """导入单个文件"""
    source = os.path.basename(file_path)
    docs = load_document(file_path)
    for doc in docs:
        doc.metadata['source'] = source
    print(f'加载文档:{source},共{len(docs)}页/段落')

    doc_type = detect_doc_type(docs,source)
    chunks = split_by_type(docs,doc_type)
    print(f'分割文档为{len(chunks)}个文本块 (切分策略: {doc_type})')

    embeddings = get_embeddings()
    print(f'文本向量化成功')

    if clear_first:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
    else:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,

        )
        vectorstore._collection.delete(where={'source': source})
        vectorstore.add_documents(chunks)   

    print(f"✅ 导入完成! 文件: {source}, 块数: {len(chunks)}")
    return len(chunks)    


def ingest_directory(directory: str, clear_first: bool = True) -> int:
    """批量导入整个目录"""
    if clear_first:
        _clear_vectorstore()

    total_chunks = 0
    docs = load_documents_from_directory(directory)
    if not docs:
        print("⚠️  没有找到支持的文档")
        return 0

    # 按文件名分组：每个文件的 doc_type 只作用于它自己的块
    groups = defaultdict(list)
    for doc in docs:
        groups[doc.metadata.get('source','')].append(doc)

    chunks = []
    for src,group in groups.items():
        doc_type = detect_doc_type(group,src)
        group_chunks = split_by_type(group,doc_type)
        print(f"✂️  {src}: {doc_type} 策略 → {len(group_chunks)} 个块")
        chunks.extend(group_chunks)

    print(f"✂️  总计分割: {len(chunks)} 个文本块")
    
    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    
    print(f"✅ 批量导入完成! 总计: {len(chunks)} 块")
    return len(chunks)

def get_ingestion_status() -> dict:
    from app.core.vectorstore import load_vectorstore
    try:
        vectorstore = load_vectorstore()
        collection = vectorstore._collection
        count = collection.count()

        results = collection.get()
        sources = set()
        if results.get('metadatas'):
            for meta in results['metadatas']:
                if meta and 'source' in meta:
                    sources.add(meta['source'])

        return{
            'total_chunks': count,
            'total_documents' : len(sources),
            'sources' : list(sources),
        }
    except Exception as e:
        return {"total_chunks": 0, "total_documents": 0, "sources": []}

def _clear_vectorstore():
    """清空向量数据库"""
    import shutil
    import chromadb

    # 先通过 ChromaDB 客户端删除集合，正确释放 SQLite 文件锁（Windows 下尤其重要）
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        for collection in client.list_collections():
            client.delete_collection(collection.name)
    except Exception as e:
        print(f"⚠️  通过客户端删除集合失败: {e}")

    # 再删除目录（ignore_errors 防止文件锁导致残留报错）
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        os.makedirs(CHROMA_DIR, exist_ok=True)

    # 父块是第二个库，必须一起清，否则清空后检索会捞出幽灵内容
    from app.core.parent_store import clear_parents
    clear_parents()
    print("🗑️  向量数据库已清空")

if __name__ == "__main__":
    # ingest_file("knowledge_docs/员工手册.pdf")
    # _clear_vectorstore()
    ingest_directory('knowledge_docs')

