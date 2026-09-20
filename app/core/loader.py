"""多格式文档加载器 —— 支持 TXT、PDF、CSV、DOCX"""
import os
from typing import List
from langchain_core.documents import Document
from app.core.parsers import (
    PARSERS, 
    SUPPORTED_EXTENSIONS,
    
)


def load_document(file_path: str) -> List[Document]:
    """根据文件后缀自动选择加载器"""
    ext = os.path.splitext(file_path)[1].lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"不支持的文件格式: {ext}, 支持的格式: {sorted(SUPPORTED_EXTENSIONS)}")
    
    return parser(file_path)



def load_documents_from_directory(directory: str) -> List[Document]:
    """批量加载目录下所有支持的文档"""
    all_docs = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext in SUPPORTED_EXTENSIONS:
            try:
                docs = load_document(file_path)
                for doc in docs:
                    doc.metadata["source"] = filename
                all_docs.extend(docs)
                print(f"[OK] 已加载: {filename} ({len(docs)} 页/段落)")
            except Exception as e:
                print(f"[FAIL] 加载失败: {filename}, 错误: {e}")

    return all_docs
