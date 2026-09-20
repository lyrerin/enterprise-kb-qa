from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from langchain_core.documents import Document
from app.core.parent_store import save_parent
import os
import re



# 中文分隔符顺序：段落 → 换行 → 句 → 分号 → 逗号 → 空格 → 硬切
SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]

def get_text_splitter(
    chunk_size : int = CHUNK_SIZE,
    chunk_overlap : int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SEPARATORS,
            length_function = len,
        )
def split_document(file_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """加载并分割单个文档"""
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(docs)
    for i,chunk in enumerate(chunks):
        chunk.metadata['source'] = file_path
        chunk.metadata['chunk_index'] = i
    return chunks

class HeadingSplitter():
    """按章节标题切；整节先存成父块，再切成带 parent_id 的子块"""
    def __init__(self,pattern:str , chunk_size: int = 400,chunk_overlap:int = 40):
        self.pattern = re.compile(pattern,re.M)
        self.inner = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,chunk_overlap = chunk_overlap,separators = SEPARATORS,
        )

    def _emit(self,parent_id:str ,content:str ,metadata:dict ,out:list):
        """存一份父块 → 把父块切成子块 → 子块带 parent_id 落进 out"""
        save_parent(parent_id,content,metadata)
        for piece in self.inner.split_text(content):
            out.append(Document(page_content=piece,
                                metadata={**metadata,'parent_id':parent_id}))

    def split(self,docs):
        out = []    
        for doc in docs:
            text = doc.page_content
            source = doc.metadata.get('source','未知')
            marks = [m.start() for m in self.pattern.finditer(text)]
            if not marks:
                out.extend(self.inner.split_documents([doc]))
                continue
            if marks[0] > 0 :
                # 第一章之前那段（公司标题等）也存一个父块，否则展开时会被丢掉
                self._emit(f'{source}#pre',text[:marks[0]].strip(),{**doc.metadata},out)
            for i ,start in enumerate(marks):
                end = marks[i + 1] if i + 1<len(marks) else len(text)
                section = text[start:end].strip()
                title = section.splitlines()[0].strip()
                content = f'[{title}]\n{section}'
                # parent_id 用确定性 id：重导同一个文件会覆盖同一行，不会越积越多
                self._emit(f'{source}#{i}',content,{**doc.metadata,'section_path':title},out)
        return out

class WholeBlockSplitter():
    """整块不切（表格、条款）：一个 Document 就是一个块"""
    def split(self,docs):
        return docs

SPLITTER_STRATEGY = {
    'handbook': lambda: HeadingSplitter(r'^第[一二三四五六七八九十]+[章节条]', chunk_size=400),
    'table': lambda: WholeBlockSplitter(),
    'default': lambda: None,
}
HEADING_MARK =re.compile(r'^第[一二三四五六七八九十]+[章节条]', re.M)

def detect_doc_type(docs,source):
    ext = os.path.splitext(source)[1].lower()
    if ext == '.csv':
        return 'table'
    text = '\n'.join([doc.page_content for doc in docs])
    if len(list(HEADING_MARK.finditer(text))) >= 3:
        return 'handbook'
    return 'default'
    

def split_by_type(docs, doc_type: str = 'default'):
    splitter = SPLITTER_STRATEGY.get(doc_type,SPLITTER_STRATEGY['default'])()
    return splitter.split(docs) if splitter else get_text_splitter().split_documents(docs)
