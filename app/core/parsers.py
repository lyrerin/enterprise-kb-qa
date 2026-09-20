import os 
from langchain_core.documents import Document
import re
PAGE_NO_PATTERNS = [
    re.compile(r"^\s*第\s*\d+\s*页\s*(/\s*共\s*\d+\s*页)?\s*$"),   # 第 3 页 / 第 3 页 / 共 12 页
    re.compile(r"^\s*[-—－]?\s*\d+\s*[-—－]?\s*$"),                  # 3 / - 3 - / — 3 —
    re.compile(r"^\s*Page\s*\d+\s*(of\s*\d+)?\s*$", re.I),           # Page 3 of 12
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),   
]

def is_page_no(line:str):
    return any(p.match(line.strip()) for p in PAGE_NO_PATTERNS)
def strip_repeated_lines(
    pages:list[str],
    min_pages:int = 3,
    ratio:float = 0.6,
    head_zone:int = 2,
    tail_zone:int = 2):
    
    pages = ['\n'.join(l for l in p.splitlines() if not is_page_no(l.strip())) for p in pages]

    if len(pages) < min_pages:
        return pages
    def zone(lines:list[str],n:int,tail:bool):
        picked = lines[-n:] if tail else lines[:n]
        return [l.strip() for l in picked if l.strip()]

    threshold = max(min_pages - 1,int(len(pages)*ratio))
    junk: set[str] = set()
    for tail in (False,True):
        counter:dict[str,int] = {}
        for page in pages:
            for line in set(zone(page.splitlines(),tail_zone if tail else head_zone,tail)):
                counter[line] = counter.get(line,0) + 1
        junk |= {l for l,c in counter.items() if c >= threshold}
    if not junk:
        return pages
    return ['\n'.join(l for l in p.splitlines() if l.strip() not in junk) for p in pages]
    
def extract_body_lines(page,header_zone:float = 0.08,footer_zone:float = 0.92):
    """位置法：按 y 坐标把页面上/下边距区域整段丢掉。
    比"跨页重复行"稳——正文里重复出现的句子不会被误删。"""
    top_limit = page.height * header_zone
    bottom_limit = page.height * footer_zone

    rows:dict[float,list] = {}
    for w in page.extract_words():
        rows.setdefault(round(w['top'],1),[]).append(w)

    out = []
    for top in sorted(rows):
        if top < top_limit or top > bottom_limit:
            continue
        line = " ".join(w['text'] for w in sorted(rows[top],key = lambda w:w['x0'])).strip()
        if line and not is_page_no(line):
            out.append(line)
    return out
def sniff_encoding(path:str):
    """自动识别文件编码"""
    raw = open(path,'rb').read(64*1024)
    for enc in ('utf-8-sig','utf-8','gbk'):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return 'utf-8'

def load_docx(path:str):
    """解析 docx 文档"""
    from docx import Document as DocxFile

    doc = DocxFile(path)
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for t in doc.tables:
        rows = [[c.text.strip() for c in r.cells]for r in t.rows]
        if not rows:
            continue
        parts.append(
             "| " + " | ".join(rows[0]) + " |\n"
            + "| " + " | ".join("---" for _ in rows[0]) + " |\n"
            + "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
        )
    return [Document(page_content = '\n\n'.join(parts),
    metadata = {'source':os.path.basename(path),"doc_type":"docx"})]

def load_pdf(path:str):
    """解析 pdf 文档"""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = ['\n'.join(extract_body_lines(p)) for p in pdf.pages]
    except Exception:
        from langchain_community.document_loaders import PyPDFLoader
        raw = PyPDFLoader(path).load()
        pages = strip_repeated_lines([d.page_content for d in raw])

    if is_scanned(path):
        return [Document(page_content='',
            metadata = {'source':os.path.basename(path),"doc_type":"scanned"})]

    return [Document(page_content = '\n\n'.join(pages),
            metadata = {'source':os.path.basename(path),"doc_type":"pdf"})]


def is_scanned(pdf_path:str):
    """判断 pdf 是否为ocr扫描件"""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    total = sum(len((p.extract_text()or '').strip()) for p in reader.pages)

    return total < 20*len(reader.pages)

def load_csv(path: str):
    """csv：一行一个 Document，列名会带进每行内容"""
    from langchain_community.document_loaders import CSVLoader
    return CSVLoader(path, encoding=sniff_encoding(path)).load()
    
def load_txt(path: str):
    """txt：一行一个 Document，列名会带进每行内容"""
    from langchain_community.document_loaders import TextLoader
    return TextLoader(path, encoding=sniff_encoding(path)).load()

PARSERS = {
    ".docx": load_docx,
    ".pdf": load_pdf,
    ".csv": load_csv,
    ".txt": load_txt,
}
SUPPORTED_EXTENSIONS = set(PARSERS)

    
    

