import os
import shutil
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models.schemas import DocumentUploadResponse, KnowledgeBaseStatus
from app.core.parsers import SUPPORTED_EXTENSIONS
from app.core.exception import AppError
from app.utils.logger import logger



router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post('/upload', response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传单个文档
    支持格式: PDF, Word (.docx/.doc), TXT, CSV
    """
    ext = os.path.splitext(file.filename or '')[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f'不支持的文件格式: {ext}, 支持的格式: {list(SUPPORTED_EXTENSIONS)}'
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    try:
        from app.core.ingestion import ingest_file
        chunks_count = ingest_file(file_path, clear_first=False)
    except AppError:
        raise
    except Exception :
        logger.exception("文档处理失败: %s", file.filename)
        raise 

    return DocumentUploadResponse(
        filename=file.filename,
        chunks_count=chunks_count,
        message=f'文档已成功导入，共{chunks_count}个文本块',
    )


@router.post('/upload/batch')
async def upload_documents(files: list[UploadFile] = File(...)):
    """批量上传文档"""
    results = []
    for file in files:
        ext = os.path.splitext(file.filename or '')[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append({
                'filename': file.filename,
                'status': 'skipped',
                'reason': f'不支持的格式: {ext}'
            })
            continue

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, 'wb') as f:
            f.write(await file.read())

        try:
            from app.core.ingestion import ingest_file
            count = ingest_file(file_path, clear_first=False)
            results.append({
                'filename': file.filename,
                'status': 'success',
                'chunks': count
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'status': 'error',
                'reason': str(e)
            })

    return {"total": len(files), "results": results}

@router.get("/status", response_model=KnowledgeBaseStatus)
async def knowledge_base_status():
    """查看知识库状态"""
    from app.core.ingestion import get_ingestion_status
    status = get_ingestion_status()
    return KnowledgeBaseStatus(**status)


@router.delete("/clear")
async def clear_knowledge_base():
    """清空知识库"""
    from app.core.ingestion import _clear_vectorstore
    _clear_vectorstore()
    return {"message": "知识库已清空"}