from contextlib import asynccontextmanager
from fastapi import FastAPI , Request
from fastapi.middleware.cors import CORSMiddleware
from app.api import documents, qa,auth
from fastapi.responses import JSONResponse
from app.utils.logger import logger
from app.core.exception import AppError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info('企业知识库问答系统启动中...')
    # 启动时检查向量数据库状态
    from app.core.ingestion import get_ingestion_status
    status = get_ingestion_status()
    logger.info(f"知识库状态: {status['total_documents']} 个文档, {status['total_chunks']} 个文本块")
    yield
    logger.info('系统关闭')

app = FastAPI(
    title="企业知识库智能问答系统",
    description="基于 RAG 的企业知识库问答系统，支持多格式文档上传和智能问答",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router,prefix = '/api/documents',tags = ['文档管理'])
app.include_router(qa.router, prefix='/api/qa', tags=['智能问答'])
app.include_router(auth.router, prefix='/api/auth', tags=['认证'])

@app.get("/")
def read_root():
    return {
        "message": "欢迎来到企业知识库问答问答系统",
        'name': '企业知识库问答问答系统',
        'version': '1.0.0',
        'docs': '/docs'
        }

@app.get('/health')
async def health_check():
    from app.core.ingestion import get_ingestion_status
    status = get_ingestion_status()
    return{
        'status':'healthy',
        'knowledge_base':status
    }
from app.core.exception import AppError
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    logger.info(f"业务异常 %s %s -> %s", request.method, request.url.path, exc.message)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    
    logger.exception(f"全局异常: {exc}")
    return JSONResponse(status_code=500, content={"detail": '服务器内部错误'})
