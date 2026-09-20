"""业务异常：给用户看安全话术，给日志留真实错误"""

class AppError(Exception):
    status_code = 500
    message = "服务器内部错误"
    
    def __init__(self, message: str | None = None):
        if message is not None:
            self.message = message
        super().__init__(self.message)

class KnowledgeError(AppError):
    status_code = 400
    message = "知识库为空，请先上传文档"

class UnsupportedFileError(AppError):
    status_code = 400
    message = "不支持的文件类型"

class PermissionDeniedError(AppError):
    status_code = 403
    message = "没有权限执行该操作"