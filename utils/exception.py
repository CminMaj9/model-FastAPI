from fastapi.responses import JSONResponse
from fastapi import Request
from utils.response import StandardResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# 捕获 422 请求参数校验错误
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=StandardResponse(
            status="error",
            message="请求参数校验失败",
            data=exc.errors()
        ).model_dump()
    )

# 捕获 FastAPI 抛出的 HTTP 异常
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=StandardResponse(
            status="error",
            message=exc.detail,
            data=None
        ).model_dump()
    )

# 捕获未处理的系统异常
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=StandardResponse(
            status="error",
            message=f"服务器内部错误：{str(exc)}",
            data=None
        ).model_dump()
    )
