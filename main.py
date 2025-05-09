from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
from router import predict, model_info
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from utils.exception import (
    validation_exception_handler,
    http_exception_handler,
    global_exception_handler
)
from utils.logger import logger
import router.upload as upload

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 正在启动 GTA-Net 服务...")
    try:
        # 获取并加载模型到全局状态
        await upload.load_model("uploads/model/model_v7_best.pth")  # 直接调用 upload 中的 load_model 函数

        # 调用上传接口来加载 node_list, scaler 和 adj 文件
        await upload.load_node_list()  # 加载 node_list 文件
        await upload.load_scaler()     # 加载 scaler 文件
        await upload.load_adj_matrix() # 加载邻接矩阵文件

        logger.info("✅ 所有模型相关组件已成功加载！")
    except Exception as e:
        logger.error(f"❌ 启动失败：{str(e)}")
        raise e
    yield

app = FastAPI(
    title="GTA-Net 模型服务",
    version="1.0.0",
    lifespan=lifespan
)

# 注册全局异常处理器
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

app.include_router(model_info.router, prefix="/model", tags=["模型信息"])
app.include_router(predict.router, prefix="/predict", tags=["预测功能"])

@app.get("/health")
def health_check():
    logger.info("🔍 健康检查接口被调用")
    return {"status": "success", "message": "服务运行正常"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)