# model_info.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from utils.response import StandardResponse
from database.models import ModelVersion
from database.session import get_db
from database.schemas import ModelVersionSchema
import app_state

router = APIRouter()

@router.get("/info", response_model=StandardResponse)
async def get_model_info(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(ModelVersion).where(ModelVersion.is_active == True).limit(1))
        model = result.scalar_one_or_none()
        if not model:
            return StandardResponse(status="fail", message="未找到启用模型")
        # 使用 Pydantic 的 model_validate 方法,这个方法用于将 ORM 对象的属性转换为 Pydantic 模型实例
        model_schema= ModelVersionSchema.model_validate(model.__dict__)
        # 将 Pydantic 模型实例序列化为字典
        return StandardResponse(status="success", message="查询成功", data=model_schema.model_dump())
    except Exception as e:
        return StandardResponse(status="error", message=str(e))
    
@router.get("/check", response_model=StandardResponse)
async def check_model_loaded():
    """
    检查当前模型是否已加载成功
    """
    if app_state.model is None:
        return StandardResponse(status="fail", message="模型尚未加载", data="完蛋了！！！")
    return StandardResponse(status="success", message="模型已成功加载", data="六百六十六！")