# schemas.py
from pydantic import BaseModel
from datetime import datetime
from pydantic import BaseModel, constr, condecimal
from typing import Optional

# 模型版本表的 Pydantic 模型
class ModelVersionSchema(BaseModel):
    id: int
    name: constr(min_length=1, max_length=100)
    description: Optional[str]
    file_path: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# 邻接矩阵表的 Pydantic 模型
class AdjMatrixSchema(BaseModel):
    id: int
    name: constr(min_length=1, max_length=100)  # 添加长度限制
    description: Optional[str]
    file_path: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# 服务区节点表的 Pydantic 模型
class ServiceNodeSchema(BaseModel):
    id: int
    node_index: int
    node_name: constr(min_length=1, max_length=100)  # 添加长度限制
    latitude: Optional[condecimal(max_digits=9, decimal_places=6)]  # 添加精度限制
    longitude: Optional[condecimal(max_digits=9, decimal_places=6)]  # 添加精度限制
    description: Optional[str]

    class Config:
        from_attributes = True

# 原始流量文件表的 Pydantic 模型
class TrafficFileSchema(BaseModel):
    id: int
    filename: str
    file_path: str
    upload_user: Optional[str]
    file_type: Optional[str]
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True