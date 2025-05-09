# pyright: reportGeneralTypeIssues=false

from typing import List, Optional, Any
from pydantic import BaseModel, conlist, Field

# ==== 类型定义 ====
FeatureVector = conlist(float, min_length=26, max_length=26)     # 每个时间点的特征向量
ContextVector = conlist(float, min_length=16, max_length=16)     # 每个时间点的上下文向量
AdjRow = conlist(float, min_length=178, max_length=178)          # 邻接矩阵一行（178维）

# ==== 请求数据结构 ====
class PredictRequest(BaseModel):
    x: List[List[List[FeatureVector]]] = Field(..., description="输入特征序列，形状为 [B, N, 7, 26]")
    context: List[List[List[ContextVector]]] = Field(..., description="上下文信息序列，形状为 [B, N, 7, 16]")

    class Config:
        extra = "forbid"

# ==== 响应结构 ====
class StandardResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None