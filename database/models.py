from sqlalchemy import Column, Integer, String, Boolean, Text, TIMESTAMP, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# 模型版本表
class ModelVersion(Base):
    __tablename__ = "model_version"

    id = Column(Integer, primary_key=True, index=True)  # 添加索引
    name = Column(String(100), nullable=False, index=True)  # 添加索引
    description = Column(Text)
    file_path = Column(Text, nullable=False)
    is_active = Column(Boolean, default=False, index=True)  # 对常用查询的字段添加索引
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)  # 添加索引

# 邻接矩阵表
class AdjMatrix(Base):
    __tablename__ = "adj_matrix"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text)
    file_path = Column(Text, nullable=False)
    is_active = Column(Boolean, default=False, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)

# 服务区节点表
class ServiceNode(Base):
    __tablename__ = "service_node"

    id = Column(Integer, primary_key=True, index=True)
    node_index = Column(Integer, unique=True, nullable=False, index=True)  # 添加索引
    node_name = Column(String(100), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    description = Column(Text)

# 原始流量文件表
class TrafficFile(Base):
    __tablename__ = "traffic_file"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    upload_user = Column(String(100))
    file_type = Column(String(10))
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
