# 系统架构

├── main.py                         # FastAPI 程序主入口
├── config.py                       # 配置文件（路径、模型参数等）
├── database
│   ├── models.py                   # SQLAlchemy ORM 模型
│   ├── schemas.py                  # Pydantic 模型定义
│   └── session.py                  # 异步 SQLAlchemy 数据库连接配置
├── model
│   └── gta_model.py                # GTA-Net v7 模型实现
├── router
│   ├── model_info.py               # 模型版本信息接口
│   └── predict.py                  # 预测接口（单日、一周）
├── utils
│   ├── db.py                       # psycopg2 连接（这个其实后面用不上，可以删了）
│   └── response.py                 # 标准响应格式定义（包括请求结构）
├── uploads                         # 上传文件统一存储位置
│   ├── adj                         # 邻接矩阵文件
│   ├── model                       # 模型文件（.pth）
│   └── traffic                     # 原始流量文件
└── requirements.txt                # Python依赖管理（后续可以添加）