# GTA-Net v7: 浙江省交通流量预测模型服务

GTA-Net v7 是一个基于图注意力网络 (GAT) 和 Transformer 的混合模型，专为高精度交通流量预测而设计。本服务采用 FastAPI 框架构建，提供了交通流量单日预测功能，能够处理复杂的时空关系和上下文信息。

## 系统架构

本项目采用模块化设计，主要结构如下：

```plaintext
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
└── requirements.txt                # Python依赖管理
```

## 核心功能

1. **混合模型架构**：结合图注意力网络 (GAT) 捕获空间依赖关系，Transformer 处理时间序列特征，并通过 ContextMLP 融合上下文信息。
2. **交通流量预测**：基于历史 7 天的交通数据，预测第 8 天的交通流量情况，支持批量处理多个位置的预测请求。
3. **模型管理**：提供模型版本管理和状态检查接口，确保服务的可靠性和可维护性。
4. **数据预处理**：包含标准化和反标准化模块，确保输入数据的一致性和预测结果的可解释性。
5. **异常处理**：完善的异常处理机制，确保服务的稳定性和用户友好的错误提示。

## 技术亮点

- **图注意力机制**：通过 GAT 捕获交通网络中节点间的复杂依赖关系，自适应学习邻接节点的重要性权重。
- **时序建模**：利用 Transformer 架构处理时间序列数据，有效捕捉长期依赖关系。
- **上下文融合**：创新地将天气、时间等上下文信息融入预测模型，提升预测准确性。
- **异步处理**：采用 FastAPI 的异步特性，高效处理并发请求，提升系统吞吐量。
- **阿里云 OSS 集成**：实现模型文件和数据的云端存储与管理，支持动态加载和更新。

## 安装与部署

### 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/MacOS
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

创建 `.env` 文件并配置以下环境变量：

```plaintext
DATABASE_URL=postgresql+asyncpg://user:password@host:port/dbname
ALIBABA_CLOUD_ACCESS_KEY_ID=your_access_key_id
ALIBABA_CLOUD_ACCESS_KEY_SECRET=your_access_key_secret
```

### 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API 文档

服务启动后，可通过以下路径访问交互式 API 文档：

```plaintext
http://localhost:8000/docs
```

### 主要接口

1. **健康检查**

   - 路径：`GET /health`
   - 功能：检查服务是否正常运行

2. **模型信息**

   - 路径：`GET /model/info`
   - 功能：获取当前活跃模型的详细信息

3. **模型状态检查**

   - 路径：`GET /model/check`
   - 功能：检查模型是否成功加载

4. **单日流量预测**

   - 路径：`POST /predict/day`

   - 功能：基于过去 7 天数据预测第 8 天的交通流量

   - 请求格式：

     ```json
     {
       "x": [[[[0.1, 0.2, ..., 0.26], ...], ...], ...],  # 输入特征 [B, N, 7, 26]
       "context": [[[[0.1, 0.2, ..., 0.16], ...], ...], ...]  # 上下文信息 [B, N, 7, 16]
     }
     ```

## 模型训练与优化

GTA-Net v7 模型的核心架构定义在model/gta_model.py中，主要包含以下组件：

- `GraphAttentionLayer`：实现图注意力机制，捕获空间依赖关系
- `TemporalTransformerBlock`：基于 Transformer 的时序建模模块
- `GAT_Transformer_ContextFusion`：整合空间、时间和上下文信息的主模型

模型训练和优化的关键参数可在config.py中配置，包括：

- 节点数量：`NUM_NODES=178`
- 输入特征维度：`INPUT_DIM=26`
- 上下文维度：`CONTEXT_DIM=16`
- 序列长度：`SEQ_LEN=7`
- GAT 隐藏层维度：`GAT_HIDDEN_DIM=64`

## 贡献与支持

欢迎对本项目提出问题、建议或贡献代码。如需帮助，请通过以下方式联系：

- 项目地址：https://github.com/your-repo/gta-net
- 问题反馈：https://github.com/your-repo/gta-net/issues

## 许可证

本项目采用MIT 许可证。