# router/upload.py
import oss2
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from model.gta_model import GAT_Transformer_ContextFusion
from config import NUM_NODES, INPUT_DIM, CONTEXT_DIM, SEQ_LEN, GAT_HIDDEN_DIM, NODE_EMB_DIM, OUTPUT_DIM, ATTN_HEADS
import app_state
import pickle
import numpy as np
import tempfile
import os
import torch
from utils.scaler import ScalerInverse

router = APIRouter()

# OSS 配置信息
ACCESS_KEY_ID = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
ACCESS_KEY_SECRET = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
ENDPOINT = "oss-cn-hangzhou.aliyuncs.com"  # 例如：oss-cn-shanghai.aliyuncs.com
BUCKET_NAME = "jaygesbucket"

auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

# 从 OSS 获取文件的函数
async def load_file_from_oss(file_name: str):
    try:
        result = bucket.get_object(file_name)
        return result.read()  # 返回二进制数据
    except Exception as e:
        raise RuntimeError(f"文件加载失败: {str(e)}")

# 加载 node_list 文件
@router.get("/load_node_list")
async def load_node_list():
    node_list_path = "uploads/node_list/node_list.pkl"
    try:
        node_list_data = await load_file_from_oss(node_list_path)
        app_state.node_list = pickle.loads(node_list_data)  # 存储到全局变量
        return JSONResponse(content={"message": "Node list 文件加载成功", "node_list_size": len(node_list_data)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载 Node list 文件失败: {str(e)}")

    
# 加载 scaler 文件
@router.get("/load_scaler")
async def load_scaler():
    scaler_path = "uploads/scaler/scaler.pkl"
    try:
        scaler_data = await load_file_from_oss(scaler_path)

        # 将字节数据写入临时文件
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(scaler_data)
            tmp_file_path = tmp_file.name

        # 使用 ScalerInverse 加载临时文件
        app_state.scaler = ScalerInverse(tmp_file_path)  # 将临时文件传递给 ScalerInverse
        
        # 删除临时文件
        os.remove(tmp_file_path)

        return {"message": "Scaler 文件加载成功", "scaler_size": len(scaler_data)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载 Scaler 文件失败: {str(e)}")

# 加载邻接矩阵文件
@router.get("/load_adj_matrix")
async def load_adj_matrix():
    adj_matrix_path = "uploads/adj_matrix/adjacency_matrix_initial.npy"
    try:
        # 获取邻接矩阵的二进制内容
        adj_matrix_data = await load_file_from_oss(adj_matrix_path)

        # 将二进制数据保存到临时文件中
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(adj_matrix_data)  # 写入二进制数据
            tmp_file_path = tmp_file.name

        # 使用 np.load() 加载临时文件
        app_state.adj = np.load(tmp_file_path)  # 保存到全局变量

        # 删除临时文件
        os.remove(tmp_file_path)

        return JSONResponse(content={"message": "邻接矩阵加载成功", "adj_matrix_size": len(adj_matrix_data)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载邻接矩阵失败: {str(e)}")

# 加载模型文件
@router.get("/load_model")
async def load_model(model_path: str):
    try:
        model_data = await load_file_from_oss(model_path)  # 从 OSS 获取模型文件的二进制数据
        
        # 将二进制数据保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(model_data)
            tmp_file_path = tmp_file.name

        # ==== 初始化模型结构 ====
        model = GAT_Transformer_ContextFusion(
            num_nodes=NUM_NODES,
            in_channels=INPUT_DIM,
            gat_hidden=GAT_HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            seq_len=SEQ_LEN,
            context_dim=CONTEXT_DIM,
            node_emb_dim=NODE_EMB_DIM,
            heads=ATTN_HEADS
        ).to(app_state.device)  # 使用 app_state.device 获取设备

        # ==== 加载模型参数 ====
        model.load_state_dict(torch.load(tmp_file_path, map_location=app_state.device))  # 使用临时文件加载模型
        model.eval()

        # 删除临时文件
        os.remove(tmp_file_path)

        # 打印模型类型
        print(f"✅ 加载的模型类型: {type(model)}")
        
        app_state.model = model  # 存储加载后的模型到全局变量

        return {"message": "模型文件加载成功", "model_size": len(model_data)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载模型文件失败: {str(e)}")