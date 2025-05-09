from fastapi import APIRouter
import torch
import numpy as np
from utils.response import StandardResponse, PredictRequest
import app_state
from utils.logger import logger

router = APIRouter()

@router.post("/day", response_model=StandardResponse, summary="单日预测", description="提供过去7天输入，预测第8天所有节点流量")
async def predict_day(request: PredictRequest):
    """
    接收过去7天的输入特征、上下文信息、邻接矩阵，调用模型预测第8天的流量。
    """

    try:
        # 检查模型是否加载
        if app_state.model is None:
            raise RuntimeError("模型未初始化，请稍后重试")

        # 将输入数据转为 PyTorch 张量，并送入指定设备
        x = torch.tensor(np.array(request.x), dtype=torch.float32).to(app_state.device)        # [B, N, 7, 26]
        context = torch.tensor(np.array(request.context), dtype=torch.float32).to(app_state.device)  # [B, N, 7, 16]
        if app_state.adj is None:
            raise RuntimeError("邻接矩阵未加载，请检查系统初始化流程")

        adj = torch.tensor(app_state.adj, dtype=torch.float32).to(app_state.device)

        print("📊 输入 x 是否有 NaN:", torch.isnan(x).any().item())
        print("📊 输入 context 是否有 NaN:", torch.isnan(context).any().item())
        print("📊 输入 adj 是否有 NaN:", torch.isnan(adj).any().item())
        
        # 禁用梯度计算，加快推理速度
        with torch.no_grad():
            y = app_state.model(x, adj, context)  # [B, N, 1]

        # 确保 y 是一个 PyTorch Tensor，并且转换为 NumPy 数组
        result = y.squeeze().cpu().numpy()  # [N]

        # 打印检查数据类型，确认反归一化操作的输入类型
        print(f"预测结果数据类型: {type(result)}")

        # 确保反归一化数据正确
        denorm = app_state.scaler.inverse(result)

        # 1. 打印反归一化结果本身
        print("🚨 denorm 数组（numpy）:", denorm)

        # 2. 检查是否含有 NaN 或 Inf
        print("🚨 是否包含 NaN:", np.isnan(denorm).any())
        print("🚨 是否包含 Inf:", np.isinf(denorm).any())

        # 3. 打印原始预测结果
        print("🚨 模型输出结果 result:", result)
        
        logger.info("✅ 单日预测完成，节点数=%d", len(result))
        return StandardResponse(status="success", message="预测成功", data=denorm.tolist())

    except Exception as e:
        logger.error(f"❌ 单日预测失败：{str(e)}")
        return StandardResponse(status="error", message=f"单日预测失败: {str(e)}")
