# app_state.py
import torch

# 说明：这些变量在程序生命周期内只会被加载一次，作为全局状态缓存使用。

model = None  # PyTorch 模型对象
scaler = None  # ScalerInverse 实例
node_list = None  # 节点列表 List[str]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adj = None  # 邻接矩阵 np.ndarray