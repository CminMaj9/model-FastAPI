# utils/scaler.py
import pickle
import numpy as np
import app_state

class ScalerInverse:
    def __init__(self, scaler_path: str):
        # 加载 scaler_dict
        with open(scaler_path, 'rb') as f:
            self.scaler_dict = pickle.load(f)
        
        # 通过 AppState 获取 node_list
        self.node_list = app_state.node_list  

    def inverse(self, y_norm: np.ndarray) -> np.ndarray:
        """
        参数：
        - y_norm: 标准化后的预测结果，形状为 [N] 或 [N, 7]

        返回：
        - 反归一化后的真实预测结果，形状与输入相同
        """
        y_denorm = np.zeros_like(y_norm)
        
        for i, node in enumerate(self.node_list):
            # 从 scaler_dict 获取节点的最小值和最大值
            min_val, max_val = self.scaler_dict.get(str(node), {}).get('当量', (0, 0))

            # 如果节点不在 scaler_dict 中，跳过
            if min_val == 0 and max_val == 0:
                continue

            # 反归一化
            if y_norm.ndim == 1:
                y_denorm[i] = y_norm[i] * (max_val - min_val) + min_val
            elif y_norm.ndim == 2:
                y_denorm[i, :] = y_norm[i, :] * (max_val - min_val) + min_val

        return y_denorm
