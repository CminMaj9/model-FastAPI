import json
import numpy as np

# 构造输入张量 [1, 178, 7, 26]  => 每个特征值填 0.1
x = np.full((1, 178, 7, 26), 0.1).tolist()

# 构造 context 张量 [1, 178, 7, 16] => 每个值填 0.2
context = np.full((1, 178, 7, 16), 0.2).tolist()

# 打包为 JSON 请求体（不再包含邻接矩阵）
data = {
    "x": x,
    "context": context
}

# 写入文件
with open("test_input_no_adj.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ 已保存 test_input_no_adj.json（不包含邻接矩阵）")