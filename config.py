from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 数据库连接
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL is not set")

# 模型参数
NUM_NODES = 178
INPUT_DIM = 26
CONTEXT_DIM = 16
SEQ_LEN = 7
OUTPUT_DIM = 1
GAT_HIDDEN_DIM = 64
NODE_EMB_DIM = 16
ATTN_HEADS = 4