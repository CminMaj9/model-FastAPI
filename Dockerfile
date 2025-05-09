# 使用官方 Python 镜像
FROM python:3.12.8

# 设置工作目录
WORKDIR /app

# 复制 .env 文件
COPY .env /app/.env

# 复制 requirements.txt 并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 运行 Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]