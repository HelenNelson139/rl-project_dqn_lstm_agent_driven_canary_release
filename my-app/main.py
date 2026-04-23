import os
import time
import random
from fastapi import FastAPI, Response
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

# Lấy kịch bản từ biến môi trường (mặc định là 'healthy')
# Các giá trị: healthy, latency_leak, error_bomb, critical_crash
SCENARIO = os.getenv("APP_SCENARIO", "healthy")
VERSION = os.getenv("APP_VERSION", "v1.0.0")

# Biến toàn cục để giả lập Memory Leak
memory_leak_list = []

@app.get("/")
async def root():
    # 1. Kịch bản: Latency Leak (Càng chạy càng chậm)
    if SCENARIO == "latency_leak":
        time.sleep(0.5 + (len(memory_leak_list) * 0.01)) 
        memory_leak_list.append(" " * 1024 * 1024) # Leak 1MB mỗi request

    # 2. Kịch bản: Critical Crash (Lỗi 500 liên tục)
    if SCENARIO == "critical_crash":
        if random.random() < 0.5:
            return Response(content="Internal Server Error", status_code=500)

    # 3. Kịch bản: Error Bomb (Chỉ lỗi khi traffic cao - giả lập bằng random)
    if SCENARIO == "error_bomb":
        if random.random() < 0.2:
            return Response(content="Service Unavailable", status_code=503)

    # 4. Kịch bản: Healthy (Phản hồi nhanh)
    return {
        "version": VERSION,
        "scenario": SCENARIO,
        "status": "online",
        "latency": "fast"
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}