import asyncio
import httpx
import random
import time

# CẤU HÌNH
# Thay đổi URL này thành URL Ingress của bạn hoặc Service DNS nội bộ
TARGET_URL = "http://my-app-stable.default.svc.cluster.local" 
CONCURRENT_REQUESTS = 10 # Số lượng request gửi song song

async def send_request(client):
    try:
        # Giả lập hành vi người dùng: random một chút thời gian chờ
        await asyncio.sleep(random.uniform(0.01, 0.1))
        response = await client.get(TARGET_URL)
        print(f"[{time.strftime('%H:%M:%S')}] Status: {response.status_code} | Version: {response.json().get('version', 'unknown')}")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    print(f"--- Bắt đầu bơm traffic vào {TARGET_URL} ---")
    async with httpx.AsyncClient() as client:
        while True:
            tasks = [send_request(client) for _ in range(CONCURRENT_REQUESTS)]
            await asyncio.gather(*tasks)
            # Nghỉ một chút để không làm sập cluster của bạn nếu cấu hình thấp
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(main())