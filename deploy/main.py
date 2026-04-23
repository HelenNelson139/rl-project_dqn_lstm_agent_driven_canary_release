import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch.nn as nn

# --- 1. KIẾN TRÚC MÔ HÌNH (DRQN) ---
class DRQN(nn.Module):
    def __init__(self, n_obs=8, n_actions=5):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(n_obs, 64) 
        self.lstm = nn.LSTM(64, 128, batch_first=True) 
        self.fc2 = nn.Linear(128, n_actions) 

    def forward(self, x, hidden=None):
        x = torch.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        # Lấy timestep cuối cùng của chuỗi (Sequence length = 10) [cite: 35, 81]
        x = self.fc2(x[:, -1, :]) 
        return x, hidden

# --- 2. CẤU HÌNH HỆ THỐNG ---
MODEL_PATH = "model_canary_drqn.pth"
SEQ_LENGTH = 10 
DEVICE = torch.device("cpu") 

app = FastAPI(title="Canary AI Agent Service")

# Load model đã huấn luyện
model = DRQN(n_obs=8, n_actions=5).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

# --- 3. ĐỊNH NGHĨA DỮ LIỆU ---
class MetricPoint(BaseModel):
    weight: float
    e_canary: float
    e_stable: float
    l_canary: float
    l_stable: float
    cpu: float
    mem: float
    rps: float

class InferenceRequest(BaseModel):
    history: List[MetricPoint]

# --- 4. ENDPOINT DỰ ĐOÁN (ĐÃ TINH CHỈNH MAPPING) ---
@app.post("/predict")
async def predict(request: InferenceRequest):
    # Kỹ thuật Padding để đảm bảo đủ Sequence Length 10 [cite: 37, 44, 92]
    if len(request.history) < SEQ_LENGTH:
        needed = SEQ_LENGTH - len(request.history)
        history_list = [request.history[0]] * needed + request.history
    else:
        history_list = request.history[-SEQ_LENGTH:]

    # Chuẩn hóa dữ liệu tương đồng với file train [cite: 30, 38, 51, 90]
    data = []
    for p in history_list:
        data.append([
            p.weight, 
            p.e_canary, 
            p.e_stable, 
            p.l_canary, 
            p.l_stable, 
            p.cpu, 
            p.mem, 
            p.rps / 1000.0 # Chuẩn hóa RPS y hệt lúc train [cite: 38, 90]
        ])
    
    input_tensor = torch.FloatTensor([data]).to(DEVICE)

    with torch.no_grad():
        q_values, _ = model(input_tensor)
        action = torch.argmax(q_values).item()

    # --- ĐỒNG BỘ HÓA VỚI ARGO ROLLOUTS TẠI ĐÂY ---
    # 0, 1: AI muốn tiến lên -> Argo trả về "Successful" để nhảy step tiếp theo [cite: 56]
    # 2, 3: AI muốn giữ nguyên hoặc lùi lại -> Argo trả về "Running" để chờ [cite: 58]
    # 4: AI muốn hủy bỏ ngay lập tức -> Argo trả về "Rollback" 
    action_mapping = {
        0: "Successful", # Fast Forward (+10%)
        1: "Successful", # Step Forward (+5%)
        2: "Running",    # Stay
        3: "Running",    # Step Back
        4: "Rollback"    # EMERGENCY ROLLBACK
    }
    
    decision = action_mapping.get(action, "Running")
    
    return {
        "action_id": action,
        "decision": decision,
        "confidence": float(torch.softmax(q_values, dim=1).max())
    }

@app.get("/health")
def health():
    return {"status": "alive"}