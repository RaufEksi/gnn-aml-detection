from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# 1. Mimarinin Birebir Kopyası (Ağırlıkları yükleyebilmek için iskelet şart)
class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FraudGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Üretim (Production) ortamında dropout kapalıdır, bu yüzden direkt katmanları geçiyoruz
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# 2. Veri Şeması (İstemciden gelen JSON'ın formatı)
class TransactionSubGraph(BaseModel):
    # Gelen veri tüm graf değil, şüpheli işlem ve onun komşularından oluşan bir "Alt Graf (Subgraph)" olmalı
    x: List[List[float]]          # Düğüm özellikleri matrisi (N adet düğüm, 165 özellik)
    edge_index: List[List[int]]   # Kenar listesi (2 satır, M adet bağlantı)
    target_node_idx: int          # Bu alt grafta tahmin yapmak istediğimiz ana işlemin indeksi

# 3. Uygulamayı ve Modeli Başlatma (Cold Start Optimizasyonu)
app = FastAPI(title="AML Fraud Detection GNN API", version="1.0")

# API'ler genelde CPU üzerinde servis edilir (Maliyet optimizasyonu)
device = torch.device('cpu') 
model = FraudGNN(in_channels=165, hidden_channels=64, out_channels=2)

# Ağırlıkları yükle ve test moduna al
try:
    model.load_state_dict(torch.load("models/gnn_fraud_model.pth", map_location=device))
    model.eval()
    print("✅ Model başarıyla belleğe yüklendi.")
except Exception as e:
    print(f"Hata: Model dosyası bulunamadı! {e}")

# 4. Tahmin Rotası (Endpoint)
@app.post("/predict")
def predict_fraud(data: TransactionSubGraph):
    try:
        # Pydantic listelerini PyTorch tensörlerine çeviriyoruz
        x_tensor = torch.tensor(data.x, dtype=torch.float)
        edge_index_tensor = torch.tensor(data.edge_index, dtype=torch.long)
        
        with torch.no_grad():
            # İleri yayılım
            out = model(x_tensor, edge_index_tensor)
            
            # Sadece hedef düğümün (target_node) çıktılarını al
            target_out = out[data.target_node_idx]
            
            # Logitleri olasılığa (0 ile 1 arası) çevir
            probabilities = F.softmax(target_out, dim=0)
            fraud_prob = probabilities[1].item() # 1: Yasadışı olma olasılığı
            
        return {
            "target_node_index": data.target_node_idx,
            "fraud_probability": round(fraud_prob, 4),
            "is_suspicious": bool(fraud_prob > 0.70), # Kesinliği (Precision) artırmak için eşik %70
            "message": "İşlem dolandırıcılık şüphesi taşıyor, onaya düştü!" if fraud_prob > 0.70 else "İşlem temiz."
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tahmin sırasında hata oluştu: {str(e)}")