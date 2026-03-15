import requests
import random

print("1. Sahte (Mock) Graf Verisi Hazırlanıyor...")
# 3 düğümlük (hesaplık) bir ağ oluşturalım. Modelimiz 165 özellik bekliyor.
num_nodes = 3
num_features = 165

# Modelin eğitildiği veri formatına benzemesi için rastgele sayılar üretiyoruz
mock_x = [[random.uniform(-2.0, 2.0) for _ in range(num_features)] for _ in range(num_nodes)]

# Düğümler arası bağlantılar (Örn: 0 numaralı cüzdan, 1 ve 2 numaralı cüzdanlara para göndersin)
mock_edge_index = [
    [0, 0], # Kaynak (Gönderen)
    [1, 2]  # Hedef (Alan)
]

# API'ye göndereceğimiz JSON paketi (Payload)
payload = {
    "x": mock_x,
    "edge_index": mock_edge_index,
    "target_node_idx": 0  # 0 numaralı işlemin dolandırıcı olup olmadığını soruyoruz
}

print("2. API'ye İstek (POST Request) Atılıyor...")
try:
    # Uvicorn sunucumuzun adresine veriyi gönderiyoruz
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    response.raise_for_status() # Eğer 200 OK dönmezse hata fırlat
    
    print("\n✅ Sunucudan Gelen Cevap:")
    print(response.json())
    
except Exception as e:
    print(f"❌ API'ye ulaşılamadı veya hata döndü: {e}")