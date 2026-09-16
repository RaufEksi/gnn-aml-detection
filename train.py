import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score, precision_score, recall_score

print("1. Veriler yükleniyor ve temizleniyor...")
# CSV'leri oku
df_features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
df_classes = pd.read_csv('data/elliptic_txs_classes.csv')
df_edges = pd.read_csv('data/elliptic_txs_edgelist.csv')

# Kolon isimlerini düzeltme (Önceki hatayı önlemek için)
df_features.rename(columns={0: 'txId', 1: 'time_step'}, inplace=True)
feature_renames = {i: f'feature_{i-2}' for i in range(2, 167) if i in df_features.columns}
df_features.rename(columns=feature_renames, inplace=True)

# Verileri birleştirme
df = pd.merge(df_features, df_classes, on='txId')

# 'unknown' (bilinmeyen) sınıfları düşürüp sayısallaştırma
df = df[df['class'] != 'unknown'].copy()
class_mapping = {'1': 1, '2': 0}  # 1: Yasadışı (Illicit), 0: Yasal (Licit)
df['class'] = df['class'].map(class_mapping)

print("2. Graf topolojisi (Düğümler ve Kenarlar) kuruluyor...")
# Düğüm ID'lerini 0'dan N-1'e ardışık eşleme
node_mapping = {tx_id: index for index, tx_id in enumerate(df['txId'].values)}
df_edges['txId1'] = df_edges['txId1'].map(node_mapping)
df_edges['txId2'] = df_edges['txId2'].map(node_mapping)
df_edges = df_edges.dropna().astype(int)

# Tensörleri oluşturma
feature_cols = [col for col in df.columns if col.startswith('feature_')]
x = torch.tensor(df[feature_cols].values, dtype=torch.float)
y = torch.tensor(df['class'].values, dtype=torch.long)
edge_index = torch.tensor(df_edges.values.T, dtype=torch.long)

# Zamansal Bölme (Temporal Split)
time_steps = torch.tensor(df['time_step'].values, dtype=torch.long)
train_mask = time_steps <= 34
test_mask = time_steps > 34

# PyG Data Objesi
data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.test_mask = test_mask

print("3. Model Mimarisi (FraudGNN) ayağa kaldırılıyor...")


class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FraudGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FraudGNN(in_channels=data.num_features, hidden_channels=64, out_channels=2).to(device)
data = data.to(device)

print("4. Eğitim başlıyor (100 Epoch)...")
# Sınıf dengesizliği için ağırlık hesaplama
train_labels = data.y[data.train_mask]
num_licit = (train_labels == 0).sum().item()
num_illicit = (train_labels == 1).sum().item()
total_train = num_licit + num_illicit
weight_licit = total_train / (2 * num_licit)
weight_illicit = total_train / (2 * num_illicit)
class_weights = torch.tensor([weight_licit, weight_illicit], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Eğitim Döngüsü
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}')

print("\n5. Test Seti Üzerinde Değerlendirme (Evaluation)...")
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"Precision (Kesinlik): {precision:.4f}")
print(f"Recall (Duyarlılık) : {recall:.4f}")
print(f"F1-Score            : {f1:.4f}")

# MLOps fazı için modeli kaydet
torch.save(model.state_dict(), 'models/gnn_fraud_model.pth')
print("\n✅ Model ağırlıkları 'models/gnn_fraud_model.pth' olarak kaydedildi!")
