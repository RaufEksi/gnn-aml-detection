# Hafif bir Python taban imajı kullanıyoruz
FROM python:3.10-slim

# Çalışma dizinini ayarlıyoruz
WORKDIR /app

# Önce sadece gereksinimleri kopyalayıp kuruyoruz (Docker cache optimizasyonu için)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Şimdi tüm proje dosyalarını (model ağırlıkları ve main.py dahil) kopyalıyoruz
COPY . .

# FastAPI'nin çalışacağı portu dışa açıyoruz
EXPOSE 8000

# Konteyner ayağa kalktığında çalıştırılacak komut
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
