# Panduan Penggunaan & Panduan Operasional MLOps

Dokumen ini menjelaskan langkah-langkah praktis untuk menjalankan, melatih, menguji, dan memantau sistem MLOps Ames Housing Price Prediction.

---

## 🚀 1. Persiapan Awal & Menjalankan Sistem

Pastikan Anda sudah menginstal **Docker Desktop** dan program Docker sedang berjalan.

### Menyalakan Seluruh Microservices
Untuk menjalankan API Backend, Streamlit Frontend, dan MLflow UI sekaligus di background, jalankan perintah berikut pada terminal di dalam direktori `housing_prices_mlops`:
```bash
docker compose --profile mlflow-ui up -d --build
```
*Catatan:*
- Flag `--build` digunakan untuk memastikan perubahan pada file *requirements* atau Dockerfile terpasang.
- Profil `--profile mlflow-ui` ditambahkan agar container MLflow UI ikut menyala di port `5001`.

### Memeriksa Status Container
Periksa apakah ketiga kontainer utama sudah menyala dan sehat:
```bash
docker ps
```
Hasil keluaran yang benar harus menampilkan kontainer `ames-frontend` (Port 8501), `ames-mlflow-ui` (Port 5001), dan `ames-predictor` (Port 8000) dengan status `healthy`.

---

## 📈 2. Melatih & Meregistrasi Model Baru (Training)

Sebelum melakukan prediksi, Anda memerlukan model terdaftar di MLflow Registry. Jika file database `mlruns` kosong atau Anda ingin memperbarui model dengan data terbaru, jalankan perintah pelatihan berikut:

```bash
docker compose --profile train run trainer
```
**Apa yang dilakukan perintah ini?**
1. Docker akan memicu container `ames-trainer` untuk menjalankan `src/train.py`.
2. Script melatih ensemble model Lasso, Ridge, CatBoost, dan XGBoost di atas data `data/train.csv`.
3. Model didaftarkan secara otomatis ke MLflow Registry dengan nama `AmesHousingModel` versi terbaru (misal: versi 1).
4. Metrik latihan disimpan secara lokal di `logs/latest_metrics.json`.

Setelah training selesai, Anda bisa membuka dashboard MLflow di **[http://localhost:5001](http://localhost:5001)** untuk melihat jalannya latihan, grafik performa, dan status model yang terdaftar.

---

## 🔮 3. Mengakses & Menggunakan Aplikasi

Akses seluruh interface layanan melalui browser Anda pada port-port berikut:

| Nama Layanan | URL Akses | Deskripsi |
| :--- | :--- | :--- |
| **Streamlit UI (Frontend)** | **[http://localhost:8501](http://localhost:8501)** | Interface utama untuk input parameter rumah atau unggah file CSV. |
| **FastAPI Swagger Docs** | **[http://localhost:8000/docs](http://localhost:8000/docs)** | Dokumentasi endpoint API interaktif untuk mencoba request langsung. |
| **MLflow Registry UI** | **[http://localhost:5001](http://localhost:5001)** | Dashboard monitoring eksperimen model dan registrasi model. |

---

## 🛠️ 4. Pengujian API Endpoint secara Manual (via Curl)

Anda dapat menguji endpoint FastAPI di terminal lokal menggunakan perintah `curl` berikut:

### A. Healthcheck Endpoint
```bash
curl -X GET http://localhost:8000/health
```
*Hasil respon:* Menampilkan status koneksi model, versi model aktif, dan total prediksi yang telah dilayani.

### B. Single Online Prediction Endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "GrLivArea": 1500,
    "LotArea": 9000,
    "Age_House": 15,
    "OverallQual": 6,
    "OverallCond": 5,
    "Neighborhood": "CollgCr",
    "TotalBsmtSF": 1000,
    "BsmtQual": "TA",
    "GarageCars": 2,
    "GarageType": "Attchd",
    "Age_Garage": 15,
    "ExterQual": "TA",
    "KitchenQual": "TA"
  }'
```
*Hasil respon:* Estimasi harga pasar properti dalam dolar USD, misalnya `{"estimated_price_usd": 127236.71, ...}`.

### C. Batch Prediction Endpoint (Mengunggah CSV)
Kirim file CSV data pengujian untuk diprediksi secara massal:
```bash
curl -X POST http://localhost:8000/predict-batch \
  -F "file=@data/test.csv"
```
*Hasil respon:* Array JSON yang berisi seluruh prediksi harga rumah per `Id`.

### D. Rollback Model (Administrasi Versi Model)
Untuk mengganti model aktif ke versi model tertentu (misalnya kembali ke versi 1) secara langsung tanpa mematikan server:
```bash
curl -X POST http://localhost:8000/rollback \
  -H "Content-Type: application/json" \
  -d '{"version": "1"}'
```

### E. Metrik Log Transaksi
Melihat log histori prediksi transaksi terakhir:
```bash
curl -X GET http://localhost:8000/metrics
```

---

## 🔍 5. Pemantauan Drift & Retraining Otomatis

Untuk memeriksa apakah data masukan pelanggan sudah melenceng jauh dari karakteristik data training awal (yang menyebabkan performa model menurun), jalankan pemantauan data drift:

```bash
docker compose --profile monitor run monitor
```
**Skenario Kerja Monitoring:**
1. Container `ames-monitor` akan membandingkan distribusi input data pada file log `logs/predictions.jsonl` dengan baseline data `data/train.csv` menggunakan statistik uji K-S Test.
2. Jika terdeteksi pergeseran statistik (*data drift*), script `src/retrain_trigger.py` akan terpicu secara otomatis untuk menjalankan kembali pipeline latihan (`src/train.py`).
3. Model baru yang diperbarui akan otomatis terdaftar di MLflow Registry untuk menggantikan model usang.

---

## 🛑 6. Mematikan Sistem

Jika Anda telah selesai melakukan pengujian dan ingin mematikan serta menghapus kontainer dari sistem memori komputer:
```bash
docker compose --profile mlflow-ui down
```
Langkah ini akan menghentikan seluruh kontainer yang sedang berjalan dengan aman tanpa menghapus data latihan `mlruns` maupun data log `logs/predictions.jsonl` Anda (karena data tersebut tersimpan di dalam volume lokal host).
