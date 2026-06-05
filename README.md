# Ames Housing MLOps Pipeline & Serving System

Sistem MLOps end-to-end untuk prediksi harga properti di Ames, Iowa. Proyek ini memisahkan UI Frontend berbasis Streamlit dengan serving backend berbasis FastAPI, dilengkapi dengan pelacakan eksperimen MLflow, containerisasi Docker, monitoring penyimpangan data (concept drift), dan pemicu pelatihan ulang otomatis (automated retraining).

---

## 🏗️ Arsitektur Sistem

Sistem ini didesain menggunakan arsitektur terdistribusi microservices yang dideploy menggunakan Docker Compose:

```
                  +-----------------------------------+
                  |          User Browser             |
                  +-----------------------------------+
                                    |
                               (Port 8501)
                                    v
                  +-----------------------------------+
                  |        Streamlit Frontend         |
                  +-----------------------------------+
                                    |
                            (Docker Bridge API)
                               (Port 8000)
                                    v
                  +-----------------------------------+
                  |          FastAPI Serving          |
                  +-----------------------------------+
                    /               |               \
                   /                |                \
  (Logs Predictions)     (Tracking Model Registry)     (Mounted Volumes)
         v                          v                          v
  [predictions.jsonl]     +--------------------+          sqlite:///mlruns
                          |    MLflow Server   |
                          |    (Port 5001)     |
                          +--------------------+
```

1. **Frontend (Streamlit):** Interface interaktif bagi pengguna untuk menginput spesifikasi rumah secara manual maupun mengunggah file CSV secara massal.
2. **Backend serving (FastAPI):** Menyediakan REST API endpoints untuk melayani prediksi online tunggal, prediksi batch (CSV), pemeriksaan kesehatan (*healthcheck*), *rollback* versi model, dan pembacaan metrik log.
3. **MLflow Tracking Server & Registry:** Bertindak sebagai repositori model untuk menyimpan parameter latihan, metrik performa, dan mendaftarkan model terpilih (*Model Registry*).
4. **Data Drift & Monitoring:** Pemantau berkala yang membaca file log prediksi untuk menghitung pergeseran statistik data masukan dan memicu retraining jika terdeteksi adanya *drift*.

---

## 📁 Peta Struktur Repositori

```
housing_prices_mlops/
├── app.py                      # Aplikasi Streamlit (Frontend UI)
├── Dockerfile                  # Struktur build Image Docker Python standar
├── docker-compose.yml          # Konfigurasi containerisasi microservices
├── requirements.txt            # Dependensi Python proyek
├── run_mlops.sh                # Script Bash otomatis untuk kontrol sistem
├── data/                       # Direktori Dataset
│   ├── train.csv               # Data latih dasar Ames Housing
│   ├── test.csv                # Data pengujian untuk batch prediction
│   ├── sample_submission.csv
│   └── data_description.txt    # Keterangan fitur dataset
├── logs/                       # File log operasional
│   ├── predictions.jsonl       # Log seluruh request dan hasil prediksi API
│   └── latest_metrics.json     # Metrik baseline hasil training terakhir
├── mlruns/                     # Database MLflow local (SQLite & Artifacts)
└── src/                        # Modul Python Backend
    ├── preprocessor.py         # Pipeline data cleansing & feature engineering modular
    ├── train.py                # Pipeline pelatihan model & pendaftaran ke MLflow
    ├── predict.py              # Pelayan API FastAPI (Endpoints, Logging)
    ├── monitor.py              # Evaluasi Concept Drift (Kolmogorov-Smirnov Test)
    └── retrain_trigger.py      # Pemicu pelatihan ulang otomatis saat drift terdeteksi
```

---

## ⚙️ Komponen Utama & Alur Kerja

### 1. Preprocessing Modular (`src/preprocessor.py`)
Menerapkan transformasi fitur anti-leakage yang konsisten:
- Penanganan nilai kosong (*Intentional NA*) seperti mengisi string `"None"` untuk area non-fasilitas (Alley, Pool, Garage, dll).
- Pemetaan ordinal (*Ordinal Encoding*) secara manual untuk rating kualitas eksterior, dapur, basement, dan garasi.
- Rekalkulasi umur rumah (`Age_House`) dan umur garasi (`Age_Garage`) berdasarkan tahun penjualan.
- Pembuatan flag biner (`Has_Pool`, `Has_Misc`, `Has_Alley`) untuk menangani sebaran data yang jomplang (*highly skewed*).

### 2. Pipeline Pelatihan & Registrasi (`src/train.py`)
- Melatih empat model dasar secara paralel: **Lasso**, **Ridge**, **CatBoost**, dan **XGBoost**.
- Menggunakan pembagian Dual-Path: Jalur model linier (dengan RobustScaler dan log-transform) & Jalur model pohon.
- Menggabungkan keempat model menjadi model ensemble dengan racikan bobot blending **30% Lasso + 30% Ridge + 20% CatBoost + 20% XGBoost**.
- Membungkus ensemble dan objek preprocessor menjadi satu objek tunggal **Custom MLflow `PythonModel`** (`AmesHousingMLflowModel`) dan mendaftarkannya ke MLflow Model Registry sebagai versi `latest`.

### 3. Serving Backend API (`src/predict.py`)
FastAPI memuat model secara dinamis dari MLflow Model Registry pada saat startup. Menyediakan 5 endpoint utama:
- `GET /health` : Mengecek status serving API, versi model aktif, dan statistik total prediksi yang sudah dilayani.
- `POST /predict` : Menerima 13 parameter properti dalam bentuk JSON, mengimputasi fitur sisa menggunakan SimpleImputer bawaan pipeline, lalu mengembalikan nilai taksiran pasar dalam USD.
- `POST /predict-batch` : Menerima unggahan file `.csv`, memproses prediksi untuk seluruh baris, mengembalikan nilai prediksi, dan mencatatnya ke dalam log prediksi.
- `POST /rollback` : Memungkinkan administrator memicu penggantian (*swap*) versi model yang aktif di memori secara real-time tanpa perlu me-restart server.
- `GET /metrics` : Membaca log `predictions.jsonl` untuk menyajikan 10 log transaksi transaksi terakhir secara langsung.

### 4. Sistem Pemantauan Drift & Retraining (`src/monitor.py` & `src/retrain_trigger.py`)
- Pemantau drift membandingkan distribusi fitur numerik masukan pada log prediksi dengan distribusi data latihan asli (`train.csv`) menggunakan uji statistik **Kolmogorov-Smirnov (K-S Test)**.
- Jika terdeteksi penyimpangan (drift) yang signifikan pada fitur penting, sistem secara otomatis akan memicu file `retrain_trigger.py` untuk menjalankan kembali proses training guna memperbarui model dengan data perilaku pasar terbaru.
