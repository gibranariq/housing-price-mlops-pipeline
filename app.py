import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

# =========================================================================
# SYSTEM CONFIG & BACKEND SETUP
# =========================================================================
st.set_page_config(
    page_title="Ames Housing AI Predictor",
    page_icon="🤖",
    layout="wide",
)

# Target URL for FastAPI backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# Get list of unique neighborhoods
@st.cache_data
def get_neighborhoods():
    try:
        df = pd.read_csv("data/train.csv")
        return sorted(df['Neighborhood'].unique())
    except Exception:
        # Fallback list of neighborhoods in Ames Iowa dataset
        return sorted([
            'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
            'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
            'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
            'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
            'Blueste'
        ])

# Quality mappings
READABLE_QUALITY_MAP = {
    "Excellent (Sangat Sempurna)": "Ex",
    "Good (Bagus)": "Gd",
    "Typical / Average (Rata-rata)": "TA",
    "Fair (Cukup)": "Fa",
    "Poor (Buruk)": "Po"
}

# =========================================================================
# PREMIUM AESTHETICS & CUSTOM STYLING (GLASSMORPHISM & DARK THEME)
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Background gradient styling */
.stApp {
    background: radial-gradient(circle at top right, #1a1b35 0%, #0a0b10 100%) !important;
    color: #e2e8f0;
}

/* Premium Header Cards */
.status-badge {
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.status-connected {
    background-color: rgba(16, 185, 129, 0.1);
    color: #34d399;
    border-color: rgba(52, 211, 153, 0.3);
}

.status-disconnected {
    background-color: rgba(239, 68, 68, 0.1);
    color: #f87171;
    border-color: rgba(248, 113, 113, 0.3);
}

/* Glassmorphic Container Cards */
.glass-container {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
}

/* Shiny gradient result card */
.result-card {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(37, 99, 235, 0.1) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.2);
    margin-top: 20px;
}

/* Custom styled predict button */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
    color: white !important;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 14px 28px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6);
    transform: translateY(-2px);
}

div.stButton > button:active {
    transform: translateY(0);
}

/* Custom font styling for form labels */
.stMarkdown p {
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# Check API Health Status
api_connected = False
model_version_info = "N/A"
predictions_served_info = "N/A"

try:
    health_response = requests.get(f"{BACKEND_URL}/health", timeout=3)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data.get("status") == "healthy":
            api_connected = True
            model_version_info = health_data.get("model_version", "latest")
            predictions_served_info = health_data.get("predictions_served", 0)
except Exception:
    pass

# Header Section
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("""
    <div style="margin-top: 10px;">
        <h1 style="background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; font-size: 2.8rem; margin-bottom: 0;">
            🤖 Ames Housing AI Predictor
        </h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 5px;">
            Sistem Prediksi Estimasi Harga Rumah Ames Iowa • Terintegrasi FastAPI & MLflow Registry
        </p>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("<div style='text-align: right; margin-top: 25px;'>", unsafe_allow_html=True)
    if api_connected:
        st.markdown(f'<span class="status-badge status-connected">● API Connected (Model {model_version_info})</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-disconnected">● API Offline / Reconnecting</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# Navigation tabs for Single vs Batch Prediction
tab1, tab2 = st.tabs([
    "🔮 Single House Estimator",
    "📁 CSV Batch Predictor"
])

# =========================================================================
# TAB 1: SINGLE HOUSE ESTIMATION
# =========================================================================
with tab1:
    if not api_connected:
        st.error("⚠️ **Koneksi backend API terputus.** Pastikan service FastAPI `predictor` telah dijalankan di port 8000.")
        
    col_inputs, col_result = st.columns([3, 2], gap="large")

    with col_inputs:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("🏡 Spesifikasi & Fitur Rumah")
        
        # Section A: Dimensi & Lokasi
        st.markdown("##### 📍 Spesifikasi Dasar")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            gr_liv_area = st.number_input("Luas Lantai Atas (GrLivArea) — *SqFt*", min_value=300, max_value=6000, value=1500)
            age_house = st.slider("Umur Bangunan Rumah (Age_House) — *Tahun*", min_value=0, max_value=150, value=15)
        with sub_col2:
            lot_area = st.number_input("Luas Tanah (LotArea) — *SqFt*", min_value=100, max_value=50000, value=9000)
            neighborhood = st.selectbox("Kluster Lingkungan / Lokasi (Neighborhood)", get_neighborhoods())
            
        st.divider()
        
        # Section B: Kualitas Bangunan
        st.markdown("##### ✨ Kualitas & Finishing")
        sub_col3, sub_col4 = st.columns(2)
        with sub_col3:
            overall_qual = st.slider("Kualitas Material & Finishing (OverallQual)", 1, 10, 6, help="1: Sangat Buruk, 10: Sangat Istimewa")
            exter_qual_label = st.selectbox("Kualitas Material Luar / Eksterior (ExterQual)", list(READABLE_QUALITY_MAP.keys()), index=2)
            exter_qual = READABLE_QUALITY_MAP[exter_qual_label]
        with sub_col4:
            overall_cond = st.slider("Rating Kondisi Fisik Rumah (OverallCond)", 1, 10, 5, help="1: Sangat Buruk, 10: Sangat Istimewa")
            kitchen_qual_label = st.selectbox("Kualitas Finishing Dapur (KitchenQual)", list(READABLE_QUALITY_MAP.keys()), index=2)
            kitchen_qual = READABLE_QUALITY_MAP[kitchen_qual_label]
            
        st.divider()
        
        # Section C: Basement
        st.markdown("##### 🕳️ Informasi Basement")
        has_basement = st.checkbox("Apakah rumah memiliki Basement?", value=True)
        if has_basement:
            sub_col5, sub_col6 = st.columns(2)
            with sub_col5:
                total_bsmt_sf = st.number_input("Total Luas Basement (TotalBsmtSF) — *SqFt*", min_value=10, max_value=4000, value=1000)
            with sub_col6:
                bsmt_qual_label = st.selectbox("Kualitas Tinggi Ruang Basement (BsmtQual)", list(READABLE_QUALITY_MAP.keys()), index=2)
                bsmt_qual = READABLE_QUALITY_MAP[bsmt_qual_label]
        else:
            total_bsmt_sf = 0
            bsmt_qual = "None"
            
        st.divider()
        
        # Section D: Garasi
        st.markdown("##### 🚗 Informasi Garasi")
        has_garage = st.checkbox("Apakah rumah memiliki Garasi?", value=True)
        if has_garage:
            sub_col7, sub_col8 = st.columns(2)
            with sub_col7:
                garage_cars = st.slider("Kapasitas Mobil di Garasi (GarageCars)", 1, 4, 2)
                garage_type = st.selectbox("Tipe Struktur Garasi (GarageType)", ['Attchd', 'Detchd', 'BuiltIn', 'Basment', '2Types', 'CarPort'])
            with sub_col8:
                age_garage = st.slider("Umur Bangunan Garasi (Age_Garage) — *Tahun*", min_value=0, max_value=150, value=15)
        else:
            garage_cars = 0
            garage_type = "None"
            age_garage = -1
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        predict_btn = st.button("🔮 Hitung Estimasi Harga Rumah", use_container_width=True, disabled=not api_connected)

    with col_result:
        st.subheader("📊 Hasil Estimasi & Analisis")
        
        if predict_btn and api_connected:
            # Construct payload
            payload = {
                "GrLivArea": int(gr_liv_area),
                "LotArea": int(lot_area),
                "Age_House": int(age_house),
                "OverallQual": int(overall_qual),
                "OverallCond": int(overall_cond),
                "Neighborhood": str(neighborhood),
                "TotalBsmtSF": int(total_bsmt_sf),
                "BsmtQual": str(bsmt_qual),
                "GarageCars": int(garage_cars),
                "GarageType": str(garage_type),
                "Age_Garage": int(age_garage),
                "ExterQual": str(exter_qual),
                "KitchenQual": str(kitchen_qual)
            }
            
            with st.spinner("Mengirimkan parameter ke model serving backend..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=5)
                    if response.status_code == 200:
                        result = response.json()
                        estimated_price = result.get("estimated_price_usd", 0.0)
                        model_ver = result.get("model_version", "unknown")
                        timestamp = result.get("timestamp", "")
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <p style="color: #a78bfa; font-weight: 600; font-size: 1.1rem; margin-bottom: 5px;">ESTIMASI NILAI PASAR PROPERTI</p>
                            <h2 style="font-size: 3.2rem; color: #ffffff; font-weight: 700; margin: 0;">${estimated_price:,.2f}</h2>
                            <p style="color: #64748b; font-size: 0.85rem; margin-top: 15px;">
                                Model Version: <b>{model_ver}</b><br>
                                Waktu Prediksi: {timestamp}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success("✅ Harga properti berhasil diestimasi menggunakan model Quad-Weighted Blending (Lasso, Ridge, CatBoost, XGBoost) yang stabil.")
                        
                        # Insight section
                        st.markdown("""
                        <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px; margin-top: 15px;">
                            <p style="margin: 0; font-size: 0.9rem; color: #94a3b8; line-height: 1.5;">
                                💡 <b>Analisis Model:</b> Estimasi dihitung menggunakan kombinasi model linear ber-regulasi dan ensemble berbasis pohon. Karakteristik utama rumah Anda (luas lantai, kualitas struktural, dan umur) diolah secara dinamis dengan penanganan data kosong dan engineering fitur otomatis langsung pada container backend.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Backend mengembalikan error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Gagal berkomunikasi dengan model backend: {e}")
        else:
            st.info("💡 Tentukan spesifikasi properti di kolom sebelah kiri dan klik tombol **Hitung Estimasi Harga Rumah** untuk melihat hasil.")

# =========================================================================
# TAB 2: CSV BATCH PREDICTION
# =========================================================================
with tab2:
    st.subheader("📁 Prediksi Batch Menggunakan File CSV")
    st.markdown("""
    Unggah file `.csv` yang berisi sekumpulan data properti Ames. Pastikan file CSV Anda memiliki kolom-kolom fitur yang sesuai dengan format data latih 
    (seperti `GrLivArea`, `LotArea`, `OverallQual`, dll). Backend secara otomatis akan melakukan imputasi missing value dan feature alignment.
    """)
    
    if not api_connected:
        st.error("⚠️ **Koneksi backend API terputus.** Pastikan service FastAPI `predictor` telah dijalankan di port 8000.")
        
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], disabled=not api_connected)
    
    if uploaded_file is not None and api_connected:
        st.success("✅ File CSV berhasil diunggah!")
        
        # Read the file to show a quick preview
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.write("##### Preview Data Masukan:")
            st.dataframe(df_preview.head(5), use_container_width=True)
            
            # Reset file pointer after reading preview
            uploaded_file.seek(0)
            
            if st.button("🔮 Mulai Prediksi Massal (Batch)", use_container_width=True):
                with st.spinner("Mengirimkan file CSV ke serving backend untuk pemrosesan paralel..."):
                    try:
                        # Prepare payload for multipart form data
                        files = {"file": (uploaded_file.name, uploaded_file.read(), "text/csv")}
                        
                        response = requests.post(f"{BACKEND_URL}/predict-batch", files=files, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            total_preds = result.get("total_predictions", 0)
                            model_ver = result.get("model_version", "unknown")
                            predictions_list = result.get("predictions", [])
                            
                            # Convert results to DataFrame
                            df_preds = pd.DataFrame(predictions_list)
                            
                            # Join prediction results with the original CSV
                            # Reset pointer to merge
                            uploaded_file.seek(0)
                            df_original = pd.read_csv(uploaded_file)
                            
                            # Merge predictions based on index or Id
                            if "Id" in df_original.columns and "Id" in df_preds.columns:
                                df_output = pd.merge(df_original, df_preds, on="Id")
                            else:
                                df_output = df_original.copy()
                                df_output["estimated_price_usd"] = df_preds["estimated_price_usd"]
                            
                            st.success(f"🎉 Sukses memproses {total_preds} baris data menggunakan model versi {model_ver}!")
                            
                            st.write("##### Preview Hasil Prediksi:")
                            show_cols = [c for c in ["Id", "estimated_price_usd", "GrLivArea", "LotArea", "OverallQual"] if c in df_output.columns]
                            st.dataframe(df_output[show_cols], use_container_width=True)
                            
                            # Create download link for the CSV
                            csv_data = df_output.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Unduh File Hasil Prediksi (.CSV)",
                                data=csv_data,
                                file_name=f"predicted_house_prices_{model_ver}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.error(f"❌ Gagal memproses batch prediction: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Gagal menghubungi serving backend: {e}")
        except Exception as e:
            st.error(f"❌ Error membaca file CSV: {e}")
