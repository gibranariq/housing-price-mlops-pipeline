import os
import pickle
import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from scipy.stats import skew

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV, RidgeCV

# Import Custom Feature Engineer yang udah kita bikin sebelumnya
from preprocessor import AmesFeatureEngineer

# Setup Logging Korporat
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AmesTrainingPipeline")

# Skenario Penyelamat Class Blending buat dipackage ke Pickle Final
class FinalQuadBlendedModel:
    def __init__(self, prep_tree, prep_linear, model_las, model_rid, model_cat, model_xgb):
        self.prep_tree = prep_tree
        self.prep_linear = prep_linear
        self.model_lasso = model_las
        self.model_ridge = model_rid
        self.model_catboost = model_cat
        self.model_xgboost = model_xgb

# Wrapper Model Custom untuk MLflow Model Registry
class AmesHousingMLflowModel(mlflow.pyfunc.PythonModel):
    def __init__(self, preprocessor, model_assets):
        self.preprocessor = preprocessor
        self.model_assets = model_assets

    def load_context(self, context):
        # Memuat high_skew_features dari artifacts MLflow
        with open(context.artifacts["high_skew_features"], "rb") as f:
            self.high_skew_features = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame):
        df_input = model_input.copy()
        
        # 1. Rekonstruksi variabel tahun mundur secara matematis untuk single request
        if 'Age_House' in df_input.columns:
            if 'YrSold' not in df_input.columns:
                df_input['YrSold'] = 2008
            if 'YearBuilt' not in df_input.columns:
                df_input['YearBuilt'] = 2008 - df_input['Age_House']
        
        if 'Age_Garage' in df_input.columns:
            if 'GarageYrBlt' not in df_input.columns:
                df_input['GarageYrBlt'] = df_input.apply(
                    lambda r: -1 if pd.isna(r['Age_Garage']) or int(r['Age_Garage']) == -1 
                    else 2008 - int(r['Age_Garage']), axis=1
                )
        
        # 2. Jalankan Preprocessing Modular AmesFeatureEngineer
        df_engineered = self.preprocessor.transform(df_input)
        
        # Extract komponen model dari asset
        prep_tree = self.model_assets.prep_tree
        prep_linear = self.model_assets.prep_linear
        model_lasso = self.model_assets.model_lasso
        model_ridge = self.model_assets.model_ridge
        model_catboost = self.model_assets.model_catboost
        model_xgboost = self.model_assets.model_xgboost
        
        # 3. Penyelarasan kolom agar sesuai dengan prep_tree
        # Kolom yang tidak diinput di FastAPI akan terisi NaN, dan akan diisi oleh SimpleImputer di pipeline
        for col in prep_tree.feature_names_in_:
            if col not in df_engineered.columns:
                df_engineered[col] = np.nan
        
        df_proc = df_engineered[prep_tree.feature_names_in_]
        
        # --- PATH A: TREE MODELS ---
        df_tree = df_proc.copy()
        X_tree = prep_tree.transform(df_tree)
        pred_cat_log = model_catboost.predict(X_tree)
        pred_xgb_log = model_xgboost.predict(X_tree)
        
        # --- PATH B: LINEAR MODELS ---
        df_linear = df_proc.copy()
        if 'Age_Garage' in df_linear.columns:
            df_linear['Age_Garage'] = df_linear['Age_Garage'].replace(np.nan, 0).replace(-1, 0)
            
        for feat in self.high_skew_features:
            if feat in df_linear.columns:
                df_linear[feat] = np.log1p(df_linear[feat].astype(float).fillna(0))
                
        X_linear = prep_linear.transform(df_linear)
        pred_las_log = model_lasso.predict(X_linear)
        pred_rid_log = model_ridge.predict(X_linear)
        
        # 4. Racikan Blending Final (30:30:20:20)
        final_log = (0.30 * pred_las_log) + (0.30 * pred_rid_log) + (0.20 * pred_cat_log) + (0.20 * pred_xgb_log)
        return np.expm1(final_log)

def run_training_pipeline(data_path="data/train.csv"):
    logger.info("====== Memulai Eksekusi ML Production Pipeline ======")
    
    # Buat direktori logs jika belum ada
    os.makedirs("logs", exist_ok=True)
    
    # 1. Load Data Mentah
    if not os.path.exists(data_path):
        logger.error(f"Data file tidak ditemukan di path: {data_path}")
        raise FileNotFoundError(f"Missing {data_path}")
        
    df_raw = pd.read_csv(data_path)
    logger.info(f"Sukses membaca dataset. Shape awal: {df_raw.shape}")
    
    # 2. Pisahkan Fitur Prediktor dan Target Variabel
    X = df_raw.drop(columns=['SalePrice'], errors='ignore')
    y = df_raw['SalePrice']
    
    # Drop Outlier ekstrem sesuai kertas riset Dean De Cock (>4000 SqFt)
    outlier_idx = X[X['GrLivArea'] > 4000].index
    if not outlier_idx.empty:
        X = X.drop(index=outlier_idx)
        y = y.drop(index=outlier_idx)
        logger.info(f"Berhasil membuang {len(outlier_idx)} baris outlier ekstrem.")
        
    # Target Log Transform biar sebaran distribusinya normal (Gaussian)
    y_log = np.log1p(y)
    
    # 3. Jalankan Tahap Preprocessing Tahap Awal via Modular Code Kustom
    fe_transformer = AmesFeatureEngineer()
    X_engineered = fe_transformer.fit_transform(X)
    
    # Ambil daftar nama kolom numerik dan nominal secara dinamis
    numerical_cols = X_engineered.select_dtypes(include=[np.number]).columns
    categorical_cols = X_engineered.select_dtypes(include=['object']).columns
    
    # =========================================================================
    # KONFIGURASI DUAL-PATH PIPELINE (TREE VS LINEAR)
    # =========================================================================
    logger.info("Membangun arsitektur Dual-Path ColumnTransformer.")
    
    # PATH A: Pipeline Khusus Model Berbasis Pohon (No Scaling, No Log Features)
    preprocessor_tree = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ], remainder='passthrough')
    
    # PATH B: Pipeline Khusus Model Linear (Pake Log Fitur Skewed & RobustScaler)
    X_linear_df = X_engineered.copy()

    if 'Age_Garage' in X_linear_df.columns:
        X_linear_df['Age_Garage'] = X_linear_df['Age_Garage'].replace(-1, 0)
        logger.info("Jalur Linear: Berhasil menjinakkan nilai -1 pada Age_Garage menjadi 0.")

    numeric_feats = X_linear_df.select_dtypes(include=[np.number]).columns
    skewed_feats = X_linear_df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew_features = list(skewed_feats[abs(skewed_feats) > 0.75].index)
    
    # Simpan list fitur high skew ke file pkl
    with open('high_skew_features.pkl', 'wb') as f:
        pickle.dump(high_skew_features, f)
    logger.info(f"Berhasil mencatat {len(high_skew_features)} fitur berkategori high skew.")
    
    for feat in high_skew_features:
        if feat in X_linear_df.columns:
            X_linear_df[feat] = np.log1p(X_linear_df[feat])
            
    preprocessor_linear = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ], remainder='passthrough')
    
    # Eksekusi Transformasi Data Latih ke wujud Matriks Siap Pakai
    X_train_tree = preprocessor_tree.fit_transform(X_engineered)
    X_train_linear = preprocessor_linear.fit_transform(X_linear_df)
    
    # =========================================================================
    # INTEGRASI EXPERIMENT TRACKING VIA MLFLOW
    # =========================================================================
    logger.info("Mengaktifkan koneksi ke MLflow Server Tracking...")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Ames_Housing_Price_Regression")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Inisialisasi Model Master (n_jobs=None / 1 untuk menghindari deadlock multiprocessing di macOS)
    model_lasso = LassoCV(alphas=np.logspace(-4, -1, 20), cv=5, max_iter=5000, random_state=42)
    model_ridge = RidgeCV(alphas=np.logspace(-2, 3, 30), cv=5)
    model_catboost = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=5, l2_leaf_reg=5, random_state=42, verbose=0)
    model_xgboost = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    
    with mlflow.start_run(run_name="Quad_Ensemble_Blending_Run"):
        logger.info("Memulai validasi silang 5-Fold Cross Validation secara simultan.")
        
        # Penampung Skor Evaluasi
        cv_rmse, cv_mae, cv_r2 = [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_linear)):
            logger.info(f"⏳ Memproses Fold {fold + 1}/5...")
            X_tr_lin, X_va_lin = X_train_linear[train_idx], X_train_linear[val_idx]
            X_tr_tree, X_va_tree = X_train_tree[train_idx], X_train_tree[val_idx]
            y_tr, y_va = y_log.iloc[train_idx], y_log.iloc[val_idx]
            
            # Fit Model
            model_lasso.fit(X_tr_lin, y_tr)
            model_ridge.fit(X_tr_lin, y_tr)
            model_catboost.fit(X_tr_tree, y_tr)
            model_xgboost.fit(X_tr_tree, y_tr)
            
            # Predict
            p_las = model_lasso.predict(X_va_lin)
            p_rid = model_ridge.predict(X_va_lin)
            p_cat = model_catboost.predict(X_va_tree)
            p_xgb = model_xgboost.predict(X_va_tree)
            
            # Blending (30:30:20:20)
            p_blend_fold = (0.30 * p_las) + (0.30 * p_rid) + (0.20 * p_cat) + (0.20 * p_xgb)
            
            cv_rmse.append(root_mean_squared_error(y_va, p_blend_fold))
            cv_mae.append(mean_absolute_error(y_va, p_blend_fold))
            cv_r2.append(r2_score(y_va, p_blend_fold))
            
            mlflow.log_metric(f"fold_{fold}_rmse", cv_rmse[-1])
            
        # Hitung Nilai Rata-rata Akhir Metrik CV
        final_cv_rmse = np.mean(cv_rmse)
        final_cv_mae = np.mean(cv_mae)
        final_cv_r2 = np.mean(cv_r2)
        
        # Hitung Adjusted R2
        n = X_train_linear.shape[0]
        p = X_train_linear.shape[1]
        final_adj_r2 = 1 - ((1 - final_cv_r2) * (n - 1) / (n - p - 1))
        
        logger.info("Proses Cross-Validation Selesai. Menghitung Real Train Metrics...")
        
        # Latih model final di 100% data bersih (Refit)
        model_lasso.fit(X_train_linear, y_log)
        model_ridge.fit(X_train_linear, y_log)
        model_catboost.fit(X_train_tree, y_log)
        model_xgboost.fit(X_train_tree, y_log)
        
        p_train_blend = (
            (0.30 * model_lasso.predict(X_train_linear)) + 
            (0.30 * model_ridge.predict(X_train_linear)) + 
            (0.20 * model_catboost.predict(X_train_tree)) + 
            (0.20 * model_xgboost.predict(X_train_tree))
        )
        final_train_rmse = root_mean_squared_error(y_log, p_train_blend)
        
        # --- LOGGING SELURUH PARAMETER & METRIK UTAMA KE SERVER MLFLOW ---
        mlflow.log_param("blending_weights", "30_30_20_20")
        mlflow.log_param("linear_scaler", "RobustScaler")
        mlflow.log_param("high_skew_threshold", 0.75)
        
        mlflow.log_metric("Final_Train_RMSE", final_train_rmse)
        mlflow.log_metric("Final_CV_RMSE", final_cv_rmse)
        mlflow.log_metric("Final_CV_MAE", final_cv_mae)
        mlflow.log_metric("Final_Adjusted_R2", final_adj_r2)
        
        logger.info(f"-> Log MLflow Sukses | Real CV RMSE: {final_cv_rmse:.5f} | Adj R²: {final_adj_r2:.4f}")
        
        # 4. PACKING COMPONENT & REGISTER TO MLFLOW REGISTRY
        best_house_predictor = FinalQuadBlendedModel(
            prep_tree=preprocessor_tree,
            prep_linear=preprocessor_linear,
            model_las=model_lasso,
            model_rid=model_ridge,
            model_cat=model_catboost,
            model_xgb=model_xgboost
        )
        
        # Bungkus ke wrapper MLflow PyFunc
        custom_mlflow_model = AmesHousingMLflowModel(
            preprocessor=fe_transformer,
            model_assets=best_house_predictor
        )
        
        # Daftarkan model dan file skewness sebagai artifact terintegrasi
        mlflow.pyfunc.log_model(
            artifact_path="ames_housing_model",
            python_model=custom_mlflow_model,
            artifacts={"high_skew_features": "high_skew_features.pkl"},
            registered_model_name="AmesHousingModel"
        )
        
        # Save metrics ke file local untuk monitoring baseline
        metrics = {"mse": float(final_cv_rmse**2), "rmse": float(final_cv_rmse), "mae": float(final_cv_mae), "r2_score": float(final_cv_r2)}
        with open("logs/latest_metrics.json", "w") as f:
            json.dump(
                {
                    **metrics,
                    "timestamp": datetime.now().isoformat(),
                    "run_id": mlflow.active_run().info.run_id,
                },
                f,
                indent=2,
            )
            
        logger.info("====== Pipeline Berhasil. Model Biner Terdaftar Kokoh di MLflow Model Registry ======")

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    run_training_pipeline(data_path)