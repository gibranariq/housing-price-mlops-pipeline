import os
import json
import logging
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from pydantic import BaseModel, Field
import mlflow.pyfunc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AmesServingAPI")

app = FastAPI(
    title="Ames Housing Price Predictor API",
    description="Production API untuk prediksi harga rumah Ames Iowa menggunakan MLflow Registry & FastAPI.",
    version="2.0.0"
)

# MLflow config
mlflow.set_tracking_uri("file:./mlruns")
MODEL_NAME = "AmesHousingModel"
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

# Global variables for model and counter
model_pipeline = None
prediction_count = 0

def load_model_from_registry():
    global model_pipeline
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        logger.info(f"🔄 Loading model from MLflow Registry: {model_uri}...")
        model_pipeline = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}. API will start, but predictions will be unavailable until model is loaded.")
        model_pipeline = None

@app.on_event("startup")
def startup_event():
    os.makedirs("logs", exist_ok=True)
    load_model_from_registry()

# Pydantic schema for single prediction (13 features)
class HousePredictionRequest(BaseModel):
    GrLivArea: int = Field(..., description="Luas lantai atas dalam SqFt", example=1500, ge=300, le=6000)
    LotArea: int = Field(..., description="Luas tanah keseluruhan dalam SqFt", example=9000, ge=100, le=50000)
    Age_House: int = Field(..., description="Umur bangunan rumah dalam satuan tahun", example=15, ge=0, le=150)
    OverallQual: int = Field(..., description="Rating kualitas material & finishing (1-10)", example=6, ge=1, le=10)
    OverallCond: int = Field(..., description="Rating kondisi fisik rumah saat ini (1-10)", example=5, ge=1, le=10)
    Neighborhood: str = Field(..., description="Nama lingkungan lokasi properti", example="CollgCr")
    TotalBsmtSF: int = Field(..., description="Total luas area basement dalam SqFt", example=1000, ge=0, le=4000)
    BsmtQual: str = Field(..., description="Kualitas tinggi basement (Ex, Gd, TA, Fa, Po, None)", example="TA")
    GarageCars: int = Field(..., description="Kapasitas mobil di dalam garasi", example=2, ge=0, le=4)
    GarageType: str = Field(..., description="Tipe struktur garasi (Attchd, Detchd, BuiltIn, dll)", example="Attchd")
    Age_Garage: int = Field(..., description="Umur bangunan garasi dalam satuan tahun (-1 jika tidak ada)", example=15, ge=-1, le=150)
    ExterQual: str = Field(..., description="Kualitas material luar/eksterior (Ex, Gd, TA, Fa)", example="TA")
    KitchenQual: str = Field(..., description="Kualitas area finishing dapur (Ex, Gd, TA, Fa)", example="TA")

class HousePredictionResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    status: str
    estimated_price_usd: float
    model_version: str
    timestamp: str

@app.get("/health", tags=["System"])
def health():
    global prediction_count
    return {
        "status": "healthy" if model_pipeline is not None else "degraded (model not loaded)",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "predictions_served": prediction_count
    }

@app.post("/predict", response_model=HousePredictionResponse, tags=["ML Predictions"])
def predict(request: HousePredictionRequest):
    global model_pipeline, prediction_count
    
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready in memory. Please train and register the model first."
        )
        
    try:
        # 1. Convert Pydantic request to dict and then to DataFrame (1 row)
        input_dict = request.model_dump()
        df_input = pd.DataFrame([input_dict])
        
        # 2. Run prediction using MLflow custom pyfunc pipeline
        prediction = model_pipeline.predict(df_input)[0]
        prediction_val = float(prediction)
        
        # 3. Log prediction to logs/predictions.jsonl
        prediction_count += 1
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_dict,
            "prediction": prediction_val,
            "model_version": MODEL_VERSION
        }
        with open("logs/predictions.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        return HousePredictionResponse(
            status="success",
            estimated_price_usd=round(prediction_val, 2),
            model_version=MODEL_VERSION,
            timestamp=log_entry["timestamp"]
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing model prediction: {str(e)}"
        )

@app.post("/predict-batch", tags=["ML Predictions"])
async def predict_batch(file: UploadFile = File(...)):
    global model_pipeline, prediction_count
    
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready in memory. Please train and register the model first."
        )
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported for batch predictions."
        )
        
    try:
        # 1. Read CSV directly into DataFrame
        df_input = pd.read_csv(file.file)
        logger.info(f"Received batch prediction request with {len(df_input)} rows.")
        
        # 2. Run prediction using the model pipeline
        predictions = model_pipeline.predict(df_input)
        
        # Convert predictions to python floats
        predictions_list = [float(p) for p in predictions]
        
        # 3. Log predictions to logs/predictions.jsonl and prepare results
        results = []
        for idx, row in df_input.iterrows():
            pred_val = predictions_list[idx]
            
            # Extract row features as dict and convert NaNs to None for JSON compliance
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "input": row_dict,
                "prediction": pred_val,
                "model_version": MODEL_VERSION
            }
            
            with open("logs/predictions.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
            prediction_count += 1
            
            # Prepare batch output item
            results.append({
                "Id": row_dict.get("Id", idx),
                "estimated_price_usd": round(pred_val, 2)
            })
            
        return {
            "status": "success",
            "model_version": MODEL_VERSION,
            "total_predictions": len(predictions_list),
            "predictions": results
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing batch prediction: {str(e)}"
        )

@app.post("/rollback", tags=["Model Administration"])
def rollback(data: dict):
    global model_pipeline, MODEL_VERSION
    try:
        version = data.get("version", "latest")
        MODEL_VERSION = str(version)
        load_model_from_registry()
        return {
            "status": "success",
            "message": f"Successfully reloaded model version {MODEL_VERSION}",
            "model_version": MODEL_VERSION
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to reload model: {str(e)}"
        )

@app.get("/metrics", tags=["System"])
def metrics():
    global prediction_count
    recent_predictions = []
    
    try:
        if os.path.exists("logs/predictions.jsonl"):
            with open("logs/predictions.jsonl", "r") as f:
                lines = f.readlines()
                # Get last 10 predictions
                for line in lines[-10:]:
                    recent_predictions.append(json.loads(line.strip()))
    except Exception as e:
        logger.error(f"Error reading prediction logs: {e}")
        
    return {
        "total_predictions_served": prediction_count,
        "recent_predictions": recent_predictions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
