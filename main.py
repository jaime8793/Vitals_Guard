from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

models: dict = {}


MODEL_FILES = {
    "metabolic": "realistic_risk_model.pkl",
    "stroke":    "stroke_model.pkl",
    "injury":    "injury_risk_model_synth.pkl",
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    for name, path in MODEL_FILES.items():
        try:
            models[name] = joblib.load(path)
            logger.info(f"✅ Loaded model: {name}")
        except Exception as exc:
            logger.error(f"🚨 Failed to load {name}: {exc}")
    yield
    models.clear()

def require_model(name: str):
    model = models.get(name)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{name}' not loaded.",
        )
    return model

app = FastAPI(title="VitalsGuard API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# 1. DIABETES (METABOLIC) PREDICTION
# ---------------------------------------------------------------------------
class MetabolicInput(BaseModel):
    # Mapping Pydantic to match 'diabetes.csv' column names exactly
    Pregnancies: int = Field(..., ge=0)
    Glucose: int = Field(..., ge=0)
    BloodPressure: int = Field(..., ge=0)
    SkinThickness: int = Field(..., ge=0)
    Insulin: int = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=1)

@app.post("/predict/metabolic")
def predict_metabolic(data: MetabolicInput):
    model = require_model("metabolic")
    
    # 1. Get raw input data
    input_data = data.model_dump()

    # 2. Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 3. Align columns exactly with what the model expects
 
    expected_cols = model.feature_names_in_
    
    for col in expected_cols:
        if col not in input_df.columns:
           
            if col == "SedentaryMinutes":
                input_df[col] = 420 
            else:
                input_df[col] = 0
                
    # Filter and reorder exactly as trained
    input_df = input_df[expected_cols]
    
    # 4. Predict (Removed the int() wrapper that was causing the crash!)
    predicted_class = str(model.predict(input_df)[0]) 
    
    # 5. Calculate Confidence Score
    # predict_proba returns an array of probabilities for each class.one.
    probabilities = model.predict_proba(input_df)[0].tolist()
    confidence_score = max(probabilities) * 100
    
    return {
        "status": "success",
        "predicted_category": predicted_class,
        "confidence": round(confidence_score, 1)
    }
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2. STROKE PREDICTION MODEL
# ---------------------------------------------------------------------------


STROKE_FEATURES_ORDER = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'
]

class StrokeInput(BaseModel):
    age: float
    hypertension: int 
    heart_disease: int 
    avg_glucose_level: float
    bmi: float
   
    Residence_type: str | None = "Urban"
    smoking_status: str | None = "never smoked"
    ever_married: int | None = 1
    work_type: str | None = "Private"
    daily_steps: int | None = 5000 

def stroke_physiological_multiplier(row: dict, steps: int, smoking_status: str) -> float:
    """Matches the VitalsGuard penalty logic, but adds Step buffer and Smoking penalty."""
    multiplier = 1.0
    if row['bmi'] > 30: multiplier += 0.2
    if row['avg_glucose_level'] > 180: multiplier += 0.3
    if row['hypertension'] == 1: multiplier += 0.3
    if row['heart_disease'] == 1: multiplier += 0.4
    
    # Even though smoking isn't in the ML model anymore, we can add it to our physiological multiplier!
    if smoking_status == "smokes": multiplier += 0.3
    elif smoking_status == "formerly smoked": multiplier += 0.1
    
    # Physiological Activity Buffer
    if steps < 3000: multiplier += 0.2
    elif steps >= 10000: multiplier -= 0.25
    elif steps >= 7000: multiplier -= 0.10
    
    return max(0.5, multiplier)

@app.post("/predict/stroke")
def predict_stroke(data: StrokeInput):
    model = require_model("stroke")
    
    # 1. Map data EXACTLY to the 5 expected ML features
    ml_input = {
        'age': data.age,
        'hypertension': data.hypertension,
        'heart_disease': data.heart_disease,
        'avg_glucose_level': data.avg_glucose_level,
        'bmi': data.bmi
    }
    
    # 2. Create DataFrame and enforce column order
    input_df = pd.DataFrame([ml_input])[STROKE_FEATURES_ORDER]
    
    # 3. Get raw probability
    raw_stroke_prob = float(model.predict_proba(input_df.values)[0][1])
    
    # 4. Apply the VitalsGuard Custom Math
    multiplier = stroke_physiological_multiplier(ml_input, data.daily_steps, data.smoking_status)
    score = raw_stroke_prob * multiplier * 100
    score = min(score, 99.0) # Cap at 99%
    
    # 5. Risk thresholds & Recommendations
    if score < 30:
        risk_level = "Low Risk"
        advice = "Vascular system is optimal. Maintain current daily step count and diet."
    elif score < 60:
        risk_level = "Moderate Risk"
        advice = "Elevated vascular stress. Consider increasing aerobic activity and monitoring glucose."
    else:
        risk_level = "High Risk"
        advice = "Critical ischemic risk factors present. Immediate cardiovascular consult required."

    return {
        "status": "success",
        "ml_base_probability": round(raw_stroke_prob * 100, 2), 
        "vitalsguard_score": round(score, 1),
        "risk_level": risk_level,
        "recommendation": advice
    }
    
# ---------------------------------------------------------------------------
# 3. ATHLETIC RECOVERY & INJURY MODEL
# ---------------------------------------------------------------------------
EXPECTED_INJURY_COLS = [
    'training_load', 'training_intensity', 'recovery_score', 'fatigue_index', 
    'stress_level', 'sleep_quality', 'age', 'bmi', 'fatigue_load', 
    'recovery_ratio', 'sleep_deficit', 'stress_fatigue', 'cumulative_load', 
    'cumulative_fatigue', 'cumulative_recovery'
]

class RecoveryInput(BaseModel):
    age: float
    bmi: float
    sleep_quality: float 
    stress_level: float  
    training_intensity: float
    # These fields are received from the UI but aren't needed by the ML model
    heart_rate: float | None = None
    hydration_level: float | None = None

def injury_risk_multiplier(data: RecoveryInput) -> float:
    """Applies VitalsGuard biological heuristics to adjust the raw ML probability."""
    multiplier = 1.0
    
    # 1. The "Exhaustion" Penalty: High intensity on terrible sleep is highly dangerous
    if data.sleep_quality < 5 and data.training_intensity >= 8:
        multiplier += 0.45  # Massive spike in risk
        
    # 2. The "Age & Load" Penalty: Older athletes need more recovery time
    if data.age > 35 and data.training_intensity >= 7:
        multiplier += 0.20
    if data.age > 50:
        multiplier += 0.15

    # 3. The "Systemic Stress" Penalty
    if data.stress_level >= 8:
        multiplier += 0.25
        
    # 4. The "Biomechanical" Penalty: High BMI increases joint shear stress
    if data.bmi > 30:
        multiplier += 0.20

    return multiplier

@app.post("/predict/recovery")
def predict_recovery(data: RecoveryInput):
    model = require_model("injury")

    # Replicate your Notebook's feature engineering
    rec_score = ((data.sleep_quality + (10 - data.stress_level)) / 20) * 100
    rec_score = max(0.0, min(100.0, rec_score)) # clamp to 0-100
    
    engineered = {
        "training_load": data.training_intensity * 10,
        "training_intensity": data.training_intensity,
        "recovery_score": rec_score,
        "fatigue_index": data.stress_level + 5,
        "stress_level": data.stress_level,
        "sleep_quality": data.sleep_quality,
        "age": data.age,
        "bmi": data.bmi,
        "fatigue_load": data.training_intensity * 1.5,
        "recovery_ratio": rec_score / 100.0,
        "sleep_deficit": max(0.0, 8.0 - data.sleep_quality),
        "stress_fatigue": data.stress_level * data.training_intensity,
        "cumulative_load": data.training_intensity * 3,
        "cumulative_fatigue": data.stress_level * 2,
        "cumulative_recovery": rec_score * 0.9,
    }
# Convert to dataframe and reorder exactly as the model expects
    input_df = pd.DataFrame([engineered])[EXPECTED_INJURY_COLS]
    
    # 1. Get the RAW probability from the XGBoost model
    raw_probability = float(model.predict_proba(input_df)[0][1])
    
    # 2. Apply the VitalsGuard Custom Math
    multiplier = injury_risk_multiplier(data)
    adjusted_prob = raw_probability * multiplier
    
    # 3. Cap the probability at 99% (so it stays realistic)
    final_probability = min(adjusted_prob, 0.99)
    
    # 4. Map to the 3-tier system using the adjusted probability
    if final_probability < 0.30:
        injury_risk = "Safe to Train"
    elif final_probability < 0.60:
        injury_risk = "Elevated Risk (Caution)"
    else:
        injury_risk = "High Risk of Injury"
    
    return {
        "status": "success",
        "ai_recovery_score": round(rec_score, 1),
        "injury_risk": injury_risk,
        "injury_probability": round(final_probability * 100, 1) 
    }