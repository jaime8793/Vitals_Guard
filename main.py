from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import logging
from typing import List, Tuple

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
        raise HTTPException(status_code=503, detail=f"Model '{name}' not loaded.")
    return model

app = FastAPI(title="VitalsGuard API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ===========================================================================
# SHARED LIFESTYLE SCHEMA
# ===========================================================================
class LifestyleModifiers(BaseModel):
    bmi: float = Field(..., ge=15, le=50)
    daily_steps: int = Field(default=5000, ge=0)
    sleep_hours: float = Field(default=7.0, ge=0, le=24)
    hydration_liters: float = Field(default=2.0, ge=0, le=10)
    stress_level: float = Field(default=5, ge=1, le=10)

# ===========================================================================
# 1. METABOLIC / DIABETES MODEL
# ===========================================================================
class MetabolicInput(LifestyleModifiers):
    Pregnancies: int = Field(..., ge=0)
    Glucose: int = Field(..., ge=0)
    BloodPressure: int = Field(..., ge=0)
    SkinThickness: int = Field(..., ge=0)
    Insulin: int = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=1)

def evaluate_metabolic(data: MetabolicInput, raw_prob: float) -> Tuple[float, str, List[dict]]:
    base_score = raw_prob * 100 
    clinical_penalty = 0.0
    lifestyle_penalty = 0.0
    insights = []
    
    if data.Glucose >= 126: clinical_penalty += 45.0
    elif data.Glucose >= 100: clinical_penalty += 25.0
    if data.BloodPressure >= 90: clinical_penalty += 15.0
    elif data.BloodPressure >= 80: clinical_penalty += 5.0
    
    if data.bmi >= 30:
        lifestyle_penalty += 20.0
        insights.append({"text": "BMI is in the obesity range. A reduction drastically improves insulin sensitivity.", "target": "bmi", "value": 24.9})
    elif data.bmi >= 25:
        lifestyle_penalty += 10.0
        
    if data.daily_steps < 6000:
        lifestyle_penalty += 15.0
        insights.append({"text": "Low step count. Walking after meals blunts glucose spikes by up to 30%.", "target": "dailySteps", "value": 8500})
    elif data.daily_steps > 10000:
        lifestyle_penalty -= 15.0 
        
    if data.sleep_hours < 6:
        lifestyle_penalty += 15.0
        insights.append({"text": f"Only {data.sleep_hours}h of sleep increases cortisol, elevating fasting blood sugar.", "target": "sleepHours", "value": 8.0})
        
    if data.stress_level >= 7:
        lifestyle_penalty += 10.0
        insights.append({"text": "High stress triggers adrenaline and cortisol, heavily contributing to insulin resistance.", "target": "stressLevel", "value": 3})

    score = min(99.0, max(5.0, base_score + clinical_penalty + lifestyle_penalty))
    
    if data.Glucose >= 126 or score >= 75: cat = "Critical Risk"
    elif clinical_penalty >= 25.0 or score >= 50: cat = "Moderate (Medical)"
    elif lifestyle_penalty > 0 or score >= 30: cat = "Elevated (Behavioral)"
    else: cat = "Healthy"
    
    if cat == "Healthy" and not insights:
        insights.append({"text": "Metabolic lifestyle markers are optimal.", "target": None, "value": None})
        
    return round(score, 1), cat, insights

@app.post("/predict/metabolic")
def predict_metabolic(data: MetabolicInput):
    model = require_model("metabolic")
    ml_data = data.model_dump(exclude={"daily_steps", "sleep_hours", "hydration_liters", "stress_level"})
    
    ml_data["BMI"] = ml_data.pop("bmi")
    input_df = pd.DataFrame([ml_data])
    
    for col in model.feature_names_in_:
        if col not in input_df.columns: input_df[col] = 0
    input_df = input_df[model.feature_names_in_]
    
    raw_prob = float(model.predict_proba(input_df)[0][1])
    score, category, insights = evaluate_metabolic(data, raw_prob)
    
    return {"status": "success", "predicted_category": category, "vitalsguard_score": score, "actionable_insights": insights}

# ===========================================================================
# 2. STROKE / VASCULAR MODEL
# ===========================================================================
class StrokeInput(LifestyleModifiers):
    age: float
    hypertension: int 
    heart_disease: int 
    avg_glucose_level: float
    smoking_status: str = "never smoked"

def evaluate_stroke(data: StrokeInput, raw_prob: float) -> Tuple[float, str, List[dict]]:
    base_score = raw_prob * 100
    clinical_penalty = 0.0
    lifestyle_penalty = 0.0
    insights = []
    
    if data.hypertension == 1 and data.avg_glucose_level > 150:
        clinical_penalty += 40.0
        insights.append({"text": "CRITICAL: Hypertension + high glucose exponentially damages blood vessels.", "target": None, "value": None})
    else:
        if data.hypertension == 1: clinical_penalty += 20.0
        if data.avg_glucose_level > 150: clinical_penalty += 15.0
        
    if data.smoking_status == "smokes":
        lifestyle_penalty += 30.0
        insights.append({"text": "Smoking severely increases clot risk.", "target": "smokingStatus", "value": "never smoked"})
    elif data.smoking_status == "formerly smoked":
        lifestyle_penalty += 10.0

    if data.hydration_liters < 1.5:
        lifestyle_penalty += 15.0
        insights.append({"text": f"Drinking {data.hydration_liters}L of water thickens blood plasma, making strokes more likely.", "target": "hydrationLiters", "value": 3.0})
    if data.sleep_hours < 6:
        lifestyle_penalty += 15.0
        insights.append({"text": "Sleep deprivation prevents nighttime blood pressure dipping.", "target": "sleepHours", "value": 8.0})
    if data.stress_level >= 8:
        lifestyle_penalty += 15.0
        insights.append({"text": "Acute stress can trigger vascular spasms.", "target": "stressLevel", "value": 3})
    if data.daily_steps >= 8000:
        lifestyle_penalty -= 15.0
        
    score = min(99.0, max(0.1, base_score + clinical_penalty + lifestyle_penalty))
    
    if score >= 60: cat = "High Risk"
    elif score >= 30: cat = "Moderate Risk"
    else: cat = "Low Risk"
    
    # FIX: Wrapped in dictionary
    if not insights: insights.append({"text": "Vascular system appears stable. Maintain hydration and current activity levels.", "target": None, "value": None})

    return round(score, 1), cat, insights

@app.post("/predict/stroke")
def predict_stroke(data: StrokeInput):
    model = require_model("stroke")
    ml_input = {'age': data.age, 'hypertension': data.hypertension, 'heart_disease': data.heart_disease, 'avg_glucose_level': data.avg_glucose_level, 'bmi': data.bmi}
    
    input_df = pd.DataFrame([ml_input])[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]
    raw_prob = float(model.predict_proba(input_df)[0][1])
    
    score, category, insights = evaluate_stroke(data, raw_prob)
    return {"status": "success", "risk_level": category, "vitalsguard_score": score, "actionable_insights": insights}

# ===========================================================================
# 3. ATHLETIC RECOVERY & INJURY MODEL
# ===========================================================================
class RecoveryInput(LifestyleModifiers):
    age: float
    training_intensity: float
    heart_rate: float | None = None

def evaluate_injury(data: RecoveryInput, raw_prob: float) -> Tuple[float, str, List[dict]]:
    base_score = raw_prob * 100
    lifestyle_penalty = 0.0
    insights = []
    
    if data.sleep_hours < 6 and data.training_intensity >= 8:
        lifestyle_penalty += 45.0
        insights.append({"text": "High-intensity training on low sleep destroys central nervous system recovery.", "target": "sleepHours", "value": 8.5})
    elif data.sleep_hours < 6:
        lifestyle_penalty += 20.0
        insights.append({"text": "Sub-optimal sleep hinders tissue repair.", "target": "sleepHours", "value": 8.0})
        
    if data.hydration_liters < 2.0:
        lifestyle_penalty += 25.0
        insights.append({"text": "Dehydration stiffens fascia and tendons, compromising elasticity.", "target": "hydrationLiters", "value": 3.5})
        
    if data.stress_level >= 7:
        lifestyle_penalty += 15.0
        insights.append({"text": "Systemic stress reduces your body's ability to adapt to training loads.", "target": "stressLevel", "value": 4})

    score = min(99.0, max(0.1, base_score + lifestyle_penalty))
    
    if score >= 60: cat = "High Risk of Injury"
    elif score >= 30: cat = "Elevated Risk (Caution)"
    else: cat = "Safe to Train"

    # FIX: Wrapped in dictionary
    if not insights: insights.append({"text": "Recovery metrics are green. You are cleared for high-intensity physical exertion.", "target": None, "value": None})

    return round(score, 1), cat, insights

@app.post("/predict/recovery")
def predict_recovery(data: RecoveryInput):
    model = require_model("stroke") 
    
    rec_score = ((data.sleep_hours/10 * 50) + ((10 - data.stress_level)/10 * 50))
    rec_score = max(0.0, min(100.0, rec_score))
    
    raw_probability = 0.15 + (data.training_intensity * 0.02)
    
    score, category, insights = evaluate_injury(data, raw_probability)
    return {
        "status": "success", 
        "ai_recovery_score": rec_score,
        "injury_risk": category, 
        "injury_probability": score, 
        "actionable_insights": insights
    }