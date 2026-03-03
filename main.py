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

def evaluate_metabolic(data: MetabolicInput, raw_prob: float) -> Tuple[float, str, List[str]]:
    base_score = raw_prob * 100 
    clinical_penalty = 0.0
    lifestyle_penalty = 0.0
    insights = []
    
    # Additive Clinical Penalties
    if data.Glucose >= 126: clinical_penalty += 45.0
    elif data.Glucose >= 100: clinical_penalty += 25.0
    if data.BloodPressure >= 90: clinical_penalty += 15.0
    elif data.BloodPressure >= 80: clinical_penalty += 5.0
    
    # Additive Lifestyle Penalties & Buffers
    if data.bmi >= 30:
        lifestyle_penalty += 20.0
        insights.append("BMI is in the obesity range. Even a 5% reduction in body weight vastly improves insulin sensitivity.")
    elif data.bmi >= 25:
        lifestyle_penalty += 10.0
        
    if data.daily_steps < 6000:
        lifestyle_penalty += 15.0
        insights.append("Low step count. Walking 15 minutes after meals can blunt glucose spikes by up to 30%.")
    elif data.daily_steps > 10000:
        lifestyle_penalty -= 15.0 # Massive buffer for high activity!
        
    if data.sleep_hours < 6:
        lifestyle_penalty += 15.0
        insights.append(f"Getting only {data.sleep_hours}h of sleep increases cortisol, which elevates fasting blood sugar. Aim for 7+ hours.")
        
    if data.stress_level >= 7:
        lifestyle_penalty += 10.0
        insights.append("High stress triggers adrenaline and cortisol, heavily contributing to systemic insulin resistance.")

    # Calculate final score additively
    score = base_score + clinical_penalty + lifestyle_penalty
    score = min(99.0, max(5.0, score))
    
    # Categorize based on new score limits
    if data.Glucose >= 126 or score >= 75: cat = "Critical Risk"
    elif clinical_penalty >= 25.0 or score >= 50: cat = "Moderate (Medical)"
    elif lifestyle_penalty > 0 or score >= 30: cat = "Elevated (Behavioral)"
    else: cat = "Healthy"
    
    if cat == "Healthy" and not insights:
        insights.append("All lifestyle and clinical markers are optimal. Keep maintaining your current routine.")
        
    return round(score, 1), cat, insights

@app.post("/predict/metabolic")
def predict_metabolic(data: MetabolicInput):
    model = require_model("metabolic")
    ml_data = data.model_dump(exclude={"daily_steps", "sleep_hours", "hydration_liters", "stress_level"})
    
    # Map back to exact ML column names
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

def evaluate_stroke(data: StrokeInput, raw_prob: float) -> Tuple[float, str, List[str]]:
    base_score = raw_prob * 100
    clinical_penalty = 0.0
    lifestyle_penalty = 0.0
    insights = []
    
    if data.hypertension == 1 and data.avg_glucose_level > 150:
        clinical_penalty += 40.0
        insights.append("CRITICAL: Combining hypertension with high glucose exponentially damages blood vessel walls.")
    else:
        if data.hypertension == 1: clinical_penalty += 20.0
        if data.avg_glucose_level > 150: clinical_penalty += 15.0
        
    if data.smoking_status == "smokes":
        lifestyle_penalty += 30.0
        insights.append("Smoking actively constricts blood vessels and thickens blood, severely increasing clot risk.")
    elif data.smoking_status == "formerly smoked":
        lifestyle_penalty += 10.0

    if data.hydration_liters < 1.5:
        lifestyle_penalty += 15.0
        insights.append(f"Drinking only {data.hydration_liters}L of water thickens blood plasma, making ischemic strokes more likely.")
    if data.sleep_hours < 6:
        lifestyle_penalty += 15.0
        insights.append("Chronic sleep deprivation prevents nighttime blood pressure dipping, a major risk factor for strokes.")
    if data.stress_level >= 8:
        lifestyle_penalty += 15.0
        insights.append("High acute stress levels can trigger vascular spasms. Consider integrating parasympathetic breathing protocols.")
    if data.daily_steps >= 8000:
        lifestyle_penalty -= 15.0
        
    score = min(99.0, max(0.1, base_score + clinical_penalty + lifestyle_penalty))
    
    if score >= 60: cat = "High Risk"
    elif score >= 30: cat = "Moderate Risk"
    else: cat = "Low Risk"
    
    if not insights: insights.append("Vascular system appears stable. Maintain hydration and current activity levels.")

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
    training_intensity: float # 1-10
    heart_rate: float | None = None

def evaluate_injury(data: RecoveryInput, raw_prob: float) -> Tuple[float, str, List[str]]:
    base_score = raw_prob * 100
    lifestyle_penalty = 0.0
    insights = []
    
    if data.sleep_hours < 6 and data.training_intensity >= 8:
        lifestyle_penalty += 45.0
        insights.append("DANGER: High-intensity training on low sleep destroys central nervous system recovery. High risk of form breakdown.")
    elif data.sleep_hours < 6:
        lifestyle_penalty += 20.0
        insights.append("Sub-optimal sleep hinders tissue repair. Growth hormone release peaks during deep sleep cycles.")
        
    if data.hydration_liters < 2.0:
        lifestyle_penalty += 25.0
        insights.append("Dehydration stiffens fascia and tendons. Muscular elasticity is severely compromised right now.")
        
    if data.stress_level >= 7:
        lifestyle_penalty += 15.0
        insights.append("Systemic stress reduces your body's ability to adapt to physical training loads. Consider active recovery.")

    if data.heart_rate and data.heart_rate > 80:
        lifestyle_penalty += 15.0
        insights.append("Elevated resting heart rate implies you are under-recovered from previous sessions.")

    score = min(99.0, max(0.1, base_score + lifestyle_penalty))
    
    if score >= 60: cat = "High Risk of Injury"
    elif score >= 30: cat = "Elevated Risk (Caution)"
    else: cat = "Safe to Train"
    
    if not insights: insights.append("Recovery metrics are green. You are cleared for high-intensity physical exertion.")

    return round(score, 1), cat, insights

@app.post("/predict/recovery")
def predict_recovery(data: RecoveryInput):
    model = require_model("stroke") # Fallback to stroke if needed just to simulate the pipeline
    
    # Recalculate AI recovery score based on universal variables
    rec_score = ((data.sleep_hours/10 * 50) + ((10 - data.stress_level)/10 * 50))
    rec_score = max(0.0, min(100.0, rec_score))
    
    # For this demo architecture, we simulate the baseline raw_prob 
    raw_probability = 0.15 + (data.training_intensity * 0.02)
    
    score, category, insights = evaluate_injury(data, raw_probability)
    return {
        "status": "success", 
        "ai_recovery_score": rec_score,
        "injury_risk": category, 
        "injury_probability": score, 
        "actionable_insights": insights
    }