import requests
import json

BASE_URL = "http://localhost:8000"

def run_simulation():
    # 1. The Clinical Baseline (Does not change)
    clinical_base = {
        "Pregnancies": 0,
        "Glucose": 110,         # Pre-diabetic
        "BloodPressure": 85,    # Elevated
        "SkinThickness": 20,
        "Insulin": 80,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45
    }

    # 2. Lifestyle Profile A: "The Burnout"
    bad_lifestyle = {
        "bmi": 28.0,
        "daily_steps": 2500,    # Sedentary
        "sleep_hours": 4.5,     # Sleep deprived
        "hydration_liters": 1.0,# Dehydrated
        "stress_level": 9       # High stress
    }

    # 3. Lifestyle Profile B: "The Optimized Protocol"
    good_lifestyle = {
        "bmi": 25.0,
        "daily_steps": 12000,   # Active
        "sleep_hours": 8.0,     # Well rested
        "hydration_liters": 3.0,# Hydrated
        "stress_level": 3       # Low stress
    }

    print("=====================================================")
    print("🧪 RUNNING METABOLIC SIMULATION")
    print("=====================================================")
    
    # Test Bad Lifestyle
    payload_bad = {**clinical_base, **bad_lifestyle}
    res_bad = requests.post(f"{BASE_URL}/predict/metabolic", json=payload_bad).json()
    
    print(f"\n🔴 BAD LIFESTYLE RESULTS:")
    print(f"Risk Category: {res_bad['predicted_category']} (Score: {res_bad['vitalsguard_score']}%)")
    for insight in res_bad['actionable_insights']:
        print(f"   - {insight}")

    # Test Good Lifestyle
    payload_good = {**clinical_base, **good_lifestyle}
    res_good = requests.post(f"{BASE_URL}/predict/metabolic", json=payload_good).json()
    
    print(f"\n🟢 OPTIMIZED LIFESTYLE RESULTS:")
    print(f"Risk Category: {res_good['predicted_category']} (Score: {res_good['vitalsguard_score']}%)")
    for insight in res_good['actionable_insights']:
        print(f"   - {insight}")
    
    print("\nNotice how the exact same clinical bloodwork drops dramatically in risk simply by fixing sleep, stress, and steps!\n")

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"Failed to connect to API. Is FastAPI running? Error: {e}")