import joblib
import pandas as pd

MODEL_PATH = "models/traffic_congestion_model.joblib"
model = joblib.load(MODEL_PATH)

def predict_congestion(
    vehicle_count,
    traffic_speed_kmh,
    road_occupancy,
    traffic_light_state,
    weather_condition,
    accident_report,
    sentiment_score,
    ride_sharing_demand,
    parking_availability,
    emission_levels,
    energy_consumption,
    hour,
    day_of_week
):
    row = pd.DataFrame([{
        "Vehicle_Count": vehicle_count,
        "Traffic_Speed_kmh": traffic_speed_kmh,
        "Road_Occupancy_%": road_occupancy,
        "Traffic_Light_State": traffic_light_state,
        "Weather_Condition": weather_condition,
        "Accident_Report": accident_report,
        "Sentiment_Score": sentiment_score,
        "Ride_Sharing_Demand": ride_sharing_demand,
        "Parking_Availability": parking_availability,
        "Emission_Levels_g_km": emission_levels,
        "Energy_Consumption_L_h": energy_consumption,
        "Hour": hour,
        "DayOfWeek": day_of_week,
        "IsWeekend": int(day_of_week >= 5),
        "PeakPeriod": int(hour in [7, 8, 9, 16, 17, 18, 19])
    }])

    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    confidence = probabilities.max()

    return prediction, confidence

if __name__ == "__main__":
    prediction, confidence = predict_congestion(
        vehicle_count=180,
        traffic_speed_kmh=42,
        road_occupancy=65,
        traffic_light_state="Red",
        weather_condition="Clear",
        accident_report=0,
        sentiment_score=0.1,
        ride_sharing_demand=60,
        parking_availability=40,
        emission_levels=120,
        energy_consumption=8,
        hour=17,
        day_of_week=2
    )
    print(f"Predicted congestion: {prediction}")
    print(f"Model confidence: {confidence:.1%}")
