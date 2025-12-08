import streamlit as st
import pandas as pd
import joblib
import datetime

# PAGE SETTINGS
st.set_page_config(page_title="Oil Filter Condition Predictor", page_icon="🛠️", layout="wide")

# GLASSMORPHISM + PREMIUM DARK UI
st.markdown("""
<style>

html, body, [class*="css"] {
    background: linear-gradient(135deg, #0d0f12 0%, #161b22 100%) !important;
    color: #e8e8e8 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* HEADER */
.header {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0px 8px 40px rgba(0,0,0,0.5);
}

.header h1 {
    font-size: 36px;
    font-weight: 800;
    margin: 0;
    color: #e8e8e8;
}

.header p {
    font-size: 16px;
    color: #9ca3af;
    margin-top: 8px;
}

/* GLASS CARD */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.35);
    transition: transform 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-6px);
}

/* INPUT LABEL */
label {
    font-weight: 600;
    color: #cfd6df !important;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    padding: 14px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #2d79c7, #3fa9f5);
    color: white;
    font-size: 18px;
    font-weight: 600;
    box-shadow: 0px 4px 20px rgba(45,121,199,0.35);
    transition: all 0.25s ease;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0px 6px 30px rgba(45,121,199,0.55);
}

/* RESULT BOX */
.result-box {
    padding: 25px;
    border-radius: 20px;
    margin-top: 25px;
    text-align: center;
    font-size: 26px;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
}

/* SIDEBAR */
.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    padding-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# LOAD MODEL
model = joblib.load("oil_filter_model_extended.pkl")

# SIDEBAR NAVIGATION
st.sidebar.markdown("<p class='sidebar-title'>📁 Navigation</p>", unsafe_allow_html=True)
pages = ["Predictor", "Dashboard", "Sensor Analytics"]
page = st.sidebar.radio("Go to:", pages)

# HEADER
st.markdown("""
<div class="header">
    <h1>🛠️ Oil Filter Condition Predictor</h1>
    <p>Advanced oil filter lifespan prediction using real sensor intelligence.</p>
</div>
""", unsafe_allow_html=True)


# PAGE 1 — PREDICTOR
if page == "Predictor":

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🚗 Vehicle & Driving Inputs")

    c1, c2 = st.columns(2)

    with c1:
        vehicle_type = st.selectbox("Vehicle Type", ["SUV", "Sedan", "Hatchback", "LCV", "Truck", "Bus"])
        engine_capacity_cc = st.number_input("Engine Capacity (cc)", 800, 8000, value=2000)

        oil_filter_change_date = st.date_input("Oil Filter Change Date")
        today = datetime.date.today()
        oil_filter_age_days = (today - oil_filter_change_date).days
        st.write(f"📅 **Oil Filter Age:** `{oil_filter_age_days}` days")

        km_after_change = st.number_input("Kilometers Driven After Filter Change", 0, 300000, value=5000)

        road_type = st.selectbox("Road Type", ["city", "highway", "offroad", "mixed"])
        load_type = st.selectbox("Load Type", ["light", "medium", "heavy"])

        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        driving_style = st.selectbox("Driving Style", ["calm", "normal", "aggressive"])

    with c2:
        st.subheader("🧪 Sensor Inputs")

        avg_oil_temperature = st.slider("Oil Temperature (°C)", 60, 140, 95)
        oil_viscosity_index = st.slider("Oil Viscosity Index", 0, 100, 70)
        engine_rpm_avg = st.slider("Average RPM", 600, 5000, 2300)
        idling_percentage = st.slider("Idling %", 0, 100, 15)
        ambient_temperature = st.slider("Ambient Temperature (°C)", -5, 60, 30)

        oil_pressure = st.slider("Oil Pressure (bar)", 0.5, 8.0, 3.5, 0.1)
        oil_level_pct = st.slider("Oil Level (%)", 0, 100, 70)
        coolant_temp = st.slider("Coolant Temperature (°°C)", 60, 120, 90)

    st.markdown("</div>", unsafe_allow_html=True)

    input_df = pd.DataFrame([{
        "vehicle_type": vehicle_type,
        "engine_capacity_cc": engine_capacity_cc,
        "oil_filter_age_days": oil_filter_age_days,
        "km_after_change": km_after_change,
        "road_type": road_type,
        "load_type": load_type,
        "avg_oil_temperature": avg_oil_temperature,
        "oil_viscosity_index": oil_viscosity_index,
        "engine_rpm_avg": engine_rpm_avg,
        "idling_percentage": idling_percentage,
        "ambient_temperature": ambient_temperature,
        "fuel_type": fuel_type,
        "driving_style": driving_style,
        "oil_pressure": oil_pressure,
        "oil_level_pct": oil_level_pct,
        "coolant_temp": coolant_temp
    }])

    if st.button("🔍 Predict Oil Filter Condition"):
        prediction = model.predict(input_df)[0]

        colors = {
            "Green": ("🟢", "#2ecc71", "Filter is in excellent condition."),
            "Light-Green": ("🟢", "#58d68d", "Filter is good – recently changed."),
            "Yellow": ("🟡", "#f1c40f", "Moderate – inspection suggested."),
            "Orange": ("🟠", "#e67e22", "Ageing – service recommended soon."),
            "Dark-Orange": ("🟠", "#d35400", "Critical – high degradation."),
            "Red": ("🔴", "#e74c3c", "Severe – replace immediately.")
        }

        icon, color, msg = colors[prediction]

        st.markdown(
            f"""
            <div class='result-box' style='color:{color}; border-color:{color};'>
                {icon}<br>
                {msg}
            </div>
            """,
            unsafe_allow_html=True
        )


# PAGE 2 — DASHBOARD
elif page == "Dashboard":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📊 Dashboard Overview")

    st.metric("Total Predictions Made", 134)
    st.metric("Critical Alerts (Red)", 11)
    st.metric("Average Filter Age", "148 days")

    demo = pd.DataFrame({
        "Condition": ["Green", "Light-Green", "Yellow", "Orange", "Dark-Orange", "Red"],
        "Count": [45, 27, 35, 20, 7, 4]
    })

    st.bar_chart(demo.set_index("Condition"))
    st.markdown("</div>", unsafe_allow_html=True)


# PAGE 3 — SENSOR ANALYTICS
elif page == "Sensor Analytics":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📈 Sensor Analytics (Demo)")

    df = pd.DataFrame({
        "Day": list(range(1, 11)),
        "Oil Temp": [80, 83, 86, 90, 93, 95, 97, 100, 103, 106],
        "Coolant Temp": [78, 80, 82, 85, 87, 90, 92, 95, 96, 98]
    })

    st.line_chart(df.set_index("Day"))
    st.info("Rising temperatures indicate higher engine thermal stress → faster oil filter degradation.")
    st.markdown("</div>", unsafe_allow_html=True)




