import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load model and data
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
df = pd.read_csv("cleaned.csv")

st.set_page_config(page_title="Health Analyzer", layout="wide")
print("model expects:" ,)
st.title("💪 Obesity Risk Analyzer")
st.markdown("Analyze how diet and lifestyle affect obesity risk")

# Sidebar inputs
st.sidebar.header("Enter Your Details")

age = st.sidebar.number_input("Age", 10, 80, 25)
height = st.sidebar.number_input("Height (cm)", 130, 220, 170)
weight = st.sidebar.number_input("Weight (kg)", 30.0, 150.0, 70.0)

faf = st.sidebar.number_input("Activity (0-7)", 0, 6, 2)
ch2o = st.sidebar.number_input("Water Intake (L)", 1, 6, 3)
tue = st.sidebar.number_input("Screen Time (0-24)", 0, 24, 2)

analyze = st.sidebar.button("Analyze")

# BMI category function

bmi = weight / (height ** 2)

#layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 User Summary")
    st.write(f"**Age:** {age}")
    st.write(f"**BMI:** {bmi:.2f}")
    st.write(f"**Water Intake:** {ch2o} L")
    st.write(f"**Activity Level:** {faf}")
    st.write(f"**Diet Risk:** {diet_risk}")

with col2:
    st.subheader("🔮 Prediction")

    if st.button("Predict Obesity Level"):
        input_data = pd.DataFrame([[age, bmi, ch2o, faf, tue, diet_risk]] , columns = features )
        result = model.predict(input_data)[0]

        st.success(f"Predicted Category: **{result}**")

        # Interpretation
        if "Obese" in result:
            st.error("⚠️ High Risk: Consider improving diet and activity level.")
        elif "Overweight" in result:
            st.warning("⚠️ Moderate Risk: Small lifestyle changes needed.")
        else:
            st.info("✅ Healthy Range: Keep maintaining your habits!")
st.markdown("---")
st.markdown("💡 **Insight:** High diet risk and low activity significantly " \
"increase obesity probability.")
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# -----------------------------
# BEFORE ANALYZE
# -----------------------------
if not analyze:

    st.subheader("Dataset Overview")

    fig1 = px.histogram(df, x="BMI", title="BMI Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="Age", y="Weight", title="Age vs Weight")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(df, x="NObeyesdad", title="Obesity Levels")
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# AFTER ANALYZE
# -----------------------------
if analyze:

    bmi = weight / ((height / 100) ** 2)
    category = bmi_category(bmi)

    col1, col2 = st.columns(2)

    # BMI Gauge
    with col1:
        st.subheader("Your BMI")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=bmi,
            title={'text': "BMI"},
            gauge={
                'axis': {'range': [10, 40]},
                'steps': [
                    {'range': [10, 18.5], 'color': "#add8e6"},
                    {'range': [18.5, 25], 'color': "#90ee90"},
                    {'range': [25, 30], 'color': "#ffd580"},
                    {'range': [30, 40], 'color': "#ff9999"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Category: **{category}**")

    # Prediction
    with col2:
        st.subheader("Prediction")

        input_dict = {feature: 0 for feature in features}
        input_dict["Age"] = age
        input_dict["Height"] = height
        input_dict["Weight"] = weight
        input_dict["FAF"] = faf
        input_dict["CH2O"] = ch2o
        input_dict["TUE"] = tue

        input_data = np.array([list(input_dict.values())])
        pred = model.predict(input_data)[0]

        st.write(f"Obesity Level: **{pred}**")

        if bmi < 18.5:
            st.warning("Increase calorie intake.")
        elif bmi < 25:
            st.success("Healthy range.")
        elif bmi < 30:
            st.warning("Improve diet and activity.")
        else:
            st.error("High risk. Take care.")

    #calorie maintainance training 
    st.subheader("🎯 Personalized Calorie Plan")
    height_m = height / 100 
    ideal_weight = 22 * (height_m ** 2)

    bmr = 10 * weight + 6.25 * height - 5 * age + 5
    maintenance_calories = bmr * 1.2

    weight_diff = weight - ideal_weight

    if weight_diff > 2:
        traget_calories = maintenance_calories - 500
        goal = "🔴 Calories Deficit (Lose Weight)"
    elif weight_diff < 2:
        traget_calories = maintenance_calories + 500
        goal = "🔵 Calories Surplus (Gain Weight) "
    else:
        traget_calories = maintenance_calories
        goal = "🟢 Maintenance"
    
    st.write(f"**Ideal Weight:** {ideal_weight:.2f} kg")
    st.write(f"**Your Weight:** {weight} kg")
    st.write(f"**Goal:** {goal}")
    st.write(f"**Recommended Daily Calories:** {int(traget_calories)} kcal")

    progress = min(max(int((weight / ideal_weight) * 100), 0), 100)
    st.progress(progress)

    st.caption("Weight vs Ideal Weight Indicator")
    # Comparison
    st.subheader("Comparison with Dataset")

    avg_bmi = df["BMI"].mean()

    comp_df = pd.DataFrame({
        "Type": ["Your BMI", "Average BMI"],
        "Value": [bmi, avg_bmi]
    })

    fig_comp = px.bar(comp_df, x="Type", y="Value", title="BMI Comparison")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Insights
    st.subheader("Dataset Insights")

    st.write(f"Average BMI: {avg_bmi:.2f}")
    st.write(f"Most Frequent BMI: {df['BMI'].mode()[0]:.2f}")
    st.write(f"Least Frequent BMI: {df['BMI'].value_counts().idxmin():.2f}")

    # Expander for visuals
    with st.expander("📊 View Dataset Analysis"):

        fig1 = px.histogram(df, x="BMI", title="BMI Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter(df, x="Age", y="Weight", title="Age vs Weight")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.histogram(df, x="NObeyesdad", title="Obesity Levels")
        st.plotly_chart(fig3, use_container_width=True)
        #theme
        st.markdown("""
        <style>
        .main {
                background-color: #0e1117;
            }
        h1,h2,h3 {
        color: #ffffff;
        }
        .cse-1d391kg {
            background-color: #111827;
        }
        </style>
        """, unsafe_allow_html=True)