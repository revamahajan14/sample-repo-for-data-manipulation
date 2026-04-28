import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
df = pd.read_csv("cleaned.csv")

st.set_page_config(page_title="Health Analyzer", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Health and Obesity Analyzer</h1>",
    unsafe_allow_html=True
)

st.sidebar.header("Enter Your Details")

age = st.sidebar.number_input("Age", 10, 80, 25)
height = st.sidebar.number_input("Height (cm)", 130, 220, 170)
weight = st.sidebar.number_input("Weight (kg)", 30.0, 150.0, 70.0)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

faf = st.sidebar.number_input("Activity (0-6)", 0, 6, 2)
ch2o = st.sidebar.number_input("Water Intake (1-8)", 1, 8, 3)
tue = st.sidebar.number_input("Screen Time (0-12)", 0, 12, 2)

analyze = st.sidebar.button("Analyze")

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"
#20240802468
if not analyze:

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        fig = px.histogram(df, x="BMI", nbins=30, title="BMI Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(df, x="Age", nbins=30, title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fig = px.histogram(df, x="Weight", nbins=30, title="Weight Distribution")
        st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)

    with c4:
        fig = px.scatter(df, x="Age", y="BMI", title="Age vs BMI", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        fig = px.scatter(df, x="Weight", y="BMI", title="Weight vs BMI", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

    obesity_counts = df["NObeyesdad"].value_counts().reset_index()
    obesity_counts.columns = ["Category", "Count"]

    fig_pie = px.pie(obesity_counts, names="Category", values="Count", title="Obesity Level Distribution")
    model = joblib.load("model.pkl")
    accuracy = joblib.load(r'C:\Users\ishum\OneDrive\Desktop\data manipulation\accuracy.pkl')

    st.subheader("📊 Model Performance")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.plotly_chart(fig_pie, use_container_width=True)

if analyze:

    bmi = weight / ((height / 100) ** 2)
    category = bmi_category(bmi)

    col1, col2 = st.columns(2)

    with col1:
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
        st.write(f"Category: {category}")

    with col2:
        mapping = {
            0: "Insufficient Weight",
            1: "Normal Weight",
            2: "Overweight Level I",
            3: "Overweight Level II",
            4: "Obesity Type I",
            5: "Obesity Type II",
            6: "Obesity Type III"
        }

        input_dict = {feature: 0 for feature in features}
        input_dict["Age"] = age
        input_dict["Height"] = height
        input_dict["Weight"] = weight
        input_dict["FAF"] = faf
        input_dict["CH2O"] = ch2o
        input_dict["TUE"] = tue

        input_data = np.array([list(input_dict.values())])

        pred_encoded = model.predict(input_data)[0]
        pred = mapping.get(int(pred_encoded), "Unknown")

        st.subheader("Prediction")
        st.write(f"Obesity Level: {pred}")

        if bmi < 18.5:
            st.warning("Increase calorie intake.")
        elif bmi < 25:
            st.success("Healthy range.")
        elif bmi < 30:
            st.warning("Improve diet and activity.")
        else:
            st.error("High risk. Take care.")

    st.subheader("Calorie Recommendation")

    if gender == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    if faf <= 1:
        activity_factor = 1.2
    elif faf <= 2:
        activity_factor = 1.375
    elif faf <= 4:
        activity_factor = 1.55
    elif faf <= 5:
        activity_factor = 1.725
    else:
        activity_factor = 1.9

    maintenance = bmr * activity_factor
    deficit = maintenance * 0.8
    surplus = maintenance * 1.15

    c11, c12, c13 = st.columns(3)

    with c11:
        st.metric("Maintenance Calories", f"{int(maintenance)} kcal")

    with c12:
        st.metric("Fat Loss (Deficit)", f"{int(deficit)} kcal")

    with c13:
        st.metric("Muscle Gain (Surplus)", f"{int(surplus)} kcal")

    if bmi < 18.5:
        st.info("You need a calorie surplus to gain weight.")
    elif bmi < 25:
        st.info("Maintain your current calorie intake.")
    else:
        st.info("A calorie deficit will help reduce weight.")

    st.subheader("Comparison")

    avg_bmi = df["BMI"].mean()

    comp_df = pd.DataFrame({
        "Type": ["Your BMI", "Average BMI"],
        "Value": [bmi, avg_bmi]
    })

    fig_comp = px.bar(comp_df, x="Type", y="Value", title="BMI Comparison")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Insights")

    c6, c7, c8 = st.columns(3)

    with c6:
        st.metric("Average BMI", f"{avg_bmi:.2f}")

    with c7:
        st.metric("Most Frequent BMI", f"{df['BMI'].mode()[0]:.2f}")

    with c8:
        st.metric("Least Frequent BMI", f"{df['BMI'].value_counts().idxmin():.2f}")

    with st.expander("Dataset Analysis"):

        c9, c10 = st.columns(2)

        with c9:
            fig = px.histogram(df, x="BMI", title="BMI Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with c10:
            fig = px.scatter(df, x="Age", y="Weight", title="Age vs Weight")
            st.plotly_chart(fig, use_container_width=True)

        obesity_counts = df["NObeyesdad"].value_counts().reset_index()
        obesity_counts.columns = ["Category", "Count"]

        fig_pie = px.pie(obesity_counts, names="Category", values="Count", title="Obesity Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
        