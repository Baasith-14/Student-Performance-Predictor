import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")

st.markdown("""
### 🤖 AI Powered Student Performance Predictor

Enter your study hours and our Machine Learning model
will estimate your expected marks instantly.

Built with:
- Python 🐍
- Scikit-Learn 🤖
- Streamlit 🌐
""")

# Load trained model
model = joblib.load("model.pkl")

# Study hours slider
hours = st.slider(
    "📚 Study Hours",
    0,
    12,
    5
)

# Predict button
if st.button("🚀 Predict Marks"):

    # Create dataframe
    data = pd.DataFrame(
        [[hours]],
        columns=["hours"]
    )

    # Predict
    prediction = model.predict(data)

    # Show result
    st.success(
        f"🎯 Expected Marks: {prediction[0]:.2f}"
    )

    # Balloons
    st.balloons()

    # Graph
    fig, ax = plt.subplots(figsize=(8, 5))

    hours_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    marks_data = [18, 24, 35, 42, 51, 60, 67, 75, 83, 92]

    ax.plot(hours_data, marks_data, marker="o")

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Marks")
    ax.set_title("Student Performance Trend")

    ax.grid(True)

    st.pyplot(fig)

st.write("Made by Baazy 😎")