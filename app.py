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
st.caption("Predict your expected marks based on study hours using AI.")        

st.markdown("""
### 🤖 AI Powered Student Performance Predictor

Enter your study hours and our Machine Learning model
will estimate your expected marks instantly.

Built with:
- Python 🐍
- Scikit-Learn 🤖
- Streamlit 🌐
""")


model = joblib.load("model.pkl")


hours = st.slider(
    "📚 Study Hours",
    0,
    12,
    5
)


if st.button("🚀 Predict Marks"):

    
    data = pd.DataFrame(
        [[hours]],
        columns=["hours"]
    )

    
    prediction = model.predict(data)

    
    st.success(
        f"🎯 Expected Marks: {prediction[0]:.2f}"
    )

    
    st.balloons()

    
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
