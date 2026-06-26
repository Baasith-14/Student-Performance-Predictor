import joblib
import pandas as pd

model = joblib.load("model.pkl")

hours = float(input("Enter study hours: "))

data = pd.DataFrame([[hours]], columns=["hours"])

prediction = model.predict(data)

print("Predicted marks:", prediction[0])