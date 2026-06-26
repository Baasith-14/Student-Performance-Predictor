import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# load dataset
data = pd.read_csv("data.csv")

# input & output
X = data[["hours"]]
y = data["marks"]

# model create
model = LinearRegression()
model.fit(X, y)

# save model
joblib.dump(model, "model.pkl")

print("Model trained & saved ✅")