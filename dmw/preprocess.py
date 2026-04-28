import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("cleaned.csv")

le = LabelEncoder()
df["NObeyesdad"] = le.fit_transform(df["NObeyesdad"])

encoders = {"NObeyesdad": le}

joblib.dump(encoders, "encoders.pkl")