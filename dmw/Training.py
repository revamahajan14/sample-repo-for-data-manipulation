import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
#20240802151
df = pd.read_csv(r'C:\Users\ishum\OneDrive\Desktop\data manipulation\dmw\cleaned.csv')

X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
#20240802151
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)

joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
joblib.dump(acc,"accuracy.pkl")