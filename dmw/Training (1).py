import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load cleaned dataset
df = pd.read_csv("cleaned.csv")

# Separate features and target
features = ['Age','BMI','CH2O','FAF','TUE','Diet_Risk']
X = df.drop("NObeyesdad", axis=1)   # input features
y = df["NObeyesdad"]                # target variable

# Save feature names (used later in Streamlit)
joblib.dump(X.columns.tolist(), "features.pkl")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved")