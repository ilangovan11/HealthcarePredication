import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
train_df = pd.read_csv(r"C:\Users\user\Desktop\HealthcarePrediction\Training.csv").drop(columns=["Unnamed: 133"], errors='ignore')
test_df = pd.read_csv(r"C:\Users\user\Desktop\HealthcarePrediction\Testing.csv")

# Prepare features and labels
X_train = train_df.drop("prognosis", axis=1)
y_train = train_df["prognosis"]
X_test = test_df.drop("prognosis", axis=1)
y_test = test_df["prognosis"]

# Encode target labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train_encoded)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
report = classification_report(y_test_encoded, y_pred, target_names=encoder.classes_)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(report)
