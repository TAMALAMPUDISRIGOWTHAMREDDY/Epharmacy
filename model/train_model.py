import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("model/dataset.csv")

# Encode symptoms and medicines
encoder = LabelEncoder()
df["medicine"] = encoder.fit_transform(df["medicine"])

# Save label encoding for later use
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Define features and labels
X = df[["symptom1", "symptom2", "symptom3"]]
y = df["medicine"]

# Convert categorical features to numerical values
X = X.apply(encoder.fit_transform)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
