import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

# Load dataset
data = pd.read_csv('model/data.csv')

# Check for missing values
print("Checking for missing values:")
print(data.isna().sum())

# Drop rows where 'medicine' column is missing (alternative: you can fill missing values using fillna)
data = data.dropna(subset=['medicine'])

# Prepare feature and target variables
# We will use 'symptom1', 'symptom2', 'symptom3' as features
# and 'medicine' as the target variable
features = data[['symptom1', 'symptom2', 'symptom3']].apply(lambda x: ' '.join(x), axis=1)
target = data['medicine']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a pipeline with a CountVectorizer and a Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Test the model's accuracy
accuracy = model.score(X_test, y_test)
print(f'Model accuracy: {accuracy * 100:.2f}%')

# Save the trained model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Optionally save the trained model to a separate file if you want
with open('train_model.pkl', 'wb') as train_model_file:
    pickle.dump(model, train_model_file)

print("Model training complete and saved.")
