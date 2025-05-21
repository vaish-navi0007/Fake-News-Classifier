# fake_news_classifier.py

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Download stopwords (only needs to run once)
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Load datasets
fake = pd.read_csv(r"C:\Users\Admin\Downloads\Fake.csv.zip")
real = pd.read_csv(r"C:\Users\Admin\Downloads\True.csv.zip")

# Add labels: 0 = fake, 1 = real
fake["label"] = 0
real["label"] = 1

# Combine and shuffle
data = pd.concat([fake, real])
data = data[['title', 'label']].dropna()
data = data.sample(frac=1).reset_index(drop=True)

# Split into features and labels
X = data["title"]
y = data["label"]

# Vectorize titles using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Show performance
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f" Model Accuracy: {accuracy * 100:.2f}%")
print("🧾 Confusion Matrix:")
print(cm)

# Prediction function
def predict_fake_news(title):
    title_vec = vectorizer.transform([title])
    result = model.predict(title_vec)
    return "✅ Real" if result[0] == 1 else "❌ Fake"

# Try it interactively
print("\n📰 Fake News Classifier is ready!")
print("Type a news headline to classify it. Type 'q' to quit.\n")

while True:
    user_input = input("Enter news headline: ")
    if user_input.lower() == 'q':
        break
    prediction = predict_fake_news(user_input)
    print(f"=>Prediction: {prediction}\n")
