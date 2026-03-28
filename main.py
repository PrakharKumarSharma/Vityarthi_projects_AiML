import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset (absolute path)
data = pd.read_csv(r"C:\python_project\dataset.csv")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

data['message'] = data['message'].apply(clean_text)

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2
)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Interactive input
while True:
    text = input("Enter message: ")
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    print("Spam" if result[0] == 1 else "Not Spam")