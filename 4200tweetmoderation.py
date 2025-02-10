# -*- coding: utf-8 -*-
"""4200TweetModeration.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Syg-vr0QbLLILniDl61pG06YWMrctbV5

Install the necessary libraries
"""

""""
!pip install pandas scikit-learn nltk seaborn matplotlib spacy
!python -m spacy download en_core_web_sm
"""

"""Import the libraries and download Natural Language Toolkit resources"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import re
import spacy
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

"""Upload the datasets"""

# Upload datasets
# from google.colab import files
# uploaded = files.upload()

"""Load and merge the datasets"""

# Load processed datasets
negative_data = pd.read_csv('../../../../Downloads/processedNegative.csv', header=None, names=['text', 'label'])
neutral_data = pd.read_csv('../../../../Downloads/processedNeutral.csv', header=None, names=['text', 'label'])
positive_data = pd.read_csv('../../../../Downloads/processedPositive.csv', header=None, names=['text', 'label'])
malicious_data = pd.read_csv('../../../../Downloads/malicious_tweets.csv', header=None, names=['text', 'label'])

# Assign binary labels
negative_data['binary_label'] = 'Malicious'
malicious_data['binary_label'] = 'Malicious'
neutral_data['binary_label'] = 'Not Malicious'
positive_data['binary_label'] = 'Not Malicious'

# Combine datasets
combined_data = pd.concat([negative_data, neutral_data, positive_data, malicious_data], ignore_index=True)

# Display dataset summary
print("Dataset Summary:")
print(combined_data['binary_label'].value_counts())

"""Preprocess the Data"""

# Load SpaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Define text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Lowercase text
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters
    doc = nlp(text)  # Process text with SpaCy
    words = [token.lemma_ for token in doc if not token.is_stop]  # Lemmatize and remove stopwords
    return " ".join(words)

# Apply cleaning
combined_data['cleaned_text'] = combined_data['text'].apply(clean_text)

# Display cleaned dataset sample
print("Cleaned Dataset Sample:")
print(combined_data.head())

"""Split the dataset"""

# Define features and labels
X = combined_data['cleaned_text']
y = combined_data['binary_label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

"""Vectorize the Data"""

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.8,
    min_df=2,
    ngram_range=(1, 2)  # Include unigrams, bigrams, and trigrams
)

# Transform training and testing sets
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF Matrix Shape (Training): {X_train_tfidf.shape}")
print(f"TF-IDF Matrix Shape (Testing): {X_test_tfidf.shape}")

"""Balance the Data"""

# Balance training data using SMOTE
smote = SMOTE(random_state=42)
X_train_tfidf_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

# Check class distribution after balancing
from collections import Counter
print("Class distribution after SMOTE:", Counter(y_train_balanced))

"""Train and Evaluate"""

# Train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train_tfidf_balanced, y_train_balanced)
mnb_preds = mnb.predict(X_test_tfidf)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=300, random_state=42)
log_reg.fit(X_train_tfidf_balanced, y_train_balanced)
log_preds = log_reg.predict(X_test_tfidf)

# Define evaluation function
def evaluate_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Malicious')
    recall = recall_score(y_true, y_pred, pos_label='Malicious')
    f1 = f1_score(y_true, y_pred, pos_label='Malicious')
    print(f"{model_name}:\n Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")

# Evaluate models
evaluate_model("Multinomial Naive Bayes", y_test, mnb_preds)
evaluate_model("Logistic Regression", y_test, log_preds)

"""Show predictions"""

# Plot prediction counts
results = pd.DataFrame({'True Label': y_test, 'Predicted Label': log_preds})
prediction_counts = results['Predicted Label'].value_counts()

plt.figure(figsize=(8, 5))
prediction_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Classification Results: Malicious vs. Not Malicious")
plt.xlabel("Prediction")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, log_preds, labels=['Not Malicious', 'Malicious'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Malicious', 'Malicious'], yticklabels=['Not Malicious', 'Malicious'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

import joblib

# Save the trained logistic regression model
joblib.dump(log_reg, "log_reg_model.pkl")
print("Logistic Regression model saved as log_reg_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")
print("TF-IDF Vectorizer saved as vectorizer.pkl")

"""Test"""

# Test predictions on custom sentences
sample_sentences = [
    "I will not be coming back to this place again",
    "The sun is bright",
    "I despise this person",
    "Thank you for your support",
    "Lunchly is disgusting",
    "You suck",
    "You were a mistake",
    "I hope you and your family have a good day",
    "Your mother is a nice lady",
    "moving on to bigger and better things… let’s find somebody faster",
    "Have yall like, never had friends before?",
    "New Art from Law's latest Volume",
    "Can’t believe I’m moving into a home this month. Lived in an apartment my entire life. So has my entire family. ",
    "Get rotated idiot"
]

sample_tfidf = vectorizer.transform(sample_sentences)
sample_preds = log_reg.predict(sample_tfidf)

# Display predictions
print("Predictions on Custom Sentences:")
for sentence, pred in zip(sample_sentences, sample_preds):
    print(f"Text: {sentence} -> Predicted: {pred}")

