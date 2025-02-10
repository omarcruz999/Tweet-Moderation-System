import tkinter as tk
from tkinter import messagebox
import joblib
import spacy
import re

# Load SpaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Load the pre-trained model and vectorizer
try:
    model = joblib.load("log_reg_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError as e:
    print("Error loading model or vectorizer:", e)
    exit()


# Define text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters
    doc = nlp(text)  # Process text with SpaCy
    words = [token.lemma_ for token in doc if not token.is_stop]  # Lemmatize and remove stopwords
    return " ".join(words)


# Function to classify the entered text
def classify_text():
    input_text = text_input.get("1.0", tk.END).strip()

    # Preprocess and vectorize the input
    cleaned_text = clean_text(input_text)
    transformed_text = vectorizer.transform([cleaned_text])

    # Predict the label
    prediction = model.predict(transformed_text)[0]
    result_label.config(text=f"Prediction: {prediction}")


# Set up the GUI
app = tk.Tk()
app.title("Tweet Moderation")
app.geometry("500x400")

# Add input text area
tk.Label(app, text="Enter Text Below:", font=("Arial", 12)).pack(pady=5)
text_input = tk.Text(app, height=10, width=50, font=("Arial", 12))
text_input.pack(pady=5)

# Add classify button
predict_button = tk.Button(app, text="Classify", command=classify_text, font=("Arial", 12), bg="lightblue")
predict_button.pack(pady=10)

# Add result label
result_label = tk.Label(app, text="Prediction: ", font=("Arial", 14), fg="darkblue")
result_label.pack(pady=10)

# Start the GUI event loop
app.mainloop()
