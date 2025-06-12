import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load processed dataset (After_Layoff + Before_Layoff for better training)
df = pd.concat([datasets["After_Layoff"], datasets["Before_Layoff"]])

# Drop NaN values in case they exist
df = df.dropna(subset=["cleaned_tweet", "Sentiment"])

# Map Sentiment labels to numerical values
sentiment_mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
df["Sentiment_Label"] = df["Sentiment"].map(sentiment_mapping)

# Split data into train & test sets
X = df["cleaned_tweet"]
y = df["Sentiment_Label"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,2))  # Increased features for better representation
X_tfidf = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize Base Models with improved parameters
models = {
    "SVM": SVC(kernel="linear", C=2, probability=True),  # Increased regularization parameter
    "Naïve Bayes": MultinomialNB(alpha=0.1),  # Tuned alpha for better smoothing
    "Logistic Regression": LogisticRegression(max_iter=300, solver='liblinear')
}

# Stacked Ensemble Model
stacked_model = StackingClassifier(
    estimators=[("SVM", models["SVM"]), ("Naïve Bayes", models["Naïve Bayes"]), ("Logistic Regression", models["Logistic Regression"])],
    final_estimator=LogisticRegression(),
    cv=5
)

models["Stacked Ensemble"] = stacked_model  # Add to models dictionary

# Train & Evaluate Models
results = {}

for model_name, model in models.items():
    print(f" Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results[model_name] = {
        "accuracy": accuracy,
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1-score": report["macro avg"]["f1-score"],
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    print(f" {model_name} - Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Convert results into a DataFrame for easier visualization
metrics_df = pd.DataFrame(results).T

# Plot Model Performance
plt.figure(figsize=(8, 5))
metrics_df[["accuracy", "precision", "recall", "f1-score"]].plot(kind="bar", figsize=(10, 6))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Display Confusion Matrices
fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Adjusted for 4 models

for idx, (model_name, result) in enumerate(results.items()):
    sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=axes[idx])
    axes[idx].set_title(f"{model_name} - Confusion Matrix")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.tight_layout()
plt.show()
