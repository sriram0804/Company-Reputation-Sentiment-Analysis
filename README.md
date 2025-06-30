# 💼 Company Reputation Sentiment Analysis

**Analyzing the Impact of Layoffs on Corporate Reputation using NLP**

This project investigates how **layoff events** affect a company's public reputation by analyzing sentiment from Twitter posts before and after the events. It uses both **classical machine learning models** (SVM, Naïve Bayes, Logistic Regression) and **deep learning (LSTM)** to classify sentiment and evaluate brand perception shifts using **Net Brand Reputation (NBR)**.

---

## 🎯 Project Objective

To determine whether layoffs significantly impact a company’s reputation by:

- Collecting and preprocessing tweets from **before and after layoff events**.
- Performing **sentiment classification** (Positive, Negative, Neutral).
- **Comparing sentiment shifts** using performance metrics and Net Brand Reputation (NBR).
- Providing a **live sentiment inference tool** for user-entered text.

---

## 🧠 Models Used

| Model | Description |
|-------|-------------|
| **SVM** (Support Vector Machine) | High-margin classifier trained on TF-IDF vectors. |
| **Naïve Bayes** | Probabilistic classifier used as a simple NLP baseline. |
| **Logistic Regression** | Linear model suitable for binary and multiclass sentiment classification. |
| **LSTM** (Long Short-Term Memory) | Deep learning model leveraging sequence learning with embeddings for better sentiment capture. |

---

## 🛠️ Features

- **📊 Sentiment Classification**
  - Trained on preprocessed tweet data using TF-IDF.
  - Classifies tweets as **Positive**, **Negative**, or **Neutral**.

- **🤖 ML and DL Model Comparison**
  - Evaluate SVM, NB, LR, and LSTM on identical datasets.
  - Compare **accuracy**, **precision**, **recall**, **F1-score**, and **loss**.

- **📉 Confusion Matrix Visualization**
  - Visual insights into model classification performance.

- **📈 Net Brand Reputation (NBR)**
  - Computes the difference in brand sentiment before and after layoffs.
  - `NBR = %Positive - %Negative`

- **📝 Inference Tool**
  - Enter your own text/tweet to get a sentiment prediction from trained models.

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct predictions. |
| **Precision** | Accuracy of positive sentiment predictions. |
| **Recall** | Ability to identify all relevant positive sentiments. |
| **F1-Score** | Harmonic mean of precision and recall. |
| **Loss** (LSTM only) | Evaluates model convergence during training. |

---

## 📂 Project Structure

Company-Reputation-Sentiment-Analysis/
├── data/ # Datasets (Before_Layoff, After_Layoff, etc.)
├── models/ # Trained ML & LSTM models
├── notebooks/ # Jupyter notebooks for experimentation
├── sentiment_predictor.py # Script for real-time inference
├── train_ml_models.py # Classical model training
├── train_lstm_model.py # LSTM model training
├── evaluation.py # Metrics and visualizations
├── requirements.txt # Required Python packages
└── README.md # Project documentation

## 🧪 Datasets Used

- `Before_Layoff.csv`: Tweets posted before company layoff announcements.
- `After_Layoff.csv`: Tweets posted after the announcement.
- Additional datasets for training and testing (`Fb_Training.csv`, `vect.csv`, etc.).

> 📌 *Tweets are cleaned and preprocessed using NLP techniques: tokenization, stopword removal, lemmatization.*
