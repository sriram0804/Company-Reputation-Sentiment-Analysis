# Company-Reputation-Sentiment-Analysis

This project aims to analyze the impact of layoffs on a company’s reputation by performing sentiment analysis on tweets posted before and after the layoff events. It uses a combination of classical machine learning models (SVM, Naïve Bayes, Logistic Regression) and deep learning (LSTM) to classify public sentiment and measure changes in brand perception.

Project Objective
To determine whether layoffs have a measurable impact on corporate reputation by:

Collecting tweets from before and after company layoff events.

Classifying sentiments as Positive, Negative, or Neutral.

Comparing sentiment shifts using classification metrics and Net Brand Reputation (NBR).

Features
Sentiment Classification using TF-IDF + SVM, NB, and LR.

LSTM-based Sentiment Model for enhanced learning of text sequences.

Model Performance Comparison using accuracy, precision, recall, and F1-score.

Confusion Matrix Visualizations.

Net Brand Reputation (NBR) to numerically capture sentiment changes.

Inference Support: Enter your own text to get sentiment prediction

Models Used

SVM --	High-margin classifier for TF-IDF text
Naïve Bayes --	Probabilistic model for baseline NLP
Logistic Regression --	Simple linear model for binary/multiclass classification
LSTM --	Sequence learning using embedding + RNN

Key Metrics
Accuracy: How often the model predicts correctly.

Precision: Accuracy of positive predictions.

Recall: Ability to find all relevant positive instances.

F1-Score: Harmonic mean of precision and recall.

Loss: Used in LSTM to evaluate training convergence.
