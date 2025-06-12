import matplotlib.pyplot as plt # Import the necessary module
import seaborn as sns

def plot_sentiment_distribution(df, title):
    sentiment_counts = df["Sentiment"].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["green", "gray", "red"])
    plt.title(title)
    plt.show()


plot_sentiment_distribution(datasets["Before_Layoff"], "Sentiment Before Layoffs")


plot_sentiment_distribution(datasets["After_Layoff"], "Sentiment After Layoffs")
