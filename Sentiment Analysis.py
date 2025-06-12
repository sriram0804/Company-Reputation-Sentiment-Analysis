from nltk.sentiment import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')


sia = SentimentIntensityAnalyzer()


def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


for name, df in datasets.items():
    if df is not None and 'cleaned_tweet' in df.columns:
        df['Sentiment'] = df['cleaned_tweet'].apply(get_sentiment)
        print(f" Sentiment analysis completed for: {name}")

datasets["After_Layoff"].head()
