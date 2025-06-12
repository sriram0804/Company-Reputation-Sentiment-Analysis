import pandas as pd

dataset_path = "/content/drive/MyDrive/dataset/"

datasets = {
    "After_Layoff_General": pd.read_csv(dataset_path + "After_Layoff_General.csv"),
    "After_Layoff": pd.read_csv(dataset_path + "After_Layoff.csv"),
    "Before_Layoff": pd.read_csv(dataset_path + "Before_Layoff.csv"),
    "Fb_Training": pd.read_excel(dataset_path + "Fb_Training.xltx"),
    "vect": pd.read_csv(dataset_path + "vect.csv")
}


print(" Datasets Loaded:", list(datasets.keys()))

import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


for name, df in datasets.items():
    if 'Tweet' in df.columns:
        df['cleaned_tweet'] = df['Tweet'].astype(str).apply(clean_text)
        print(f" Cleaned text in dataset: {name}")
