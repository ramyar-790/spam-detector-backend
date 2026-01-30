import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def train_model(csv_path="messages.csv"):
    # Load dataset
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df[["message", "label"]]
    df.dropna(inplace=True)

    # Vectorization
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["message"])

    # Train model
    model = MultinomialNB()
    model.fit(X, df["label"])

    return model, vectorizer
