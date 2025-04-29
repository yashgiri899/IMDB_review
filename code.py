import pandas as pd
import numpy as np
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', quoting=3)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(lemmatized)

    df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)
    df["review"] = df["review"].apply(clean_text)
    return df


def vectorize_text(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    return X, y


def build_model(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    DATA_PATH = "./data/IMDB Dataset.csv"

    df = load_data(DATA_PATH)
    df = preprocess_text(df)
    X, y = vectorize_text(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(X.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred = model.predict(X_test).round()
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
