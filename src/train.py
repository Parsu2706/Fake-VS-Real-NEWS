import os
import pickle
import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score


import matplotlib.pyplot as plt
import seaborn as sns

def main():
    try:
        print(f"[DEBUG] Current working directory: {os.getcwd()}")

        # Load data
        data_path = "processed_data.csv"
        df = pd.read_csv(data_path)

        # Drop rows with NaN in 'clean_text' column
        df = df.dropna(subset=['clean_text'])

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['clean_text'])
        y = df['label']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # RandomForestClassifier
        rf = RandomForestClassifier(random_state=42)

        # Smaller, efficient hyperparameter space
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }

        # Randomized Search (50 random combinations)
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=50,
            cv=3,
            n_jobs=-1,
            verbose=2,
            scoring='accuracy',
            random_state=42
        )

        search.fit(X_train, y_train)

        print("\nRandomizedSearch Completed!")
        print("Best Parameters:", search.best_params_)
        print("Best CV Accuracy:", round(search.best_score_ * 100, 2), "%")

        # Test Set Evaluation
        y_pred = search.predict(X_test)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.2%}")

        # Save model and vectorizer
        os.makedirs("models", exist_ok=True)
        with open("models/best_rf_model.pkl", "wb") as f:
            pickle.dump(search.best_estimator_, f)

        with open("models/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        print("Model and vectorizer saved in 'models/'")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
