# Fake vs Real News Detection using Machine Learning

This project focuses on detecting whether a news article is fake or real using Natural Language Processing (NLP) and Machine Learning. It uses TF-IDF vectorization and a Random Forest Classifier to perform binary classification. The project also includes a user-friendly Streamlit web app for interacting with the model — allowing users to visualize the dataset, predict article authenticity, and evaluate model performance in real-time.

Note: Some large files (e.g., datasets and model files) are not included in this repository to comply with GitHub’s file size limits.  
To access the dataset, visit the Kaggle source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## Project Highlights

- Model Used: Random Forest Classifier  
- Accuracy Achieved: ~98% on holdout test set  
- Web App: Built using Streamlit with interactive UI  
- Visualizations: Label/subject distribution, word clouds, confusion matrix  
- Text Analysis: TF-IDF vectorization and NLTK-based preprocessing  
- Deployment-Ready: Modular and scalable structure for future enhancements

---

## Key Features

### 1. Dataset Overview

- Preview top 10 rows  
- View data types of each column  
- Check for missing/null values  
- Display sample headlines and preprocessed text  
- Display article publication date range

### 2. Visualization Dashboard

- Bar chart: Fake vs Real article count  
- Bar chart: News subject/category frequency  
- Word clouds for most frequent words (Real and Fake)

### 3. News Authenticity Prediction

- Select or input article content  
- Automatically preprocess and vectorize input  
- Predict whether the article is “Fake News” or “Real News”

### 4. Model Evaluation

- Dataset is split 80/20 for training/testing  
- Outputs include:  
  - Accuracy score  
  - Precision, Recall, and F1-Score  
  - Confusion Matrix and Classification Report (visualized with heatmap)



Core Concepts and Formulas
### TF-IDF (Term Frequency - Inverse Document Frequency)  
**TF-IDF(t, d)** = TF(t, d) × log(N / DF(t))

- **TF(t, d)**: Frequency of term *t* in document *d*  
- **DF(t)**: Number of documents containing term *t*  
- **N**: Total number of documents in the corpus

TF-IDF helps identify important words in a document by balancing how often a term appears in that document vs. how common it is across all documents.

This helps eliminate common but uninformative words, giving more importance to discriminative terms.

🔹 Evaluation Metrics
### Accuracy
**Accuracy** = (TP + TN) / (TP + TN + FP + FN)

### Precision
**Precision** = TP / (TP + FP)

### Recall
**Recall** = TP / (TP + FN)

### F1-Score
**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
​
