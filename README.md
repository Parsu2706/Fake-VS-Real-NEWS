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

---

## Project Structure

Fake-VS-Real-NEWS/
│
├── data/
│ └── raw/raw.csv # Raw data (excluded from repo)
│
├── models/
│ ├── best_rf_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── src/
│ ├── data_loader.py
│ ├── preprocess.py
│ ├── train.py
│ └── init.py
│
├── notebooks/
│ └── fake_news.ipynb
│
├── app.py # Streamlit web application
├── run_preprocessing.py # Run data preprocessing
├── requirements.txt
└── README.md

Core Concepts and Formulas
🔹 TF-IDF Vectorization
TF-IDF (Term Frequency-Inverse Document Frequency) weighs a word’s importance across documents:

TF-IDF
(
𝑡
,
𝑑
)
=
TF
(
𝑡
,
𝑑
)
×
log
⁡
(
𝑁
𝐷
𝐹
(
𝑡
)
)
TF-IDF(t,d)=TF(t,d)×log( 
DF(t)
N
​
 )
TF(t, d): Frequency of term t in document d

DF(t): Number of documents containing term t

N: Total number of documents in corpus

This helps eliminate common but uninformative words, giving more importance to discriminative terms.

🔹 Evaluation Metrics
Accuracy

Accuracy
=
𝑇
𝑃
+
𝑇
𝑁
𝑇
𝑃
+
𝑇
𝑁
+
𝐹
𝑃
+
𝐹
𝑁
Accuracy= 
TP+TN+FP+FN
TP+TN
​
 
Precision

Precision
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑃
Precision= 
TP+FP
TP
​
 
Recall

Recall
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑁
Recall= 
TP+FN
TP
​
 
F1-Score

F1
=
2
×
Precision
×
Recall
Precision
+
Recall
F1=2× 
Precision+Recall
Precision×Recall
​
