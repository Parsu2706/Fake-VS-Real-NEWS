# Fake vs Real News Detection using Machine Learning

This project focuses on detecting whether a news article is **fake** or **real** using Natural Language Processing (NLP) and Machine Learning. It uses TF-IDF vectorization and a **Random Forest Classifier** to perform binary classification. The project also includes a user-friendly **Streamlit web app** for interacting with the model — allowing users to visualize the dataset, predict article authenticity, and evaluate model performance in real-time.

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
- Preview the top 10 rows of the dataset  
- View column data types  
- Check for missing values  
- See sample news titles and cleaned text  
- Display the publication date range  

### 2. Visualization Dashboard
- Bar Chart: Fake vs Real news count  
- Bar Chart: News subject/category frequency  
- Word Cloud: Top frequent words from cleaned text  
  
### 3. News Authenticity Prediction
- Select a specific article row to view its title  
- Uses the cleaned version of the text for prediction  
- Outputs result as: "Fake News" or "Real News"  

### 4. Model Evaluation
- Splits dataset internally (80/20) for evaluation  
- Displays:  
  - Accuracy score  
  - Classification report (precision, recall, F1-score)  
  - Confusion matrix heatmap  

---

## Tools and Technologies Used

| Category        | Tools & Libraries                                              |
|----------------|----------------------------------------------------------------|
| Language        | Python                                                         |
| ML Algorithms   | Random Forest (Scikit-learn)                                   |
| Text Processing | TF-IDF, NLTK, Stopwords                                        |
| Visualization   | Matplotlib, Seaborn, WordCloud                                 |
| Frontend        | Streamlit                                                      |
| Others          | Pandas, NumPy, Pickle, OS, Regex                               |

---

## Dataset Details

**Source:** e.g., Kaggle or custom dataset

**Columns:**

- `title`: News headline  
- `text`: Full content of the news article  
- `clean_text`: Preprocessed text used for machine learning model  
- `label`: Binary label (0 = Fake, 1 = Real)  
- `subject`: News category (e.g., worldnews, politics)  
- `date`: Date of publication  

---

## Skills Demonstrated

- End-to-end machine learning pipeline development  
- Natural Language Processing (NLP) cleaning using tokenization, stopword removal  
- TF-IDF based feature engineering  
- Binary classification using Random Forest  
- Dashboard development using Streamlit  
- Model evaluation using confusion matrix and classification report  
- Modularized and maintainable codebase structure  

