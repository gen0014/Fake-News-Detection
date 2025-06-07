# 📰 Fake News Detection Using NLP & Machine Learning

This project aims to automatically classify news articles as **Real** or **Fake** using Natural Language Processing (NLP) and a Machine Learning model.  
A **Streamlit web app** is included to allow users to test articles via text input or file upload (`.txt`, `.docx`, `.pdf`).

---

## 🔍 Problem Statement

With the rise of misinformation and fake news on social media and online platforms, there is a critical need for automated fake news detection systems.  
This project solves that by using a labeled dataset to train a machine learning classifier that can distinguish between real and fake news articles.

---

## ✅ Features

- Predict whether a news article is REAL or FAKE
- Upload `.txt`, `.docx`, or `.pdf` files for automatic classification
- Clean and preprocess news content using NLP techniques
- Web interface built using **Streamlit**
- Supports both manual input and file-based input

---

## 🧠 Model & Methodology

- **Preprocessing**: Tokenization, stopword removal, stemming
- **Feature Extraction**: TF-IDF Vectorization
- **Model Used**: Random Forest Classifier
- **Accuracy Achieved**: ~95% on test data

---

## 🛠️ Tech Stack

| Component      | Tool / Library             |
|----------------|----------------------------|
| Programming    | Python                     |
| NLP            | NLTK                       |
| ML             | scikit-learn               |
| Vectorization  | TfidfVectorizer            |
| Web App        | Streamlit                  |
| File Parsing   | python-docx, PyPDF2        |

---

## Install dependencies - 
pip install -r requirements.txt
pip install python-docx PyPDF2

## Run the Streamlit app-
streamlit run app.py

---

🔄 Input Options
* Paste article text in the input box

* OR upload a .txt, .docx, or .pdf file

* Example files are available in the news_examples/ folder.

📈 Results Snapshot
* Accuracy: 95%

* Algorithm: Random Forest

* Input Format: Cleaned text → TF-IDF → Classification

🔮 Future Scope
* Switch to transformer-based models (e.g., BERT)

* Add real-time social media integration

* Support multilingual detection

* Visual analytics for fake news trends

📚 References
* Kaggle Dataset - https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

* scikit-learn

* NLTK

* Streamlit

