📰 Fake News Detection using NLP & Machine Learning
This project aims to identify and classify fake news articles using Natural Language Processing (NLP) and Machine Learning. A simple and interactive Streamlit web app is deployed for real-time predictions.

🔍 Problem Statement
With the rise of online platforms and social media, fake news has become a major concern. This project addresses the need for an automated system to detect fake news articles and prevent the spread of misinformation.

✅ Features
Detects whether a news article is REAL or FAKE

Trained on a labeled dataset from Kaggle

Includes NLP preprocessing: tokenization, stemming, stopword removal

Uses TF-IDF for text vectorization

Built with Random Forest Classifier

Deployed as a Streamlit app for easy usage

📁 Dataset
Dataset used:
🗂️ Fake and Real News Dataset (Kaggle)

Fake.csv - Fake news articles

True.csv - Real news articles

Combined and labeled as 0 (fake) and 1 (real)

🧪 Tech Stack
Tool / Library	Purpose
Python	Programming language
Pandas, NumPy	Data manipulation
scikit-learn	Machine learning models
NLTK	Text preprocessing (NLP)
TF-IDF	Text vectorization
Streamlit	Web application framework
Pickle	Saving trained models

🚀 How to Run the Project
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Note: Make sure NLTK stopwords are downloaded in the first run.

3. Train the model (if not already trained)
bash
Copy
Edit
python train_model.py
4. Launch the Streamlit app
bash
Copy
Edit
streamlit run app.py
🧠 Model Training Details
Preprocessing: Lowercase conversion, punctuation removal, stopword filtering, stemming

Vectorization: TF-IDF (Top 5000 features)

Model: Random Forest Classifier

Accuracy Achieved: ~95% on test set

🌐 Streamlit App Preview
Input Text	Prediction
“Breaking: Vaccine proven 100% effective”	✅ REAL
“Aliens landed in US – Government hides it”	🚨 FAKE

📈 Results & Evaluation
Accuracy: ~95%

Precision, Recall: Shown in classification report

Streamlit UI provides real-time prediction feedback

🔮 Future Scope
Use BERT or other transformer-based models

Multilingual fake news detection

Real-time social media integration

Classify satire, clickbait, or biased articles

📚 References
Kaggle Dataset

Scikit-learn Documentation

NLTK Documentation

Streamlit Docs

