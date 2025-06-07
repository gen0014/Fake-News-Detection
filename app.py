import streamlit as st # type: ignore
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article manually or upload a `.txt` file to check if it's Real or Fake.")

# Text area input
user_input = st.text_area("Enter News Text (or upload below)", height=250)

# File uploader
uploaded_file = st.file_uploader("Or Upload a Text File (.txt)", type=['txt'])

if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    st.text_area("Uploaded File Content", file_content, height=250)
    user_input = file_content  # Override user input if file uploaded

# Prediction button
if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter or upload some text.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.success("✅ This news is **REAL**.")
        else:
            st.error("🚨 This news is **FAKE**.")
