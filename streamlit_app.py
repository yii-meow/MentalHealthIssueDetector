import streamlit as st
from joblib import load

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer 

import re

lr_model = load('logistic_regression_model.joblib')
vectorizer = load('vectorizer.joblib')

def main():
    st.title('Mental Health Issue Detector')
    
    # Text input for user to enter a sentence
    user_input = st.text_input("Enter a sentence:")
    
    # Button to trigger prediction
    if st.button("Predict"):
        if user_input:
            # Get prediction
            result = predict_sentiment(user_input)
            
            # Display result
            if result == 1:
                st.write("The sentence is classified as: Depressed")
            else:
                st.write("The sentence is classified as: Not Depressed")
                
    # File uploader for analyzing sentences from a file
    st.header("Upload a text file")
    uploaded_file = st.file_uploader("Choose a file", type=["txt"])

    # If file is uploaded
    if uploaded_file is not None:
        # Read uploaded file
        text = uploaded_file.getvalue().decode("utf-8")

        # Split text into sentences and analyze each sentence
        sentences = text.split('\n')
        st.write("Sentences to analyze:")
        for sentence in sentences:
            st.write(sentence)
            result = predict_sentiment(sentence)
            st.write("Result: ", "Depressed" if result == 1 else "Not Depressed")
    
# Function to predict whether the sentence is depressed or not
def predict_sentiment(sentence):
    # Preprocess the input sentence
    processed_sentence = preprocess(sentence)
    
    # Predict using the pre-trained model
    prediction = lr_model.predict(processed_sentence)
    
    return prediction[0]

# Preprocess function to clean and tokenize text
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    
    text = text.lower()
    text = text.replace(r'[^\w\s]+','')
    word = ' '.join([lemmatizer.lemmatize(i, pos='v') for i in word_tokenize(text) if i not in stop_words])
    return vectorizer.transform([word]).toarray()
    
if __name__ == '__main__':
    main()