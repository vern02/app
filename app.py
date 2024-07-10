import os
import streamlit as st
import joblib
from streamlit_extras.let_it_rain import rain

# Define the full path to the TF-IDF vectorizer and Logistics regression model files
tfidf_path = "/Users/vernsin/TfIdf_Vectorizer.joblib"
logreg_path = "/Users/vernsin/Logistics_Regression_Model.joblib"

# Check if the files exist before trying to load them
if os.path.exists(tfidf_path):
    # Load the TF-IDF vectorizer
    vectorizer = joblib.load(tfidf_path)
else:
    st.error("TF-IDF vectorizer file not found!")

if os.path.exists(logreg_path):
    # Load the Logistics Regression model
    best_model = joblib.load(logreg_path)
else:
    st.error("Logistics Regression model file not found!")

def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = best_model.predict(text_vectorized)
    return prediction

st.title("Senti Analyzer")

with st.form("sentiment_form"):
    user_input = st.text_area("Enter a movie review:")
    
    submitted = st.form_submit_button("Predict Sentiment")

    if submitted and user_input.strip():  # Check if the user input is not empty
        sentiment = predict_sentiment(user_input)

        if sentiment == 1:
            sentiment_label = "Positive"
            st.success("The given movie review has positive sentiments! 😊")
            rain(
                emoji = "😊",
                font_size = 20,
                falling_speed = 2,
                animation_length = "infinite")
        else:
            sentiment_label = "Negative"
            st.warning("The given movie review has negative sentiments! 😞")
            rain(
                emoji = "😞",
                font_size = 20,
                falling_speed = 2,
                animation_length = "infinite")

        st.write(f'Sentiment: {sentiment_label}')
    elif submitted:
        st.warning("Please enter a movie review before predicting sentiment.")
