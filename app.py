import streamlit as st  
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)


with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Title of the app
st.title("Text Classification App")

# Description
st.markdown("""
    Welcome to the **Text Classification App**! 🎉  
    Simply type or paste your text into the box below, and our model will analyze it to predict its category.  
    Explore the power of machine learning in action!
""")


# Get user input
user_input = st.text_area("Enter Text for Classification:")


if user_input:
    
    user_input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vectorized)
    

    
    st.write(f"Predicted Category: **{prediction[0]}**")



