

import os
import streamlit as st
import numpy as np
import pickle
import re
import google.generativeai as genai

# Set API Key directly (replace with your actual key)
api_key = "AIzaSyCi5rNWXoIXa2wWosI1cEWwCT4eGDPTz5w"  # Replace with your actual API key
genai.configure(api_key=api_key)

# Load the models, vectorizer, and label encoders for the existing classification
with open('category_classifier_model.pkl', 'rb') as model_file:
    model_category = pickle.load(model_file)

with open('maturity_level_classifier_model.pkl', 'rb') as model_file:
    model_maturity = pickle.load(model_file)

with open('context_classifier_model.pkl', 'rb') as model_file:
    model_context = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('label_encoder_category.pkl', 'rb') as encoder_file:
    label_encoder_category = pickle.load(encoder_file)

with open('label_encoder_maturity.pkl', 'rb') as encoder_file:
    label_encoder_maturity = pickle.load(encoder_file)

with open('label_encoder_context.pkl', 'rb') as encoder_file:
    label_encoder_context = pickle.load(encoder_file)

# Define the prediction function for joke attributes
def predict_joke_attributes(joke_text):
    # Clean the joke text
    joke_text_clean = re.sub(r'\s+', ' ', joke_text)
    joke_text_clean = re.sub(r'[^\w\s]', '', joke_text_clean)
    joke_text_clean = joke_text_clean.lower()

    # Transform text to feature vectors
    seq = vectorizer.transform([joke_text_clean])

    # Predict the category, maturity level, and context
    pred_category = model_category.predict(seq)
    pred_maturity = model_maturity.predict(seq)
    pred_context = model_context.predict(seq)

    category = label_encoder_category.inverse_transform(pred_category)[0]
    maturity_level = label_encoder_maturity.inverse_transform(pred_maturity)[0]
    context = label_encoder_context.inverse_transform(pred_context)[0]

    return category, maturity_level, context

# Function to generate joke using Gemini AI
def generate_joke(language, joke_type, joke_context):
    chat_session = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
    ).start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    "your job is to generate a joke based on the information provided by the user,"
                    " you stick on that information and generate a joke based on that."
                    " The main focus is to generate a joke based on the language. The options are Hindi, Malayalam, and English."
                    " If you cannot generate a joke based on this particular language, say that."
                    " The next input by the user is the joke type and joke context."
                    " In case you cannot generate a joke from these, say that and ask the user to rechange their options.\n",
                ],
            },
            {
                "role": "model",
                "parts": [
                    "Okay, I'm ready to generate jokes! Please tell me:\n\n"
                    "1. **What language do you want the joke in?** (Hindi, Malayalam, or English)\n"
                    "2. **What type of joke do you want?** (e.g., puns, knock-knock jokes, observational humor, etc.) \n"
                    "3. **What is the joke context?** (e.g., about animals, food, daily life, etc.)\n"
                    "Once you give me these details, I'll do my best to craft a funny joke for you! 😊\n",
                ],
            },
        ]
    )

    user_input = f"Language: {language}, Joke Type: {joke_type}, Context: {joke_context}"
    response = chat_session.send_message(user_input)
    return response.text

# Start the joke category classifier model for new integration
def start_classifier_chat():
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        },
        system_instruction=(
            "what you have to do is user will input a joke and you have to say which category "
            "that joke belongs to. Give only one single category, don't give multiple categories. "
            "If the input is not a joke, say that. If you cannot classify it, say it's not a joke. "
            "Respond with the joke category only."
        ),
    )

    chat_session = model.start_chat(
        history=[
            {"role": "user", "parts": ["hi\n"]},
            {"role": "model", "parts": ["That's not a joke. 😊 \n"]},
            {"role": "user", "parts": ["Why did the scarecrow win an award?\n\nBecause he was outstanding in his field!\n"]},
            {"role": "model", "parts": ["Pun \n"]},
        ]
    )
    
    return chat_session

# Streamlit interface
st.title('Joke Classification and Generation')

# Create tabs for joke classification and joke generation
tab1, tab2 = st.tabs(["Joke Classification", "AI Joke Generator"])

# Joke classification tab using the existing models
with tab1:
    st.header('Joke Classification with Traditional Models')
    joke_text = st.text_area('Enter a joke:')
    if st.button('Predict'):
        if joke_text:
            category, maturity_level, context = predict_joke_attributes(joke_text)
            st.write(f'The joke belongs to the category: {category}')
            st.write(f'The joke belongs to the maturity level: {maturity_level}')
            st.write(f'The joke is suitable for: {context}')
        else:
            st.write('Please enter a joke to predict.')

    st.header("Joke Classification with AI Model")
    joke_input = st.text_area("Enter your joke here for AI classification:", height=100)
    classify_button = st.button("Classify Joke")

    if classify_button and joke_input:
        classifier_chat_session = start_classifier_chat()
        response = classifier_chat_session.send_message(joke_input)
        st.write(f"Category: **{response.text.strip()}**")

# Joke generation tab
with tab2:
    st.header("AI Joke Generator")

    # Dropdowns for user inputs
    language = st.selectbox("Select Language", options=["English", "Hindi", "Malayalam"])
    joke_type = st.selectbox(
        "Select the Joke Type", 
        options=[
            "Puns", 
            "Knock-knock jokes", 
            "Observational humor", 
            "One-liners", 
            "Dad jokes", 
            "Dark humor", 
            "Surreal humor", 
            "Wordplay"
        ]
    )
    joke_context = st.selectbox(
        "Select the Joke Context", 
        options=[
            "Animals", 
            "Food", 
            "Daily life", 
            "Technology", 
            "Relationships", 
            "Work", 
            "School", 
            "Travel", 
            "Weather", 
            "Sports"
        ]
    )

    # Generate joke button
    if st.button("Generate Joke"):
        with st.spinner("Generating your joke..."):
            joke = generate_joke(language, joke_type, joke_context)
        st.write("Here's your joke:")
        st.write(joke)
