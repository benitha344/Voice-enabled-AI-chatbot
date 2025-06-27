import streamlit as st
import random
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from tensorflow.keras.models import load_model

# Load trained model and data
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()
intents = json.load(open("intents.json", encoding="utf-8"))
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

# Preprocess input
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand."

# Streamlit UI
st.set_page_config(page_title="RIT Chatbot", layout="centered")
st.title("🎓 RIT Chatbot (Text Only)")
st.markdown("Ask your questions about RIT below:")

# Chat input
user_input = st.text_input("You: ")

if st.button("Ask"):
    if user_input:
        intents_list = predict_class(user_input)
        response = get_response(intents_list, intents)
        st.success(f"🤖 Bot: {response}")
