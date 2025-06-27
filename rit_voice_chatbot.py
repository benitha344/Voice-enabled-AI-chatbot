# ✅ RIT Voice Chatbot in Streamlit (Trained model must exist)
import streamlit as st
import json
import random
import numpy as np
import pickle
import speech_recognition as sr
import pyttsx3
import threading
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from tensorflow.keras.models import load_model

# Initialize tools
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Load trained model and data
intents = json.load(open("intents.json", encoding="utf-8"))
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

# NLP helper functions
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

def get_response(intents_list):
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand."

import threading
import pyttsx3

def speak(text):
    def _speak():
        engine = pyttsx3.init()  # reinitialize on every call
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()  # ensure it's cleaned up
    threading.Thread(target=_speak).start()

def listen():
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            st.info("🎤 Listening...")
            audio = recognizer.listen(source)
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        return f"[Voice Error] {str(e)}"

# Streamlit UI
st.set_page_config(page_title="RIT Voice Chatbot", layout="centered")
st.title("🎓 RIT Voice Chatbot")
st.markdown("Ask questions using mic or text. The bot will speak back!")

# Session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Text input box
user_text = st.text_input("💬 Type your message:")
if st.button("Send Text"):
    st.session_state.history.append(("You", user_text))
    intents_list = predict_class(user_text)
    response = get_response(intents_list)
    st.session_state.history.append(("Bot", response))
    speak(response)

if st.button("🎤 Speak to Bot"):
    user_voice = listen()
    st.session_state.history.append(("You (voice)", user_voice))
    if "[Voice Error]" not in user_voice:
        intents_list = predict_class(user_voice)
        response = get_response(intents_list)
        st.session_state.history.append(("Bot", response))  # ✅ SAME HERE
        speak(response)                                     # ✅ ADD HERE
    else:
        st.warning(user_voice)

# Show conversation
st.divider()
st.subheader("🧠 Conversation")
for sender, message in st.session_state.history:
    if sender == "You" or sender == "You (voice)":
        st.markdown(f"**👤 {sender}:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")
