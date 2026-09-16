import pickle
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Load trained model
# -----------------------------
model = load_model("model.h5", compile=False)

# -----------------------------
# Load tokenizer
# -----------------------------
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Twitter Tweets Sentiment Analysis")

tweet = st.text_area("Enter the Tweet:")

if st.button("Predict Sentiment"):

    if not tweet.strip():
        st.warning("Please enter a tweet.")
    else:

        # Tokenize the tweet
        sequences = tokenizer.texts_to_sequences([tweet])

        # Pad sequence
        sequences = pad_sequences(
            sequences,
            padding="post",
            maxlen=166
        )

        # Make prediction
        prediction = model.predict(sequences, verbose=0)

        # Get predicted class
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Sentiment mapping
        sentiment_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        sentiment = sentiment_map[predicted_class]

        st.success(f"Sentiment: {sentiment}")
