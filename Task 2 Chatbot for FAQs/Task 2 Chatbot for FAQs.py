# CODEALPHA
# Task#2: Chatbot for FAQs
# Objective: Create a chatbot that can answer frequently asked questions (FAQs) about a particular topic or product. Use natural language processing (NLP) techniques and pre-built libraries like NLTK or SpaCy to understand and generate responses.

# This is a Console Based Chatbot.
# *********************************** CHATBOT FOR FAQs *********************************************

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string
import warnings

warnings.filterwarnings('ignore')

# Download NLTK resources if not already downloaded
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample FAQs (can be modified)
faqs = {
    "What is your return policy?": "You can return the product within 30 days of purchase for a full refund.",
    "What are your shipping options?": "We offer standard, expedited, and overnight shipping options.",
    "How can I track my order?": "You can track your order using the tracking link sent to your email.",
    "What payment methods do you accept?": "We accept credit cards, debit cards, PayPal, and Apple Pay.",
    "Do you offer customer support?": "Yes, our customer support is available 24/7 via chat, email, or phone."
}

# Preprocess text (remove punctuation and lowercase)
def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text

# Tokenize and lemmatize text
def lemmatize(text):
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

# Initialize preprocessed questions and answers
questions = [preprocess(lemmatize(q)) for q in faqs.keys()]
answers = list(faqs.values())

# Define a function to get the most relevant response
def get_response(user_input):
    user_input = preprocess(lemmatize(user_input))
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions + [user_input])
    similarity = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    max_index = np.argmax(similarity)
    
    if similarity[max_index] > 0.2:  # Set a threshold for similarity
        return answers[max_index]
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

# Chatbot interaction
def chatbot():
    print("Chatbot: Hi! I'm here to answer your questions. Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()

# incomplete, hsve to down load scikit learn and nltk library after making some free space. 
# download error due to os error: no space left
