import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Dataset for Training
data = {
    'text': [
        "Hey, are we still meeting for lunch today?",
        "Can you send me the notes from the lecture?",
        "CONGRATULATIONS! You won a $1000 Walmart gift card. Click here to claim!",
        "URGENT: Your account has been suspended. Verify your bank details now.",
        "Free ringtones for your phone, text back to opt in.",
        "Dear friend, I hope you are doing well.",
        "Get a free iPhone today by clicking this link!",
        "Are you free to talk on the phone right now?"
    ],
    'label': ['ham', 'ham', 'spam', 'spam', 'spam', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)

# 2. Text Preprocessing (Counting word frequencies)
cv = CountVectorizer()
X = cv.fit_transform(df['text'])
y = df['label']

# 3. Train the Naive Bayes Model
model = MultinomialNB()
model.fit(X, y)

# 4. Streamlit Web Interface Layout
st.title("🛡️ AI SMS Spam Detector")
st.write("Enter a message below to test if the ML model thinks it is Safe or Spam.")

user_input = st.text_area("Type your message:")

if st.button("Check Message"):
    if user_input:
        # Convert user input to match trained features
        data_features = cv.transform([user_input])
        prediction = model.predict(data_features)
        
        # Display the result on the screen
        if prediction == 'spam':
            st.error("🚨 WARNING: This looks like a SPAM message!")
        else:
            st.success("✅ SAFE: This looks like a normal message.")
    else:
        st.warning("Please enter a message first!")
