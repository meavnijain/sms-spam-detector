# 🛡️ AI-Powered SMS Spam Detector

A production-ready Machine Learning web application that classifies incoming text messages into **Spam** or **Safe (Ham)**. This project leverages natural language processing (NLP) tokenization and a probabilistic classification architecture to analyze text streams with minimal runtime latency.

🚀 **[Live Demo Link](https://sms-spam-detector-byavnijain.streamlit.app/)**

---

## 📊 How It Works (The Core Logic)

The background pipeline transforms text into mathematical structures using a two-step approach:

1. **Text Preprocessing & Feature Extraction (`CountVectorizer`):** 
   Much like a frequency array approach in low-level programming (e.g., tracking characters via `freq[ch - 'a']++`), `CountVectorizer` scans the entire text dataset to tokenize inputs. It constructs a global sparse matrix tracking the frequency counts of distinct words rather than singular characters.
   
2. **Probabilistic Classification (`MultinomialNB`):** 
   Using the extracted word frequency vectors, a **Multinomial Naive Bayes** model calculates the joint probability of a message belonging to a specific class (Spam vs. Ham) based on individual word weights.

---

## 🛠️ Tech Stack & Architecture

- **Language:** Python 3
- **Machine Learning & NLP:** Scikit-Learn, Pandas
- **Frontend & Deployment Infrastructure:** Streamlit Cloud
- **Version Control:** Git & GitHub

---

## ⚙️ Local Installation & Execution

To run this repository on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd sms-spam-detector
   ```

2. **Install the dependencies:**
   Ensure you have Python installed, then execute:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```

---

## 📈 Future Enhancements
- Expand the static training array into a dynamic dataset via the **UCI SMS Spam Collection dataset**.
- Implement TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to adjust weights for common vocabulary.
- Integrate advanced stop-word filtering to ignore structural noise (like punctuation and common filler words).
