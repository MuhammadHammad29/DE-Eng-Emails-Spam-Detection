# 📧 DE/EN Multilingual Spam Classifier

A multilingual email spam classification system that detects **Spam vs Ham** in **German and English** using classical machine learning models, TF-IDF vectorization, and an optional **cross-lingual translation pipeline**.  
The system is deployed as an **interactive Streamlit web application**.

---

## 🚀 Project Overview

This project implements an end-to-end **spam detection pipeline** supporting:
- German and English emails
- Automatic language detection
- Optional German → English translation
- Multiple machine learning models
- Real-time predictions via Streamlit

For German emails, the system performs **dual evaluation**:
- Native German model
- English model applied on translated text

This allows analysis of **cross-lingual robustness and model consistency**.

---

## ✨ Features

- 🌍 Multilingual support (German & English)
- 🔍 Automatic language detection
- 🔁 German → English translation using MarianMT
- 🧠 Multiple ML models with selection option
- 📊 Probability-based predictions
- 📱 Interactive Streamlit interface
- 📈 Evaluation using standard ML metrics

---

## 🧠 Models Implemented

Each language uses the following classifiers:

| Model | Description |
|------|------------|
| Logistic Regression | Balanced, interpretable, best overall performer |
| Multinomial Naïve Bayes | High precision, simple probabilistic model |
| Calibrated Linear SVC | Strong separation with probability calibration |

---

## 📂 Dataset

- **Total samples:** 3,790 emails  
- **Ham:** 2,997  
- **Spam:** 793  

The dataset includes:
- Legitimate service emails
- Phishing attempts
- Lottery and prize scams
- Account suspension messages
- Synthetic spam templates

English data is generated via **machine translation** for cross-lingual experiments.

---

## ⚙️ Preprocessing Pipeline

1. Convert text to lowercase  
2. Mask URLs → `<URL>`  
3. Mask email addresses → `<EMAIL>`  
4. Mask numbers → `<NUM>`  
5. Normalize whitespace  
6. TF-IDF vectorization (unigrams + bigrams)

Separate TF-IDF vectorizers are trained for:
- German
- English

---

## 🔄 Translation Layer

- Uses **Helsinki-NLP MarianMT (opus-mt-de-en)**
- Applied only when German text is detected
- Enables cross-lingual comparison without training multilingual transformers

---

## 🧪 Training & Evaluation

- **Train/Test Split:** 80% / 20% (stratified)
- **Metrics used:**
  - Accuracy
  - Precision (Spam)
  - Recall (Spam)
  - F1-Score
  - AUC-ROC
  - Specificity (Ham)
  - Negative Predictive Value (NPV)

**Best results achieved:**
- Accuracy: **98.55%**
- AUC-ROC: **0.986**

---

## 🖥️ Streamlit Application

### Workflow
1. User pastes an email
2. Language is auto-detected
3. If German:
   - German model runs on original text
   - English model runs on translated text
4. If English:
   - English model runs directly
5. Predictions and probabilities are displayed
6. Translation can be viewed optionally

---

## 📊 Model Selection

The app allows selecting:
- Logistic Regression
- Multinomial Naïve Bayes
- Calibrated Linear SVC

This enables direct comparison of model behavior.

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- HuggingFace Transformers
- PyTorch
- Streamlit
- Joblib

---

## ▶️ How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
2️⃣ Run the app
```bash
streamlit run app.py
```
3️⃣ Open in browser
```bash
http://localhost:8501
```
---
## 📁 Required Files

Ensure the following files are present in the project directory:

app.py
tfidf_de.pkl
tfidf_en.pkl
model_lr_de.pkl
model_nb_de.pkl
model_svc_de.pkl
model_lr_en.pkl
model_nb_en.pkl
model_svc_en.pkl
---

## 📌 Key Takeaways
Classical ML models remain highly effective for text classification
TF-IDF + Logistic Regression is a strong baseline
Probability calibration is essential for SVMs
Translation enables practical cross-lingual NLP
Lightweight models are suitable for real-time deployment
---

## 🚀 Future Work
Add transformer-based multilingual classifiers
Extend to more languages
Cloud deployment (Docker / HuggingFace Spaces)
Incremental learning with new emails
---

## 🎓 Academic Use
This project is suitable for:
Final Year Project (FYP)
Machine Learning / NLP coursework
Viva and academic presentations
Demonstration of end-to-end ML systems
---