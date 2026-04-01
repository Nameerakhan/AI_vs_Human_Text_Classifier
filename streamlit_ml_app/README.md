
# 🤖 ML Text Classification App

The **ML Text Classification App** is a Streamlit-based interactive tool that allows users to explore, compare, and visualize the predictions of various machine learning models trained for binary sentiment analysis (Positive vs Negative). It supports single input prediction, batch processing via files, and model-to-model comparison.

---

## 🚀 Features

- Predict sentiment using multiple models:
  - 📈 Support Vector Machine (Logistic Regression)
  - 🎯 Decision Tree
  - 🔥 AdaBoost Classifier
  - 🧠 Convolutional Neural Network (CNN)
  - 🔁 Long Short-Term Memory (LSTM)
  - 🔂 Recurrent Neural Network (RNN)
- Get confidence scores and probability distributions
- Batch text classification via file upload
- Compare predictions and visualizations across models
- Confusion matrices and agreement analysis
- Modern dark UI with responsive layout

---

## 🗂 Project Structure

```
streamlit_ml_app/
│
├── models/                         
│   ├── sentiment_analysis_pipeline.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── svm_best_model.pkl
│   ├── decision_tree_best_model.pkl
│   ├── adaboost_model.pkl
│   ├── cnn_ai_human_classifier.keras
│   ├── lstm_ai_human_classifier.keras
│   ├── rnn_ai_human_classifier.keras
│   └── tokenizer.pkl
│__ README.md
├── app.py                          
├── requirements.txt                
└── sample_data/                    
    ├── sample_texts.txt
    └── sample_data.csv
```

---

## ⚙ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/streamlit_ml_app.git
cd streamlit_ml_app
```

### 2. Create and Activate Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. ▶ Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📦 Requirements

Make sure your `requirements.txt` includes the following:

```
streamlit
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.7.1
seaborn>=0.12.2
joblib>=1.3.2
tensorflow>=2.12.0
```

> ✅ `tensorflow` is now required to support deep learning models (`.keras`)

---

## 📘 Code Highlights

- **`make_prediction()`** handles logic for both traditional and deep learning models
- **`.keras` models** use Keras with padded sequences and a saved tokenizer
- **`get_available_models()`** dynamically detects loaded models
- **Custom CSS** provides unified dark/light themes
- **Modular layout** supports scaling to more models

Example:

```python
elif model_choice == "CNN_Model":
    tokenizer = load_tokenizer()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = cnn_model.predict(padded)
```

---

## 💡 How to Use

1. Choose a model from the sidebar (SVM, CNN, LSTM, etc.)
2. Enter or upload your input text
3. Click Predict / Compare / Process
4. Visualize:
   - Sentiment predictions
   - Confidence & probability metrics
   - Confusion matrices and model agreement

---

## 🛠 Known Issues

| Problem                               | Solution                                           |
|--------------------------------------|----------------------------------------------------|
| ROC Curve not shown                  | Not applicable for single-input predictions       |
| Models not loading                   | Ensure `.pkl` and `.keras` files exist in `/models/` |
| Tokenizer mismatch                   | Use the exact tokenizer used during training      |
| TensorFlow not installed             | Run `pip install tensorflow`                      |

---

## 🔮 Future Enhancements

- 📤 Export predictions as PDF reports
- 🤗 Add transformer models (e.g., BERT, DistilBERT)
- 🧠 Dynamic integration of any Keras-based model
- 🧪 Evaluate models on live user feedback

---

**Nameera Khan**   
📫 Email: [namkhan@ttu.edu](mailto:namkhan@ttu.edu)  
🔗 [LinkedIn](https://www.linkedin.com/in/nameera-khan-b6963b23b/)
