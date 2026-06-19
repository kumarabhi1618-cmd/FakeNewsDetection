# 🔍 Fake News Detection

A transformer-based binary text classification system to detect misinformation in news articles, benchmarked against classical ML baselines and deployed as a live web application.

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://fakenewsdetection-8xvbprorxytapr3izcxdz9.streamlit.app/)
[![Hugging Face](https://img.shields.io/badge/Model-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/abhishekkumar7/FakeNewsDetectorBERT)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/kumarabhi1618-cmd/FakeNewsDetection)

---

## 📌 Project Overview

| | |
|---|---|
| **Task** | Binary classification — Real vs Fake news |
| **Dataset** | ISOT Fake News Dataset (Fake.csv + True.csv) |
| **Model** | Fine-tuned `bert-base-uncased` (110M parameters) |
| **Deployment** | Streamlit Community Cloud |
| **Model Registry** | Hugging Face Hub — `abhishekkumar7/FakeNewsDetectorBERT` |

---

## 🖥️ Live Demo

🔗 **[https://fakenewsdetection-8xvbprorxytapr3izcxdz9.streamlit.app/](https://fakenewsdetection-8xvbprorxytapr3izcxdz9.streamlit.app/)**

Paste any news article and the fine-tuned BERT model classifies it as **Real** or **Fake** with a confidence score and probability breakdown.

---

## 📊 Results

### BERT (Fine-tuned)
| Metric | Score |
|---|---|
| Accuracy | 99.77% |
| Macro F1 | 0.9977 |
| MCC | 0.9955 |
| AUC-ROC | 1.0000 |
| Best Val F1 | 0.9984 |
| Test Samples | 4,419 |
| Epochs | 1 |

### Classical ML Baselines
| Model | Accuracy | F1 | MCC | AUC-ROC |
|---|---|---|---|---|
| SVM + TF-IDF | 99.16% | 0.9916 | 0.9832 | 0.9987 |
| LR + TF-IDF | 98.87% | 0.9887 | 0.9774 | 0.9987 |
| SVM + BoW | 98.48% | 0.9848 | 0.9697 | 0.9963 |
| LR + BoW | 98.23% | 0.9823 | 0.9647 | 0.9964 |
| NB + BoW | 95.47% | 0.9547 | 0.9094 | 0.9809 |
| NB + TF-IDF | 95.23% | 0.9522 | 0.9044 | 0.9901 |

> BERT contextual embeddings outperform the best classical baseline (SVM + TF-IDF) by **+0.61% F1** and **+0.0123 MCC**, confirming the representational advantage of transformer architectures for NLP classification.

---

## 🏗️ Pipeline Architecture

### Deep Learning Pipeline (BERT)
```
Raw Text
   │
   ▼
Preprocessing (dateline/URL/HTML removal → lowercase → spaCy lemmatization)
   │  stopwords KEPT — BERT needs syntactic context for self-attention
   ▼
BertTokenizerFast (max_length=512, padding, truncation)
   │
   ▼
bert-base-uncased → BertForSequenceClassification (2-class head)
   │
   ▼
Softmax → P(Real), P(Fake) + confidence score
```

### Classical ML Pipeline (BoW / TF-IDF)
```
Raw Text
   │
   ▼
Preprocessing (same regex steps → spaCy lemmatization)
   │  stopwords REMOVED — reduces dimensionality for sparse representations
   ▼
CountVectorizer (BoW) or TfidfVectorizer
   │  fit on TRAIN only — no data leakage
   ▼
Logistic Regression / LinearSVC / Complement Naive Bayes
   │  GridSearchCV + StratifiedKFold
   ▼
Prediction + calibrated probability
```

---

## ⚙️ BERT Training Configuration

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Loss | Weighted cross-entropy (class-imbalance correction) |
| Precision | fp16 mixed-precision |
| Effective batch size | 32 (via gradient accumulation) |
| Early stopping | patience=2 on validation macro-F1 |
| Hardware | Kaggle T4 GPU |

---

## 📁 Repository Structure

```
FakeNewsDetection/
├── app.py                                  # Streamlit inference app
├── requirements.txt                        # Dependencies
├── Fake_News_Detection_BoW_TFIDF.ipynb     # Classical ML pipeline (Kaggle-ready)
├── Fake_News_Detection_BERT_Complete.ipynb # BERT fine-tuning pipeline (Kaggle-ready)
└── Fake_News_Model_Comparison.ipynb        # Model benchmarking & results
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/kumarabhi1618-cmd/FakeNewsDetection.git
cd FakeNewsDetection
pip install -r requirements.txt
streamlit run app.py
```

> On first run, the app downloads the fine-tuned model (~440 MB) from Hugging Face Hub and caches it. No local model files needed.

---

## 🤗 Model on Hugging Face

The fine-tuned model is publicly hosted at:
**[abhishekkumar7/FakeNewsDetectorBERT](https://huggingface.co/abhishekkumar7/FakeNewsDetectorBERT)**

Load it directly in Python:
```python
from transformers import BertTokenizerFast, BertForSequenceClassification

tokenizer = BertTokenizerFast.from_pretrained("abhishekkumar7/FakeNewsDetectorBERT")
model = BertForSequenceClassification.from_pretrained("abhishekkumar7/FakeNewsDetectorBERT")
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, Hugging Face Transformers |
| Classical ML | scikit-learn |
| NLP Preprocessing | spaCy, NLTK |
| Training Platform | Kaggle (T4 GPU) |
| Model Registry | Hugging Face Hub |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## 📂 Dataset

**ISOT Fake News Dataset** — University of Victoria  
- `True.csv` — 21,417 real news articles (Reuters)  
- `Fake.csv` — 23,481 fake news articles  
- Split: 80% train / 10% val / 10% test (stratified)

---

*PPOC Cell, IIT Kanpur — May 2025*
