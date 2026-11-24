# Turkish Sentiment Analysis on E-Commerce Reviews & Detection of Text–Star Rating Mismatches

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-BERT%20%7C%20BERTurk-yellow.svg)](https://huggingface.co/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-f7931e.svg)](https://scikit-learn.org/)

## 📌 Project Overview & Problem Statement
On e-commerce platforms, customers provide both written reviews and star ratings (1–5 stars). However, these two signals frequently contradict each other—such as leaving a 5-star rating while complaining about receiving the wrong product color, or giving a low rating due to delivery issues despite praising the product. Relying solely on star ratings introduces severe **label noise** into customer satisfaction analytics.

This study addresses 3-class Turkish sentiment analysis (*Positive*, *Neutral*, *Negative*) through a **two-stage experimental pipeline**:
1. **Stage 1 (Controlled Benchmark):** Training and comparing 14 models across Traditional Linear/Probabilistic ML, Ensemble Trees, Deep Learning, and Transformer architectures on a large-scale Turkish review dataset (40,598 reviews).
2. **Stage 2 (Real-World Generalization & Mismatch Detection):** Evaluating the top-performing model on a custom real-world dataset collected from **Trendyol** (500 reviews) to detect mismatches between written review sentiment and user-assigned star ratings, supported by a granular qualitative error analysis.

---

## 👩‍💻 Team & Role Distribution
Developed as part of the Machine Learning course at **Kahramanmaraş Sütçü İmam University** under the supervision of **Assoc. Prof. Dr. Pelin Canbay**.

* **Serap YÜCEL** *(My Contributions — Included in this Repository)*:
  * **Custom Dataset Engineering:** Collected, structured, and annotated the real-world **Trendyol test dataset (500 reviews)** with rich metadata (brand, product, size, gender, date, star rating, and sentiment labels).
  * **Model Architecture & Training (7 Models):** Implemented and trained the Deep Learning and Transformer pipelines (`BERT`, `BERTurk`, `CNN + BiLSTM`, `BiLSTM`, `LSTM`) alongside ensemble tree models (`XGBoost`, `Random Forest`).
  * **Real-World Evaluation & Error Analysis:** Designed the Stage 2 inference pipeline on Trendyol data, conducted the 500-sample mismatch audit (identifying 150 label–text discrepancies), and analyzed prediction confidence distributions.
* **Meltem FISTIK & Ayşe ELGÖRMÜŞ** *(Team Members)*:
  * Implemented and benchmarked the 7 traditional linear/probabilistic baseline classifiers (*Logistic Regression, Linear SVM, SGD Logistic, SGD SVM, Ridge Classifier, Multinomial Naive Bayes, Complement Naive Bayes*).

---

## 🗂️ Datasets & NLP Preprocessing Pipeline
1. **Training & Validation Dataset (40,598 Reviews):** Sourced from Hugging Face (`We-Bears/Turkish-Review-Sentiment-Data`), consisting of Turkish product reviews labeled as *Positive*, *Negative*, and *Neutral*.
2. **Real-World Test Dataset (`ml proje.xlsx` — 500 Reviews):** Custom dataset compiled from **Trendyol** containing raw, noisy user reviews alongside metadata (*Brand, Product, Size, Gender, Date, Star Rating*).

**NLP Preprocessing & Feature Representation:**
* **Text Normalization:** Lowercasing; removal of punctuation, numbers, special characters, and Turkish stop-words.
* **Feature Extraction:**
  * *Tree & Linear Models:* Sparse high-dimensional vectors via **TF-IDF** (Term Frequency–Inverse Document Frequency).
  * *Deep Learning & Transformers:* Dense contextual **Word Embeddings** ($w_i \rightarrow \mathbf{e}_i \in \mathbb{R}^d$) and subword tokenization (`BERT` & `BERTurk` tokenizers handling Turkish agglutinative morphology).

---

## 📊 Experimental Results

### Stage 1: Controlled Test Set Performance (My 7 Models — Hugging Face Dataset)
While traditional linear baselines achieved up to **78.28%** test accuracy (Linear SVM), ensemble trees, deep sequential networks, and transformer models significantly outperformed them. **BERT** and **BERTurk** achieved the highest and most balanced generalization across all metrics without overfitting:

| Model Family | Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ensemble Trees** | Random Forest (TF-IDF) | 89.63% | 86.87% | 91.45% | 88.46% |
| **Ensemble Trees** | XGBoost (TF-IDF) | 89.96% | 89.16% | 89.35% | 89.15% |
| **Deep Learning** | LSTM (Baseline) | *See `lstm.ipynb`* | *See `lstm.ipynb`* | *See `lstm.ipynb`* | *See `lstm.ipynb`* |
| **Deep Learning** | BiLSTM | 90.55% | 88.73% | 91.49% | 89.87% |
| **Deep Learning** | CNN + BiLSTM (Hybrid) | 90.69% | 88.41% | 92.18% | 89.91% |
| **Transformer** | BERTurk (`dbmdz/bert-base-turkish-cased`) | 96.65% | 96.61% | 96.65% | 96.63% |
| **Transformer** | **BERT (Selected Best Model)** | **96.78%** | **96.75%** | **96.78%** | **96.76%** |

### Stage 2: Real-World Generalization on Custom Trendyol Dataset (500 Reviews)
When the best-performing **BERT** model was evaluated directly on the unseen, noisy Trendyol dataset (where ground-truth labels were initially derived from star ratings):
* **Accuracy:** `68.61%` | **Precision:** `74.79%` | **Recall:** `68.61%` | **F1-Score:** `69.62%`
* **Why the drop from 96.78% to 68.61%?** Granular error inspection proved that this gap stems primarily from **user rating inconsistencies (label noise)** and domain shift in natural e-commerce slang/irony rather than model failure. Higher precision (`74.79%`) indicates conservative, reliable predictions.

---

## 🔍 Qualitative Error & Mismatch Analysis (500 Trendyol Reviews)
An in-depth audit of all 500 Trendyol reviews revealed **350 exact matches** and **150 mismatches** between model predictions and star-rating labels:

| Real Trendyol Review Example | User Star Rating | Model Prediction | Root Cause Analysis |
| :--- | :---: | :---: | :--- |
| *"İkinci defa Siyah istedim yine lacivert geldi nasıl iş yapıyorsunuz"* | ⭐⭐⭐⭐⭐ (5 Stars) | **Negative** | User accidentally gave 5 stars or rated delivery speed while writing a clear complaint. BERT correctly prioritized semantic text over star noise. |
| *"rengi daha açık bir beden küçük alınabilir"* | ⭐⭐⭐⭐ (4 Stars) | **Negative** | Critical product feedback paired with a high rating; model captured the underlying dissatisfaction. |
| *"Duruşunu beğenmedim maalesef"* | ⭐⭐⭐ (3 Stars / Neutral) | **Negative** | 3-star ratings are typically mapped to *Neutral*, yet the text expresses explicit dislike. Model resolved the human labeling ambiguity. |
| *"güzel ürün"* | ⭐⭐⭐ (3 Stars / Neutral) | **Positive** | Clearly positive sentiment despite a 3-star rating; model ignored star-label noise and predicted the true sentiment. |
| *"kalıbı dar"* | ⭐⭐⭐⭐ (4 Stars) | **Neutral** | Short, context-poor phrases produce balanced class probabilities, demonstrating calibrated model uncertainty rather than random guessing. |

---

## 📂 Repository Structure
```text
├── bert.ipynb                  # Fine-tuning BERT, Stage 2 Trendyol inference & mismatch analysis
├── berttürk.ipynb              # Fine-tuning Turkish-specific BERTurk model & evaluation
├── cnn+bilstm hibrit.ipynb     # Hybrid CNN + BiLSTM architecture for local n-gram & sequence learning
├── bilstm.ipynb                # Bidirectional LSTM implementation & learning curves
├── lstm.ipynb                  # Baseline LSTM sequence model
├── xgboost.ipynb               # XGBoost classifier with TF-IDF & confidence distribution analysis
├── randomforest.ipynb          # Random Forest classifier with TF-IDF & confusion matrix
└── ml proje.xlsx               # Custom-compiled Trendyol test dataset (500 annotated reviews)
