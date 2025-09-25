# 📰 Fake News Detection with RoBERTa + CNN-LSTM (FakeStack Replication)

> **Jupyter Notebook Project** — Binary classification of news articles (real vs fake) using **RoBERTa** embeddings + **CNN with skip connections** + **LSTM**. Built to replicate and extend the FakeStack architecture in a single, runnable notebook.

---

## 📌 Overview

- Re-implements the FakeStack approach in a **single Jupyter notebook**, swapping **BERT → RoBERTa** for richer contextual embeddings.
- Architecture: **RoBERTa (Hugging Face)** → **CNN (with skip connections)** → **LSTM** → **sigmoid classifier**.
- Training tricks: **weighted BCE loss** for class imbalance + **dynamic thresholding** to maximize F1.

**Paper (PDF):** see [`/docs/NLP_Project-2.pdf`](docs/NLP_Project-2.pdf)

---

## 🧱 What’s in this repo

- `project_v1_8_new.ipynb` — end-to-end code: data prep, model, training, evaluation, exports.
- `docs/NLP_Project-2.pdf` — final report/paper describing methods, experiments, and findings.
---

## ⚡ Key Features

- **RoBERTa embeddings (`roberta-base`)** for contextual text representation.
- **CNN with skip connections** for local pattern extraction and feature reuse.
- **LSTM** for temporal/sequential dependencies.
- **Weighted BCEWithLogitsLoss** to handle class imbalance.
- **Dynamic threshold search** for best F1 (instead of fixed 0.5).



