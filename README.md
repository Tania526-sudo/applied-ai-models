# applied-ai-models
# Applied AI Models Portfolio (ML/DL • NLP • CV • Recommenders)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-%20-success)
![Pandas](https://img.shields.io/badge/Pandas-%20-success)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![NLP](https://img.shields.io/badge/NLP-IMDB%20%7C%20Summarization-purple)
![Computer%20Vision](https://img.shields.io/badge/Computer%20Vision-Fashion--MNIST-teal)
![Recommenders](https://img.shields.io/badge/Recommenders-SVD%20%7C%20NMF-informational)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)

**Job-oriented portfolio** showcasing practical implementations of Machine Learning & Deep Learning tasks with clear evaluation, model comparison, and reproducible experiments.  
Includes **classical ML pipelines**, **clustering + PCA**, **recommender systems**, and **deep learning** for **Computer Vision** and **NLP**.

---

## Project tags (keywords)
`python` `machine-learning` `deep-learning` `ai` `data-science`  
`numpy` `pandas` `matplotlib` `seaborn`  
`scikit-learn` `svm` `random-forest` `feature-engineering`  
`kmeans` `pca` `clustering`  
`recommender-systems` `surprise` `svd` `svdpp` `nmf`  
`tensorflow` `keras` `cnn` `transfer-learning` `vgg16` `fashion-mnist`  
`nlp` `imdb` `rnn` `lstm` `bilstm` `text-summarization` `nltk` `spacy`

---

## Why this repo (for hiring)
This repository demonstrates:
- **End-to-end ML workflow**: data loading → preprocessing → training → evaluation → conclusions
- **Model comparison beyond accuracy**: `classification_report`, confusion matrices, learning curves, cross-validation where applicable
- **Reproducibility**: fixed seeds, consistent preprocessing, modular helper code
- **Engineering mindset**: structured repo (`notebooks/` + `src/`), clean code style, readable experiments

---

## Repository structure
- `notebooks/` — all experiments and solutions (executed without errors)
- `src/` — reusable helpers (seeding, visualization, metrics, model builders)
- `data/` — local datasets (kept empty / git-ignored)

---

## Highlights (quick navigation)
### Classical ML: Human Activity Recognition (accelerometer)
- `05_human_activity_recognition_svm_rf.ipynb`
  - Time-domain feature engineering
  - **SVM vs RandomForest**
  - `classification_report` + confusion matrix comparison

### Clustering: KMeans + PCA visualization
- `06_kmeans_clustering_pca_visualization.ipynb`
  - Elbow method
  - PCA to 2D for high-dimensional dataset
  - Cluster visualization & interpretation

### Recommender Systems: Matrix Factorization
- `07_recommender_systems_matrix_factorization.ipynb`
  - Surprise: **SVD / SVD++ / NMF**
  - Cross-validation and best model selection (RMSE/MAE)
  - Simple Top-N recommendation demo

### Computer Vision: Fashion-MNIST (Keras)
- `08_fashion_mnist_mlp_baseline.ipynb` — Dense MLP baseline (target ≥ ~0.91)
- `09_fashion_mnist_cnn_architecture_comparison.ipynb` — CNN architecture + comparison to MLP
- `10_fashion_mnist_transfer_learning_vgg16.ipynb` — Transfer learning **VGG16** (feature extraction + fine-tuning)

### NLP: IMDB Sentiment + Summarization (NLTK & spaCy)
- `11_imdb_sentiment_rnn_lstm_bilstm.ipynb`
  - SimpleRNN vs LSTM vs BiLSTM (+ optional deep variant)
  - classification report + training diagnostics
- `12_text_summarization_nltk_spacy.ipynb`
  - Extractive summarization using **NLTK** and **spaCy**
  - Sentence scoring with word frequencies + `heapq.nlargest`

---

## Tech stack
**Core:** Python 3.10+, NumPy, Pandas  
**Visualization:** Matplotlib, Seaborn  
**ML:** scikit-learn  
**Deep Learning:** TensorFlow / Keras  
**NLP:** NLTK, spaCy  
**Recommenders:** scikit-surprise

Links:
- NumPy: https://numpy.org/
- Pandas: https://pandas.pydata.org/
- scikit-learn: https://scikit-learn.org/stable/
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- NLTK: https://www.nltk.org/
- spaCy: https://spacy.io/
- Surprise: https://surpriselib.com/

---

## Setup

### Option A — Local (pip)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
