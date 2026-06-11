# Machine Learning & Data Science Fundamentals

Beginner-friendly Jupyter notebooks for Python, data preprocessing, classical machine learning, deep learning, and NLP.

## Notebook folders

| Folder | Topics |
|--------|--------|
| [`notebooks/01-python/`](notebooks/01-python/) | Core Python, data structures, OOP, standard library |
| [`notebooks/02-data-science/`](notebooks/02-data-science/) | Missing values, encoding, scaling, train/test split |
| [`notebooks/03-regression/`](notebooks/03-regression/) | Linear & polynomial regression, SVR, random forest |
| [`notebooks/04-classification/`](notebooks/04-classification/) | Logistic regression, KNN, Naive Bayes, SVM, trees, model comparison |
| [`notebooks/05-clustering/`](notebooks/05-clustering/) | K-means clustering |
| [`notebooks/06-deep-learning/`](notebooks/06-deep-learning/) | Neural networks, churn/audiobooks, MNIST |
| [`notebooks/07-nlp/`](notebooks/07-nlp/) | Text preprocessing, sentiment, NER, LDA, spaCy (`01`–`10`) |

**Datasets:** [`datasets/`](datasets/) — see [`datasets/README.md`](datasets/README.md) for file descriptions.

## How to run

1. Clone this repo  
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. For NLP notebook **10** (spaCy):
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. Launch Jupyter and open a topic folder:
   ```bash
   jupyter notebook notebooks/
   ```

**NLP notebooks (`02`–`10`):** Open notebooks from `notebooks/07-nlp/` so `nlp_helpers.py` imports correctly (Jupyter sets the working directory to that folder).

---

Give a ⭐ if you find this useful!
