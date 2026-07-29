# LLM Sustainability Analysis

This project investigates how prompt characteristics, user interaction patterns, and semantic properties influence efficiency-related outcomes in large language model (LLM) interactions.

The analysis combines exploratory data analysis, semantic modeling, feature engineering, and machine learning to identify factors associated with efficient and successful LLM usage.

---

# 📌 Project Overview

Large Language Models are increasingly integrated into everyday workflows. However, different usage patterns can lead to substantial differences in computational requirements, response behavior, and interaction efficiency.

This project analyzes real-world LLM conversations to understand how prompt design and user behavior influence:

- token consumption
- response characteristics
- refusal behavior
- efficiency-related outcomes

The goal is to identify measurable patterns that can support more efficient LLM usage.

---

# ❓ Research Questions

- How does prompt structure influence token consumption and response length?
- Are certain tasks or topics associated with more efficient LLM interactions?
- Can semantic properties of prompts improve prediction of response behavior?
- Can machine learning models identify factors associated with successful, refusal-free responses?

---

# 📊 Data Sources

This project uses a combination of:

- Open dialogue datasets (ShareGPT-style conversations)
- Synthetic prompt experiments for controlled comparisons

Data processing pipeline:


---

# ⚙️ Methodology

## Exploratory Data Analysis

Analysis of:

- prompt length distributions
- token usage patterns
- interaction complexity
- task and topic distributions

## Semantic Analysis

Feature extraction using:

- Sentence Transformers for text embeddings
- Embedding novelty measures
- CountVectorizer
- Topic modeling
- BERTopic
- UMAP dimensionality reduction

## Machine Learning Modeling

Prediction models were developed to analyze which features contribute to efficient LLM behavior.

Three feature configurations were compared:

### Model v01: Structural Features

Includes:

- prompt length
- question count
- spelling error rate
- instruction-related features

### Model v02: Structural + Semantic Features

Additionally includes:

- task type
- topic category
- embedding novelty
- topic probabilities

### Model v03: Structural + Semantic + Embeddings

Adds dense text embeddings to capture latent semantic patterns.

Models:

- Logistic Regression
- Linear Regression
- XGBoost
- Random Forest (planned comparison)

Model interpretation:

- SHAP analysis
- coefficient analysis
- permutation importance

---

# 📈 Target Variables

## Refusal-Free Response Prediction

A binary classification target was created based on explicit refusal patterns:

Examples:

- "I can't"
- "I cannot"
- "I'm sorry"
- "As an AI"
- "I do not have access"

Target:

- `1`: response without detected refusal pattern
- `0`: response containing refusal pattern

## Cost Proxy

Resource consumption is approximated using:

- response token counts
- prompt length
- interaction complexity

Actual energy consumption is not directly measured.

---

# 📊 Key Results

Feature ablation experiments show that increasingly rich feature representations improve prediction performance.

| Model | Features | ROC AUC | F1 |
|---|---|---:|---:|
| v01 | Structural features | ~0.59 | ~0.65 |
| v02 | Structural + semantic features | ~0.63 | ~0.69 |
| v03 | + embeddings | ~0.73 | ~0.83 |

Main findings:

- Structural prompt features provide a limited but measurable signal.
- Task and topic information provide additional predictive value.
- Embeddings provide the largest performance improvement by capturing latent semantic patterns.

---

# 🧠 Key Tools & Libraries

Python ecosystem:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

NLP:

- sentence-transformers
- tiktoken
- BERTopic
- UMAP
- CountVectorizer

Machine Learning:

- Logistic Regression
- Linear Regression
- XGBoost
- Random Forest
- SHAP

---

# 📌 Expected Outcomes

The project aims to provide:

- identification of inefficient LLM usage patterns
- quantitative models of prompt efficiency
- evidence-based recommendations for more efficient prompting
- better understanding of how semantic prompt properties influence LLM behavior

---

# ⚠️ Limitations

This project uses proxy measures for LLM resource consumption.

Actual energy usage depends on factors such as:

- model architecture
- hardware infrastructure
- inference optimization
- provider-specific implementation

Therefore, token usage and interaction complexity are used as measurable approximations rather than direct energy measurements.

