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

01_raw

└── Original downloaded datasets

02_processed

└── Cleaned, filtered, and language-selected conversations

03_features

└── Engineered conversation-level feature tables


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

A binary classification approach was used to predict whether an LLM response contains a detected refusal pattern.

Three feature configurations were evaluated:

### Model v01: Structural Features

Includes:

- prompt length
- question count
- spelling error rate
- instruction-related features

### Model v02: Structural + Semantic Features

Adds:

- task type
- topic category
- embedding novelty
- topic probability features

### Model v03: Structural + Semantic + Embedding Features

Adds dense text embeddings to capture latent semantic information from prompts.

Model:

- Logistic Regression

Model interpretation:

- model coefficients
- SHAP feature importance analysis

---

# 📈 Target Variables

## Refusal-Free Response Prediction

Two binary classification targets were created based on explicit refusal patterns:

```python
refusal_pattern = (
    r"(i\s+cannot"
    r"|i\s+can't"
    r"|i\s+do\s+not\s+have\s+access"
    r"|i\s+don't\s+have\s+access"
    r"|i['’]?m\s+sorry.*?(cannot|can't)"
    r"|as\s+an\s+ai\s+language\s+model.*?(cannot|can't))"
)

capability_pattern = (
    r"(i\s+(do\s+not|don't)\s+have\s+access"
    r"|i\s+cannot\s+access"
    r"|i\s+cannot\s+browse"
    r"|i\s+cannot\s+interact"
    r"|i\s+cannot\s+connect"
    r"|i\s+cannot\s+directly\s+"
    r"|i\s+do\s+not\s+have\s+the\s+ability\s+to"
    r"|i\s+don't\s+have\s+the\s+ability\s+to"
    r"|i\s+am\s+unable\s+to\s+)"
)
```

Targets:

- `1`: response without detected refusal pattern
- `0`: response containing refusal pattern


## Length First Response Prediction

```python
"target_cost" = "first_response_tokens"
"log_target_cost" = np.log1p(df["target_cost"])

```


## Cost Proxy

Resource consumption is approximated using:

- response token counts
- prompt length
- interaction complexity

Actual energy consumption is not directly measured.

---

# 📊 Key Results

### Refusal Target 1: target_no_refusal

Feature ablation experiments show that increasingly rich feature representations improve prediction performance.

| Model | Features | ROC AUC | F1 |
|---|---|---:|---:|
| v01 | Structural features | ~0.57 | ~0.65 |
| v02 | Structural + semantic features | ~0.60 | ~0.67 |
| v03 | + embeddings | ~0.73 | ~0.84 |

Main findings:

- Structural prompt features provide a limited predictive power.
- Task and topic information improve slightly model performance.
- Embeddings provide the largest performance improvement by capturing latent semantic patterns.


### Refusal Target 2: target_no_capability

| Model | Features | ROC AUC | F1 |
|---|---|---:|---:|
| v01 | Structural features | ~0.56 | ~0.64 |
| v02 | Structural + semantic features | ~0.59 | ~0.65 |
| v03 | + embeddings | ~0.72 | ~0.88 |


Main findings:

- Structural prompt features provide a limited but measurable signal.
- Task and topic information provide additional predictive value.
- The integration of semantic embeddings does the prediction quality increase significantly.



### Cost Target: Length First Response

| Model | Features | r2 Score | alpha |
|---|---|---:|---:|
| v01 | Structural features | ~0.06| 10|
| v02 | Structural + semantic features | ~0.11 | 10 |
| v03 | + embeddings | ~0.32 | 100 |


Main findings:

- Structural prompt features provide a limited signal.
- Task and topic information improves performance.
- The integration of semantic embeddings does the prediction quality increase significantly.


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

# 🔭 Future Ideas

Potential extensions include:

- comparison with non-linear models such as XGBoost or Random Forest
- calibration and threshold optimization for classification
- evaluation on additional datasets

# ⚠️ Limitations

This project uses proxy measures for LLM resource consumption.

Actual energy usage depends on factors such as:

- model architecture
- hardware infrastructure
- inference optimization
- provider-specific implementation

Therefore, token usage and interaction complexity are used as measurable approximations rather than direct energy measurements.

