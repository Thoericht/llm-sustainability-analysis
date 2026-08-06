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

- Ridge Regression
- Logistic Regression Classifier
- Random Forest Classifier

### Model Evaluation and Interpretation
The models were evaluated using task-specific performance metrics:

Regression: r2 Score , MAE, and RMSE
Classification: ROC-AUC, precision, recall, and F1-score

Model interpretation was conducted using:

- Model coefficients for Ridge Regression and Logistic Regression
- SHAP feature importance analysis to assess the contribution of individual features to model predictions
- Grouped SHAP importance to evaluate the contribution of original feature groups, such as structural features, semantic metadata, and embeddings

---


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


### Model v04: Dense Embedding Features only 

Adds dense text embeddings to capture latent semantic information from prompts.


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

- `1`: response containing pattern
- `0`: response without detected pattern

## Cost Proxy Prediction

```python
"target_cost" = "first_response_tokens"
"log_target_cost" = np.log1p(df["target_cost"])

```
Actual energy consumption is not directly measured.


---

# 📊 Key Results

## Logistic Regression
Overall, logistic regression showed limited predictive performance. The inclusion of embeddings substantially improved the ROC-AUC for refusal patterns, from 0.56 to 0.75, and for capability-related responses, from 0.62 to 0.71. However, performance in detecting clarification questions remained at chance level, with a ROC-AUC of approximately 0.49.

## Ridge Regression
Ridge regression showed no meaningful predictive power for logarithmic costs across any of the feature sets, with r2 scores values ranging from −0.0003 to −0.0155. In particular, the inclusion of embeddings did not improve model performance, although embeddings showed the highest aggregated feature importance in the SHAP analysis.

## Random Forest Classifier
The Random Forest models show more promising initial results. Their evaluation and interpretation are currently ongoing.



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

- comparison with non-linear models such as XGBoost or Random Forest (in process)
- calibration and threshold optimization for classification

# ⚠️ Limitations

This project uses proxy measures for LLM resource consumption.

Actual energy usage depends on factors such as:

- model architecture
- hardware infrastructure
- inference optimization
- provider-specific implementation

Therefore, token usage and interaction complexity are used as measurable approximations rather than direct energy measurements.

