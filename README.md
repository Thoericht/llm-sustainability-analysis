# LLM Sustainability Analysis

This project explores how prompt characteristics, interaction patterns, and semantic properties relate to resource-related outcomes and response behavior in large language model (LLM) interactions.

The project combines exploratory data analysis, feature engineering, semantic representations, and initial machine-learning models. It is documented as a portfolio analysis and as a basis for possible future experiments on more efficient LLM usage.

> **Project status:** The data preparation, exploratory analysis, feature engineering, and initial modeling have been completed. Additional controlled experiments and a deeper comparison of models were not pursued further because the project became increasingly time- and computationally intensive. These extensions remain possible directions for future work.

---

## Project Overview

Large Language Models are increasingly integrated into everyday workflows. Different prompt and interaction patterns may be associated with differences in response length, token usage, and response behavior.

The project investigates measurable relationships between user prompts and selected outcomes, including:

- token consumption and response length
- interaction complexity
- refusal and capability-related response patterns
- semantic properties of prompts
- proxy measures related to computational efficiency

The project does **not** directly measure energy consumption. Instead, token usage and interaction characteristics are treated as observable proxies that can help structure an initial analysis of LLM sustainability.

## Research Questions

The analysis was guided by the following questions:

- How are prompt structure and interaction characteristics related to token consumption and response length?
- Are certain tasks or topics associated with different response patterns?
- Do semantic prompt representations provide additional information about refusal-related responses?
- Which feature groups appear most informative for predicting selected response outcomes?

These questions frame the analysis; they should not be interpreted as claims that all proposed relationships were conclusively established.

## Data and Processing

The project uses an open dialogue dataset containing ShareGPT-style conversations. The data were cleaned, filtered, and restricted to the language selected for the analysis.

The processing structure is organized as follows:

```text
01_raw/
└── Original downloaded datasets

02_processed/
└── Cleaned, filtered, and language-selected conversations

03_features/
└── Engineered conversation-level feature tables
```

The notebooks document the main steps from data preparation and exploration to feature construction and modeling.

## Methodology

### Exploratory Data Analysis

The exploratory analysis examines, among other aspects:

- prompt and response length distributions
- token usage patterns
- interaction and conversation complexity
- task and topic distributions
- response and refusal-related patterns

### Feature Engineering and Semantic Analysis

The feature set combines structural and semantic information, including:

- prompt length and question count
- spelling- and instruction-related features
- task and topic information
- embedding novelty measures
- topic probabilities
- dense text embeddings generated with Sentence Transformers
- CountVectorizer-based representations
- topic-modeling outputs and UMAP projections for exploration

### Initial Machine-Learning Models

The project contains initial applications of:

- Logistic Regression for binary response-pattern targets
- Ridge Regression for a token-based cost proxy
- Random Forest classification

The models were evaluated with task-specific metrics such as ROC-AUC, precision, recall, F1-score, MAE, RMSE, and R². Coefficients and SHAP-based feature importance were used where applicable to support model interpretation.

### Feature Configurations

Several feature configurations were compared conceptually and, for selected models, empirically:

- **Structural features:** prompt length, question count, spelling-related and instruction-related features
- **Structural plus semantic features:** structural features supplemented with task, topic, novelty, and topic-probability features
- **Structural plus semantic plus embedding features:** the preceding features supplemented with dense text embeddings
- **Embedding-only features:** dense text embeddings used as the main predictor set

## Target Variables

### Refusal- and Capability-Related Responses

Binary targets were derived from explicit textual patterns in model responses. These targets indicate whether a response contained selected refusal or capability-related phrases.

A value of `1` indicates that a relevant pattern was detected; a value of `0` indicates that it was not detected. The targets are therefore pattern-based operationalizations and should not be understood as complete semantic classifications of refusals or model capabilities.

### Token-Based Cost Proxy

The first response's token count was used as a measurable proxy for response-related computational demand:

```python
target_cost = first_response_tokens
log_target_cost = np.log1p(target_cost)
```

This is not a measurement of actual energy consumption. Energy use also depends on model architecture, hardware, batching, inference optimization, and provider-specific infrastructure.

## Results Summary

### Logistic Regression

The initial logistic-regression models showed limited predictive performance overall. Adding embeddings improved ROC-AUC for the analyzed refusal-related targets in the documented comparisons. In contrast, the clarification-question target remained close to chance level.

These results suggest that semantic representations may contain useful information for some response-pattern targets, while also showing that predictive performance depends strongly on the target definition and feature configuration.

### Ridge Regression

The Ridge models did not show meaningful predictive power for the logarithmically transformed token-based cost proxy in the documented comparisons. Adding embeddings did not improve the overall regression performance, even though embedding features showed comparatively high aggregated SHAP importance in the interpretation analysis.

This difference illustrates why feature importance should not be interpreted as equivalent to out-of-sample predictive performance.

### Random Forest

Random Forest models were explored as a non-linear alternative for classification. A more extensive evaluation, tuning, calibration, and interpretation were not completed because the project became too time- and computationally intensive to continue within the available scope.

## Project Status and Scope

The completed work provides:

- a reproducible data-preparation and feature-engineering workflow
- exploratory analysis of prompt, conversation, and response characteristics
- initial semantic feature construction
- baseline regression and classification models
- a first assessment of the limits of token-based sustainability proxies

The following work was **not completed**:

- controlled synthetic prompt experiments
- systematic hyperparameter tuning and cross-validation across all model variants
- a comprehensive comparison of Random Forest, XGBoost, and other non-linear models
- calibration and threshold optimization for classification
- direct measurement or estimation of actual energy consumption
- causal conclusions about the effects of prompt design

The project was intentionally paused at this stage so that work could continue on another project. The unfinished components are retained as possible extensions rather than presented as completed results.

## Future Ideas

Possible directions for future work include:

- conducting controlled prompt experiments with matched tasks and systematically varied prompt properties
- evaluating whether shorter prompts or responses actually correspond to lower energy use under specified technical assumptions
- comparing linear, tree-based, and gradient-boosting models using consistent validation procedures
- tuning classification thresholds and assessing probability calibration
- improving refusal and capability detection with manual annotation or more robust language models
- analyzing model-, provider-, and infrastructure-specific differences
- combining token counts with published or measured energy estimates
- testing whether prompt optimization changes response quality, not only response length

## Tools and Libraries

- Python, pandas, NumPy
- scikit-learn
- matplotlib and seaborn
- Sentence Transformers
- tiktoken
- BERTopic and UMAP
- CountVectorizer
- SHAP
- Jupyter notebooks

## Limitations

The analysis is observational and based on an open dialogue dataset. It can identify associations in the available data but cannot establish causal effects.

The refusal and capability targets depend on manually specified text patterns and may miss implicit or differently phrased responses. The token-based cost proxy is also an approximation and cannot replace direct energy measurements.

Consequently, the findings should be interpreted as an exploratory analysis of measurable indicators related to LLM sustainability and response behavior, rather than as a complete assessment of the environmental impact of LLM use.
