# LLM Sustainability Analysis

This project explores how prompt characteristics, interaction patterns, and semantic properties relate to response behavior and token-based resource proxies in large language model (LLM) interactions.

It combines exploratory data analysis, feature engineering, semantic representations, and initial machine-learning models. The project is documented as a portfolio analysis and as a basis for future experiments on more efficient LLM usage.

> **Project status:** Data preparation, exploratory analysis, feature engineering, and initial modeling have been completed. The current results are exploratory. Further controlled experiments, systematic tuning, calibration, and direct energy measurement remain future work.

## Project Overview

The analysis investigates measurable relationships between user prompts and selected outcomes, including:

- token consumption and response length
- interaction complexity
- refusal-, capability-, and clarification-related response patterns
- semantic properties of prompts
- proxy measures related to computational efficiency

The project does **not** directly measure energy consumption. Token usage and interaction characteristics are treated as observable proxies that can support an initial sustainability analysis.

## Research Questions

- How are prompt structure and interaction characteristics related to token consumption and response behavior?
- Are certain tasks or topics associated with different response patterns?
- Do semantic prompt representations provide additional information for predicting response outcomes?
- Which feature groups appear most informative for the selected targets?

These questions guide the analysis and should not be interpreted as causal claims.

## Data and Processing

The project uses an open dialogue dataset containing ShareGPT-style conversations. The data were cleaned, filtered, and restricted to the language selected for the analysis.

```text
01_raw/
└── Original downloaded datasets

02_processed/
└── Cleaned, filtered, and language-selected conversations

03_features/
└── Engineered conversation-level feature tables
```

## Methodology

### Exploratory Data Analysis

The exploratory analysis examines:

- prompt and response length distributions
- token usage patterns
- interaction and conversation complexity
- task and topic distributions
- refusal- and response-related patterns

### Feature Engineering and Semantic Analysis

The feature set combines structural and semantic information:

- prompt length and question count
- spelling- and instruction-related features
- task and topic information
- embedding novelty and topic probabilities
- dense text embeddings generated with Sentence Transformers
- CountVectorizer representations
- topic-modeling outputs and UMAP projections for exploration

### Initial Machine-Learning Models

The project includes initial applications of:

- logistic regression for binary response-pattern targets
- random forest classification
- ridge regression for a token-based cost proxy

Classification models were evaluated with ROC-AUC, Average Precision, precision, recall, and F1-score. Regression models were evaluated with MAE, RMSE, and R². Coefficients and SHAP-based feature importance were used for interpretation where appropriate.

## Feature Configurations

- **v01 — Structural features:** prompt length, question count, spelling-related features, and instruction flags.
- **v02 — Structural plus semantic features:** v01 plus task type, topic category, embedding novelty, and topic probability.
- **v03 — Structural plus semantic plus embeddings:** v02 plus 384 dense text-embedding dimensions.
- **v04 — Embeddings only:** dense text embeddings as the predictor set.

## Target Variables

### Response-Pattern Targets

Binary targets were derived from explicit textual patterns in model responses. They represent selected refusal-, capability-, or clarification-related response patterns.

A value of `1` indicates that the relevant pattern was detected; `0` indicates that it was not detected. These targets are pattern-based operationalizations and should not be understood as complete semantic classifications.

### Token-Based Cost Proxy

The first response's token count was used as a measurable proxy for response-related computational demand:

```python
target_cost = first_response_tokens
log_target_cost = np.log1p(target_cost)
```

This is not a direct measurement of energy consumption. Actual energy use also depends on model architecture, hardware, batching, inference optimization, and provider infrastructure.

## Results Summary

### Logistic Regression

The v03 logistic-regression models achieved the strongest overall performance for the three analyzed targets:

| Target | Test ROC-AUC | Test Average Precision | Test F1 | Test Precision | Test Recall |
|---|---:|---:|---:|---:|---:|
| `target_refusal` | 0.753 | 0.075 | 0.093 | 0.050 | 0.614 |
| `target_capability` | 0.755 | 0.040 | 0.060 | 0.032 | 0.536 |
| `target_needs_clarification` | 0.768 | 0.386 | 0.408 | 0.290 | 0.686 |

`target_needs_clarification` is the most predictable target. The low precision for the other targets reflects class imbalance and the default classification threshold. The models should therefore primarily be used for ranking and prioritization unless thresholds are tuned for a specific application.

Feature-level SHAP analysis for v03 showed that `embedding` was the dominant aggregated feature, followed by `log_first_prompt_tokens`, `task_type`, and `topic_cat`. The embedding result combines all 384 dimensions and therefore represents block-level importance rather than equally important or directly interpretable dimensions.

### Ridge Regression

The Ridge models did not show meaningful predictive power for the log-transformed token-based cost proxy in the documented comparisons. Adding embeddings did not improve overall regression performance, although embedding features showed comparatively high aggregated SHAP importance. This illustrates that feature importance should not be interpreted as equivalent to out-of-sample predictive performance.

### Random Forest

Random Forest models were explored as a non-linear alternative for classification. The available analysis indicates that v03 generally provided the strongest overall performance across the evaluated targets, while different versions could produce higher recall for specific targets. A more extensive evaluation, tuning, calibration, and interpretation were not completed within the project scope.

## Project Status and Scope

The completed work provides:

- a reproducible data-preparation and feature-engineering workflow
- exploratory analysis of prompt, conversation, and response characteristics
- semantic feature construction
- baseline regression and classification models
- coefficient- and SHAP-based model interpretation
- an initial assessment of token-based sustainability proxies

The following work remains incomplete:

- controlled synthetic prompt experiments
- systematic hyperparameter tuning across all model variants
- repeated or nested cross-validation
- probability calibration and classification-threshold optimization
- comprehensive comparison of non-linear models
- direct measurement or estimation of actual energy consumption
- causal analysis of prompt design effects

The project was paused at this stage so work could continue on another project. These unfinished components are retained as possible extensions rather than presented as completed results.

## Future Ideas

- Conduct controlled prompt experiments with matched tasks and systematically varied prompt properties.
- Evaluate whether shorter prompts or responses correspond to lower energy use under explicit technical assumptions.
- Compare linear, tree-based, and gradient-boosting models using consistent validation procedures.
- Tune classification thresholds and assess probability calibration.
- Improve refusal and capability detection with manual annotation or more robust language models.
- Analyze model-, provider-, and infrastructure-specific differences.
- Combine token counts with published or measured energy estimates.
- Test whether prompt optimization changes response quality, not only response length.

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

The analysis is observational and based on an open dialogue dataset. It identifies associations in the available data but cannot establish causal effects.

The response-pattern targets depend on manually specified text patterns and may miss implicit or differently phrased behaviors. The token-based cost proxy is an approximation and cannot replace direct energy measurements.

The findings should therefore be interpreted as an exploratory analysis of measurable indicators related to LLM sustainability and response behavior, not as a complete assessment of the environmental impact of LLM use.