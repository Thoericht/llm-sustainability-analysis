# Analysis Summary: Logistic Regression

## Objective

The objective of this analysis was to identify which prompt characteristics are associated with successful responses without explicit refusal patterns.

The target variable was defined based on the absence of common refusal phrases:

- "I can't"
- "I cannot"
- "I'm sorry"
- "As an AI"
- "I do not have access"

A value of `1` indicates a response without detected refusal patterns, while `0` indicates a response containing a refusal pattern.

---

# Modeling Approach

Three feature configurations were evaluated to measure the incremental value of different information sources:

## Model v01: Structural Features

The baseline model used only prompt structure features:

- log-transformed prompt length
- log-transformed question count
- spelling error rate
- instruction-related binary features:
  - role instruction
  - audience/level instruction
  - formatting instruction

## Model v02: Structural + Semantic Features

The second model additionally incorporated semantic metadata:

- task type
- topic category
- embedding novelty
- topic probability

## Model v03: Structural + Semantic + Embedding Features

The final model extended v02 with dense text embeddings (384 dimensions), allowing the model to capture latent semantic patterns beyond manually engineered features.

All models used a Logistic Regression classifier with preprocessing pipelines including:

- standardization of numerical features
- one-hot encoding of categorical variables
- class weighting to account for class imbalance

Model evaluation was performed using stratified 5-fold cross-validation and a final holdout test set.

---

# Model Performance

| Model | Feature Set | CV ROC AUC | Test ROC AUC | CV F1 | Test F1 | Test Recall |
|---|---|---:|---:|---:|---:|---:|
| v01 | Structural features | 0.59 | 0.59 | 0.66 | 0.65 | 0.49 |
| v02 | Structural + semantic features | 0.62 | 0.63 | 0.69 | 0.69 | 0.54 |
| v03 | + embeddings | 0.74 | 0.73 | 0.83 | 0.83 | 0.72 |

The close agreement between cross-validation and test performance indicates that the models generalize well and that the improvements are not caused by a favorable data split.

---

# Key Findings

## 1. Structural Features Provide Limited Predictive Signal

The baseline model achieved only moderate predictive performance (ROC AUC ≈ 0.59).

The most important structural features were:

- `log_first_prompt_tokens`
- `has_role_instruction`
- `log_question_count`

This indicates that prompt complexity and length contain some information about response behavior, but structural characteristics alone are insufficient for reliable prediction.

---

## 2. Semantic Metadata Improves Prediction Moderately

Adding task and topic information improved performance:

- ROC AUC increased from approximately 0.59 to 0.63.
- Recall improved from approximately 0.49 to 0.54.

Important semantic features included:

- task type
- topic category
- embedding novelty

However, their overall contribution remained relatively small compared with the embedding representation.

---

## 3. Embeddings Provide the Largest Performance Improvement

The inclusion of text embeddings resulted in the strongest improvement:

- ROC AUC increased from 0.63 to approximately 0.73.
- F1 increased from approximately 0.69 to 0.83.
- Recall increased substantially from approximately 0.54 to 0.72.

This indicates that latent semantic representations capture important patterns related to whether a response contains refusal behavior.

The embeddings should not be interpreted through individual dimensions. Instead, their predictive value comes from the combined representation of semantic patterns across all embedding dimensions.

---

# Feature Importance Analysis

SHAP-based feature importance showed that:

| Feature Group | Relative Importance |
|---|---:|
| Prompt length | Highest |
| Question count | Moderate |
| Embeddings | Additional semantic signal |
| Topic category | Additional context |
| Task type | Smaller contribution |
| Instruction flags | Limited contribution |

The results suggest that refusal behavior is influenced by multiple interacting factors:

- prompt complexity
- semantic characteristics
- task context
- latent meaning captured by embeddings

---

# Interpretation of Model Behavior

The model does not predict whether an answer is factually correct or useful. Instead, it predicts whether the generated response avoids explicit refusal patterns.

Therefore:

- A predicted successful response may still contain incorrect information.
- A refusal-free response does not necessarily indicate task completion quality.
- The model captures response behavior rather than answer quality.

---

# Limitations

Several limitations should be considered:

1. **Target definition**
   
   The target is based on pattern matching of refusal phrases. More subtle refusals or indirect limitations may not be detected.

2. **Class imbalance**

   Successful responses dominate the dataset. Therefore, accuracy is not an informative metric; ROC AUC, F1, precision, and recall provide more meaningful evaluation.

3. **Embedding interpretability**

   Individual embedding dimensions lack direct semantic interpretation. Their importance should be analyzed collectively.

4. **Potential dataset bias**

   Task categories and topics may reflect the underlying dataset composition rather than causal relationships.

---

# Conclusion

The analysis demonstrates that predicting refusal-free responses requires semantic information beyond basic prompt structure. Structural features provide a useful baseline, while semantic metadata offers incremental improvements. Dense text embeddings deliver the largest performance gain, indicating that latent semantic representations capture important patterns associated with refusal behavior.

The final model achieves strong predictive performance (ROC AUC ≈ 0.73, F1 ≈ 0.83) while maintaining high precision, suggesting that combining interpretable prompt features with embedding-based representations provides an effective approach for modeling response behavior.