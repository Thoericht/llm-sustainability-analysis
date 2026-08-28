# Analysis Summary: Random Forest Models

## Objective

The objective of this analysis was to evaluate how well Random Forest classifiers can predict three different response-related outcomes:

- `target_refusal`
- `target_capability`
- `target_needs_clarification`

The models were designed to identify prompt characteristics associated with these response behaviors. Four feature configurations were compared to measure the incremental value of structural, engineered semantic, and dense embedding features.

---

## Modeling Approach

Four Random Forest configurations were evaluated:

### Model v01: Structural Features

The baseline model used only prompt structure features:

- log-transformed prompt length
- log-transformed question count
- spelling error rate
- instruction-related binary features:
  - role instruction
  - audience or level instruction
  - formatting instruction

### Model v02: Structural + Semantic Features

The second model added engineered semantic metadata:

- task type
- topic category
- embedding novelty
- topic probability

This configuration was intended to capture semantic context while retaining interpretable structural features.

### Model v03: Structural + Semantic + Embedding Features

The third model extended v02 with dense text embeddings containing 384 dimensions. This allowed the Random Forest to model nonlinear relationships between latent semantic patterns and the target outcomes.

### Model v04: Embedding-Only Features

The final model used only the 384-dimensional text embeddings. This configuration measured how much predictive information was contained in the embeddings alone, without structural or manually engineered semantic features.

All models used a `RandomForestClassifier` with:

- 500 trees
- square-root feature sampling
- balanced class weights
- minimum split size of 5
- minimum leaf size of 2
- fixed random state for reproducibility
- parallel processing

Model evaluation used stratified 5-fold cross-validation and a final stratified holdout test set. In addition to ROC AUC and F1, average precision was included because the target classes are imbalanced.

---

## Model Performance

### Target: Refusal

| Model | Feature Set | CV ROC AUC | Test ROC AUC | CV F1 | Test F1 | Test Precision | Test Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| v01 | Structural features | 0.61 | 0.67 | 0.08 | 0.11 | 0.07 | 0.39 |
| v02 | Structural + semantic features | 0.70 | 0.74 | 0.22 | 0.35 | 0.74 | 0.23 |
| v03 | + embeddings | 0.80 | 0.83 | 0.30 | 0.43 | 0.78 | 0.30 |
| v04 | Embeddings only | 0.80 | 0.83 | 0.29 | 0.41 | 0.66 | 0.30 |

The structural baseline performed relatively weakly, with a test ROC AUC of 0.67 and a test F1 score of only 0.11. Adding semantic metadata improved the test ROC AUC to 0.74 and substantially increased precision.

The strongest overall performance was achieved by v03 and v04. Both reached a test ROC AUC of approximately 0.83. Model v03 achieved a slightly higher F1 score than v04, while v04 achieved marginally higher ROC AUC. This indicates that embeddings contain most of the predictive information for identifying refusal-related outcomes.

However, recall remained relatively low at approximately 0.30. The models therefore identified refusal cases with reasonable precision but failed to detect a considerable proportion of all refusal cases.

### Target: Capability

| Model | Feature Set | CV ROC AUC | Test ROC AUC | CV F1 | Test F1 | Test Precision | Test Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| v01 | Structural features | 0.69 | 0.72 | 0.06 | 0.06 | 0.03 | 0.34 |
| v02 | Structural + semantic features | 0.73 | 0.79 | 0.26 | 0.25 | 0.88 | 0.14 |
| v03 | + embeddings | 0.82 | 0.83 | 0.33 | 0.40 | 0.81 | 0.27 |
| v04 | Embeddings only | 0.81 | 0.81 | 0.31 | 0.36 | 0.57 | 0.27 |

For the capability target, the baseline model achieved a test ROC AUC of 0.72 but produced very low precision and F1. This indicates that structural features alone were not sufficient to distinguish capability-related cases reliably.

Adding semantic metadata improved the test ROC AUC to 0.79. The model achieved very high precision of 0.88, but recall decreased to 0.14. In other words, v02 made relatively few positive predictions, but those predictions were often correct.

The best balance was achieved by v03. Its test ROC AUC was 0.83, with an F1 score of 0.40, precision of 0.81, and recall of 0.27. The embedding-only model performed somewhat worse, suggesting that structural and engineered semantic features still provide useful complementary information when combined with embeddings.

### Target: Needs Clarification

| Model | Feature Set | CV ROC AUC | Test ROC AUC | CV F1 | Test F1 | Test Precision | Test Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| v01 | Structural features | 0.67 | 0.67 | 0.34 | 0.34 | 0.30 | 0.39 |
| v02 | Structural + semantic features | 0.79 | 0.80 | 0.43 | 0.45 | 0.64 | 0.35 |
| v03 | + embeddings | 0.86 | 0.86 | 0.43 | 0.46 | 0.87 | 0.31 |
| v04 | Embeddings only | 0.86 | 0.86 | 0.43 | 0.45 | 0.86 | 0.31 |

The clarification target was the easiest outcome to predict. Even the structural baseline achieved a test ROC AUC of 0.67 and an F1 score of 0.34.

Adding semantic metadata resulted in a substantial improvement, increasing the test ROC AUC from 0.67 to 0.80. The addition of embeddings improved performance further, with v03 reaching a test ROC AUC of 0.86 and an average precision of 0.63.

The embedding-based models achieved very high precision, approximately 0.87, but recall remained around 0.31. Consequently, the models were highly reliable when predicting that clarification was needed, but they detected only about one-third of all clarification cases at the default classification threshold.

---

## Key Findings

### 1. Structural Features Provide a Limited Baseline

Structural features alone provided some predictive signal, but their performance varied by target.

The structural baseline was weakest for predicting refusal and capability outcomes:

- Refusal: test ROC AUC of 0.67
- Capability: test ROC AUC of 0.72
- Needs clarification: test ROC AUC of 0.67

The results suggest that prompt length, question count, spelling errors, and instruction flags contain useful information, but they cannot adequately represent the meaning and context of a prompt.

For refusal and capability prediction, the low F1 scores also show that a model can achieve moderate ROC AUC while still performing poorly at the default classification threshold, particularly when the positive class is rare.

### 2. Semantic Metadata Provides a Meaningful Improvement

Adding task type, topic category, embedding novelty, and topic probability consistently improved model performance.

The largest improvements from v01 to v02 occurred for:

- refusal: test ROC AUC increased from 0.67 to 0.74
- capability: test ROC AUC increased from 0.72 to 0.79
- needs clarification: test ROC AUC increased from 0.67 to 0.80

This shows that task and topic information provide relevant context that is not captured by prompt structure alone.

The improvement was particularly strong for the clarification target, suggesting that the type and semantic context of a prompt are closely related to whether clarification may be required.

### 3. Embeddings Provide the Largest Overall Performance Gain

Dense embeddings produced the strongest improvement for all three targets.

Compared with v02, adding embeddings increased test ROC AUC to:

- 0.83 for refusal
- 0.83 for capability
- 0.86 for needs clarification

The largest gain occurred for refusal prediction, where test ROC AUC increased from 0.74 to 0.83. This indicates that latent semantic features capture important patterns associated with refusal behavior that are not represented by manually engineered features.

The comparison between v03 and v04 provides additional insight:

- For refusal, v04 slightly exceeded v03 in ROC AUC, but v03 had a higher F1 score and precision.
- For capability, v03 outperformed v04, indicating that structural and engineered semantic features add complementary information.
- For clarification, v03 and v04 performed almost identically.

Overall, embeddings appear to contain most of the predictive signal, but combining them with interpretable prompt features can still improve classification behavior and provide additional explanatory context.

### 4. The Models Favor Precision Over Recall

A consistent pattern across the Random Forest models is relatively high precision combined with modest recall.

This is especially visible in the strongest models:

- Refusal v03: precision 0.78, recall 0.30
- Capability v03: precision 0.81, recall 0.27
- Clarification v03: precision 0.87, recall 0.31

The models therefore tend to make conservative positive predictions. When they identify a refusal, capability-related outcome, or clarification case, the prediction is often correct. However, many positive cases remain undetected.

This behavior may be partly related to class imbalance and the use of the default probability threshold of 0.5. If the practical objective is to detect more positive cases, threshold optimization could increase recall at the expense of precision.

---

## Feature Importance Analysis

Random Forest feature importance and SHAP analysis were used to examine which variables contributed to the predictions.

The results indicate that predictive information came from several complementary sources:

| Feature Group | Relative Contribution |
|---|---|
| Dense embeddings | Strongest semantic signal in the expanded models |
| Prompt length | Important interpretable structural signal |
| Question count | Additional structural information |
| Topic category | Relevant contextual information |
| Task type | Useful semantic context |
| Embedding novelty | Additional information about semantic distance |
| Instruction flags | More limited contribution |

The dense embedding dimensions should not be interpreted individually as directly meaningful concepts. Their predictive value results from the combined representation of semantic patterns across all 384 dimensions.

For interpretability, it is therefore more appropriate to discuss embeddings as a feature group and use SHAP aggregation or grouped importance rather than assigning a concrete meaning to individual embedding coordinates.

Prompt length and question count remain useful because they provide an interpretable summary of prompt complexity. Nevertheless, the performance comparison shows that these structural features cannot replace semantic representations.

---

## Interpretation of Model Behavior

The Random Forest models predict response-related categories, not general answer quality.

A positive or negative prediction does not necessarily indicate:

- that the response was factually correct
- that the response fully completed the task
- that a refusal was appropriate
- that clarification would improve the final answer
- that the underlying prompt was objectively difficult

For example, a prompt may be classified as likely to require clarification because it is semantically ambiguous, but the model does not determine what clarification question should be asked.

Similarly, a predicted refusal-related outcome indicates an association with refusal behavior in the dataset. It does not establish that a refusal is necessary or desirable in that particular case.

The model results should therefore be interpreted as behavioral pattern prediction rather than causal explanation.

---

## Limitations

Several limitations should be considered:

1. **Target definition**

   The target variables depend on how refusal, capability, and clarification behavior were operationalized. Any labeling errors or ambiguity in these definitions will affect model performance.

2. **Class imbalance**

   The target classes are imbalanced, particularly for refusal and capability. Accuracy would therefore be misleading. ROC AUC, average precision, precision, recall, and F1 provide a more informative evaluation.

3. **Threshold dependence**

   F1, precision, and recall are calculated using the default classification threshold. The strong precision and relatively low recall suggest that threshold tuning could be useful depending on the intended application.

4. **Embedding interpretability**

   Individual embedding dimensions do not have a direct semantic interpretation. Their importance should be analyzed collectively, for example through grouped SHAP values, permutation importance, or dimensionality-reduction-based diagnostics.

5. **Potential dataset bias**

   The models may learn patterns related to the composition of the dataset, including the distribution of tasks, topics, prompt styles, and target labels. These associations should not automatically be interpreted as causal relationships.

6. **Single holdout split**

   Although the close agreement between cross-validation and test results supports generalization, the test evaluation is based on one holdout split. Additional repeated splits or external validation would provide stronger evidence of robustness.

7. **Target-specific data preparation**

   Embedding novelty was calculated using the training-set centroid separately for each target. This avoids using test information when constructing that feature, but it also means that the resulting feature is target-specific and should be reproduced carefully during deployment.

---

## Conclusion

The Random Forest analysis demonstrates that predicting response-related behavior requires semantic information beyond basic prompt structure.

Structural features provide a useful but limited baseline. Engineered semantic metadata produces a clear improvement, particularly for predicting whether clarification may be needed. Dense text embeddings deliver the largest overall performance gain, raising test ROC AUC to approximately:

- 0.83 for refusal
- 0.83 for capability
- 0.86 for needs clarification

The clarification target was the easiest to predict, while the capability target remained more difficult when evaluated using F1, precision, and recall.

The final models generally achieved high precision but relatively modest recall. This means that they made conservative predictions: positive predictions were often reliable, but many positive cases were missed. For applications where detecting as many cases as possible is more important than avoiding false positives, probability-threshold optimization should be considered.

Overall, the results support combining interpretable prompt features with dense embeddings. Embeddings capture the strongest latent semantic signal, while structural and engineered semantic variables add context that can improve performance and support model interpretation. As with the logistic regression analysis, these models predict response behavior rather than factual correctness, usefulness, or the appropriateness of the model’s response.