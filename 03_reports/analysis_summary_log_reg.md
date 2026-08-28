# Analysis Summary: Logistic Regression

## Objective

The objective was to model three response-behavior outcomes: `target_refusal`, `target_capability`, and `target_needs_clarification`. The models predict labeled response behavior, not factual correctness, usefulness, or overall answer quality.

## Modeling Approach

Four feature configurations were evaluated:

- **v01:** Structural prompt features, including transformed prompt length, question count, orthographic error rate, and instruction flags.
- **v02:** v01 plus semantic metadata: `task_type`, `topic_cat`, `embedding_novelty`, and `topic_prob`.
- **v03:** v02 plus 384 dense embedding dimensions.
- **v04:** Dense embeddings only.

All models used logistic regression with standardized numeric and embedding features, one-hot encoded categorical variables, class weighting, stratified 5-fold cross-validation, and a stratified holdout test set.

## Model Performance

The v03 specification provides the strongest overall performance for all three targets:

| Target | Test ROC-AUC | Test Average Precision | Test F1 | Test Precision | Test Recall |
|---|---:|---:|---:|---:|---:|
| `target_refusal` | 0.753 | 0.075 | 0.093 | 0.050 | 0.614 |
| `target_capability` | 0.755 | 0.040 | 0.060 | 0.032 | 0.536 |
| `target_needs_clarification` | 0.768 | 0.386 | 0.408 | 0.290 | 0.686 |

`target_needs_clarification` is the most predictable outcome. The low precision for the other two targets reflects severe class imbalance and the default classification threshold. The models should therefore be used primarily for ranking and prioritization unless the threshold is tuned for a specific application.

## SHAP Feature Importance

SHAP analysis was performed for the v03 models and aggregated to original feature names. The results show a consistent pattern:

- `embedding` is the dominant feature, accounting for approximately 97.5% of total absolute SHAP importance. This combines all 384 embedding dimensions.
- `log_first_prompt_tokens` is the strongest non-embedding feature.
- `task_type` and `topic_cat` provide meaningful additional context.
- `log_question_count`, `topic_prob`, `embedding_novelty`, and `orthographic_error_rate` make smaller contributions.
- `has_role_instruction`, `has_format_instruction`, and `has_audience_or_level_instruction` have comparatively limited global importance.

The embedding result should be interpreted as block-level importance: it shows that the combined semantic representation is highly influential, not that every embedding dimension is equally important or directly interpretable. Categorical features aggregate the contributions of their one-hot encoded levels.

## Interpretation

The results indicate that response behavior depends primarily on semantic content, with prompt length, task type, and topic category adding complementary information. Structural features alone provide a limited baseline, while adding semantic metadata improves performance and adding embeddings produces the largest gain.

The models predict whether a labeled response exhibits a particular behavior; they do not establish causal relationships. A predicted refusal-free response may still be incorrect or low quality, and a predicted positive case should not be treated as a definitive classification without considering the precision–recall trade-off.

## Limitations

- The targets depend on existing labels and may not capture every subtle or indirect response behavior.
- Strong class imbalance makes accuracy unsuitable as the main metric.
- Aggregated embedding importance is influenced by the 384-dimensional representation.
- Task and topic variables may reflect dataset composition rather than causal effects.
- Threshold tuning, probability calibration, and repeated or nested cross-validation would improve the reliability of deployment-oriented conclusions.

## Conclusion

The v03 logistic-regression models are the preferred configurations because they combine the strongest predictive performance with a transparent linear model structure. SHAP analysis shows that embeddings provide the main predictive signal, while prompt length, task type, topic category, and other engineered features contribute complementary information. The models are best interpreted as tools for ranking and analyzing response behavior rather than as definitive measures of answer quality.