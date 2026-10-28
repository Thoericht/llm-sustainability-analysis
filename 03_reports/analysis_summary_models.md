# Analysis summary: baseline models

## Objective

Predict three regex-based response patterns from prompt information: `target_refusal` (2.3% positive), `target_capability` (1.0%) and `target_needs_clarification` (14.5%), on 46,429 conversations. The models predict labelled response patterns, not answer quality, and show associations, not causal effects.

## Setup

- **Models:** Logistic Regression (L2, balanced class weights, standardised inputs) and Random Forest (500 trees, balanced class weights, min leaf size 2).
- **Feature sets:** v01 structural, v02 structural + task/topic/novelty, v03 v02 + 384-dimensional embeddings, v04 embeddings only.
- **Evaluation:** stratified 20% holdout split and stratified 5-fold cross-validation on the training part. Rows with the same prompt (19.3% of prompts occur more than once) always stay on the same side of a split. `embedding_novelty` uses training embeddings only.
- **Metrics:** ROC-AUC and Average Precision (AP). The AP of a random ranking equals the positive rate (0.023, 0.010, 0.145). F1, precision and recall at the default 0.5 threshold are stored in `results_log_reg.csv` and `results_random_forest.csv` but are not used for conclusions (see below).

## Results

CV = mean ROC-AUC over 5 folds, Test = ROC-AUC on the holdout, AP = test Average Precision.

| Target | Set | LR CV | LR Test | LR AP | RF CV | RF Test | RF AP |
|---|---|---:|---:|---:|---:|---:|---:|
| refusal | v01 | 0.556 | 0.562 | 0.027 | 0.528 | 0.560 | 0.026 |
| refusal | v02 | 0.578 | 0.571 | 0.029 | 0.622 | 0.647 | 0.057 |
| refusal | v03 | 0.694 | 0.704 | 0.053 | 0.706 | 0.713 | 0.071 |
| refusal | v04 | 0.693 | 0.705 | 0.054 | 0.702 | 0.701 | 0.066 |
| capability | v01 | 0.636 | 0.634 | 0.020 | 0.613 | 0.601 | 0.014 |
| capability | v02 | 0.644 | 0.607 | 0.016 | 0.655 | 0.650 | 0.054 |
| capability | v03 | 0.721 | 0.747 | 0.028 | 0.719 | 0.781 | 0.086 |
| capability | v04 | 0.719 | 0.759 | 0.030 | 0.716 | 0.730 | 0.054 |
| clarification | v01 | 0.576 | 0.580 | 0.194 | 0.563 | 0.573 | 0.211 |
| clarification | v02 | 0.677 | 0.684 | 0.251 | 0.699 | 0.696 | 0.306 |
| clarification | v03 | 0.761 | 0.778 | 0.434 | 0.771 | 0.775 | 0.446 |
| clarification | v04 | 0.758 | 0.773 | 0.431 | 0.765 | 0.770 | 0.439 |

## Findings

1. **Embeddings carry the predictive signal.** Adding them (v02 to v03) raises the cross-validated ROC-AUC by roughly 0.07 to 0.12 for every target and both models. v04 (embeddings only) is as good as v03: the cross-validated differences are at most 0.006 ROC-AUC, which is within noise.
2. **Engineered prompt features help only the non-linear model.** For the Random Forest, v02 clearly beats v01 (for example refusal 0.622 vs. 0.528 CV ROC-AUC); for Logistic Regression, v01 and v02 are almost identical for refusal and capability. Once embeddings are included, the engineered features add nothing measurable.
3. **Logistic Regression and Random Forest perform alike.** With v03, the cross-validated ROC-AUC differs by at most 0.012 between the two models. With a random split that kept duplicate prompts on both sides, the Random Forest had looked far better (test ROC-AUC about 0.83 vs. 0.75); this advantage disappears with the grouped split. It was most likely caused by the Random Forest recognising repeated prompts.
4. **Absolute performance is moderate.** Clarification is the most predictable target (AP about 0.43 to 0.45 against a base rate of 0.145). Refusal and capability are only weakly predictable (CV ROC-AUC around 0.70 to 0.72, AP of a few percent, about 2 to 3 times the base rate). The capability holdout has only about 100 positives: its test ROC-AUC (0.75 to 0.78) is higher than the cross-validated value (0.72), and the cross-validated value is more reliable.
5. **Threshold-based metrics are not meaningful here.** At the default threshold the Random Forest predicts almost no positives (recall 0.02 for refusal with v02 to v04; for capability v03 it predicts none), while Logistic Regression with balanced weights predicts many (recall around 0.5 to 0.7) with very low precision. Thresholds were not tuned, so the models are only useful for ranking.

## Interpretation of v03 (Logistic Regression)

v03 is used for interpretation because it is the only set with interpretable prompt features and its cross-validated scores are equal to those of v04.

- **Coefficients (clarification):** Summarisation (odds ratio 0.63) and explanation prompts (0.83) lower the odds of a clarification pattern; writing/generation (1.30) and roleplay (1.22) raise them; a role instruction raises them by a factor of 1.25. This matches the exploratory analysis. Individual embedding coefficients have no direct meaning.
- **SHAP:** The embedding block accounts for 96% to 97% of the summed mean absolute SHAP value for all three targets. This is largely a counting effect of 384 columns. Per column, the average embedding dimension (mean |SHAP| 0.06 to 0.20) is less important than the strongest single prompt features: prompt length (0.55 for refusal, 0.56 for capability), question count and role instruction for capability (0.32 and 0.29), and topic category for capability (0.17 on average over its 4 columns). For clarification, prompt length and question count (about 0.09 each) lie in the same range as the average embedding dimension (0.06). Summing absolute values per column also overstates the block effect.

## Interpretation of v03 (Random Forest)

SHAP values (TreeExplainer on probabilities, 2,000 test rows) give the same overall picture as for Logistic Regression. Shares of the summed mean absolute SHAP value:

| Feature | refusal | capability | clarification |
|---|---:|---:|---:|
| embeddings (384 columns) | 97.7% | 95.3% | 92.7% |
| log_first_prompt_tokens | 0.8% | 2.6% | 0.8% |
| embedding_novelty | 0.6% | 0.6% | 4.4% |
| task_type | 0.4% | 1.0% | 0.5% |
| has_role_instruction | 0.0% | 0.0% | 1.1% |

- The embedding block dominates again, with the same counting caveat (384 columns against 1 to 9 for the others). The shares are more informative than the absolute values, which are small because they are on the probability scale of rare targets.
- Among the other features, prompt length matters most for refusal and capability. For clarification, the distance of a prompt from the average prompt (`embedding_novelty`, 4.4%) and a role instruction (1.1%) stand out, which agrees with the role-instruction finding of the exploratory analysis.
- Topic category and topic probability contribute almost nothing, consistent with the weak topic structure.
- Logistic Regression and Random Forest rank the secondary features somewhat differently (for example topic category is visible in the Logistic Regression, not in the Random Forest). Differences in these small shares should not be interpreted.

## Limitations

- One holdout split per target; no confidence intervals or repeated cross-validation. Differences below about 0.01 to 0.03 ROC-AUC should not be interpreted.
- Near-duplicate prompts (paraphrases) can still end up on both sides of a split.
- The targets are regex-based and overlap (refusal and capability share phrases), and the cost target contains HTML markup (see README).
- Topic features are weak: 96% of prompts fall into two topic categories.
- No threshold tuning or probability calibration.
