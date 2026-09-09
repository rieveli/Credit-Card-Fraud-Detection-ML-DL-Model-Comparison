
-----------
# Credit Card Fraud Detection — ML/DL Model Comparison

An exploratory project comparing classical ML and deep learning approaches for detecting fraudulent credit card transactions on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 fraudulent — a ~578:1 class imbalance).

> **Status: exploratory / learning project**, originally built as a university coursework project to compare ML/DL techniques on an imbalanced classification problem — not a production-ready fraud system. A few real bugs affect the final conclusions — see "Known Issues" below before treating any result here as final.

## What's in here

- Baseline models: Logistic Regression, Random Forest
- Imbalance handling: class weights, SMOTE oversampling
- Gradient boosting: XGBoost, LightGBM
- Deep learning: a simple feedforward NN and a deeper MLP with batch norm, dropout, and early stopping
- Model interpretability: SHAP values on the XGBoost model
- Validation: held-out test set plus 3-fold and 5-fold stratified cross-validation
- Visualizations: confusion matrices, ROC curves, precision-recall curves, feature correlation heatmap, class distribution, PCA + K-Means clustering

## What you can learn from this

- **Why accuracy is a useless metric here.** With fraud at 0.17% of transactions, a model that predicts "not fraud" every time scores 99.83% accuracy while catching zero fraud. Every evaluation in this notebook uses precision, recall, F1, and ROC-AUC instead — that's the right instinct on any heavily imbalanced classification problem, not just fraud.
- **The precision/recall trade-off is not academic here — it's a business decision.** High recall with low precision (like this notebook's Logistic Regression run: 92% recall, 6% precision) means catching almost all fraud but burying a review team in false alarms — roughly 15 false flags for every real fraud case. High precision with lower recall means fewer false alarms but missed fraud. Which one is "better" depends entirely on what a false positive costs you versus what a missed fraud costs you — there's no metric that answers that for you.
- **Imbalance-handling techniques behave differently across model families.** `class_weight="balanced"` pushed Logistic Regression's recall up but tanked its precision. SMOTE (oversampling the minority class) gave Random Forest a more balanced trade-off. The same technique doesn't transfer predictably across algorithms — it has to be tuned per model.
- **Gradient-boosted trees (XGBoost, Random Forest) outperformed both linear models and the neural nets here** on the precision/recall balance that actually matters for fraud triage — a reasonable prior for structured/tabular data in general, though not a universal law.
- **Cross-validation at different fold counts is a real robustness check**, not just a formality — comparing 3-fold vs. 5-fold vs. held-out test metrics side by side is a good habit for catching a result that only looked good by luck on one split.
- **SHAP is worth having in your toolkit** for actually explaining *why* a model flagged something as fraud, which matters in a domain where "the model said so" isn't an acceptable answer to a customer or regulator.

## Known Issues (fix before trusting the conclusions)

1. **The "best model" pick is wrong.** The notebook selects a winner by recall alone, which crowns Logistic Regression (Recall = 0.918) despite its precision being only 0.061 — meaning roughly 15 out of 16 fraud alerts it raises are false alarms. XGBoost (Precision 0.89 / Recall 0.85, F1 ≈ 0.87) and Random Forest (Precision 0.96 / Recall 0.77) are the actually usable detectors in this comparison. **Fix:** select by F1-score or a cost-weighted metric that reflects the real cost of a false positive vs. a false negative, not recall alone.
2. **LightGBM is misconfigured, not a fair result.** The hundreds of "No further splits with positive gain" warnings and the resulting precision collapse (as low as 0.008) indicate `scale_pos_weight` is fighting LightGBM's leaf-wise growth on very few positive examples. **Fix:** tune `num_leaves`, `min_child_samples`, and `max_depth` down, or use `is_unbalance=True` instead of manually computed `scale_pos_weight`.
3. **KNN's ROC-AUC is hardcoded to 0**, not computed. The code passes `y_pred` (0s and 1s) instead of a probability score wherever KNN is evaluated. **Fix:** use `model.predict_proba(X)[:, 1]` for KNN like every other model — it does support it.

## Requirements

`numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `tensorflow`/`keras`, `matplotlib`, `seaborn`, `shap`, `imbalanced-learn` (for SMOTE), `joblib`.

## License

MIT — see [LICENSE](LICENSE). The dataset itself is subject to Kaggle's own terms; it isn't redistributed in this repo.

## Using this as a starting point

If you're adapting this for your own fraud/anomaly-detection project, the useful skeleton to keep is: stratified train/test split → try multiple model families → evaluate with precision/recall/F1/ROC-AUC (never accuracy alone) → cross-validate → explain with SHAP. Swap in the three fixes above before trusting any "best model" conclusion it produces.
