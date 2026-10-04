# AI-Based Spam Email Detection

Compares classical ML (TF-IDF word + char n-grams + hand-crafted features with
Naive Bayes / Logistic Regression / SVM / Random Forest) against a fine-tuned
DistilBERT, with error analysis, threshold tuning and a Streamlit demo.

## Run
```bash
pip install -r requirements.txt
# put a CSV with columns text,label in data/spam.csv  (e.g. Enron-Spam, SpamAssassin, Kaggle spam datasets)
python train_classical.py data/spam.csv     # ~minutes; writes models/ and results/
python train_bert.py data/spam.csv          # optional, GPU recommended (Colab)
streamlit run app.py
```

## What to put in the report
1. Problem and why false positives matter more than false negatives
2. Dataset: size, class balance, examples, cleaning steps
3. Features: TF-IDF words, char n-grams, meta features (URLs, caps, "!" etc.)
4. Models and 5-fold CV table (`results/cv_comparison.csv`)
5. Test results: confusion matrix, ROC/PR (`results/`)
6. Error analysis: read `results/misclassified.csv` and explain *why* they fail
7. Threshold trade-off (`results/threshold_sweep.csv`)
8. Explainability: top spam/ham words (`results/top_words.csv`) and the app's per-email view
9. Limitations (dataset age, adversarial spam, language) and future work

## Extension ideas
- Test on a *different* dataset than you trained on to show generalisation
- Add header features (sender domain, reply-to mismatch) and URL analysis
- Adversarial test: obfuscated words ("fr33", "c1ick") and see what breaks
- Deploy on Streamlit Community Cloud / Hugging Face Spaces
