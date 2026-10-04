"""Classical ML spam detector: compares 4 models with cross-validation,
evaluates the winner on a held-out test set, saves plots, errors and models.

Usage: python train_classical.py data.csv      (CSV needs columns: text,label)
label can be spam/ham or 1/0.
"""
import os, re, sys, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                             RocCurveDisplay, PrecisionRecallDisplay)

os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)

def meta_features(texts):
    """Hand-crafted signals spammers leave behind, all scaled to 0-1."""
    rows = []
    for t in texts:
        t = str(t); n = max(len(t), 1)
        rows.append([
            min(len(t) / 2000, 1),
            min(len(re.findall(r"http\S+|www\.\S+", t)) / 5, 1),
            min(t.count("!") / 5, 1),
            sum(c.isupper() for c in t) / n,
            sum(c.isdigit() for c in t) / n,
            float(bool(re.search(r"[$£€]|free|win|prize|urgent|click", t, re.I))),
        ])
    return np.array(rows)

def features():
    return make_union(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True),
        FunctionTransformer(meta_features),
    )

MODELS = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=2000, C=10, class_weight="balanced"),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=42),
}

def main(path):
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    y = df["label"].astype(str).str.lower().isin(["spam", "1"]).astype(int).values
    X = df["text"].astype(str).to_numpy(dtype=object)
    print(f"{len(df)} emails, {y.mean():.1%} spam")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    rows = []
    for name, clf in MODELS.items():
        pipe = make_pipeline(features(), clf)
        cv = cross_validate(pipe, X_tr, y_tr, cv=skf, scoring=["precision", "recall", "f1"], n_jobs=1)
        rows.append({"model": name, **{m: cv[f"test_{m}"].mean() for m in ["precision", "recall", "f1"]},
                     "f1_std": cv["test_f1"].std()})
        print(rows[-1])
    table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    table.to_csv("results/cv_comparison.csv", index=False)
    print("\n", table.round(4).to_string(index=False))

    best = table.iloc[0]["model"]
    print(f"\nBest model: {best}")
    pipe = make_pipeline(features(), MODELS[best]).fit(X_tr, y_tr)
    pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    print(classification_report(y_te, pred, target_names=["ham", "spam"], digits=4))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["ham", "spam"], cmap="Blues")
    plt.title(f"{best} - confusion matrix"); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_te, proba, ax=ax[0]); PrecisionRecallDisplay.from_predictions(y_te, proba, ax=ax[1])
    plt.tight_layout(); plt.savefig("results/roc_pr.png", dpi=150); plt.close()

    errs = pd.DataFrame({"text": X_te, "true": y_te, "pred": pred, "p_spam": proba})
    errs[errs.true != errs.pred].to_csv("results/misclassified.csv", index=False)

    # threshold sweep: false positives (ham -> spam) are the costly mistake
    sweep = []
    for th in np.arange(0.1, 0.95, 0.05):
        p = (proba >= th).astype(int)
        sweep.append({"threshold": round(th, 2), "false_pos": int(((p == 1) & (y_te == 0)).sum()),
                      "false_neg": int(((p == 0) & (y_te == 1)).sum())})
    pd.DataFrame(sweep).to_csv("results/threshold_sweep.csv", index=False)

    joblib.dump(pipe, "models/best_model.joblib")
    # simple word-level LR kept for explainability in the app
    explain = make_pipeline(TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2),
                            LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X_tr, y_tr)
    joblib.dump(explain, "models/explain_model.joblib")
    coefs = explain[-1].coef_[0]; names = explain[0].get_feature_names_out()
    top = pd.DataFrame({"word": names, "weight": coefs}).sort_values("weight")
    pd.concat([top.head(15), top.tail(15)]).to_csv("results/top_words.csv", index=False)
    print("Saved models/ and results/")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/spam.csv")
