"""Streamlit demo: streamlit run app.py"""
import os, joblib, pandas as pd, streamlit as st
from train_classical import meta_features  # needed to unpickle the pipeline

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")
st.title("📧 AI Spam Email Detector")

@st.cache_resource
def load():
    best = joblib.load("models/best_model.joblib")
    explain = joblib.load("models/explain_model.joblib")
    bert = None
    if os.path.isdir("models/bert"):
        from transformers import pipeline
        bert = pipeline("text-classification", model="models/bert", truncation=True, max_length=256)
    return best, explain, bert

best, explain, bert = load()
threshold = st.sidebar.slider("Spam threshold", 0.1, 0.95, 0.5, 0.05,
                              help="Raise it to avoid flagging real emails (fewer false positives).")
text = st.text_area("Paste an email (subject + body):", height=220)

if st.button("Analyze", type="primary") and text.strip():
    p = best.predict_proba([text])[0, 1]
    c1, c2 = st.columns(2)
    c1.metric("Classical model - spam probability", f"{p:.1%}")
    _ = c1.error("🚨 SPAM") if p >= threshold else c1.success("✅ Not spam")
    if bert:
        r = bert(text)[0]; ps = r["score"] if r["label"] == "spam" else 1 - r["score"]
        c2.metric("DistilBERT - spam probability", f"{ps:.1%}")
        _ = c2.error("🚨 SPAM") if ps >= threshold else c2.success("✅ Not spam")

    vec, clf = explain[0], explain[-1]
    X = vec.transform([text]); names = vec.get_feature_names_out()
    contrib = {names[i]: X[0, i] * clf.coef_[0][i] for i in X.nonzero()[1]}
    if contrib:
        s = pd.Series(contrib).sort_values()
        st.subheader("Why? Words pushing the decision")
        st.caption("Positive = towards spam, negative = towards legitimate")
        st.bar_chart(pd.concat([s.head(6), s.tail(6)]).drop_duplicates())
