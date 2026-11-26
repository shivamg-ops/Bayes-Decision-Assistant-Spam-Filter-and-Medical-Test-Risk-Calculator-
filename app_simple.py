"""
Bayes Decision Assistant - Simple & Clean Streamlit App
MCA Semester I Mini Project: Mathematical Foundations with Python (Roll No. 26 to 31)

Features:
1. SMS Spam Detection (Scratch Naive Bayes with Laplace Smoothing)
2. Medical Test Risk Calculator (Bayes' Theorem & Base Rate Neglect)
3. Natural Frequency Table (Cohort of 1,000)
4. Monte Carlo Simulation (100,000 Individuals with NumPy)
"""

import math
import re
import string
from collections import defaultdict
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CORE LOGIC: TEXT CLEANING & NAIVE BAYES (FROM SCRATCH) ---
def tokenize(text):
    """Lowercase, remove punctuation/numbers, and extract words."""
    text = text.lower()
    text = re.sub(r'https?://\S+|\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [w for w in text.split() if len(w) > 1]

class SimpleNaiveBayes:
    """Multinomial Naive Bayes Classifier with Laplace Smoothing (No scikit-learn)."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.priors = {}
        self.word_counts = {'ham': defaultdict(int), 'spam': defaultdict(int)}
        self.total_words = {'ham': 0, 'spam': 0}
        self.vocab = set()

    def fit(self, messages, labels):
        total = len(messages)
        for msg, lbl in zip(messages, labels):
            tokens = tokenize(msg)
            for t in tokens:
                self.word_counts[lbl][t] += 1
                self.total_words[lbl] += 1
                self.vocab.add(t)
        
        # Prior Probabilities P(Class)
        spam_count = labels.count('spam')
        self.priors['spam'] = spam_count / total
        self.priors['ham'] = 1.0 - self.priors['spam']
        self.vocab_size = len(self.vocab)

    def predict_proba(self, text):
        """Calculates log-likelihoods and converts to normalized percentage probabilities."""
        tokens = tokenize(text)
        log_probs = {}
        for c in ['ham', 'spam']:
            log_p = math.log(self.priors[c])
            den = self.total_words[c] + (self.alpha * self.vocab_size)
            for t in tokens:
                num = self.word_counts[c].get(t, 0) + self.alpha
                log_p += math.log(num / den)
            log_probs[c] = log_p
        
        # Softmax normalization to avoid underflow
        max_log = max(log_probs.values())
        exp_ham = math.exp(log_probs['ham'] - max_log)
        exp_spam = math.exp(log_probs['spam'] - max_log)
        total_exp = exp_ham + exp_spam
        return {'ham': (exp_ham / total_exp) * 100, 'spam': (exp_spam / total_exp) * 100}

# --- 2. LOAD DATASET & TRAIN MODEL (CACHED) ---
@st.cache_resource
def get_trained_model():
    path = "data/sms_spam_cleaned.csv"
    df = pd.read_csv(path)
    # 80-20 Train-Test Split
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8
    train_df, test_df = df[mask], df[~mask]
    
    nb = SimpleNaiveBayes(alpha=1.0)
    nb.fit(train_df['message'].tolist(), train_df['label'].tolist())
    
    # Calculate test accuracy
    correct = sum(1 for _, row in test_df.iterrows() 
                  if (nb.predict_proba(row['message'])['spam'] >= 50) == (row['label'] == 'spam'))
    accuracy = (correct / len(test_df)) * 100
    return nb, df, accuracy

model, df, test_acc = get_trained_model()

# --- 3. STREAMLIT APP UI ---
st.set_page_config(page_title="Bayes Decision Assistant", page_icon="⚖️", layout="wide")
st.title("⚖️ Bayes Decision Assistant")
st.caption("MCA Sem I Mini Project (Roll No. 26 to 31) | Pure NumPy & Python (No Sklearn)")

tabs = st.tabs([
    "📩 1. SMS Spam Filter", 
    "🩺 2. Medical Risk Calculator", 
    "👥 3. Natural Frequency Table", 
    "🎲 4. Monte Carlo Simulation"
])

# ================= TAB 1: SMS SPAM FILTER =================
with tabs[0]:
    st.subheader("Multinomial Naive Bayes SMS Classifier")
    st.info(f"Model trained on **{len(df):,} messages**. Test Accuracy: **{test_acc:.2f}%**")
    
    sample_text = st.selectbox(
        "Choose a sample message or write your own below:",
        [
            "Custom Message",
            "Congratulations! You won a $1,000 Walmart gift card. Call now to claim!",
            "Hey bro, are we meeting at the college library at 4 PM today?",
            "Urgent: You won 2,000,000 cash prize in lottery. Text WIN to 8080."
        ]
    )
    user_msg = st.text_area("Enter SMS text:", value="" if sample_text == "Custom Message" else sample_text)
    
    if st.button("Classify SMS", type="primary"):
        if not user_msg.strip():
            st.warning("Please type a message first.")
        else:
            probs = model.predict_proba(user_msg)
            is_spam = probs['spam'] >= 50.0
            
            if is_spam:
                st.error(f"🚨 **SPAM DETECTED** (Probability: {probs['spam']:.2f}%)")
            else:
                st.success(f"✅ **LEGITIMATE (HAM)** (Probability: {probs['ham']:.2f}%)")
            
            c1, c2 = st.columns(2)
            c1.metric("P(Spam | Message)", f"{probs['spam']:.2f}%")
            c2.metric("P(Ham | Message)", f"{probs['ham']:.2f}%")

# ================= TAB 2: MEDICAL TEST CALCULATOR =================
with tabs[1]:
    st.subheader("Medical Test Risk Calculator (Bayes' Theorem)")
    st.write("Understand **Base Rate Neglect**: Why accurate tests can still mean low probability if disease is rare.")
    
    col1, col2 = st.columns(2)
    with col1:
        sens = st.slider("Sensitivity / True Positive Rate P(+|D) (%)", 50.0, 100.0, 95.0, 0.5)
        spec = st.slider("Specificity / True Negative Rate P(-|~D) (%)", 50.0, 100.0, 90.0, 0.5)
        prev = st.slider("Disease Prevalence P(D) (%)", 0.1, 20.0, 1.0, 0.1)
    
    # Pure Bayes Calculation
    p_d = prev / 100.0
    p_pos_d = sens / 100.0
    p_pos_not_d = (100.0 - spec) / 100.0
    
    # Law of Total Probability
    p_pos = (p_pos_d * p_d) + (p_pos_not_d * (1.0 - p_d))
    ppv = (p_pos_d * p_d) / p_pos * 100.0
    
    with col2:
        st.metric("P(Disease | Positive Test)", f"{ppv:.2f}%")
        st.warning(f"**Interpretation:** Even with {sens}% accuracy, someone who tests positive only has a **{ppv:.2f}%** chance of actually being sick because the disease is rare! **{100-ppv:.2f}%** are false alarms.")
        
        with st.expander("View Bayes Formula"):
            st.latex(r"P(D \mid +) = \frac{\text{Sensitivity} \times \text{Prevalence}}{(\text{Sensitivity} \times \text{Prevalence}) + ((1 - \text{Specificity}) \times (1 - \text{Prevalence}))}")

# ================= TAB 3: NATURAL FREQUENCY TABLE =================
with tabs[2]:
    st.subheader("Natural Frequency Table (Cohort of 1,000 People)")
    st.write("Explaining the Bayes result in simple plain words:")
    
    pop = 1000
    sick = pop * p_d
    healthy = pop * (1.0 - p_d)
    tp = sick * p_pos_d
    fp = healthy * p_pos_not_d
    fn = sick * (1.0 - p_pos_d)
    tn = healthy * (spec / 100.0)
    
    table_data = pd.DataFrame({
        "Category": ["Test Positive (+)", "Test Negative (-)", "Total"],
        "Sick (Disease)": [f"{tp:.1f} (True +)", f"{fn:.1f} (False -)", f"{sick:.1f}"],
        "Healthy (No Disease)": [f"{fp:.1f} (False +)", f"{tn:.1f} (True -)", f"{healthy:.1f}"],
        "Total Tests": [f"{tp + fp:.1f}", f"{fn + tn:.1f}", f"{pop}"]
    })
    st.table(table_data)
    st.info(f"👉 Out of **{tp+fp:.1f} people who test positive**, only **{tp:.1f}** are truly sick. Therefore: {tp:.1f} / {tp+fp:.1f} = **{ppv:.2f}%**.")

# ================= TAB 4: MONTE CARLO SIMULATION =================
with tabs[3]:
    st.subheader("Monte Carlo Simulation (100,000 Virtual People)")
    st.write("Using NumPy's random module to empirically verify the theoretical Bayes probability.")
    
    if st.button("Run Simulation (N = 100,000)", type="primary"):
        np.random.seed(42)
        N = 100000
        # 1. Assign disease based on prevalence
        has_disease = np.random.rand(N) < p_d
        # 2. Simulate test outcomes
        test_pos = np.zeros(N, dtype=bool)
        test_pos[has_disease] = np.random.rand(np.sum(has_disease)) < p_pos_d
        test_pos[~has_disease] = np.random.rand(np.sum(~has_disease)) < p_pos_not_d
        
        sim_tp = np.sum(has_disease & test_pos)
        sim_fp = np.sum((~has_disease) & test_pos)
        sim_ppv = (sim_tp / (sim_tp + sim_fp)) * 100.0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Theoretical Bayes PPV", f"{ppv:.2f}%")
        c2.metric("Monte Carlo Simulated PPV", f"{sim_ppv:.2f}%")
        c3.metric("Difference", f"{abs(sim_ppv - ppv):.4f}%")
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["Theoretical Bayes", "Monte Carlo (N=100k)"], [ppv, sim_ppv], color=['#2563EB', '#DC2626'], width=0.4)
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, max(ppv, sim_ppv) * 1.3)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + 0.12, p.get_height() + 0.3), fontweight='bold')
        st.pyplot(fig)
        st.success("The simulation matches the theoretical Bayes calculation with almost zero error!")

