
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Clinical Coding AI Demo",
    layout="wide"
)

st.title("🧠 Clinical Coding AI Demo Platform")
st.markdown("Hybrid Rule-based + Fuzzy + AI Semantic Coding Engine")

# --------------------------------------------------
# MOCK DICTIONARIES
# --------------------------------------------------

mock_ae_dictionary = [
    {"pt": "Headache", "soc": "Nervous system disorders"},
    {"pt": "Migraine", "soc": "Nervous system disorders"},
    {"pt": "Chest pain", "soc": "Cardiac disorders"},
    {"pt": "Myocardial infarction", "soc": "Cardiac disorders"},
    {"pt": "Nausea", "soc": "Gastrointestinal disorders"},
    {"pt": "Vomiting", "soc": "Gastrointestinal disorders"},
]

mock_cm_dictionary = [
    {"trade": "Crocin", "substance": "Paracetamol", "atc": "N02BE01"},
    {"trade": "Augmentin", "substance": "Amoxicillin + Clavulanic acid", "atc": "J01CR02"},
    {"trade": "Insulin", "substance": "Insulin human", "atc": "A10AB01"},
    {"trade": "Metformin", "substance": "Metformin", "atc": "A10BA02"},
]

MEMORY_FILE = "feedback_memory.json"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_data
def compute_embeddings():
    pt_terms = [entry["pt"] for entry in mock_ae_dictionary]
    embeddings = model.encode(pt_terms, convert_to_tensor=True)
    return pt_terms, embeddings

pt_terms, pt_embeddings = compute_embeddings()

# --------------------------------------------------
# MEMORY FUNCTIONS
# --------------------------------------------------

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f)

# --------------------------------------------------
# AE HYBRID CODING
# --------------------------------------------------

def hybrid_ae_code(text):

    memory = load_memory()
    text_lower = text.lower()

    if text_lower in memory:
        pt = memory[text_lower]
        soc = next(item["soc"] for item in mock_ae_dictionary if item["pt"] == pt)
        return pt, soc, 0.99

    if "heart attack" in text_lower:
        return "Myocardial infarction", "Cardiac disorders", 0.95

    fuzzy_scores = [
        (entry["pt"], fuzz.token_sort_ratio(text, entry["pt"]) / 100)
        for entry in mock_ae_dictionary
    ]
    fuzzy_pt, fuzzy_score = max(fuzzy_scores, key=lambda x: x[1])

    query_embedding = model.encode(text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, pt_embeddings)[0]
    best_idx = int(np.argmax(cos_scores))
    ai_pt = pt_terms[best_idx]
    ai_score = float(cos_scores[best_idx])

    combined = {}
    combined[fuzzy_pt] = fuzzy_score
    combined[ai_pt] = combined.get(ai_pt, 0) + ai_score

    best_pt = max(combined, key=combined.get)
    soc = next(item["soc"] for item in mock_ae_dictionary if item["pt"] == best_pt)
    confidence = round(combined[best_pt] / 2, 2)

    return best_pt, soc, confidence

# --------------------------------------------------
# CM CODING
# --------------------------------------------------

def hybrid_cm_code(text):
    scores = [
        (entry, fuzz.token_sort_ratio(text, entry["trade"]) / 100)
        for entry in mock_cm_dictionary
    ]
    best_entry, score = max(scores, key=lambda x: x[1])
    return best_entry["trade"], best_entry["substance"], best_entry["atc"], round(score, 2)

# --------------------------------------------------
# SIDEBAR MENU
# --------------------------------------------------

menu = st.sidebar.selectbox(
    "Select Module",
    ["AE Coding", "CM Coding", "Batch Upload", "Dashboard"]
)

# --------------------------------------------------
# AE CODING UI
# --------------------------------------------------

if menu == "AE Coding":

    st.header("Adverse Event Coding")

    user_input = st.text_input("Enter AE verbatim")

    if user_input:
        pt, soc, conf = hybrid_ae_code(user_input)

        st.subheader("AI Suggestion")
        st.write("PT:", pt)
        st.write("SOC:", soc)
        st.write("Confidence:", conf)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Approve"):
                st.success("Approved")

        with col2:
            correct_pt = st.selectbox("Select Correct PT (if disapprove)", pt_terms)
            if st.button("Save Correction"):
                memory = load_memory()
                memory[user_input.lower()] = correct_pt
                save_memory(memory)
                st.success("Feedback stored successfully")

# --------------------------------------------------
# CM CODING UI
# --------------------------------------------------

elif menu == "CM Coding":

    st.header("Concomitant Medication Coding")

    user_input = st.text_input("Enter Drug Name")

    if user_input:
        trade, substance, atc, conf = hybrid_cm_code(user_input)

        st.subheader("AI Suggestion")
        st.write("Trade:", trade)
        st.write("Substance:", substance)
        st.write("ATC Code:", atc)
        st.write("Confidence:", conf)

# --------------------------------------------------
# BATCH UPLOAD UI
# --------------------------------------------------

elif menu == "Batch Upload":

    st.header("Batch AE Coding")

    uploaded_file = st.file_uploader(
        "Upload Excel File (Columns: Verbatim, Human_PT)",
        type=["xlsx"]
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        results = []

        for _, row in df.iterrows():
            pt, soc, conf = hybrid_ae_code(row["Verbatim"])
            match = pt == row["Human_PT"]

            results.append({
                "Verbatim": row["Verbatim"],
                "Human_PT": row["Human_PT"],
                "AI_PT": pt,
                "Confidence": conf,
                "Match": match
            })

        result_df = pd.DataFrame(results)

        accuracy = result_df["Match"].mean()

        st.subheader(f"Accuracy: {round(accuracy * 100, 2)}%")
        st.dataframe(result_df)

        result_df.to_excel("batch_output.xlsx", index=False)

# --------------------------------------------------
# DASHBOARD UI
# --------------------------------------------------

elif menu == "Dashboard":

    st.header("Performance Dashboard")

    if os.path.exists("batch_output.xlsx"):
        df = pd.read_excel("batch_output.xlsx")
        accuracy = df["Match"].mean()

        st.metric("Overall Accuracy", f"{round(accuracy * 100, 2)}%")

        fig1, ax1 = plt.subplots()
        df["Match"].value_counts().plot(kind="bar", ax=ax1)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        df["Confidence"].hist(bins=10, ax=ax2)
        st.pyplot(fig2)

    else:
        st.warning("Run Batch Upload first.")
