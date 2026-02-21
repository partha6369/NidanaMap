import subprocess
import sys

def install_if_missing(pip_package, import_name=None):
    import_name = import_name or pip_package
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", pip_package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
# Attempt to install 'icd10-cm' only if not present
install_if_missing("icd10-cm")
install_if_missing("rapidfuzz")
install_if_missing("icdcodex")
install_if_missing("pandas")
install_if_missing("gradio")
install_if_missing("nltk")

import gradio as gr
import pandas as pd
import icd10
import re
import os
from icdcodex import icd2vec, hierarchy
from rapidfuzz import process, fuzz
from datetime import datetime

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Ensure required data is downloaded (only needed once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# ========== Setup: Load ICD Hierarchy and Train Embeddings ==========
# icd_graph, icd_code_list = hierarchy.icd10cm()
# from icdcodex import hierarchy
icd_graph, icd_code_list = hierarchy.icd10cm("2025")  # or "2024", etc.

embedder = icd2vec.Icd2Vec(num_embedding_dimensions=64, workers=-1)
embedder.fit(icd_graph, icd_code_list)

# ========== Build ICD Reference Table with Descriptions ==========
records = []
for code in icd_code_list:
    try:
        obj = icd10.find(code)
        desc = obj.description if obj else ""
    except:
        desc = ""
    records.append({'code': code, 'description': desc})

icd_ref = pd.DataFrame(records)

# ========== Function to Find Matching ICD-10 Code ==========
def find_icd_with_embedding(diagnosis, ref_df, top_k=3):
    candidates = process.extract(
        diagnosis,
        ref_df['description'],
        scorer=fuzz.token_sort_ratio,
        limit=top_k
    )
    matches = []
    for desc, score, idx in candidates:
        code = ref_df.loc[idx, 'code']
        vec_diag = embedder.to_vec([code])[0]
        matches.append({
            'icd_code': code,
            'icd_description': desc,
            'text_score': round(score, 5),
            'justification': f"Matched '{desc}' ({score}%) based on fuzzy logic and ICD-10 vector proximity."
        })
    return matches

# Try to load the PayPal URL from the environment; if missing, use a placeholder
paypal_url = os.getenv("PAYPAL_URL", "https://www.paypal.com/donate/dummy-link")

# ========== Gradio Interface Logic ==========
def preprocess_input(text):
    """
    Preprocess the diagnosis text by:
    - Lowercasing
    - Removing special characters
    - Normalising whitespace
    - Lemmatising each word
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = nltk.word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)
    
def format_results(diagnosis, matches):
    output_md = f"### 📋 Results for: **{diagnosis}**\n\n"
    for match in matches:
        output_md += f"""
**ICD-10 Code:** `{match['icd_code']}`  
**Description:** {match['icd_description']}  
**Fuzzy Score:** `{match['text_score']:.5f}`  
**Justification:** {match['justification']}

---
"""
    return output_md

def process_diagnosis(input_text):
    if not input_text.strip():
        return "❌ Please enter a diagnosis to search."

    initial_input = input_text.strip()
    cleaned_input = preprocess_input(initial_input)
    matches = find_icd_with_embedding(cleaned_input, icd_ref, top_k=3)
    result = format_results(initial_input, matches)
    return result

def clear_input():
    return "", ""

def save_suggestion(suggestion):
    if not suggestion.strip():
        return "⚠️ Please enter a suggestion.", None

    # Save to a visible path
    today_str = datetime.now().strftime("%Y-%m-%d")
    base_dir = "/home/user/suggestions"
    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, f"suggestions_{today_str}.txt")

    # Append suggestion
    with open(filepath, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%H:%M:%S")
        f.write(f"[{timestamp}] {suggestion.strip()}\n")

    return "✅ Suggestion saved!", filepath

APP_TITLE = "🩺 NidanaMap: ICD-10 Diagnosis Mapper"
APP_DESCRIPTION = (
    "NidanaMap is an intelligent ICD-10 diagnosis mapper designed to bridge clinical expressions with their corresponding ICD-10 codes."
    "Whether you’re a medical practitioner, coder, researcher, or student, this tool simplifies the task of identifying, validating, and mapping diagnostic phrases to globally recognised classifications." 
    "Fast, reliable, and intuitive — NidanaMap helps streamline documentation, billing, and data analysis with clinical accuracy."
)
    
# ========== Build Gradio UI ==========
with gr.Blocks() as app:
    # Title and Description
    gr.HTML(
        f"""
        <p style='text-align: center; font-size: 40px; font-weight: bold;'>{APP_TITLE}</p>
        <p style='text-align: center; font-size: 20px; color: #555;'><sub>{APP_DESCRIPTION}</sub></p>
        <hr>
        """
    )
    
    with gr.Row():
        input_box = gr.Textbox(label="Enter a diagnosis", placeholder="e.g., Diabetes mellitus without complications")
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")
    with gr.Row():
        output_box = gr.Markdown(value="", show_copy_button=True)

    submit_btn.click(process_diagnosis, inputs=input_box, outputs=output_box)
    clear_btn.click(fn=clear_input, inputs=[], outputs=[input_box, output_box])

    with gr.Row():
        suggestion_input = gr.Textbox(label="💡 Suggest a Feature")
        suggestion_submit = gr.Button("Submit Suggestion")
    
    with gr.Row():
        suggestion_status = gr.Textbox(label="Status", interactive=False)
        suggestion_file = gr.File(label="📥 Download Today's Suggestions", visible=True)

    suggestion_submit.click(
        fn=save_suggestion,
        inputs=suggestion_input,
        outputs=[suggestion_status, suggestion_file]
    )
    
    with gr.Row():
        gr.HTML(f"""
        <a href="{paypal_url}" target="_blank">
            <button style="background-color:#0070ba;color:white;border:none;padding:10px 20px;
            font-size:16px;border-radius:5px;cursor:pointer;margin-top:10px;">
                ❤️ Support Research via PayPal
            </button>
        </a>
        """)

    with gr.Row():
        gr.HTML("""
        <a href="https://huggingface.co/spaces/partha6369/partha-research-centre"
           target="_blank"
           rel="noopener noreferrer">
            <button style="
                background-color:#111827;
                color:white;
                border:none;
                padding:10px 20px;
                font-size:16px;
                border-radius:8px;
                cursor:pointer;
                margin-top:10px;">
                🔗 Dr Partha Majumdar's Research Centre
            </button>
        </a>
        """)

if __name__ == "__main__":
    # Determine if running on Hugging Face Spaces
    on_spaces = os.environ.get("SPACE_ID") is not None
    
    # Launch the app conditionally
    app.launch(share=not on_spaces)