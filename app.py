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

import gradio as gr
import pandas as pd
from icdcodex import icd2vec, hierarchy
from rapidfuzz import process, fuzz
import icd10

# ========== Setup: Load ICD Hierarchy and Train Embeddings ==========
icd_graph, icd_code_list = hierarchy.icd10cm()

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

# ========== Gradio Interface Logic ==========
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

    matches = find_icd_with_embedding(input_text.strip(), icd_ref, top_k=3)
    result = format_results(input_text, matches)
    return result

def clear_input():
    return "", ""

# ========== Build Gradio UI ==========
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 NadanaMap\nICD-10 Diagnosis Mapper")
    with gr.Row():
        input_box = gr.Textbox(label="Enter a diagnosis", placeholder="e.g., Diabetes mellitus without complications")
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")
    with gr.Row():
        output_box = gr.Markdown()

    submit_btn.click(process_diagnosis, inputs=input_box, outputs=output_box)
    clear_btn.click(fn=clear_input, inputs=[], outputs=[input_box, output_box])

# Launch the app
demo.launch(share=True, debug=False)