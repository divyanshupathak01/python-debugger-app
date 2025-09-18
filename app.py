# app.py

import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from bs4 import BeautifulSoup

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Debugging Helper",
    page_icon="🤖",
    layout="wide"
)

# --- Model and Data Loading ---
# This decorator caches the models and data, so they only load once when the app starts.
# This is crucial for performance.
@st.cache_resource
def load_assets():
    """Loads all necessary models, indexes, and dataframes."""
    print("Loading assets for the first time...")
    # Load the Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load the FAISS index
    index = faiss.read_index("faiss_index.bin")
    
    # Load the dataframes
    questions_df = pd.read_csv('processed_questions.csv')
    answers_df = pd.read_csv('processed_answers.csv')
    
    print("Assets loaded successfully.")
    return model, index, questions_df, answers_df

model, index, df_questions, df_answers = load_assets()


# --- Core Functions ---
def find_similar_questions(query, top_k=10):
    """Encodes a query and searches the FAISS index for top_k similar questions."""
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    
    results = df_questions.iloc[indices[0]].copy()
    results['SimilarityDistance'] = distances[0] # Lower distance is better
    return results

def get_suggested_solution(question_id):
    """Retrieves the code snippets from the best answer for a given question ID."""
    try:
        accepted_answer_id = df_questions.loc[df_questions['QuestionId'] == question_id, 'AcceptedAnswerId'].iloc[0]
        answer_body = df_answers.loc[df_answers['AnswerId'] == accepted_answer_id, 'AnswerBody'].iloc[0]
        soup = BeautifulSoup(answer_body, 'html.parser')
        code_blocks = [code.get_text() for code in soup.find_all('code')]
        return code_blocks if code_blocks else ["No code blocks found in the best answer."]
    except (IndexError, KeyError):
        return ["Could not find a corresponding answer for this question."]

# --- User Interface ---
st.title("🤖 AI-Powered Python Debugging Helper")
st.markdown("This tool uses **Semantic Search** to find the most relevant Stack Overflow posts for your Python errors. It's powered by Sentence-Transformers and FAISS for fast, context-aware results.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    user_code = st.text_area("Paste your Python code here:", height=300, placeholder="import pandas as pd\n...")
with col2:
    user_error = st.text_area("Paste the full error message here:", height=300, placeholder="KeyError: 'column_name'")

if st.button("Find Solution", type="primary", use_container_width=True):
    if user_code and user_error:
        with st.spinner("🧠 Analyzing your query and searching for the best solutions..."):
            user_query = user_code + "\n" + user_error
            top_matches = find_similar_questions(user_query)
            
            st.markdown("---")
            st.subheader("🏆 Top Results")
            
            # Display the top match as the main suggestion
            top_question = top_matches.iloc[0]
            st.success(f"**Best Match:** [{top_question['Title']}](https://stackoverflow.com/q/{top_question['QuestionId']})")
            
            with st.expander("💡 **View Suggested Solution Code**", expanded=True):
                solution_code = get_suggested_solution(top_question['QuestionId'])
                if solution_code and "No code blocks found" not in solution_code[0]:
                    for i, snippet in enumerate(solution_code):
                        st.code(snippet, language='python', line_numbers=True)
                else:
                    st.warning("No code snippets were found in the top-ranked answer.")
            
            # Display other similar questions
            with st.expander("➕ **View Other Similar Questions**"):
                for i, row in top_matches.iloc[1:].iterrows():
                    st.markdown(f"* [{row['Title']}](https://stackoverflow.com/q/{row['QuestionId']})")

    else:
        st.error("Please enter both your code and the error message to get a solution.")