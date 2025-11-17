import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (Dark Theme Friendly)
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #4A90E2; /* Brighter blue for dark theme */
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #262730; /* Darker background */
        color: #FAFAFA; /* Light text color */
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4A90E2; /* Brighter blue border */
        margin: 1rem 0;
    }
    .source-box {
        background-color: #1E1E1E; /* Slightly different dark background */
        color: #D4D4D4; /* Light grey text color */
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #3A3A3A; /* Subtle border */
        margin: 0.5rem 0;
    }
    /* Ensure Streamlit's default dark theme text color is overridden in our boxes */
    .answer-box p, .source-box p {
        color: #FAFAFA !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">🤖 RAG Q&A System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API URL input
    api_url = "https://rag-api-service-ajbermudezh22.onrender.com"
    st.markdown(f"✅ Connected to API: `{api_url}`")
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This is a **Retrieval-Augmented Generation (RAG)** system.
    
    Ask questions about your documents, and the AI will:
    1. Search through your knowledge base
    2. Find relevant information
    3. Generate an answer based on the retrieved context
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 API Status")
    
    # Check API health
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ API is running")
        else:
            st.error("❌ API returned an error")
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to API")
        st.info(f"Make sure your API is running at: {api_url}")

# Main content area
st.markdown("### 💬 Ask a Question")

# Text input for question
question = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="e.g., What is this document about?",
    help="Type your question about the documents in the knowledge base"
)

# Submit button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    submit_button = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

# Clear functionality
if clear_button:
    st.rerun()

# Process question when submitted
if submit_button and question.strip():
    with st.spinner("🔍 Searching knowledge base and generating answer..."):
        try:
            # Make API request
            response = requests.post(
                f"{api_url}/ask",
                json={"question": question},
                timeout=60  # Longer timeout for LLM processing
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display answer
                st.markdown("---")
                st.markdown("### ✅ Answer")
                st.markdown(f'<div class="answer-box">{data["answer"]}</div>', unsafe_allow_html=True)
                
                # Display sources
                st.markdown("### 📚 Sources")
                st.markdown("These are the document chunks used to generate the answer:")
                
                for i, source in enumerate(data["sources"], 1):
                    with st.expander(f"Source {i}: {source['source']}"):
                        st.markdown(f'<div class="source-box">{source["content_preview"]}</div>', unsafe_allow_html=True)
                
            else:
                st.error(f"❌ Error: {response.status_code}")
                st.json(response.json() if response.content else {"error": "Unknown error"})
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to connect to API: {str(e)}")
            st.info("Make sure your FastAPI server is running.")

elif submit_button and not question.strip():
    st.warning("⚠️ Please enter a question.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 2rem;'>"
    "Built with Streamlit, FastAPI, LangChain, and OpenAI"
    "</div>",
    unsafe_allow_html=True
)