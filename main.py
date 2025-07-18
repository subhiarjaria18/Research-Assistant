import streamlit as st
import os
import tempfile
import json
from research_agent import ResearchAgent
from config import Config
import pandas as pd

# Page config
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 30px;
    }
    .section-header {
        color: #A23B72;
        border-bottom: 2px solid #A23B72;
        padding-bottom: 5px;
        margin-top: 20px;
    }
    .info-box {
        background-color: #F18F01;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #C73E1D;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #A23B72;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None

def setup_sidebar():
    st.sidebar.title("🔧 Configuration")
    
    provider = st.sidebar.selectbox(
        "Select LLM Provider",
        ["together", "groq"],
        help="Choose your preferred LLM provider"
    )
    
    config = Config()

    if provider == "together":
        models = list(config.TOGETHER_MODELS.keys())
        default_model = "deepseek-ai/DeepSeek-V3"
    else:
        models = list(config.GROQ_MODELS.keys())
        default_model = "llama-3.1-8b-instant"
    
    model = st.sidebar.selectbox(
        "Select Model",
        models,
        index=models.index(default_model) if default_model in models else 0
    )
    
    api_key_label = f"{provider.upper()} API Key"
    api_key = st.sidebar.text_input(
        api_key_label,
        type="password",
        help=f"Enter your {provider} API key"
    )
    
    if api_key:
        if provider == "together":
            os.environ["TOGETHER_API_KEY"] = api_key
        else:
            os.environ["GROQ_API_KEY"] = api_key
    
    if api_key and (not st.session_state.agent or 
                    st.session_state.agent.llm.provider != provider):
        try:
            st.session_state.agent = ResearchAgent(llm_provider=provider, model=model)
            st.sidebar.success(f"✅ {provider} initialized successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error initializing {provider}: {str(e)}")
    
    return provider, model, api_key

def display_analysis_results(results):
    if not results or 'error' in results:
        st.error(f"Error: {results.get('error', 'Unknown error occurred')}")
        return
    
    tabs = st.tabs([
        "📄 Summary", 
        "🎯 Objective", 
        "📖 Introduction", 
        "🔬 Methodology", 
        "📊 Results", 
        "📚 Citations", 
        "🔍 Research Gap", 
        "🔗 Similar Papers"
    ])
    
    with tabs[0]:
        st.markdown("### 📄 Paper Summary")
        st.write(results.get('summary', 'No summary available'))
    
    with tabs[1]:
        st.markdown("### 🎯 Research Objective")
        st.write(results.get('objective', 'No objective identified'))
    
    with tabs[2]:
        st.markdown("### 📖 Introduction")
        st.write(results.get('introduction', 'No introduction summary available'))
    
    with tabs[3]:
        st.markdown("### 🔬 Methodology")
        st.write(results.get('methodology', 'No methodology summary available'))
    
    with tabs[4]:
        st.markdown("### 📊 Results")
        st.write(results.get('results', 'No results summary available'))
    
    with tabs[5]:
        st.markdown("### 📚 Citations")
        citations = results.get('citations', {})
        if isinstance(citations, dict):
            st.write(f"**Total Citations:** {citations.get('citation_count', 0)}")
            if citations.get('in_text_citations'):
                st.write("**In-text Citations:**")
                for citation in citations['in_text_citations'][:20]:
                    st.write(f"- {citation}")
            if citations.get('reference_list'):
                st.write("**Reference List:**")
                for ref in citations['reference_list'][:10]:
                    st.write(f"- {ref}")
        else:
            st.write("Citations not in structured format")
    
    with tabs[6]:
        st.markdown("### 🔍 Research Gap")
        st.write(results.get('research_gap', 'No research gap identified'))
    
    with tabs[7]:
        st.markdown("### 🔗 Similar Papers")
        similar_papers = results.get('similar_papers', [])
        if similar_papers:
            for i, paper in enumerate(similar_papers[:5], 1):
                st.write(f"**{i}. {paper.get('title', 'Unknown Title')}**")
                st.write(f"Authors: {', '.join(paper.get('authors', []))}")
                st.write(f"Source: {paper.get('source', 'Unknown')}")
                st.write(f"Published: {paper.get('published', 'Unknown')}")
                if paper.get('url'):
                    st.write(f"URL: {paper['url']}")
                st.write(f"Abstract: {paper.get('abstract', 'No abstract available')[:200]}...")
                st.write("---")
        else:
            st.write("No similar papers found")

def main():
    initialize_session_state()
    
    st.markdown("<h1 class='main-header'>🔬 Research Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Upload a research paper and get comprehensive analysis including summary, citations, objectives, methodology, results, and more!</p>", unsafe_allow_html=True)
    
    provider, model, api_key = setup_sidebar()
    
    if not api_key:
        st.warning("⚠️ Please enter your API key in the sidebar to continue.")
        return
    
    if not st.session_state.agent:
        st.error("❌ Failed to initialize the research agent. Please check your API key.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📄 Upload Research Paper")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            if st.button("🔍 Analyze Paper", type="primary"):
                with st.spinner("Analyzing paper... This may take a few minutes."):
                    try:
                        results = st.session_state.agent.analyze_paper(tmp_file_path)
                        st.session_state.analysis_results = results
                        st.session_state.collection_name = results.get('collection_name')
                        os.unlink(tmp_file_path)
                        st.success("✅ Analysis completed successfully!")
                    except Exception as e:
                        st.error(f"❌ Error analyzing paper: {str(e)}")
                        os.unlink(tmp_file_path)
    
    with col2:
        st.markdown("### 💡 Features")
        st.markdown("""
        - **📄 Comprehensive Summary**
        - **🎯 Research Objectives**
        - **📖 Introduction Analysis**
        - **🔬 Methodology Overview**
        - **📊 Results Summary**
        - **📚 Citation Extraction**
        - **🔍 Research Gap Identification**
        - **🔗 Similar Papers Discovery**
        - **💬 Interactive Q&A**
        """)
    
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("<h2 class='section-header'>📊 Analysis Results</h2>", unsafe_allow_html=True)
        display_analysis_results(st.session_state.analysis_results)
        
        st.markdown("---")
        st.markdown("<h2 class='section-header'>💬 Ask Questions</h2>", unsafe_allow_html=True)
        
        question = st.text_input("Ask a question about the paper:", placeholder="e.g., What is the main contribution of this paper?")
        
        if st.button("🤔 Get Answer") and question:
            with st.spinner("Searching for answer..."):
                try:
                    answer = st.session_state.agent.query_paper(
                        question,
                        st.session_state.collection_name
                    )
                    st.markdown("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 0.8em;'>"
        "Research Assistant - Powered by AI | Built with Streamlit, LangChain, and ChromaDB"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
