import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# --- 1. Page Configuration ---
st.set_page_config(page_title="Aura AI", page_icon="🌌", layout="wide")

# --- 2. Custom CSS UI Polishing ---
st.markdown("""
<style>
    /* Hide default Streamlit headers and footers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global container padding */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }
    
    /* Premium Gradient Header */
    .brand-title {
        background: linear-gradient(135deg, #7F00FF, #E100FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0rem;
    }
    
    .brand-subtitle {
        font-size: 1.1rem;
        color: #8A99AD;
        margin-bottom: 2.5rem;
    }
    
    /* Styled metric card container */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Initialize Session States ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# --- 4. Sidebar / Control Center ---
with st.sidebar:
    st.markdown("### 🪐 Aura Workspace")
    st.caption("Manage your local context and data storage securely.")
    st.markdown("---")
    
    uploaded_files = st.file_uploader("Upload reference documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        uploaded_file_names = sorted([f.name for f in uploaded_files])
        if "current_file_names" not in st.session_state or st.session_state.current_file_names != uploaded_file_names:
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.session_state.current_file_names = uploaded_file_names
            st.session_state.total_chunks = 0

        # Processing Engine
        if st.session_state.vector_store is None:
            with st.spinner("Parsing documents into vector space..."):
                all_flashcards = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_path = temp_file.name

                    pages = PyPDFLoader(temp_path).load()
                    file_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(pages)
                    all_flashcards.extend(file_chunks)
                    os.remove(temp_path)

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector_store = Chroma.from_documents(all_flashcards, embeddings)
                st.session_state.total_chunks = len(all_flashcards)
                st.toast('Knowledge updated!', icon='✨')

        brain = Ollama(model="llama3")
        manager = ConversationalRetrievalChain.from_llm(
            llm=brain,
            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=st.session_state.memory
        )

        # Workspace Statistics Dashboard
        st.markdown("### 📊 Live Analytics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Files Loaded", value=len(uploaded_files))
        with col2:
            st.metric(label="Text Blocks", value=st.session_state.total_chunks)
            
        st.markdown("---")
        with st.expander("📂 Indexed File Index", expanded=False):
            for name in st.session_state.current_file_names:
                st.markdown(f"`{name}`")
        
        st.markdown("##")
        if st.button("🗑️ Reset Chat Thread", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.rerun()
    else:
        st.session_state.vector_store = None
        st.session_state.current_file_names = []
        st.session_state.chat_history = []
        st.session_state.total_chunks = 0
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- 5. Main Application Interface ---
st.markdown('<h1 class="brand-title">Aura</h1>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">An elegant, sandboxed interface for distributed document analysis.</p>', unsafe_allow_html=True)

if uploaded_files:
    # Chat Viewport
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            avatar_icon = "👤" if msg["role"] == "user" else "🌌"
            with st.chat_message(msg["role"], avatar=avatar_icon):
                st.markdown(msg["content"])

    # Chat Entry Bar
    if question := st.chat_input("Ask Aura anything about the loaded data..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant", avatar="🌌"):
            with st.spinner("Deconstructing prompt..."):
                response = manager.invoke({"question": question})
                answer = response["answer"]
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    # Premium Empty State Design
    st.markdown("""
    <div style="background-color: rgba(255,255,255,0.02); padding: 40px; border-radius: 16px; border: 1px dashed rgba(255,255,255,0.1); text-align: center;">
        <h3 style="margin-top: 0;">System is currently idle</h3>
        <p style="color: #8A99AD; max-width: 500px; margin: 0 auto 20px auto;">
            Aura requires context to function. Please upload one or more PDF manuals, research notes, or scripts using the sidebar menu to begin an active session.
        </p>
    </div>
    """, unsafe_allow_html=True)