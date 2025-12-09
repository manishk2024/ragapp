import streamlit as st
import os
import tempfile
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

# --- 1. Page Config ---
st.set_page_config(page_title="RAG with Llama 3.2", layout="wide")
st.title("🦙 Llama 3.2 RAG: Chat with your Docs")

# --- 2. Initialize Session State ---
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# --- 3. Helper Functions ---
def load_and_process_file(uploaded_file):
    """
    Saves uploaded file to a temp path and loads it using the appropriate loader.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        else:
            st.error("Unsupported file type!")
            return []
        
        return loader.load()
    finally:
        os.remove(tmp_path)  # Clean up temp file

def create_vector_db(documents):
    """
    Chunks documents and creates a vector store.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)
    
    # Using HuggingFace embeddings (runs locally, good performance)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Create Vector Store
    vectors = Chroma.from_documents(
        documents=final_documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectors

# --- 4. Sidebar: File Upload ---
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            st.session_state.processing = True
            with st.spinner("Processing documents... this may take a moment."):
                all_docs = []
                for file in uploaded_files:
                    docs = load_and_process_file(file)
                    all_docs.extend(docs)
                
                if all_docs:
                    st.session_state.vectors = create_vector_db(all_docs)
                    st.success("Documents processed and database ready!")
                st.session_state.processing = False
        else:
            st.warning("Please upload at least one file.")

# --- 5. Main Chat Interface ---
if st.session_state.vectors:
    st.write("Ask a question based on your uploaded documents:")
    user_input = st.text_input("Your Query:")

    if user_input:
        with st.spinner("Generating answer with Llama 3.2..."):
            try:
                # Initialize LLM
                llm = ChatOllama(model=LLM_MODEL)

                # Create Retrieval Chain
                prompt = ChatPromptTemplate.from_template(
                    """
                    Answer the following question based ONLY on the provided context.
                    If the answer is not in the context, say "I don't know based on these documents."
                    
                    <context>
                    {context}
                    </context>

                    Question: {input}
                    """
                )
                
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                response = retrieval_chain.invoke({"input": user_input})
                
                st.markdown("### Answer:")
                st.write(response["answer"])
                
                # Optional: Show source documents
                with st.expander("View Source Context"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"**Source {i+1}:** {doc.page_content[:200]}...")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
else:
    st.info("👈 Upload and process documents in the sidebar to get started.")