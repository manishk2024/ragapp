import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

# --- 1. Data Ingestion and Processing ---
def process_document(file_path):
    """
    Loads a PDF document, splits it into chunks.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    return texts

# --- 2. Vector Store Creation ---
def create_vector_store(texts):
    """
    Creates a FAISS vector store from the given texts.
    """
    # Use a popular text embedding model
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# --- 3. Retriever and LLM Integration ---
def create_rag_pipeline(vectorstore, llama_model):
    """
    Creates a RAG pipeline with a retriever and a Llama model.
    """
    retriever = vectorstore.as_retriever()
    
    # Create a Hugging Face pipeline for the Llama model
    hf_pipeline = pipeline(
        "text-generation",
        model=llama_model,
        tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf"), # Use a valid Llama tokenizer
        max_length=2048,
        trust_remote_code=True,
    )
    
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- 4. Chatbot UI ---
def create_chatbot_ui(qa_chain):
    """
    Creates a Gradio UI for the chatbot.
    """
    def chatbot_response(message, history):
        response = qa_chain({"query": message})
        return response["result"]

    iface = gr.ChatInterface(
        chatbot_response,
        chatbot=gr.Chatbot(height=500),
        title="Llama RAG Chatbot",
        description="Ask questions about your PDF document.",
        theme="soft",
    )
    return iface

# --- Main execution ---
def start_app(file):
    """
    Starts the RAG chatbot application.
    """
    if file is None:
        return "Please upload a PDF file."

    pdf_path = file.name
    
    # 1. Process the document
    print("Processing document...")
    texts = process_document(pdf_path)
    
    # 2. Create the vector store
    print("Creating vector store...")
    vectorstore = create_vector_store(texts)
    
    # 3. Load the Llama model
    print("Loading Llama model...")
    # It is recommended to use a quantized model for local execution
    model_id = "TheBloke/Llama-2-7B-Chat-GPTQ"
    llama_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. Create the RAG pipeline
    print("Creating RAG pipeline...")
    qa_chain = create_rag_pipeline(vectorstore, llama_model)
    
    # 5. Create and launch the Gradio UI
    print("Launching Gradio UI...")
    iface = create_chatbot_ui(qa_chain)
    iface.launch(share=True)

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## Llama RAG Chatbot")
        gr.Markdown("Upload a PDF file and start asking questions about its content.")
        file_input = gr.File(label="Upload PDF")
        start_button = gr.Button("Start Chatbot")
        
        start_button.click(start_app, inputs=[file_input], outputs=None)

    demo.launch()
