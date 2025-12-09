# 🦙 Llama 3.2 RAG Web App

A local Retrieval-Augmented Generation (RAG) application that allows you to chat with your own documents (PDF, DOCX, TXT) using the **Llama 3.2** AI model. This application runs entirely locally on your machine using **Ollama**, ensuring complete data privacy.

## 🏗️ Architecture

![Llama 3.2 RAG Architecture](assets/rag-app.png.png)

The application follows a standard RAG pipeline:
1.  **Ingest:** Documents are loaded and split into chunks.
2.  **Embed:** Chunks are converted to vectors using HuggingFace embeddings.
3.  **Store:** Vectors are stored locally in ChromaDB.
4.  **Retrieve:** User queries fetch relevant context from the database.
5.  **Generate:** Llama 3.2 generates an answer using the retrieved context.

## 🚀 Features

* **100% Local & Private:** No data leaves your machine; powered by local LLMs via Ollama.
* **Multi-Format Support:** Upload and chat with `.pdf`, `.docx`, and `.txt` files.
* **Vector Search:** Uses `ChromaDB` and `HuggingFace Embeddings` for accurate context retrieval.
* **Interactive UI:** Built with **Streamlit** for a clean, chat-based interface.
* **Source Citation:** See exactly which part of your document the AI used to generate the answer.

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.9+** installed on your system.
2.  **[Ollama](https://ollama.com/)** installed and running.

## 📦 Installation

### 1. Set up Ollama
Ensure you have the Llama 3.2 model pulled locally. Open your terminal and run:

```bash
ollama pull llama3.2:3b
```

### 2. Install Dependencies
Run the following command to install the necessary Python libraries.

```bash
pip3 install streamlit langchain-community langchain-chroma langchain-ollama langchain-huggingface pypdf docx2txt chromadb sentence-transformers
```
- streamlit: The web interface.
- langchain-*: Frameworks for RAG, PDF/Text processing, and connecting to Ollama.
- docx2txt: For processing Word documents.
- pypdf: For processing PDF files.
- chromadb: The local vector database to store your document data.

### 3. Running the App

Execute below command to run the app:
```bash
python3 -m streamlit run app.py
```
