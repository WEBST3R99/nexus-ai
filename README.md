# 🌌 Aura — AI-Powered PDF Chatbot

Aura is an AI-powered chatbot that lets you upload multiple PDF documents and ask questions about their contents. Instead of manually searching through lengthy documents, you can have a conversation with your files and get context-aware answers.

Built with Python, LangChain, and locally running LLMs, Aura combines document processing, semantic search, and conversational memory into a simple interface.

## ✨ Features

- 📄 **Multiple PDF Uploads** — Upload and query multiple documents in one session.
- 🔍 **Semantic Search** — Find relevant information using HuggingFace embeddings and ChromaDB.
- 🤖 **AI-Powered Responses** — Generate answers using the Llama 3 model through Ollama.
- 💬 **Conversational Memory** — Ask follow-up questions while maintaining conversation context.
- 📊 **Document Analytics** — View uploaded files and the number of processed text chunks.
- 🎨 **Interactive UI** — A clean, custom-styled interface built with Streamlit.
- 🖥️ **Local LLM Execution** — Run the language model locally using Ollama.

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core application logic |
| Streamlit | User interface |
| LangChain | RAG pipeline and conversational retrieval |
| Ollama | Local LLM inference |
| Llama 3 | Answer generation |
| HuggingFace Embeddings | Text embeddings using `all-MiniLM-L6-v2` |
| ChromaDB | Vector storage and semantic retrieval |
| PyPDF | PDF document loading |

## ⚙️ How It Works

1. **Upload:** The user uploads one or more PDF documents.
2. **Extract:** Text is extracted from the PDFs and divided into smaller chunks.
3. **Embed:** HuggingFace generates embeddings for the document chunks.
4. **Index:** ChromaDB stores the embedded chunks for semantic retrieval.
5. **Retrieve:** When a question is asked, the system retrieves the four most relevant chunks.
6. **Generate:** Llama 3 uses the retrieved context and conversation history to generate an answer.
7. **Respond:** The answer appears in the Streamlit chat interface.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/WEBST3R99/nexus-ai.git
cd nexus-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install streamlit langchain-community langchain-text-splitters langchain-classic chromadb sentence-transformers pypdf
```

### 4. Set Up Ollama

Install Ollama, then download the Llama 3 model:

```bash
ollama pull llama3
```

Make sure the Ollama service is running.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to start using Aura.

## 💡 Example Use Cases

- Ask questions about research papers.
- Find specific information in study materials.
- Query technical documentation.
- Explore lengthy PDF reports without manually searching every page.

## 🔮 Future Improvements

- Support for DOCX and TXT documents.
- Source citations with page numbers.
- Improved retrieval accuracy through reranking.
- Persistent document indexing across sessions.
- Exportable chat history.

## 👨‍💻 Author

**Soumit Das**

[GitHub](https://github.com/WEBST3R99)

---

*Built to explore practical applications of Retrieval-Augmented Generation (RAG), local LLMs, and semantic search.*
