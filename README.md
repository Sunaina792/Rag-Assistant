# Rag-Assistant (DocuMind AI) 🧠

A complete, beginner-friendly Retrieval-Augmented Generation (RAG) chatbot with a real-time pipeline visualizer. Upload documents (PDF, TXT) and ask questions grounded strictly in your content.

---

## 🚀 Features

- **Document Processing**: Fast extraction from PDF and plain text documents with page and source tracking.
- **Smart Chunking**: Recursive character chunking (800 chars, 200 overlap) preserving page and file metadata.
- **Semantic Embeddings**: 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace with pure-Python cosine similarity.
- **Dual Retrieval Mode**: Semantic vector search with automatic BM25 fallback.
- **High-Speed Inference**: Powered by the Groq API with automatic model failover.
- **Pipeline Visualizer**: Real-time visual architecture flow, live execution timing for each step (ms), and a chunk inspector.
- **Clean Chat UI**: Responsive dark-mode interface with structured markdown answers, sources, and document management.
- **Python 3.14+ Ready**: Pure Python implementation with zero C++ compilation or wheel compatibility issues.

---

## 📁 Project Structure

```
Rag-Assistant/
├── app.py               # FastAPI backend & routes (/upload, /chat, /status, /reset)
├── document_loader.py   # PDF & TXT text extraction
├── text_splitter.py     # Recursive text chunking
├── embeddings.py        # HuggingFace semantic embeddings & cosine similarity
├── vector_store.py      # In-memory semantic vector store with BM25 fallback
├── rag_pipeline.py      # Retrieval orchestration, Groq LLM client & timing
├── prompts.py           # Grounded system prompt template
├── static/
│   └── index.html       # Chatbot UI & Pipeline Visualizer
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── .gitignore           # Git ignore configuration
└── README.md            # Documentation
```

---

## 🛠️ Installation & Setup (Windows / Mac / Linux)

### 1. Clone the Repository
```bash
git clone https://github.com/Sunaina792/Rag-Assistant.git
cd Rag-Assistant
```

### 2. Create and Activate a Virtual Environment
```powershell
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root folder:
```powershell
copy .env.example .env
```

Open `.env` and add your **Groq API Key** (get one free at [console.groq.com](https://console.groq.com)):
```env
GROQ_API_KEY=gsk_your_actual_groq_key_here
```

*(Optional)* You can also add a free Hugging Face token:
```env
HF_TOKEN=hf_your_token_here
```

---

## ▶️ Running the Application

Start the FastAPI application with Uvicorn:
```powershell
python -m uvicorn app:app --reload --port 8000
```

Open your browser and visit:
👉 **http://localhost:8000**

---

## 💡 How to Use

1. **Upload Documents**: Drag and drop PDF or TXT files in the left sidebar.
2. **Ask Questions**: Type your query in the chat input.
3. **Inspect the Pipeline**: Switch to the **"🔬 Pipeline Visualizer"** tab to see:
   - Visual architecture nodes glow as the request flows through.
   - Exact execution time for retrieval, context assembly, and LLM inference.
   - Top retrieved text chunks and similarity scores.
4. **Manage Knowledge Base**: Clear chat or reset the index anytime from the sidebar.

---

## 📜 License

MIT License. Free to use, adapt, and build upon!
