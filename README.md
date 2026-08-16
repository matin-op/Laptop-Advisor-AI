# 💻 Laptop Advisor AI

A RAG-powered chatbot that answers laptop-related questions using a curated knowledge base — built with LangChain, Gemini, ChromaDB, and Streamlit.

Ask it things like *"Which laptop is best for coding?"* or *"Best gaming laptop under budget?"* and it retrieves relevant context from a local knowledge base before generating a grounded answer, with sources shown for transparency.

---

## ✨ Features

- 🔍 **Retrieval-Augmented Generation** — answers are grounded in a custom laptop knowledge base, not just the model's raw knowledge
- 📚 **Source transparency** — every answer shows the exact chunks it was generated from
- 💬 **Chat interface** — persistent conversation history within a session
- ⚡ **Suggestion chips** — quick-start prompts for first-time users
- 🎨 **Custom UI** — styled chat bubbles and a clean, modern layout
- 🛡️ **Input sanitization** — user input is escaped before rendering to prevent HTML/script injection

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| LLM | Google Gemini (`gemini-3-flash-preview`) via `langchain-google-genai` |
| Embeddings | Gemini Embeddings (`gemini-embedding-2`) |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| UI | Streamlit |

---

## 📂 Project Structure

```
laptop-advisor-ai/
├── main.py            # Streamlit app + RAG pipeline
├── data.txt            # Knowledge base source document
├── chroma_store/        # Persisted vector store (generated on first run)
├── requirements.txt
├── .env                 # GEMINI_API_KEY (not committed)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/laptop-advisor-ai.git
cd laptop-advisor-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file one level above `main.py` (or adjust the path in `init_rag_pipeline()`) with:
```
GEMINI_API_KEY=your_gemini_api_key_here
```
Get a free key from [Google AI Studio](https://aistudio.google.com/).

### 4. Add your knowledge base
Place your laptop data in `data.txt` (plain text, one entry/section per laptop or topic works best for chunking).

### 5. Run the app
```bash
streamlit run main.py
```

The vector store is built automatically on first run and persisted to `chroma_store/` — subsequent runs reuse it instead of re-embedding.

---

## 🧠 How It Works

1. `data.txt` is loaded and split into ~1000-character chunks with 100-character overlap.
2. Chunks are embedded using Gemini Embeddings and stored in ChromaDB.
3. On each user question, the top 3 most relevant chunks are retrieved.
4. Retrieved context + the question are passed to Gemini via a LangChain prompt template.
5. The model's answer is streamed back to the chat UI, along with the source chunks used.

---

## 🔭 Known Limitations / Next Steps

- No conversational memory in retrieval — each question is retrieved independently, so follow-up questions like *"what about a cheaper one?"* won't resolve pronouns from prior turns.
- Knowledge base is a single flat text file — no structured metadata (price, brand, specs) for filtered retrieval yet.
- No evaluation harness for retrieval quality (e.g. faithfulness/relevance scoring).

---

## 📄 License

MIT — feel free to fork and adapt.
