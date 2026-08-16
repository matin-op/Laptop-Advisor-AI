import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import html
from dotenv import load_dotenv

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💻 Laptop Advisor AI",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .header-container {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .header-container h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .header-container p {
        font-size: 1.05rem;
        opacity: 0.7;
    }

    /* ── Chat bubbles ── */
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        margin-bottom: 0.75rem;
        line-height: 1.6;
        font-size: 0.97rem;
        animation: fadeSlideIn 0.35s ease-out;
    }
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #fff;
        margin-left: 15%;
        border-bottom-right-radius: 0.25rem;
    }
    .chat-message.assistant {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-right: 15%;
        border-bottom-left-radius: 0.25rem;
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Source cards ── */
    .source-card {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        margin-top: 0.4rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ── Suggestion chips ── */
    .suggestions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin-top: 1rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        opacity: 0.4;
        font-size: 0.82rem;
    }

    /* Hide default streamlit header/footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── RAG Pipeline (cached so it only runs once) ─────────────────────────────
@st.cache_resource(show_spinner="🔧 Loading AI models & knowledge base…")
def init_rag_pipeline():
    """Initialize LLM, embeddings, vectorstore, and chain — once."""
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    # Load & split the document
    loader = TextLoader(
        os.path.join(os.path.dirname(__file__), "data.txt"), encoding="utf-8"
    )
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(raw_documents)

    persist_dir = os.path.join(os.path.dirname(__file__), "chroma_store")

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        # Store already built — reuse it instead of re-embedding & duplicating
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a friendly and knowledgeable laptop advisor. "
            "Use the following context to answer the user's question. "
            "If the context doesn't contain enough information, say so honestly. "
            "Format your answer clearly with bullet points or short paragraphs.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    return retriever, chain


def ask_question(question: str) -> dict:
    """Run the RAG pipeline and return the answer + source docs."""
    retriever, chain = init_rag_pipeline()

    relevant_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    answer = chain.invoke({"context": context, "question": question})

    return {"answer": answer, "sources": relevant_docs}


# ── Session State ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-container">
        <h1>💻 Laptop Advisor AI</h1>
        <p>Ask me anything about laptops — I'll find the perfect match for you.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Render chat history ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role_class = "user" if msg["role"] == "user" else "assistant"
    st.markdown(
        f'<div class="chat-message {role_class}">{html.escape(msg["content"])}</div>',
        unsafe_allow_html=True,
    )
    # Show source snippets for assistant messages
    if msg["role"] == "assistant" and msg.get("sources"):
        with st.expander("📚 View sources used"):
            for i, src in enumerate(msg["sources"], 1):
                st.markdown(
                    f'<div class="source-card"><strong>Source {i}:</strong> {html.escape(src)}</div>',
                    unsafe_allow_html=True,
                )

# ── Suggestion chips (only shown when chat is empty) ───────────────────────
if not st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    suggestions = [
        "Which laptop is best for coding?",
        "Best gaming laptop under budget?",
        "MacBook vs Windows for students?",
        "Best laptop for video editing?",
    ]
    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions):
        if col.button(suggestion, use_container_width=True):
            st.session_state["pending_question"] = suggestion
            st.rerun()

# ── Chat Input ──────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about laptops…")

# Handle suggestion chip click
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(
        f'<div class="chat-message user">{html.escape(user_input)}</div>',
        unsafe_allow_html=True,
    )

    # Get AI response
    with st.spinner("🤔 Thinking…"):
        try:
            result = ask_question(user_input)
        except Exception as e:
            st.error(f"⚠️ Something went wrong while generating a response: {e}")
            st.stop()

    # Append assistant message (rendering happens on rerun via history loop)
    source_texts = [doc.page_content for doc in result["sources"]]
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": source_texts,
    })

    st.rerun()

# ── Sidebar: Clear chat ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "**Laptop Advisor AI** uses RAG (Retrieval-Augmented Generation) "
        "to search a curated laptop database and answer your questions "
        "with AI-powered recommendations."
    )

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Powered by Gemini + LangChain + ChromaDB</div>',
    unsafe_allow_html=True,
)