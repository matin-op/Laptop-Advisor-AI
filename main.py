import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

import os
import html
import math
import re
from dotenv import load_dotenv


# ═══════════════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="💻 Laptop Advisor AI",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Premium CSS — Dark Glassmorphism Theme
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(160deg, #0f0f1a 0%, #1a1025 40%, #0d1117 100%) !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, header, footer { visibility: hidden; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(139,92,246,0.25);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.45); }

    /* ── Hero Header ── */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 0.75rem;
    }
    .hero-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #8b5cf6, #06b6d4, #ec4899, #8b5cf6);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease infinite;
        margin: 0 0 0.2rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.4);
        font-weight: 300;
    }
    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin-top: 0.7rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.22rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 500;
        background: rgba(139,92,246,0.07);
        border: 1px solid rgba(139,92,246,0.16);
        color: rgba(255,255,255,0.55);
        transition: all 0.3s ease;
    }
    .hero-badge:hover {
        background: rgba(139,92,246,0.14);
        border-color: rgba(139,92,246,0.3);
        color: rgba(255,255,255,0.8);
    }

    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50%      { background-position: 100% 50%; }
    }

    /* ── Chat Messages (Streamlit native overrides) ── */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 1rem !important;
        backdrop-filter: blur(16px) !important;
        animation: msgSlideIn 0.4s cubic-bezier(0.22,1,0.36,1);
        margin-bottom: 0.55rem !important;
    }

    @keyframes msgSlideIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Tool-Call Cards ── */
    .tool-card {
        background: linear-gradient(135deg, rgba(139,92,246,0.07), rgba(6,182,212,0.05));
        border: 1px solid rgba(139,92,246,0.18);
        border-radius: 0.65rem;
        padding: 0.5rem 0.85rem;
        margin-bottom: 0.55rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.8rem;
        animation: toolPop 0.45s cubic-bezier(0.22,1,0.36,1);
    }
    .tool-icon  { font-size: 1rem; }
    .tool-name  { font-weight: 600; color: #a78bfa; }
    .tool-args  {
        color: rgba(255,255,255,0.35);
        font-size: 0.74rem;
        margin-left: auto;
        max-width: 55%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    @keyframes toolPop {
        from { opacity: 0; transform: scale(0.96) translateX(-6px); }
        to   { opacity: 1; transform: scale(1) translateX(0); }
    }

    /* ── Source Cards ── */
    .source-card {
        background: rgba(6,182,212,0.04);
        border: 1px solid rgba(6,182,212,0.1);
        border-radius: 0.55rem;
        padding: 0.55rem 0.85rem;
        margin-top: 0.3rem;
        font-size: 0.8rem;
        line-height: 1.55;
        color: rgba(255,255,255,0.7);
    }

    /* ── Suggestion Chips ── */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: rgba(255,255,255,0.025) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 9999px !important;
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.78rem !important;
        font-weight: 400 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.4rem 0.85rem !important;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
        backdrop-filter: blur(8px) !important;
        white-space: nowrap !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: rgba(139,92,246,0.1) !important;
        border-color: rgba(139,92,246,0.3) !important;
        color: #e2e8f0 !important;
        box-shadow: 0 0 20px rgba(139,92,246,0.1) !important;
        transform: translateY(-2px) !important;
    }

    /* ── Chat Input ── */
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 0.75rem !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(139,92,246,0.3) !important;
        box-shadow: 0 0 0 3px rgba(139,92,246,0.06) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #13101e 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.04) !important;
    }
    .sidebar-brand {
        text-align: center;
        padding: 1.2rem 0 0.4rem;
    }
    .sidebar-brand h2 {
        font-size: 1.15rem;
        font-weight: 700;
        background: linear-gradient(135deg, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sidebar-stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.4rem;
        margin: 0.6rem 0;
    }
    .sidebar-stat {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 0.55rem;
        padding: 0.45rem 0.5rem;
        text-align: center;
    }
    .sidebar-stat .s-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #8b5cf6;
        line-height: 1.2;
    }
    .sidebar-stat .s-lbl {
        font-size: 0.64rem;
        color: rgba(255,255,255,0.35);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.1rem;
    }

    /* ── Expander ── */
    .stExpander {
        border: 1px solid rgba(255,255,255,0.04) !important;
        border-radius: 0.55rem !important;
        background: transparent !important;
    }

    /* ── Glass Divider ── */
    .glass-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.18), transparent);
        margin: 1.25rem 0;
        border: none;
    }

    /* ── Tools Showcase (empty state) ── */
    .tools-row {
        display: flex;
        gap: 0.55rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .tool-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.28rem 0.65rem;
        border-radius: 0.45rem;
        font-size: 0.7rem;
        font-weight: 500;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        color: rgba(255,255,255,0.4);
    }

    /* ── Bottom Input Bar (force dark) ── */
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    .stBottom,
    .stChatInput {
        background: #0d0d18 !important;
        background-color: #0d0d18 !important;
    }
    [data-testid="stBottom"] > div {
        background: #0d0d18 !important;
    }
    [data-testid="stBottomBlockContainer"] {
        background: #0d0d18 !important;
        padding-top: 0.5rem !important;
    }

    /* ── Streamlit block containers (transparent) ── */
    [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlock"],
    .block-container {
        background: transparent !important;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.18);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Load Laptop Database (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_laptop_database():
    """Parse data.txt to extract structured laptop records."""
    data_path = os.path.join(os.path.dirname(__file__), "data.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()

    ns = {}
    exec(content, ns)                       # noqa: S102  — trusted local file
    docs     = ns["documents"]
    ids_list = ns["ids"]
    metas    = ns["etadatas"]               # variable name in the original data.txt

    laptops = []
    for doc, lid, meta in zip(docs, ids_list, metas):
        name = doc.split(":")[0].strip()
        laptops.append({"name": name, "id": lid, "description": doc, **meta})
    return laptops


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Definitions
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def search_laptops(
    brand: str = "",
    min_price: int = 0,
    max_price: int = 99999,
    category: str = "",
    min_ram: int = 0,
    operating_system: str = "",
    min_rating: float = 0.0,
) -> str:
    """Search the laptop database with filters. Use when the user wants to
    find laptops matching criteria like brand, price range, category, RAM,
    operating system, or rating.

    Args:
        brand: Brand filter (Dell, Apple, ASUS, Lenovo, HP, MSI, Razer, Acer,
            Samsung, LG, Framework, Huawei, Gigabyte, etc.). Empty = all.
        min_price: Minimum price in USD (default 0).
        max_price: Maximum price in USD (default 99999).
        category: Category filter (Gaming, Ultrabook, Business, Workstation,
            2-in-1, Budget, Chromebook, Productivity, Rugged). Empty = all.
        min_ram: Minimum RAM in GB (default 0).
        operating_system: OS filter (Windows, macOS, ChromeOS). Empty = all.
        min_rating: Minimum star rating 0.0–5.0 (default 0.0).

    Returns:
        A formatted list of matching laptops with specs.
    """
    laptops = load_laptop_database()
    hits = []
    for lp in laptops:
        if brand and lp["brand"].lower() != brand.lower():
            continue
        if lp["price"] < min_price or lp["price"] > max_price:
            continue
        if category and lp["category"].lower() != category.lower():
            continue
        if lp["ram_gb"] < min_ram:
            continue
        if operating_system and lp["os"].lower() != operating_system.lower():
            continue
        if lp["rating"] < min_rating:
            continue
        hits.append(lp)

    if not hits:
        return "No laptops found matching those criteria."

    hits.sort(key=lambda x: (-x["rating"], x["price"]))
    lines = [f"Found {len(hits)} laptop(s):\n"]
    for h in hits[:10]:
        lines.append(
            f"• **{h['name']}** — ${h['price']:,} | "
            f"{h['ram_gb']}GB RAM | {h['storage_gb']}GB SSD | "
            f"{h['category']} | {h['os']} | ⭐ {h['rating']}"
        )
    if len(hits) > 10:
        lines.append(f"\n…and {len(hits) - 10} more.")
    return "\n".join(lines)


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use for price calculations,
    EMI/loan payments, tax, savings, or price-per-spec comparisons.

    EMI formula: P * r * (1+r)**n / ((1+r)**n - 1)
    where P = principal, r = monthly interest rate (annual/12/100), n = months.

    Args:
        expression: A Python math expression. Available helpers include
            sqrt, ceil, floor, round, abs, pow, log, log10, pi, e.

    Returns:
        The numeric result of the calculation.
    """
    safe_names = {
        "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
        "round": round, "abs": abs, "pow": pow,
        "log": math.log, "log10": math.log10,
        "pi": math.pi, "e": math.e,
    }
    try:
        cleaned = re.sub(r"[^0-9+\-*/()._ \w]", "", expression)
        result = eval(cleaned, {"__builtins__": {}}, safe_names)   # noqa: S307
        if isinstance(result, float):
            result = round(result, 2)
        return f"Result: {result}"
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def compare_laptops(laptop_names: str) -> str:
    """Compare two or three laptops side by side. Provide comma-separated
    laptop names or partial names.

    Args:
        laptop_names: Comma-separated names, e.g.
            'MacBook Air M3, Dell XPS 15, ThinkPad X1 Carbon'.

    Returns:
        A markdown comparison table with key specs.
    """
    queries = [n.strip() for n in laptop_names.split(",") if n.strip()]
    laptops = load_laptop_database()
    matched = []

    for query in queries[:3]:
        q_lower = query.lower()
        best, best_score = None, 0
        for lp in laptops:
            words = q_lower.split()
            score = sum(1 for w in words if w in lp["name"].lower())
            if score > best_score:
                best_score, best = score, lp
        if best and best_score > 0:
            matched.append(best)

    if len(matched) < 2:
        return "Could not find enough matching laptops — please use more specific names."

    hdr = "| Spec | " + " | ".join(m["name"] for m in matched) + " |"
    sep = "|---|" + "|".join(["---"] * len(matched)) + "|"
    rows = [
        "| **Brand** | "   + " | ".join(m["brand"] for m in matched) + " |",
        "| **Price** | "   + " | ".join(f"${m['price']:,}" for m in matched) + " |",
        "| **RAM** | "     + " | ".join(f"{m['ram_gb']}GB" for m in matched) + " |",
        "| **Storage** | " + " | ".join(f"{m['storage_gb']}GB" for m in matched) + " |",
        "| **Category** | "+ " | ".join(m["category"] for m in matched) + " |",
        "| **OS** | "      + " | ".join(m["os"] for m in matched) + " |",
        "| **Rating** | "  + " | ".join(f"⭐ {m['rating']}" for m in matched) + " |",
    ]
    return "\n".join([hdr, sep, *rows])


# ── Tool registry ──────────────────────────────────────────────────────────

TOOLS     = [search_laptops, calculator, compare_laptops]
TOOLS_MAP = {t.name: t for t in TOOLS}

TOOL_ICONS  = {"search_laptops": "🔍", "calculator": "🧮", "compare_laptops": "📊"}
TOOL_LABELS = {
    "search_laptops": "Database Search",
    "calculator":     "Calculator",
    "compare_laptops":"Comparison",
}


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Pipeline (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔧 Loading AI models & knowledge base…")
def init_rag_pipeline():
    """Initialize LLM (with tools), embeddings, vectorstore, retriever."""
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    api_key = os.environ["GEMINI_API_KEY"]

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key,
    )

    # Load & chunk the knowledge base
    loader = TextLoader(
        os.path.join(os.path.dirname(__file__), "data.txt"), encoding="utf-8",
    )
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks   = splitter.split_documents(raw_docs)

    persist_dir = os.path.join(os.path.dirname(__file__), "chroma_store")

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
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
    return retriever, llm_with_tools


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Loop
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable laptop advisor. "
    "Use the Context below **and** your tools to give the best possible answer.\n\n"
    "• If the user asks to *find* or *filter* laptops by specs → call `search_laptops`.\n"
    "• If the user asks for a *calculation* (EMI, tax, savings) → call `calculator`.\n"
    "• If the user wants to *compare* specific laptops → call `compare_laptops`.\n"
    "• Otherwise answer directly from the context.\n\n"
    "Format answers clearly with bullet points or short paragraphs.\n\n"
    "Context:\n{context}"
)


def run_agent(question: str) -> dict:
    """Execute the RAG + tool-calling agent loop and return the result."""
    retriever, llm_with_tools = init_rag_pipeline()

    # 1. Retrieve relevant context
    relevant_docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in relevant_docs)

    # 2. Build message history
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=question),
    ]

    tool_calls_log: list[dict] = []
    max_iters = 5

    # 3. Agentic loop — keep going until no more tool calls
    for _ in range(max_iters):
        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            break                          # final text answer

        messages.append(response)          # record the AIMessage with tool_calls

        for tc in response.tool_calls:
            fn = TOOLS_MAP.get(tc["name"])
            try:
                result = fn.invoke(tc["args"]) if fn else f"Unknown tool: {tc['name']}"
            except Exception as exc:
                result = f"Tool error: {exc}"

            tool_calls_log.append({
                "name":   tc["name"],
                "args":   tc["args"],
                "result": str(result),
            })
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    # Extract text — Gemini can return content as str OR list of dicts
    raw = response.content
    if isinstance(raw, list):
        answer = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        )
    else:
        answer = raw or ""

    answer = answer.strip() or "Sorry, I couldn't generate a response. Please try again."

    return {
        "answer":     answer,
        "sources":    relevant_docs,
        "tool_calls": tool_calls_log,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _tool_card_html(tc: dict) -> str:
    """Return HTML for a single tool-call card."""
    icon  = TOOL_ICONS.get(tc["name"], "🔧")
    label = TOOL_LABELS.get(tc["name"], tc["name"])
    parts = []
    for k, v in tc["args"].items():
        if v and str(v) not in ("0", "0.0", "99999", ""):
            parts.append(f"{k}={v}")
    args_str = ", ".join(parts) if parts else "default filters"
    return (
        f'<div class="tool-card">'
        f'<span class="tool-icon">{icon}</span>'
        f'<span class="tool-name">{html.escape(label)}</span>'
        f'<span class="tool-args">{html.escape(args_str)}</span>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []


# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-header">
    <h1>💻 Laptop Advisor AI</h1>
    <p class="hero-subtitle">
        Smart recommendations powered by RAG &amp; function calling
    </p>
    <div class="hero-badges">
        <span class="hero-badge">🔍 Search</span>
        <span class="hero-badge">🧮 Calculate</span>
        <span class="hero-badge">📊 Compare</span>
        <span class="hero-badge">🧠 RAG</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Chat History
# ═══════════════════════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        # ── Tool-call cards (assistant only) ──
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                st.markdown(_tool_card_html(tc), unsafe_allow_html=True)
                with st.expander(
                    f"View {TOOL_LABELS.get(tc['name'], tc['name'])} result"
                ):
                    st.markdown(tc["result"])

        # ── Message body ──
        st.markdown(msg["content"])

        # ── Sources (assistant only) ──
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 View sources used"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>Source {i}:</strong> {html.escape(src)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# Suggestion Chips (empty-state only)
# ═══════════════════════════════════════════════════════════════════════════════

if not st.session_state.messages:
    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)

    suggestions = [
        "Best laptop for coding?",
        "Compare laptops",
        "Gaming laptops",
        "EMI for a laptop?",
    ]
    cols = st.columns(len(suggestions))
    for col, sug in zip(cols, suggestions):
        if col.button(sug, use_container_width=True):
            st.session_state["pending_question"] = sug
            st.rerun()

    st.markdown("""
    <div class="tools-row">
        <span class="tool-pill">🔍 Smart Search</span>
        <span class="tool-pill">🧮 Price Calculator</span>
        <span class="tool-pill">📊 Side-by-Side Compare</span>
        <span class="tool-pill">🧠 RAG Answers</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Chat Input & Agent Execution
# ═══════════════════════════════════════════════════════════════════════════════

user_input = st.chat_input("Ask about laptops — search, compare, calculate…")

# Handle suggestion-chip click
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")

if user_input:
    # ── Record & render the user message ──
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # ── Run the agent ──
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Thinking…"):
            try:
                result = run_agent(user_input)
            except Exception as e:
                st.error(f"⚠️ Something went wrong: {e}")
                st.stop()

        # Tool cards
        for tc in result.get("tool_calls", []):
            st.markdown(_tool_card_html(tc), unsafe_allow_html=True)
            with st.expander(
                f"View {TOOL_LABELS.get(tc['name'], tc['name'])} result"
            ):
                st.markdown(tc["result"])

        # Answer
        st.markdown(result["answer"])

        # Sources
        source_texts = [doc.page_content for doc in result["sources"]]
        if source_texts:
            with st.expander("📚 View sources used"):
                for i, src in enumerate(source_texts, 1):
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>Source {i}:</strong> {html.escape(src)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Record assistant message ──
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    result["answer"],
        "sources":    source_texts,
        "tool_calls": result.get("tool_calls", []),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><h2>💻 Laptop Advisor</h2></div>',
        unsafe_allow_html=True,
    )

    db         = load_laptop_database()
    brands     = sorted(set(lp["brand"] for lp in db))
    categories = sorted(set(lp["category"] for lp in db))
    prices     = [lp["price"] for lp in db]

    st.markdown(f"""
    <div class="sidebar-stat-grid">
        <div class="sidebar-stat">
            <div class="s-val">{len(db)}</div>
            <div class="s-lbl">Laptops</div>
        </div>
        <div class="sidebar-stat">
            <div class="s-val">{len(brands)}</div>
            <div class="s-lbl">Brands</div>
        </div>
        <div class="sidebar-stat">
            <div class="s-val">{len(categories)}</div>
            <div class="s-lbl">Categories</div>
        </div>
        <div class="sidebar-stat">
            <div class="s-val">${min(prices):,}</div>
            <div class="s-lbl">Lowest</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⚙️ Actions")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.markdown(
        "**How it works:** Uses **RAG** to retrieve relevant laptop data, "
        "then **function calling** lets the AI search the database, run "
        "calculations, and compare laptops side-by-side — all in one chat."
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.72rem;opacity:0.3;text-align:center;">'
        "Built with Gemini · LangChain · ChromaDB</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="app-footer">'
    "Powered by Gemini · LangChain · ChromaDB · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
