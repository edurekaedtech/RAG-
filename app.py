import streamlit as st
from openai import OpenAI

from rag.utils import get_env
from rag.ingest import ingest
from rag.retrieve import get_index_and_metadata, retrieve
from rag.prompts import DOMAIN_SYSTEM_PROMPTS, RAG_USER_TEMPLATE

# centralized key handling helpers
from auth import get_api_key

APP_TITLE = "RAG App with Streamlit"

# set_page_config must come BEFORE any st.* UI calls (like st.title)
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
    <style>
    .small-muted { font-size: 0.85rem; opacity: 0.75; }
    .pill { display:inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; border: 1px solid rgba(255,255,255,0.15); }
    .block { padding: 1rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.03); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption(
    "Deploy a RAG app with FAISS, prompt specialization, and optimization controls (chunking, top-k, caching, model choice)."
)

# Sidebar: Controls
with st.sidebar:
    st.header("RAG Controls")

    domain_mode = st.selectbox(
        "Assistant specialization",
        options=list(DOMAIN_SYSTEM_PROMPTS.keys()),
        index=1,
        help="Changes system behavior and tone for different domains.",
    )

    model = st.selectbox(
        "LLM model",
        options=["gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
        index=0,
        help="Choose speed/cost vs quality.",
    )

    st.subheader("Retrieval + Chunking")
    chunk_size = st.slider(
        "Chunk size", 300, 1500, 900, 50,
        help="Larger chunks keep context; smaller chunks improve precision."
    )
    overlap = st.slider(
        "Chunk overlap", 0, 400, 150, 10,
        help="Overlap prevents splitting important info across chunks."
    )
    top_k = st.slider(
        "Top-k retrieval", 1, 10, 4, 1,
        help="Higher k may increase recall but add noise + cost."
    )

    st.divider()

    st.subheader("Storage")
    docs_path = st.text_input("Docs folder", value="data/docs")
    persist_dir = st.text_input("FAISS persist dir", value="data/faiss_db")
    collection_name = st.text_input("Collection", value="demo_docs")

    st.divider()

    # Pre-fill from env just to help local use
    env_key = ""
    try:
        env_key = get_env("OPENAI_API_KEY", "")
    except Exception:
        env_key = ""

    # If key already stored in session use it; else try env; else blank
    default_key = st.session_state.get("OPENAI_API_KEY") or env_key

    api_key_input = st.text_input(
        "OPENAI_API_KEY",
        value=default_key,
        type="password",
        help="For Streamlit Cloud: prefer Secrets. If not using Secrets, paste key here (stored only for this session).",
    )

    # If user typed a key, store it in session_state for this run/session
    if api_key_input:
        st.session_state["OPENAI_API_KEY"] = api_key_input

    # Now resolve the final key: secrets/env/session (auth.py) OR sidebar input
    api_key = api_key_input or get_api_key("OPENAI_API_KEY") or get_api_key("API_KEY")

    if api_key:
        st.markdown('<span class="pill">API key loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill">API key missing</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<div class='small-muted'>Tip: For a course demo, keep <b>chunk_size=800–1000</b>, "
        "<b>overlap=100–200</b>, and <b>top_k=3–5</b>.</div>",
        unsafe_allow_html=True,
    )

# Stop if key not provided anywhere
if not api_key:
    st.warning("Please add your OpenAI API key in Streamlit Secrets or paste it in the sidebar.")
    st.stop()

# Use the resolved key everywhere below
client = OpenAI(api_key=api_key)

# Cached resources
@st.cache_resource
def load_faiss_index(persist_dir: str, collection_name: str, api_key: str):
    # api_key included to avoid confusing cache behavior across environments
    return get_index_and_metadata(persist_dir, collection_name, api_key)

@st.cache_data
def cached_retrieve(question: str, k: int, persist_dir: str, collection_name: str, api_key: str):
    index, metadatas, _ = load_faiss_index(persist_dir, collection_name, api_key)
    return retrieve(index, metadatas, question, api_key, k=k)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant", "content":...}]
if "last_context" not in st.session_state:
    st.session_state.last_context = ""

# Top layout: Ingest + Health
top_left, top_right = st.columns([1.1, 0.9], vertical_alignment="top")

with top_left:
    st.markdown("### 1) Ingest documents")
    st.markdown("<div class='block'>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([0.35, 0.35, 0.3])
    with c1:
        ingest_click = st.button("Ingest / Re-ingest", use_container_width=True)
    with c2:
        clear_cache = st.button("Clear caches", use_container_width=True)
    with c3:
        clear_chat = st.button("Clear chat", use_container_width=True)

    if clear_cache:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caches cleared. (Reloading index on next query.)")

    if clear_chat:
        st.session_state.messages = []
        st.session_state.last_context = ""
        st.success("Chat cleared.")

    if ingest_click:
        with st.spinner("Embedding + storing chunks into FAISS..."):
            result = ingest(
                docs_path=docs_path,
                persist_dir=persist_dir,
                collection_name=collection_name,
                openai_api_key=api_key,   # ensure we use the resolved key
                chunk_size=chunk_size,
                overlap=overlap,
            )
        st.success(f"Done. Chunks added: {result.get('chunks_added', 0)}")
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        "<div class='small-muted'>Change chunk size / overlap and re-ingest to demonstrate how retrieval quality changes.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    st.markdown("### App status")
    st.markdown("<div class='block'>", unsafe_allow_html=True)

    st.write("**Specialization:**", domain_mode)
    st.write("**Model:**", model)
    st.write("**Chunking:**", f"{chunk_size} chars, overlap {overlap}")
    st.write("**Top-k:**", top_k)
    st.write("**Docs folder:**", docs_path)
    st.write("**FAISS dir:**", persist_dir)
    st.write("**Collection:**", collection_name)

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# Main Questions and Answers
left, right = st.columns([1.1, 0.9], vertical_alignment="top")

with left:
    st.markdown("### 2) Ask questions (RAG + FAISS)")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask something from your uploaded docs…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        try:
            with st.spinner("Retrieving relevant context..."):
                contexts = cached_retrieve(
                    question=question,
                    k=top_k,
                    persist_dir=persist_dir,
                    collection_name=collection_name,
                    api_key=api_key,  # resolved key
                )

            if not contexts:
                raise ValueError("No chunks found. Ingest docs first, or add files to your docs folder.")

            context_block = "\n\n---\n\n".join(contexts)
            st.session_state.last_context = context_block

            system_prompt = DOMAIN_SYSTEM_PROMPTS[domain_mode]
            user_prompt = RAG_USER_TEMPLATE.format(context=context_block, question=question)

            with st.spinner("Generating answer..."):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )

            answer = resp.choices[0].message.content.strip()

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

        except Exception as e:
            st.error(f"Error: {e}")

with right:
    st.markdown("### Retrieved context & sources")
    st.markdown("<div class='block'>", unsafe_allow_html=True)

    if st.session_state.last_context:
        st.text_area(
            "Context used for the most recent answer",
            value=st.session_state.last_context,
            height=520,
        )
        st.markdown(
            "<div class='small-muted'>Teaching moment: show learners how top-k and chunking changes this context block.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Ask a question to see the retrieved context here.")

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption(
    "Optimization checklist: tune chunking, top-k, prompt specialization, caching, embedding model, "
    "and consider reranking for higher precision. Using FAISS for fast similarity search."
)
