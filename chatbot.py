import os
import tempfile
import streamlit as st
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")


st.set_page_config(page_title="Fusion RAG Multi-PDF Chatbot", layout="wide")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "bm25_docs_text" not in st.session_state:
    st.session_state.bm25_docs_text = []

if "bm25_docs_meta" not in st.session_state:
    st.session_state.bm25_docs_meta = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

with st.sidebar:
    st.title("📂 Upload & Process PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
    process_clicked = st.button("Process PDFs")

if process_clicked:
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            try:
                documents = []
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in st.session_state.processed_files:
                        continue
                    # Save temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_path = temp_file.name

                    try:
                        loader = PyMuPDFLoader(temp_path)
                        pdf_docs = loader.load()
                        if not pdf_docs:
                            st.warning(f"No text found in {uploaded_file.name}. Skipping.")
                            continue

                        for doc in pdf_docs:
                            if not doc.page_content.strip():
                                continue
                            doc.metadata["source"] = uploaded_file.name
                            documents.append(doc)

                        st.session_state.processed_files.append(uploaded_file.name)

                    except Exception as e:
                        st.warning(f"Failed to process {uploaded_file.name}: {e}")
                    finally:
                        os.unlink(temp_path)

                if not documents:
                    st.error("No valid text extracted from uploaded PDFs.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    st.session_state.chunks = chunks
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    st.session_state.vector_store = vector_store
                    documents_text = [doc.page_content for doc in chunks]
                    tokenized_corpus = [doc.lower().split() for doc in documents_text]
                    bm25 = BM25Okapi(tokenized_corpus)
                    st.session_state.bm25 = bm25
                    st.session_state.bm25_docs_text = documents_text
                    st.session_state.bm25_docs_meta = [doc.metadata for doc in chunks]

                    st.success(f"Processed {len(st.session_state.processed_files)} PDFs into {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Error during PDF processing: {e}")
    else:
        st.warning("Please upload at least one PDF.")


if st.session_state.processed_files:
    st.subheader("Processed Files:")
    for idx, fname in enumerate(st.session_state.processed_files):
        st.write(f"{idx+1}. {fname}")

st.title("💬 Chat with your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.markdown(f"**Document:** {src['title']}")
                    snippet = src['content'][:300].replace("\n", " ") + "..."
                    st.markdown(f"**Excerpt:** {snippet}")


if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.vector_store or not st.session_state.bm25:
        st.error("Please upload and process PDFs first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

                    dense_retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                    dense_docs = dense_retriever.get_relevant_documents(prompt)

                    query_tokens = prompt.lower().split()
                    bm25_top_n_texts = st.session_state.bm25.get_top_n(query_tokens, st.session_state.bm25_docs_text, n=5)

                    bm25_docs = []
                    bm25_text_set = set()
                    for text in bm25_top_n_texts:
                        if text not in bm25_text_set:
                            bm25_text_set.add(text)
                            idx = st.session_state.bm25_docs_text.index(text)
                            meta = st.session_state.bm25_docs_meta[idx]
                            class DummyDoc:
                                def __init__(self, content, metadata):
                                    self.page_content = content
                                    self.metadata = metadata
                            bm25_docs.append(DummyDoc(text, meta))
                    combined_docs = []
                    seen_texts = set()
                    for doc in dense_docs + bm25_docs:
                        if doc.page_content not in seen_texts:
                            combined_docs.append(doc)
                            seen_texts.add(doc.page_content)
                        if len(combined_docs) >= 5:
                            break

                    if not combined_docs:
                        response = "Sorry, I couldn't find any relevant information in the uploaded documents."
                        sources = []
                    else:
                        context_text = "\n\n".join(doc.page_content for doc in combined_docs)
                        template = """
You are an assistant for question-answering tasks. Use ONLY the following context to answer the user's question.
If the answer is not contained within the context, respond: 'Sorry I couldn't provide any relevant information in the uploaded documents.'
Do NOT attempt to answer from outside knowledge.

Context:
{context}

Question: {question}

Provide a detailed and accurate answer based ONLY on the above context.
"""
                        prompt_template = ChatPromptTemplate.from_template(template)

                        rag_chain = (
                            {"context": lambda x: context_text, "question": RunnablePassthrough()}
                            | prompt_template
                            | llm
                            | StrOutputParser()
                        )

                        response = rag_chain.invoke(prompt)

                        sources = []
                        for doc in combined_docs[:3]:  # Show top 3 sources
                            sources.append({
                                "title": doc.metadata.get("source", "Unknown"),
                                "content": doc.page_content
                            })

                    st.write(response)

                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.markdown(f"**Document:** {src['title']}")
                                snippet = src['content'][:300].replace("\n", " ") + "..."
                                st.markdown(f"**Excerpt:** {snippet}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
