import streamlit as st
import tempfile
import os
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="RAG Document QA", layout="wide")
st.title(" Document Question Answering System (RAG)")
st.write("Upload a PDF and ask questions based on the document.")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.success("PDF uploaded successfully!")

    # -----------------------------
    # Load PDF
    # -----------------------------
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    st.write(f"Total Pages Loaded: {len(documents)}")

    # -----------------------------
    # Split Text into Chunks
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    st.write(f"Total Chunks Created: {len(chunks)}")

    # -----------------------------
    # Create Embeddings
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # Create FAISS Vector Store
    # -----------------------------
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success("Vector database created successfully!")

    # -----------------------------
    # Load GPT-2 Model
    # -----------------------------
    device = 0 if torch.cuda.is_available() else -1

    model_name = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=200,   #  Only controls output length
    do_sample=True,
    temperature=0.7
)

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # -----------------------------
    # Create RetrievalQA Chain
    # -----------------------------
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # -----------------------------
    # Ask Question
    # -----------------------------
    query = st.text_input("Ask your question:")

    if query:

        with st.spinner("Generating answer..."):
            response = qa_chain.invoke({"query": query})
            answer = response["result"]

        st.subheader("Final Answer")
        st.write(answer)

        st.subheader("Retrieved Chunks with Similarity Score")

        docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)

        for i, (doc, score) in enumerate(docs_with_scores):
            st.markdown(f"###  Chunk {i+1}")
            st.write(doc.page_content)
            st.write(f"Similarity Score: {score}")
            st.markdown("---")

    # Delete temporary file
    os.remove(temp_path)
