# 🤖 Retrieval-Augmented Generation (RAG) Project

## 📌 Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers user queries by retrieving relevant information from a PDF document and generating responses using a language model.

---

## 🎯 Objective

To build an intelligent system that:

* Retrieves relevant content from documents
* Generates accurate answers using an LLM
* Combines retrieval + generation for better results

---

## 🚀 Features

* 📄 PDF document loading and processing
* ✂️ Text chunking for better retrieval
* 🔎 Semantic search using embeddings
* 🤖 LLM-based answer generation
* 📊 Improved response accuracy using RAG

---

## 🛠️ Technologies Used

* Python
* LangChain
* OpenAI / LLM APIs
* Vector Database (for embeddings)

---

## 📁 Project Structure

```id="f9k2xp"
├── app.py                 # Main application file
├── requirements.txt      # Dependencies
├── paracetamol2.pdf      # Input document
```

---

## ▶️ How to Run

### 1. Install dependencies

```id="m4z8qp"
pip install -r requirements.txt
```

### 2. Run the application

```id="t8n3vx"
python app.py
```

---

## 🔄 Workflow

1. Load PDF document
2. Split text into chunks
3. Convert text into embeddings
4. Store embeddings in vector database
5. Retrieve relevant chunks based on query
6. Generate response using LLM

---

## 📌 Example Use Case

* Ask questions about the PDF (e.g., medicine details)
* System retrieves relevant content
* Generates accurate answer using LLM

---

## 👩‍💻 Author

**Nageshwari**
B.Tech Computer Science Engineering

---

## ⭐ Note

This project demonstrates the practical implementation of RAG (Retrieval-Augmented Generation), a key concept in modern Generative AI systems.
