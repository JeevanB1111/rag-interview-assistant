# 🚀 RAG Interview Assistant

A production-style Retrieval-Augmented Generation (RAG) system built using FastAPI, FAISS, SentenceTransformers, and Gemini API for domain-specific CS interview Q&A.

---

## 📌 Overview

This project implements an end-to-end RAG pipeline to answer computer science interview questions using semantic search and grounded LLM responses.

Instead of directly prompting an LLM, the system retrieves relevant document chunks using vector similarity search and provides context-aware, hallucination-controlled answers.

---

## 🏗 Architecture

```mermaid
flowchart TD

A[User Query from Web UI]
B[FastAPI Backend /ask Route]
C[SentenceTransformer Embedding]
D[FAISS Vector Database]
E[Top Relevant Chunks]
F[Gemini LLM - Context Grounded Prompt]
G[Final Answer Displayed]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
