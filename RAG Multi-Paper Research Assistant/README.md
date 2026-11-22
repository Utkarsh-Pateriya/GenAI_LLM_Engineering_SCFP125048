# RAG Multi-Paper Research Assistant

## Overview
This project focuses on building a Retrieval-Augmented Generation (RAG) assistant that can answer questions using multiple research papers. It retrieves relevant information from COVID-19 vaccine research papers and provides answers with paper names and page number citations.

## Key Features
- Load and process multiple research paper PDFs.
- Extract paper titles, page numbers, and document content.
- Split documents into meaningful chunks for better retrieval.
- Store embeddings in a persistent Chroma vector database.
- Support single-paper and cross-paper search modes.
- Generate answers with source and page citations.
- Prevent the model from following instructions found inside retrieved documents.
- Evaluate the assistant using factual, comparative, and out-of-scope questions.

## Technologies Used
- Python
- LangChain
- Chroma DB
- Hugging Face Embeddings
- Groq API (ChatGroq)
- Docling
- Kaggle Notebook

## Learning Outcome
Learned how to build a multi-document RAG system with semantic search, persistent vector storage, paper-specific retrieval, cross-paper comparison, source citations, and answer evaluation.
