# Lab 7 – Agentic RAG on Union Budget 2026-27

## Overview
This project focuses on building an Agentic Retrieval-Augmented Generation (RAG) system using the Union Budget 2026-27 speech PDF. It allows users to ask questions and receive answers based on relevant information retrieved from the document.

## Key Features
- Load, clean, and split the Union Budget PDF into chunks.
- Store document embeddings in a persistent Chroma vector database.
- Retrieve relevant information using semantic search.
- Use an LLM-powered agent to decide when document search is required.
- Provide answers with page number citations.
- Use a calculator tool for safe arithmetic operations.
- Maintain conversation history through an interactive chat loop.
- Evaluate the agent using predefined test queries.

## Technologies Used
- Python
- LangChain
- Chroma DB
- Hugging Face Embeddings
- Groq API (ChatGroq)
- Kaggle Notebook

## Learning Outcome
Learned how to build an Agentic RAG system using document retrieval, vector databases, LLM tool calling, conversation history, and calculator tools to answer questions from the Union Budget document.
