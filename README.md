# RAG-Powered-Assistant-Project
# Harry Potter RAG-Powered Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers natural language questions based on the seven Harry Potter books

The project processes the books, creates page-level embeddings, stores them in Qdrant, retrieves the most relevant pages for a user query, and uses Gemini to generate an answer based only on the retrieved context and uses groq to decide the route of the user question

## Project Overview

The system consists of three main components:

1. Data preparation and RAG pipeline using Jupyter Notebook
2. FastAPI backend for query processing and RAG execution
3. frontend for user interaction

## Features

1. Questions and answers based on all seven Harry Potter books
2.  Page-level document chunking
3.  Semantic search using embeddings
4.  Vector storage and retrieval with Qdrant
5.  Query routing using Groq
6.  RAG answer generation using Gemini
7.  Retrieved source pages returned with each answer
8.  Chitchat handling
9.  Off-topic query handling
10.  FastAPI health-check endpoint
11.  Custom frontend
12.  Retrieval evaluation using Precision and Recall
13.  Use llm to judge another llm answer

## Dataset

The project uses the seven Harry Potter books combined into a single PDF

The dataset has the following characteristics:

- Source: Seven Harry Potter books
- Format: PDF
- Total pages: 3623
- Page chunks: 3568

The PDF is converted to Markdown and then split into individual page-level chunks

Each chunk contains:

- Book name
- Page number
- Page content

## RAG Pipeline

The RAG pipeline consists of several stages

### 1. Document Preparation

PyMuPDF is used to extract the text from the PDF and convert it into Markdown

Each page is identified using its page number

### 2. Chunking

The document is split page by page

This allows the system to preserve the original book and page information and return it as a source

### 3. Embeddings

The project uses the ( intfloat/multilingual-e5-large ) embedding model

Each page is converted into a 1024-dimensional vector

Passages are embedded using:

```text
passage: {content}
```

User queries are embedded using:

```text
query: {query}
```

### 4. Vector Database

The embeddings are stored in Qdrant using cosine similarity

The collection contains 3568 page embeddings

Each stored vector contains metadata including:

- Book name
- Page number
- Page content

### 5. Query Routing

Groq classifies each user message into one of three routes:

- retrieve
- chitchat
- off-topic

The retrieve ->  route is used for Harry Potter-related questions.

The chitchat -> route handles greetings, thanks, and casual conversation.

The off-topic -> route handles questions unrelated to the Harry Potter books.

### 6. Retrieval

For a ( retrieve query ), the question is converted into an embedding using the same embedding model.

Qdrant performs semantic vector search and retrieves the most relevant page-level chunks based on cosine similarity.

The number of retrieved results is controlled using the ( TOP_K ) environment variable.

### 7. Context Construction

The retrieved pages are combined into a context containing the relevant book names, page numbers, and text content.

This context is provided to the answer generation model.

### 8. Answer Generation

Gemini generates the final answer using only the retrieved context.

The model is instructed not to use outside knowledge, make assumptions, or generate unsupported information.

If the retrieved context is not sufficient to answer the question, Gemini is instructed to respond with:

```text
I do not know
```

### 9. Response

The FastAPI backend returns:

- The original query
- The detected route
- The generated answer
- The retrieved sources

Each source contains:

- Book name
- Page number
- Similarity score

The frontend displays the generated answer and its supporting sources.

## Technologies

### Programming Language

- Python

### Document Processing

- PyMuPDF

### Embeddings

- Sentence Transformers
- intfloat/multilingual-e5-large

### Vector Database

- Qdrant

### Language Models

- Groq for query routing and chitchat
- Gemini for RAG answer generation

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Frontend

## Running the Project

### Step 1: Prepare the Data

Place the source PDF in the project environment:

( harrypotter.pdf )

And The rag_pipeline.ipynb notebook processes the PDF, creates page-level chunks, generates embeddings, stores them in Qdrant, and evaluates retrieval performance.

### Step 2: Start the FastAPI Server

### Step 3: Open the API Documentation

FastAPI provides interactive Swagger documentation

### Step 4: Run the Frontend

Open the frontend in a browser while the FastAPI server is running.

The frontend sends user queries to the ( /query ) endpoint and displays the generated answer and retrieved sources.

## API Endpoints

### GET `/`

Checks that the API is running.

The endpoint returns the API name and the docs

### GET `/health`

Returns the server health status

### POST `/query`

Processes a user query through the RAG pipeline.

Example request:


{
  "query": "Who is Harry Potter's godfather?"
}

The response contains:

- query
- route
- answer
- sources

Each source contains:

- book_name
- page_number
- score

## Evaluation

The retrieval system was evaluated using manually defined ground-truth questions and calculate the Precision and Recall.

The project also includes answer-grounding evaluation using llm as a judge to check whether generated answers are supported by the retrieved context.
