"""
Harry Potter RAG API

This FastAPI:
1. Classifies the user's query using Groq
2. Retrieves relevant pages from Qdrant
3. Generates an answer using Gemini based only on retrieved context
4. Returns the answer together with the retrieved sources
"""

# ============================= Imports =============================

import os

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from qdrant_client import QdrantClient

from sentence_transformers import SentenceTransformer

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


# ============================= Setup =============================

# Load environment variables from the .env file
load_dotenv()

# Create the FastAPI application
app = FastAPI(title="Harry Potter RAG API")


# Enable CORS so the frontend can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================= Configuration =============================

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")


# Embedding model
# This is the same model that was used to create the embeddings stored in Qdrant
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large",
)


# Gemini configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Groq configuration
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Number of results retrieved from Qdrant
TOP_K = int(os.getenv("TOP_K"))


# ============================= Models and Clients =============================

# Loading the same embedding model used during the data preparation stage
model = SentenceTransformer(EMBEDDING_MODEL)


# Create Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# Gemini is used for the final RAG answer
gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)


# Groq is used for query routing and chitchat
groq_llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)


# ============================= Schemas =============================

class QueryRequest(BaseModel):
    """
    Request body received from the frontend
    """
    query: str


class Source(BaseModel):
    """
    Information about a retrieved source page
    """
    book_name: str
    page_number: int
    score: float


class QueryResponse(BaseModel):
    """
    Response returned to the frontend
    """
    query: str
    route: str
    answer: str
    sources: list[Source]


# ============================= Endpoints =============================

@app.get("/")
def root():
    """
    Basic endpoint to check that the API is running
    """
    return {
        "name": "Harry Potter RAG API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """
    Health-check endpoint used by the frontend
    to verify that the server is running
    """
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Main RAG endpoint

    """
    # ============================= Query Routing =============================

    # Groq classifies the user's query into exactly one route:
    # - retrieve: question related to the Harry Potter books
    # - chitchat: greetings, thanks, or casual conversation
    # - off-topic: unrelated questions

    ROUTER_SYSTEM_PROMPT = """
    You are the query router for a Harry Potter books chatbot.

    Your task is to classify the user's message into exactly one of these categories:

    retrieve - Use this category when the user is asking for information about the Harry Potter books, including characters, events, locations, objects, creatures, relationships, or story details

    chitchat - Use this category for greetings, thank-you messages, or casual conversation that does not require searching the books

    off-topic - Use this category when the message is unrelated to the Harry Potter books

    Return only the category name:
    retrieve
    chitchat
    or
    off-topic

    Do not provide explanations or any additional text"""



    route = groq_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=request.query),
    ]).text.strip().lower()


    # Make sure the router returns a valid route.
    if route not in {"retrieve", "chitchat", "off-topic"}:
        route = "off-topic"


    # ============================= Chitchat =============================

    # Chitchat does not need a database search
    # We simply ask Groq to respond in a friendly way

    if route == "chitchat":

        CHITCHAT_SYSTEM_PROMPT = """
        You are a friendly assistant for a Harry Potter book chatbot.

        Respond briefly and naturally to greetings, thanks, and casual conversation.

        Keep the response friendly, helpful, and concise.
        """


        response = groq_llm.invoke([
            SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
            HumanMessage(content=request.query),
        ])


        return QueryResponse(
            query=request.query,
            route=route,
            answer=response.text,
            sources=[],
        )


    # ============================= Off-topic =============================

    # If the question is unrelated to Harry Potter and there is no need to search Qdrant or call Gemini

    if route == "off-topic":

        return QueryResponse(
            query=request.query,
            route=route,
            answer="I can only answer questions about the Harry Potter books.",
            sources=[],
        )


    # ============================= Query Embedding =============================

    # The query embedded using the Same embedding model used when creating the vectors
    # "query:" is used because multilingual-e5-large distinguishes between queries and passages
    query_vector = model.encode(
        [f"query: {request.query}"],
        normalize_embeddings=True,
    )[0].tolist()


    # ============================= Retrieval from Qdrant =============================

    # Search for the TOP_K most similar pages in the vector database

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    ).points


    # ============================= Build Context =============================

    # Convert the retrieved pages into a context string that will be provided to Gemini

    context = "\n\n".join(
        f"Book: {result.payload['book_name']}\n"
        f"Page: {result.payload['page_number']}\n"
        f"Content: {result.payload['content']}"
        for result in results
    )


    # ============================= Answer Generation =============================

    # Gemini generates the final answer
    # The model is instructed to use only the retrieved context and If the context does not contain enough information
    # it will say "I do not know"

    RAG_SYSTEM_PROMPT = """
    Answer the user's question using only the provided context from the Harry Potter books.

    If the provided context does not contain enough information to answer the question, say exactly:

    I do not know

    Do not use outside knowledge.
    Do not make up or assume information that is not present in the context.

    Keep the answer concise and directly answer the user's question.
    """


    response = gemini_llm.invoke([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{request.query}"
        ),
    ])


    # ============================= Return Response =============================

    # Return:
    # - Original query
    # - Detected route
    # - Generated answer
    # - Retrieved sources

    return QueryResponse(
        query=request.query,
        route=route,
        answer=response.text,
        sources=[
            Source(
                book_name=result.payload["book_name"],
                page_number=result.payload["page_number"],
                score=result.score,
            )
            for result in results
        ],
    )