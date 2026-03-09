import os
import re
import random
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag_chain import build_chain, is_greeting, is_goodbye, get_random_greeting, IRRELEVANT_RESPONSE

load_dotenv()

# Global chain instance
chat_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG chain on startup"""
    global chat_chain
    print("Loading RAG chain...")
    chat_chain, _ = build_chain()
    print("RAG chain loaded successfully!")
    yield

    chat_chain = None

app = FastAPI(
    title="KIIT Nexus Chatbot API",
    description="REST API for KIIT University and KIIT Nexus chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation tracking")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Chatbot's response")
    session_id: str = Field(..., description="Session ID for the conversation")
    sources: Optional[list[str]] = Field(None, description="Source documents used for the response")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to KIIT Nexus Chatbot API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the API is healthy and RAG chain is loaded"""
    if chat_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not loaded")
    return HealthResponse(
        status="healthy",
        message="KIIT Nexus Chatbot API is running"
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint for interacting with the chatbot.
    
    - **message**: User's query or message
    - **session_id**: Optional session identifier for conversation continuity
    
    Returns the chatbot's response along with source documents if available.
    """
    user_message = request.message.strip()
    session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    # Handle greetings
    if is_greeting(user_message):
        return ChatResponse(
            response=get_random_greeting(),
            session_id=session_id,
            sources=[]
        )
    
    # Handle goodbyes
    if is_goodbye(user_message):
        return ChatResponse(
            response="Goodbye! Feel free to return if you have more questions about KIIT. Have a great day!",
            session_id=session_id,
            sources=[]
        )
    
    try:
        # Invoke the RAG chain
        result = chat_chain.invoke({"question": user_message})
        
        # Extract response and sources
        response_text = result.get("answer", "I apologize, but I couldn't generate a response. Please try again.")
        source_documents = result.get("source_documents", [])
        
        # Extract unique source metadata
        sources = []
        seen_sources = set()
        for doc in source_documents:
            source_info = doc.metadata.get("source", "Unknown")
            if source_info not in seen_sources:
                sources.append(source_info)
                seen_sources.add(source_info)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            sources=sources[:5]  # Limit to 5 unique sources
        )
        
    except Exception as e:
        print(f"Error processing chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your request: {str(e)}"
        )


@app.get("/greeting", tags=["Utilities"])
async def get_greeting():
    """Get a random greeting from the chatbot"""
    return {
        "greeting": get_random_greeting()
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=port,
        reload=(port == 8000)
    )
