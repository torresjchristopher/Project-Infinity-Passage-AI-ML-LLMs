"""
RAG System FastAPI Backend
Production-grade retrieval-augmented generation API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
from typing import List, Optional
from datetime import datetime
import json

from app.schemas.models import (
    QueryRequest, MessageResponse, DocumentInput, ConversationResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Production-grade Retrieval-Augmented Generation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (use database in production)
conversations: dict = {}
documents: list = []


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "service": "RAG System API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Read file
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Store document
        doc_id = str(uuid.uuid4())
        doc_entry = {
            "id": doc_id,
            "filename": file.filename,
            "size": len(content),
            "uploaded_at": datetime.now().isoformat(),
            "chunks": 5  # Placeholder
        }
        documents.append(doc_entry)
        
        return {
            "success": True,
            "document_id": doc_id,
            "filename": file.filename,
            "size": len(content),
            "message": "Document uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Initialize conversation if new
        if conversation_id not in conversations:
            conversations[conversation_id] = {
                "id": conversation_id,
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
        
        # Simulate RAG response (in production, use actual RAG chain)
        answer = f"This is a simulated response to: '{request.query}'"
        
        sources = [
            {
                "filename": "document1.pdf",
                "chunk": 2,
                "relevance": 0.95
            }
        ]
        
        # Store message in conversation
        message = {
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().isoformat(),
            "sources": sources
        }
        conversations[conversation_id]["messages"].append(message)
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "answer": answer,
            "sources": sources,
            "confidence": 0.92,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversations[conversation_id]


@app.post("/conversation/{conversation_id}/clear")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    if conversation_id in conversations:
        conversations[conversation_id]["messages"] = []
        return {"success": True, "message": "Conversation cleared"}
    
    raise HTTPException(status_code=404, detail="Conversation not found")


@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    return {
        "count": len(documents),
        "documents": documents
    }


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document"""
    global documents
    documents = [doc for doc in documents if doc["id"] != doc_id]
    
    return {"success": True, "message": f"Document {doc_id} deleted"}


@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for streaming responses"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process query
            query = message_data.get("query")
            
            # Simulate streaming response
            response = f"Processing query: {query}"
            
            # Send response
            await websocket.send_json({
                "type": "response",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
