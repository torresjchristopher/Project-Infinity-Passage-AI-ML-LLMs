"""
RAG System Data Models and Schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# Request/Response Models

class DocumentInput(BaseModel):
    """Document upload schema"""
    filename: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    """Query request schema"""
    query: str
    conversation_id: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)
    use_memory: bool = True


class MessageResponse(BaseModel):
    """Chat message response"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = None
    confidence: Optional[float] = None


class ConversationResponse(BaseModel):
    """Conversation history"""
    conversation_id: str
    messages: List[MessageResponse]
    created_at: datetime
    updated_at: datetime


class DocumentMetadata(BaseModel):
    """Document metadata"""
    filename: str
    chunk_id: str
    chunk_number: int
    total_chunks: int
    source_url: Optional[str] = None
    upload_date: datetime


class EmbeddingRequest(BaseModel):
    """Embedding request"""
    texts: List[str]
    model: str = "text-embedding-3-small"


class RAGConfig(BaseModel):
    """RAG system configuration"""
    vector_store_type: str = "pinecone"  # pinecone, weaviate, milvus
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_length: int = 4000
    temperature: float = 0.7
    max_tokens: int = 1000
    top_k_retrieval: int = 5
    memory_type: str = "conversation"  # conversation, summary
    memory_max_messages: int = 10
