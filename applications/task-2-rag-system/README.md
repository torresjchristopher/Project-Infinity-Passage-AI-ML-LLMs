# 🤖 RAG System: Retrieval-Augmented Generation

**Production-Grade Full-Stack AI Application**

> *Intelligent Document Q&A with Streaming Responses, Memory Management, and Real-Time Chat*

## 🎯 Overview

A complete, production-ready Retrieval-Augmented Generation (RAG) system combining:
- 🧠 **Advanced LLM Integration** - GPT-4 with Langchain orchestration
- 📚 **Vector Database** - Pinecone/Weaviate for semantic search
- ⚡ **Real-Time Chat** - WebSocket streaming responses
- 💬 **Conversation Memory** - Context-aware multi-turn dialogues
- 📄 **Document Processing** - Intelligent chunking and embedding
- 🎨 **React Frontend** - Beautiful, responsive chat interface
- 🔐 **Security** - API authentication and document access control

## 🚀 Quick Start

### Prerequisites

```bash
# Required
- Python 3.9+
- Node.js 16+
- OpenAI API key
- Pinecone account (or alternative vector DB)
```

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=...
export PINECONE_ENVIRONMENT=gcp-starter

# Run server
python main.py
# Server runs at http://localhost:8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
# Opens at http://localhost:3000
```

## 🏗️ Architecture

### Backend Architecture

```
FastAPI Server
├── Document Upload & Processing
│   ├── Chunk generation
│   ├── Embedding generation
│   └── Vector indexing
├── RAG Chain (Langchain)
│   ├── Document retrieval
│   ├── Memory management
│   └── LLM integration
├── Conversation Management
│   └── Multi-turn dialogue
└── WebSocket Streaming
    └── Real-time responses
```

### Frontend Architecture

```
React Application
├── Chat Interface
│   ├── Message display
│   ├── Streaming responses
│   └── Source citations
├── Document Management
│   ├── Upload form
│   └── Document list
├── Conversation Management
│   └── History sidebar
└── Utilities
    ├── API client
    └── WebSocket manager
```

## 📋 API Endpoints

### Health & Status

```bash
GET /
GET /health
```

### Document Management

```bash
# Upload document
POST /upload
Content-Type: multipart/form-data
{file: <binary>}

# List documents
GET /documents

# Delete document
DELETE /documents/{doc_id}
```

### Query & Chat

```bash
# Single query
POST /query
{
  "query": "What is...",
  "conversation_id": "optional-id",
  "top_k": 3
}

# Get conversation history
GET /conversation/{conversation_id}

# Clear conversation
POST /conversation/{conversation_id}/clear
```

### Real-Time Streaming

```bash
# WebSocket endpoint
WS /ws/{conversation_id}

# Message format
{
  "query": "Your question here"
}
```

## 🔌 Integration Examples

### Example 1: Document Upload and Query

```python
import requests

# Upload document
files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/upload', files=files)
doc_id = response.json()['document_id']

# Query
query_data = {
    "query": "What is mentioned about X?",
    "conversation_id": "conv-123"
}
response = requests.post('http://localhost:8000/query', json=query_data)
print(response.json()['answer'])
```

### Example 2: Multi-Turn Conversation

```python
conversation_id = "conv-user-123"

# First turn
q1 = {"query": "What is RAG?", "conversation_id": conversation_id}
r1 = requests.post('http://localhost:8000/query', json=q1)

# Second turn (uses previous context)
q2 = {"query": "How does it improve search?", "conversation_id": conversation_id}
r2 = requests.post('http://localhost:8000/query', json=q2)

# Get full conversation
conv = requests.get(f'http://localhost:8000/conversation/{conversation_id}')
print(conv.json())
```

### Example 3: Real-Time Streaming with WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/conv-123');

ws.onopen = () => {
  ws.send(JSON.stringify({
    query: "Explain RAG systems"
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Streaming response:", data.content);
};
```

## 📊 RAG System Details

### How RAG Works

1. **Document Ingestion**
   ```
   PDF/TXT → Text Extraction → Chunking → Embedding → Vector Store
   ```

2. **Query Processing**
   ```
   User Query → Embedding → Vector Search → Retrieve Relevant Chunks
   ```

3. **Response Generation**
   ```
   Retrieved Context + Query + History → LLM → Grounded Response
   ```

### Key Components

#### Document Processing
- **Chunking**: Smart splitting with overlap for context preservation
- **Embedding**: OpenAI embeddings (text-embedding-3-small/large)
- **Indexing**: Fast vector search with Pinecone

#### Conversation Memory
- **Buffer Memory**: Keeps last N messages for context
- **Summary Memory**: Summarizes old conversations for efficiency
- **Hybrid**: Combines both for optimal performance

#### LLM Chain
- **Retrieval**: Top-k most relevant document chunks
- **Context**: Formatted with sources and citations
- **Generation**: GPT-4 for high-quality responses

## ⚙️ Configuration

### Backend Configuration

```python
# .env file
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=gcp-starter

# Advanced settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CONTEXT_LENGTH=4000
TEMPERATURE=0.7
MAX_TOKENS=1000
TOP_K_RETRIEVAL=5
MEMORY_MAX_MESSAGES=10
```

### Frontend Configuration

```javascript
// src/config.js
export const API_BASE_URL = 'http://localhost:8000';
export const WS_BASE_URL = 'ws://localhost:8000';
export const STREAMING_ENABLED = true;
export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
```

## 🔐 Security

### Authentication

```python
# Add authentication middleware
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/query")
async def query_documents(request: QueryRequest, credentials: HTTPAuthCredentials = Depends(security)):
    # Validate token
    token = credentials.credentials
    # ...
```

### File Validation

```python
# Only allow specific file types
ALLOWED_TYPES = ['.pdf', '.txt', '.md']

@app.post("/upload")
async def upload_document(file: UploadFile):
    if not file.filename.endswith(tuple(ALLOWED_TYPES)):
        raise HTTPException(400, "File type not allowed")
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query_documents(request: QueryRequest):
    # ...
```

## 📈 Performance

### Benchmarks

| Operation | Latency | Notes |
|-----------|---------|-------|
| Document Upload (1MB) | 2-3s | Includes embedding generation |
| Vector Search | 100-200ms | 5 results from index |
| LLM Response | 1-3s | Streaming enabled |
| Full Query | 2-4s | Total end-to-end |

### Optimization Tips

1. **Embedding Caching**
   ```python
   @cache
   def get_embeddings(text):
       return embeddings.embed_query(text)
   ```

2. **Batch Processing**
   ```python
   # Process multiple documents
   embeddings.embed_documents(documents)
   ```

3. **Vector Store Optimization**
   - Use namespaces for document separation
   - Index filtering by metadata
   - Lazy loading of large documents

## 📚 Examples

### Example Document: AI Research Paper

```markdown
# Attention is All You Need

## Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...

## Introduction
Recurrent neural networks, long short-term memory and gated recurrent neural networks...
```

### Example Conversation

**User**: "What is the main contribution of this paper?"  
**RAG**: "The paper introduces the Transformer architecture, which relies entirely on self-attention mechanisms rather than recurrence or convolution."

**User**: "How does it improve on previous approaches?"  
**RAG**: "The Transformer enables better parallelization and reaches new state-of-the-art performance on machine translation tasks..."

## 🐛 Troubleshooting

### Vector DB Connection Issues

```bash
# Test Pinecone connection
python -c "import pinecone; pinecone.init()"

# Check API key
echo $PINECONE_API_KEY
```

### LLM Response Errors

```bash
# Verify OpenAI API key
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

### WebSocket Connection Issues

```javascript
// Check connection
const ws = new WebSocket('ws://localhost:8000/ws/test');
ws.onopen = () => console.log('Connected');
ws.onerror = (e) => console.error('Error:', e);
```

## 🚀 Production Deployment

### Docker

```bash
# Build backend image
docker build -f docker/Dockerfile.backend -t rag-backend .

# Build frontend image
docker build -f docker/Dockerfile.frontend -t rag-frontend .

# Run with compose
docker-compose up
```

### Environment Variables for Production

```bash
# Security
SECRET_KEY=your-secret-key
ALLOWED_ORIGINS=https://yourdomain.com

# API Keys
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...

# Database
DATABASE_URL=postgresql://user:pass@db:5432/rag

# Vector Store
VECTOR_STORE_NAMESPACE=production
VECTOR_STORE_INDEX=rag-prod

# Performance
MAX_WORKERS=4
BATCH_SIZE=32
```

## 📖 Documentation

- **[Langchain Docs](https://docs.langchain.com/)** - Chain orchestration
- **[OpenAI API](https://platform.openai.com/docs/)** - LLM integration
- **[Pinecone Guide](https://docs.pinecone.io/)** - Vector database
- **[FastAPI Docs](https://fastapi.tiangolo.com/)** - Backend framework

## 🤝 Contributing

Areas for enhancement:
- Additional vector stores (Weaviate, Milvus, Qdrant)
- Multi-modal support (images, audio)
- Advanced retrieval (reranking, fusion)
- Analytics and monitoring
- Custom fine-tuning workflows

## 📄 License

MIT License

## ✉️ Support

Issues and questions:
- Check documentation
- Review example workflows
- Create GitHub issues
- Contact support

---

## 🎓 Learning Outcomes

After implementing this system, you'll understand:

✅ **RAG Architecture** - Complete understanding of retrieval-augmented generation  
✅ **LLM Integration** - How to integrate GPT-4 with Langchain  
✅ **Vector Databases** - Semantic search and embeddings  
✅ **Full-Stack AI** - Building production AI applications  
✅ **Real-Time Systems** - WebSocket streaming and async processing  
✅ **Memory Management** - Multi-turn conversation context  
✅ **Production Readiness** - Security, scaling, monitoring  

---

**Built with ❤️ as part of Project Infinity Passage: AI/ML & LLMs Benchmarks**

*Master RAG systems. Lead the AI revolution. Transform how AI accesses knowledge.* 🚀🤖
