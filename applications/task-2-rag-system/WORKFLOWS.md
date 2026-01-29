# Sample RAG System Query Workflows

## Workflow 1: Academic Paper Analysis

### Scenario
You have uploaded a research paper on "Attention is All You Need" and want to understand the key concepts.

### Conversation Flow

**Query 1**: "What is the main contribution of this paper?"
```
Expected Response:
The main contribution is the introduction of the Transformer architecture, which relies entirely 
on self-attention mechanisms rather than recurrence or convolution. This new architecture outperforms 
previous sequence transduction models on machine translation tasks.
```

**Query 2**: "How does the self-attention mechanism work?"
```
Expected Response:
Self-attention, also known as intra-attention, is a mechanism that relates different positions 
of a single sequence in order to compute a representation of the sequence. The attention mechanism 
uses queries, keys, and values to compute attention weights that determine how much each token 
should attend to every other token.
```

**Query 3**: "What are the advantages over RNNs?"
```
Expected Response:
The Transformer overcomes RNN limitations in several ways:
1. Parallelization - Unlike RNNs, all positions can be processed in parallel
2. Long-range dependencies - Constant path length between positions
3. Computational efficiency - More parallelizable operations
4. Better performance - Achieves state-of-the-art results on translation tasks
```

---

## Workflow 2: Technical Documentation Q&A

### Scenario
Product documentation is uploaded, and users ask technical questions.

### Example Conversation

**Query 1**: "How do I set up authentication?"
**Query 2**: "What are the available authentication methods?"
**Query 3**: "Can I use both OAuth and API keys simultaneously?"

---

## Workflow 3: Legal Document Review

### Scenario
Contract or legal documents are uploaded for analysis.

### Example Conversation

**Query 1**: "What are the key obligations in this agreement?"
**Query 2**: "What is the termination clause?"
**Query 3**: "What happens if there's a dispute?"

---

## Workflow 4: Multiple Document Analysis

### Scenario
Multiple research papers are uploaded. System performs cross-document analysis.

### Example Conversation

**Query 1**: "Compare the approaches in Paper A and Paper B"
**Query 2**: "Which methodology is more effective?"
**Query 3**: "How do these papers relate to recent findings?"

---

## API Integration Example: Python Client

```python
import requests
import json

BASE_URL = "http://localhost:8000"

class RAGClient:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.conversation_id = None
    
    def upload_document(self, file_path):
        """Upload a document to the RAG system"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f'{self.base_url}/upload', files=files)
            return response.json()
    
    def query(self, question):
        """Ask a question about uploaded documents"""
        payload = {
            'query': question,
            'conversation_id': self.conversation_id,
            'top_k': 5
        }
        response = requests.post(f'{self.base_url}/query', json=payload)
        return response.json()
    
    def get_conversation(self):
        """Get full conversation history"""
        response = requests.get(f'{self.base_url}/conversation/{self.conversation_id}')
        return response.json()
    
    def new_conversation(self):
        """Start a new conversation"""
        self.conversation_id = f"conv-{int(time.time())}"

# Usage
client = RAGClient()
client.new_conversation()

# Upload document
result = client.upload_document('research_paper.pdf')
print(f"Uploaded: {result['document_id']}")

# Ask questions
answer1 = client.query("What is the main topic of this paper?")
print(f"Answer: {answer1['answer']}")

answer2 = client.query("What are the key findings?")
print(f"Answer: {answer2['answer']}")

# View conversation
conversation = client.get_conversation()
print(f"Full conversation: {json.dumps(conversation, indent=2)}")
```

---

## CLI Usage Example

```bash
# Start backend
cd backend
python main.py

# In another terminal, start frontend
cd frontend
npm start

# Open browser
open http://localhost:3000
```

### Backend API Health Check
```bash
curl http://localhost:8000/health
# Response: {"status": "healthy"}
```

### Upload Document via CLI
```bash
curl -F "file=@document.pdf" http://localhost:8000/upload
```

### Query via CLI
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "conversation_id": "conv-123"
  }'
```

---

## Performance Considerations

### Document Sizes
- Small documents (< 1MB): ~1-2 seconds to process
- Medium documents (1-5MB): ~2-5 seconds to process
- Large documents (5-10MB): ~5-10 seconds to process

### Query Response Times
- Simple questions: ~1-2 seconds
- Complex questions requiring analysis: ~2-4 seconds
- Multi-turn conversations: ~1-3 seconds (faster due to caching)

### Optimization Tips
1. **Chunk size**: Larger chunks (1500 tokens) for high-level summaries
2. **Chunk size**: Smaller chunks (500 tokens) for detailed analysis
3. **Top-K**: Use top_k=3 for speed, top_k=5-10 for accuracy
4. **Memory**: Buffer memory for short conversations, summary for long ones

---

## Common Use Cases

### 1. Research and Academic
- Analyzing papers
- Comparing methodologies
- Understanding proofs and derivations

### 2. Business and Finance
- Contract analysis
- Financial report review
- Compliance checking

### 3. Product and Documentation
- API documentation assistance
- Code examples and snippets
- Troubleshooting guides

### 4. Content and Knowledge Management
- Knowledge base Q&A
- Customer support automation
- Internal wiki searching

### 5. Data Analysis
- Dataset documentation
- Data dictionary reference
- Analysis result interpretation
