"""
RAG Chain Implementation using Langchain
Core retrieval-augmented generation logic
"""

from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import os
from datetime import datetime


class RAGChain:
    """
    Retrieval-Augmented Generation Chain
    Combines document retrieval with LLM for grounded question answering
    """
    
    def __init__(
        self,
        vector_store,
        llm_model: str = "gpt-4",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        memory_type: str = "conversation"
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_type = memory_type
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize memory
        self.memory = self._initialize_memory()
        
        # Initialize RAG chain
        self.chain = self._build_chain()
    
    def _initialize_memory(self):
        """Initialize conversation memory"""
        if self.memory_type == "conversation":
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                k=10  # Keep last 10 messages
            )
        elif self.memory_type == "summary":
            return ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history"
            )
        else:
            raise ValueError(f"Unknown memory type: {self.memory_type}")
    
    def _build_chain(self):
        """Build the RAG chain"""
        # Custom prompt template with sources
        prompt_template = """
        Use the following context and conversation history to answer the question.
        If you don't know the answer, say so. Include relevant sources.
        
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Answer: Provide a detailed, accurate answer with citations.
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create retrieval QA chain
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return chain
    
    def query(self, question: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            conversation_id: Optional conversation ID for context
            
        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            result = self.chain({"query": question})
            
            # Extract sources
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    sources.append({
                        "content": doc.page_content[:500],
                        "metadata": doc.metadata
                    })
            
            return {
                "success": True,
                "answer": result["result"],
                "sources": sources,
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": None,
                "sources": []
            }
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
    
    def get_memory_summary(self) -> str:
        """Get summary of conversation memory"""
        if hasattr(self.memory, 'buffer'):
            return self.memory.buffer
        elif hasattr(self.memory, 'summary'):
            return self.memory.summary
        return ""


class DocumentIndexer:
    """
    Index documents in vector store
    """
    
    def __init__(self, vector_store, embedding_model: str = "text-embedding-3-small"):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def index_documents(self, documents: List[Document]) -> bool:
        """Index documents in vector store"""
        try:
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vector_store.add_documents(split_docs)
            
            return True
        except Exception as e:
            print(f"Error indexing documents: {e}")
            return False
    
    def index_texts(self, texts: List[str], metadatas: List[Dict] = None) -> bool:
        """Index raw texts in vector store"""
        try:
            embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Create documents from texts
            documents = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(texts, metadatas or [{}] * len(texts))
            ]
            
            # Split and index
            split_docs = self.text_splitter.split_documents(documents)
            self.vector_store.add_documents(split_docs)
            
            return True
        except Exception as e:
            print(f"Error indexing texts: {e}")
            return False
