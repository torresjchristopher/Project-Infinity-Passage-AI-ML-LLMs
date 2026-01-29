"""
Document processing utilities
"""

import os
from typing import List, Tuple
from pathlib import Path
import hashlib


class DocumentProcessor:
    """Process and chunk documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_document(self, file_path: str) -> str:
        """Load document from file"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_ext == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            except ImportError:
                raise ImportError("PyPDF2 required for PDF support: pip install PyPDF2")
        elif file_ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def chunk_text(self, text: str, filename: str) -> List[Tuple[str, dict]]:
        """Split text into chunks with metadata"""
        chunks = []
        words = text.split()
        current_chunk = []
        chunk_number = 0
        
        for word in words:
            current_chunk.append(word)
            chunk_text = " ".join(current_chunk)
            
            if len(chunk_text.split()) >= self.chunk_size:
                # Save chunk
                chunk_id = self._generate_chunk_id(filename, chunk_number)
                metadata = {
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "chunk_number": chunk_number,
                }
                chunks.append((chunk_text, metadata))
                
                # Create overlap
                overlap_words = self.chunk_overlap // 4  # Rough estimate
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                chunk_number += 1
        
        # Save remaining
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = self._generate_chunk_id(filename, chunk_number)
            metadata = {
                "filename": filename,
                "chunk_id": chunk_id,
                "chunk_number": chunk_number,
            }
            chunks.append((chunk_text, metadata))
        
        # Update total chunks
        for chunk, metadata in chunks:
            metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    @staticmethod
    def _generate_chunk_id(filename: str, chunk_number: int) -> str:
        """Generate unique chunk ID"""
        hash_input = f"{filename}_{chunk_number}".encode()
        hash_hex = hashlib.md5(hash_input).hexdigest()[:8]
        return f"{Path(filename).stem}_{chunk_number}_{hash_hex}"
    
    def process_document(self, file_path: str) -> List[Tuple[str, dict]]:
        """Complete document processing pipeline"""
        filename = Path(file_path).name
        text = self.load_document(file_path)
        chunks = self.chunk_text(text, filename)
        return chunks


class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except ImportError:
            raise ImportError("openai library required: pip install openai")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            # Sort by index to maintain order
            embeddings = [None] * len(texts)
            for item in response.data:
                embeddings[item.index] = item.embedding
            
            return embeddings
        except ImportError:
            raise ImportError("openai library required: pip install openai")


from typing import Optional
