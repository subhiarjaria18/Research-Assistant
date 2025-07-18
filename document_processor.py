import os
import re
import PyPDF2
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from config import Config
import uuid
import json

class DocumentProcessor:
    def __init__(self):
        self.config = Config()
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=self.config.VECTOR_STORE_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into chunks"""
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap or self.config.CHUNK_OVERLAP
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citations from text"""
        # Pattern for common citation formats
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\(([^)]+,\s*\d{4})\)',  # (Author, 2023)
            r'\b([A-Z][a-z]+\s+et\s+al\.,\s*\d{4})\b',  # Author et al., 2023
            r'\b([A-Z][a-z]+\s+and\s+[A-Z][a-z]+,\s*\d{4})\b',  # Author and Author, 2023
            r'\b([A-Z][a-z]+,\s*\d{4})\b'  # Author, 2023
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from research paper"""
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'conclusion': '',
            'references': ''
        }
        
        # Define section patterns
        section_patterns = {
            'abstract': r'(?i)abstract\s*(.+?)(?=\n\n|\nintroduction|\n1\.|\nkeywords)',
            'introduction': r'(?i)(?:introduction|1\.?\s*introduction)\s*(.+?)(?=\n\n|\n(?:2\.|methodology|method|literature))',
            'methodology': r'(?i)(?:methodology|methods?|2\.?\s*(?:methodology|methods?))\s*(.+?)(?=\n\n|\n(?:3\.|results|findings))',
            'results': r'(?i)(?:results|findings|3\.?\s*(?:results|findings))\s*(.+?)(?=\n\n|\n(?:4\.|discussion|conclusion))',
            'conclusion': r'(?i)(?:conclusion|conclusions?|4\.?\s*conclusion)\s*(.+?)(?=\n\n|\nreferences|\nbibliography)',
            'references': r'(?i)(?:references|bibliography)\s*(.+?)(?=\n\n|\Z)'
        }
        
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[section] = match.group(1).strip()
        
        return sections
    
    def store_document(self, file_path: str, metadata: Dict[str, Any] = None):
        """Store document in vector database"""
        # Extract text
        text = self.extract_text_from_pdf(file_path)
        
        # Extract sections
        sections = self.extract_sections(text)
        
        # Extract citations
        citations = self.extract_citations(text)
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        # Create collection
        collection_name = f"doc_{uuid.uuid4().hex[:8]}"
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"source": file_path}
        )
        
        # Store chunks with embeddings
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_id": i,
                "source": file_path,
                "citations": json.dumps(citations),
                "sections": json.dumps(sections),
                **(metadata or {})
            }
            
            collection.add(
                documents=[chunk],
                metadatas=[chunk_metadata],
                ids=[f"{collection_name}_{i}"]
            )
        
        return collection_name, sections, citations
    
    def search_documents(self, query: str, collection_name: str = None, n_results: int = 5) -> List[Dict]:
        """Search documents in vector database"""
        if collection_name:
            collections = [self.client.get_collection(collection_name)]
        else:
            collections = [self.client.get_collection(col.name) for col in self.client.list_collections()]
        
        results = []
        for collection in collections:
            try:
                search_results = collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
                
                for i, doc in enumerate(search_results['documents'][0]):
                    results.append({
                        'content': doc,
                        'metadata': search_results['metadatas'][0][i],
                        'distance': search_results['distances'][0][i] if 'distances' in search_results else None
                    })
            except Exception as e:
                print(f"Error searching collection {collection.name}: {e}")
        
        return sorted(results, key=lambda x: x['distance'] or 0)[:n_results]