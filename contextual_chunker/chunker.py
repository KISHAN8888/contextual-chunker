"""
Document Chunker Module
Handles PDF processing, text extraction, chunking, and context generation.
Saves results to JSON for later use in retrieval.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    
    # API Keys
    openai_api_key: Optional[str] = None
    
    # Context generation settings
    context_provider: str = "openai"  # "openai"
    context_model: str = "gpt-4o-mini"
    max_context_tokens: int = 1000
    context_temperature: float = 0.0
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "recursive"  # "recursive" or "semantic"
    
    # Processing settings
    parallel_threads: int = 5
    
    # Output settings
    output_dir: str = "./chunked_documents"
    save_contexts: bool = True


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    content: str
    doc_id: str
    chunk_id: str
    original_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass 
class ContextualChunk(DocumentChunk):
    """Extended chunk with contextual information."""
    contextualized_content: str = ""
    combined_content: str = ""
    context_generation_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PDFProcessor:
    """Handles PDF text extraction."""
    
    def __init__(self, use_pymupdf: bool = True):
        self.use_pymupdf = use_pymupdf
        
        if use_pymupdf:
            try:
                import fitz
                self.fitz = fitz
            except ImportError:
                print("PyMuPDF not available, falling back to PyPDF2")
                self.use_pymupdf = False
        
        if not self.use_pymupdf:
            try:
                import PyPDF2
                self.PyPDF2 = PyPDF2
            except ImportError:
                raise ImportError("Either PyMuPDF or PyPDF2 is required for PDF processing")
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF."""
        if self.use_pymupdf:
            return self._extract_with_pymupdf(file_path)
        else:
            return self._extract_with_pypdf2(file_path)
    
    def _extract_with_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF."""
        doc = self.fitz.open(file_path)
        text = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
            text += "\n\n"
        
        doc.close()
        return text.strip()
    
    def _extract_with_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2."""
        text = ""
        
        with open(file_path, 'rb') as file:
            pdf_reader = self.PyPDF2.PdfReader(file)
            
            for page in pdf_reader.pages:
                text += page.extract_text()
                text += "\n\n"
        
        return text.strip()
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF."""
        return {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'file_size': os.path.getsize(file_path),
            'processor': 'PyMuPDF' if self.use_pymupdf else 'PyPDF2'
        }


class TextProcessor:
    """Handles plain text file processing."""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from text file."""
        return {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'file_size': os.path.getsize(file_path),
            'processor': 'TextProcessor'
        }


class RecursiveTextSplitter:
    """Recursive character-based text splitter."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Split text into chunks."""
        chunks = self._split_text_recursive(text)
        
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                content=chunk_text.strip(),
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{i:04d}",
                original_index=i,
                metadata=metadata or {}
            )
            document_chunks.append(chunk)
        
        return document_chunks
    
    def _split_text_recursive(self, text: str) -> List[str]:
        """Recursively split text using different separators."""
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try different separators
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                
                chunks = []
                current_chunk = ""
                
                for split in splits:
                    test_chunk = current_chunk + (separator if current_chunk else "") + split
                    
                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        if len(split) > self.chunk_size:
                            # Recursively split long pieces
                            sub_chunks = self._split_text_recursive(split)
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = split
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                return chunks
        
        # Fallback: force split
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_end = min(i + self.chunk_size, len(text))
            chunks.append(text[i:chunk_end])
        
        return chunks


class SemanticTextSplitter:
    """Semantic-based text splitter that preserves meaning."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Split text semantically."""
        import re
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle long paragraphs by splitting on sentences
                if len(paragraph) > self.chunk_size:
                    sentences = re.split(r'[.!?]+', paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                            
                        test_sentence_chunk = temp_chunk + (". " if temp_chunk else "") + sentence.strip()
                        
                        if len(test_sentence_chunk) <= self.chunk_size:
                            temp_chunk = test_sentence_chunk
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk + ".")
                            temp_chunk = sentence.strip()
                    
                    if temp_chunk:
                        current_chunk = temp_chunk + "."
                    else:
                        current_chunk = ""
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Convert to DocumentChunk objects
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                content=chunk_text,
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{i:04d}",
                original_index=i,
                metadata=metadata or {}
            )
            document_chunks.append(chunk)
        
        return document_chunks


class OpenAIContextGenerator:
    """Generate context using OpenAI."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                 max_tokens: int = 1000, temperature: float = 0.0):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate_context(self, document: str, chunk: str) -> tuple[str, Dict[str, Any]]:
        """Generate context for a chunk."""
        
        system_prompt = "You are a helpful assistant that generates concise context for document chunks to improve search retrieval. Given a document and a specific chunk from that document, provide a short, succinct context that situates the chunk within the overall document. Answer only with the context and nothing else."
        
        user_prompt = f"""Document:
{document}

Chunk to contextualize:
{chunk}

Provide a short context for this chunk within the document:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            context = response.choices[0].message.content.strip()
            
            usage_info = {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'model': self.model,
                'provider': 'openai'
            }
            
            return context, usage_info
            
        except Exception as e:
            print(f"Error generating context with OpenAI: {e}")
            context = f"This chunk is from the document and relates to the main document theme."
            usage_info = {'error': str(e), 'provider': 'openai'}
            return context, usage_info


class DocumentChunker:
    """Main document chunking class."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        
        # Initialize processors
        self.pdf_processor = PDFProcessor()
        self.text_processor = TextProcessor()
        
        # Initialize text splitter
        if config.chunking_strategy == "recursive":
            self.text_splitter = RecursiveTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        elif config.chunking_strategy == "semantic":
            self.text_splitter = SemanticTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {config.chunking_strategy}")
        
        # Initialize context generator
        if config.save_contexts:
            if config.context_provider == "openai":
                if not config.openai_api_key:
                    raise ValueError("OpenAI API key required for context generation")
                self.context_generator = OpenAIContextGenerator(
                    api_key=config.openai_api_key,
                    model=config.context_model,
                    max_tokens=config.max_context_tokens,
                    temperature=config.context_temperature
                )
            else:
                raise ValueError(f"Unknown context provider: {config.context_provider}")
        else:
            self.context_generator = None
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def process_pdf_files(self, pdf_paths: List[str], doc_ids: List[str] = None) -> Dict[str, Any]:
        """Process PDF files and return chunked results."""
        if doc_ids is None:
            doc_ids = [Path(path).stem for path in pdf_paths]
        
        if len(pdf_paths) != len(doc_ids):
            raise ValueError("Number of PDF paths and doc IDs must match")
        
        print(f"Processing {len(pdf_paths)} PDF files...")
        
        all_results = {
            'documents': [],
            'chunks': [],
            'total_chunks': 0,
            'processing_info': {
                'chunking_strategy': self.config.chunking_strategy,
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'context_generation': self.config.save_contexts,
                'context_provider': self.config.context_provider if self.config.save_contexts else None
            }
        }
        
        for pdf_path, doc_id in zip(pdf_paths, doc_ids):
            print(f"Processing {pdf_path}...")
            
            # Extract text and metadata
            text = self.pdf_processor.extract_text(pdf_path)
            metadata = self.pdf_processor.extract_metadata(pdf_path)
            
            # Create basic chunks
            chunks = self.text_splitter.split_text(text, doc_id, metadata)
            
            # Generate contexts if enabled
            if self.config.save_contexts and self.context_generator:
                contextual_chunks = self._add_contexts_to_chunks(text, chunks)
            else:
                contextual_chunks = [
                    ContextualChunk(
                        content=chunk.content,
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        original_index=chunk.original_index,
                        metadata=chunk.metadata,
                        contextualized_content="",
                        combined_content=chunk.content,
                        context_generation_info={}
                    ) for chunk in chunks
                ]
            
            # Store document info
            document_info = {
                'doc_id': doc_id,
                'source_file': pdf_path,
                'metadata': metadata,
                'full_text': text,
                'chunk_count': len(contextual_chunks)
            }
            
            all_results['documents'].append(document_info)
            all_results['chunks'].extend([chunk.to_dict() for chunk in contextual_chunks])
            all_results['total_chunks'] += len(contextual_chunks)
        
        return all_results
    
    def process_text_files(self, text_paths: List[str], doc_ids: List[str] = None) -> Dict[str, Any]:
        """Process text files and return chunked results."""
        if doc_ids is None:
            doc_ids = [Path(path).stem for path in text_paths]
        
        if len(text_paths) != len(doc_ids):
            raise ValueError("Number of text paths and doc IDs must match")
        
        print(f"Processing {len(text_paths)} text files...")
        
        all_results = {
            'documents': [],
            'chunks': [],
            'total_chunks': 0,
            'processing_info': {
                'chunking_strategy': self.config.chunking_strategy,
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'context_generation': self.config.save_contexts,
                'context_provider': self.config.context_provider if self.config.save_contexts else None
            }
        }
        
        for text_path, doc_id in zip(text_paths, doc_ids):
            print(f"Processing {text_path}...")
            
            # Extract text and metadata
            text = self.text_processor.extract_text(text_path)
            metadata = self.text_processor.extract_metadata(text_path)
            
            # Create basic chunks
            chunks = self.text_splitter.split_text(text, doc_id, metadata)
            
            # Generate contexts if enabled
            if self.config.save_contexts and self.context_generator:
                contextual_chunks = self._add_contexts_to_chunks(text, chunks)
            else:
                contextual_chunks = [
                    ContextualChunk(
                        content=chunk.content,
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        original_index=chunk.original_index,
                        metadata=chunk.metadata,
                        contextualized_content="",
                        combined_content=chunk.content,
                        context_generation_info={}
                    ) for chunk in chunks
                ]
            
            # Store document info
            document_info = {
                'doc_id': doc_id,
                'source_file': text_path,
                'metadata': metadata,
                'full_text': text,
                'chunk_count': len(contextual_chunks)
            }
            
            all_results['documents'].append(document_info)
            all_results['chunks'].extend([chunk.to_dict() for chunk in contextual_chunks])
            all_results['total_chunks'] += len(contextual_chunks)
        
        return all_results
    
    def process_directory(self, directory_path: str, file_extensions: List[str] = None, 
                         recursive: bool = True) -> Dict[str, Any]:
        """Process all documents in a directory."""
        if file_extensions is None:
            file_extensions = ['.pdf', '.txt', '.md']
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Find all files
        all_files = []
        pattern = "**/*" if recursive else "*"
        
        for ext in file_extensions:
            files = list(directory.glob(f"{pattern}{ext}"))
            all_files.extend(files)
        
        if not all_files:
            print(f"No files found with extensions {file_extensions} in {directory_path}")
            return {'documents': [], 'chunks': [], 'total_chunks': 0}
        
        # Separate by file type
        pdf_files = [f for f in all_files if f.suffix.lower() == '.pdf']
        text_files = [f for f in all_files if f.suffix.lower() in ['.txt', '.md']]
        
        all_results = {
            'documents': [],
            'chunks': [],
            'total_chunks': 0,
            'processing_info': {
                'chunking_strategy': self.config.chunking_strategy,
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'context_generation': self.config.save_contexts,
                'context_provider': self.config.context_provider if self.config.save_contexts else None
            }
        }
        
        # Process PDFs
        if pdf_files:
            pdf_paths = [str(f) for f in pdf_files]
            doc_ids = [f.stem for f in pdf_files]
            pdf_results = self.process_pdf_files(pdf_paths, doc_ids)
            
            all_results['documents'].extend(pdf_results['documents'])
            all_results['chunks'].extend(pdf_results['chunks'])
            all_results['total_chunks'] += pdf_results['total_chunks']
        
        # Process text files
        if text_files:
            text_paths = [str(f) for f in text_files]
            doc_ids = [f.stem for f in text_files]
            text_results = self.process_text_files(text_paths, doc_ids)
            
            all_results['documents'].extend(text_results['documents'])
            all_results['chunks'].extend(text_results['chunks'])
            all_results['total_chunks'] += text_results['total_chunks']
        
        return all_results
    
    def _add_contexts_to_chunks(self, document_text: str, chunks: List[DocumentChunk]) -> List[ContextualChunk]:
        """Add contexts to chunks using parallel processing."""
        def process_chunk(chunk: DocumentChunk) -> ContextualChunk:
            context, usage_info = self.context_generator.generate_context(document_text, chunk.content)
            
            return ContextualChunk(
                content=chunk.content,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                original_index=chunk.original_index,
                metadata=chunk.metadata,
                contextualized_content=context,
                combined_content=f"{chunk.content}\n\n{context}",
                context_generation_info=usage_info
            )
        
        # Process chunks in parallel
        contextual_chunks = []
        with ThreadPoolExecutor(max_workers=self.config.parallel_threads) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating contexts"):
                contextual_chunk = future.result()
                contextual_chunks.append(contextual_chunk)
        
        # Sort by original index to maintain order
        contextual_chunks.sort(key=lambda x: x.original_index)
        return contextual_chunks
    
    def save_results(self, results: Dict[str, Any], output_filename: str = None) -> str:
        """Save chunking results to JSON file."""
        if output_filename is None:
            output_filename = f"chunked_documents_{len(results['documents'])}docs_{results['total_chunks']}chunks.json"
        
        output_path = os.path.join(self.config.output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")
        return output_path


# Utility function to create default config
def create_chunking_config(**kwargs) -> ChunkingConfig:
    """Create a chunking configuration with environment variables."""
    config_dict = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
    }
    config_dict.update(kwargs)
    
    return ChunkingConfig(**config_dict)

def save_chunks_to_txt(json_file_path, output_txt_path):
    """Save chunks from JSON to a text file."""
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Open the output text file
    with open(output_txt_path, 'w', encoding='utf-8') as out_file:
        # Iterate through each chunk
        for chunk in data['chunks']:
            # Write chunk header with chunk_id
            out_file.write(f"=== Chunk {chunk['chunk_id']} ===\n")
            
            # Write the combined_content
            out_file.write(chunk['combined_content'])
            
            # Add some spacing between chunks
            out_file.write("\n\n")

# # Example usage
# if __name__ == "__main__":
#     # Example chunking workflow
#     config = create_chunking_config(
#         context_provider="openai",
#         context_model="gpt-4.1-nano",
#         chunk_size=1500,
#         chunk_overlap=100,
#         chunking_strategy="recursive",
#         save_contexts=True,
#         parallel_threads=3,
#         output_dir="./chunked_output"
#     )
    
#     chunker = DocumentChunker(config)
    
#     # Process PDFs
#     pdf_files = [r"C:\Users\shash\Downloads\pathrag.pdf"]
#     results = chunker.process_pdf_files(pdf_files)
    
#     # Or process entire directory
#     # results = chunker.process_directory("./documents")
    
#     # Save results
#     output_file = chunker.save_results(results)

#     json_file_path = output_file  # Replace with your JSON file path
#     output_txt_path = 'pathrag_chunks.txt'
#     save_chunks_to_txt(json_file_path, output_txt_path)
    
#     print(f"Successfully saved all chunks to txt file {output_txt_path}")

#     print(f"Chunking complete! Results saved to: {output_file}")            