"""
Contextual Chunker - A comprehensive document chunking library for RAG applications
"""

from .chunker import (
    DocumentChunker,
    ChunkingConfig,
    DocumentChunk,
    ContextualChunk,
    PDFProcessor,
    TextProcessor,
    RecursiveTextSplitter,
    SemanticTextSplitter,
    OpenAIContextGenerator,
    create_chunking_config,
    save_chunks_to_txt
)

__version__ = "0.1.0"
__author__ = "Kishan Tripathi"
__email__ = "kishantripathi888@gmail.com"

__all__ = [
    "DocumentChunker",
    "ChunkingConfig", 
    "DocumentChunk",
    "ContextualChunk",
    "PDFProcessor",
    "TextProcessor",
    "RecursiveTextSplitter",
    "SemanticTextSplitter",
    "OpenAIContextGenerator",
    "create_chunking_config",
    "save_chunks_to_txt"
]