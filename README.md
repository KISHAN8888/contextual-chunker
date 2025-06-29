# Contextual Chunker

A comprehensive Python library for document chunking with intelligent context generation, designed specifically for RAG (Retrieval-Augmented Generation) applications.

## Features

- **Multiple Document Formats**: Support for PDF and text files (.txt, .md)
- **Flexible Chunking Strategies**: Recursive and semantic text splitting
- **Context Generation**: AI-powered context generation using OpenAI models
- **Parallel Processing**: Multi-threaded context generation for efficiency
- **Multiple Output Formats**: JSON and plain text output options
- **CLI Interface**: Easy-to-use command-line interface
- **Configurable**: Extensive configuration options for different use cases

## Installation

```bash
pip install contextual-chunker
```

## Quick Start

### Using as a Library

```python
from contextual_chunker import DocumentChunker, create_chunking_config

# Create configuration
config = create_chunking_config(
    openai_api_key="your-openai-api-key",
    chunk_size=1500,
    chunk_overlap=100,
    chunking_strategy="recursive",
    save_contexts=True
)

# Initialize chunker
chunker = DocumentChunker(config)

# Process PDF files
results = chunker.process_pdf_files(["document.pdf"])

# Or process a directory
results = chunker.process_directory("./documents")

# Save results
output_file = chunker.save_results(results)
print(f"Results saved to: {output_file}")
```

### Using the CLI

```bash
# Process a single PDF file
contextual-chunker document.pdf --chunk-size 1000 --output-dir ./output

# Process a directory with custom settings
contextual-chunker ./documents --strategy semantic --chunk-size 1500 --save-txt

# Process without context generation
contextual-chunker ./documents --no-context --chunk-size 800
```

## Configuration Options

### ChunkingConfig Parameters

- `openai_api_key`: Your OpenAI API key (required for context generation)
- `chunk_size`: Maximum size of each chunk in characters (default: 1000)
- `chunk_overlap`: Overlap between chunks in characters (default: 200)
- `chunking_strategy`: "recursive" or "semantic" (default: "recursive")
- `save_contexts`: Enable AI context generation (default: True)
- `context_model`: OpenAI model for context generation (default: "gpt-4o-mini")
- `parallel_threads`: Number of threads for parallel processing (default: 5)
- `output_dir`: Directory for output files (default: "./chunked_documents")

## Chunking Strategies

### Recursive Text Splitter
Splits text using a hierarchy of separators (paragraphs → sentences → words → characters) while respecting chunk size limits.

### Semantic Text Splitter
Preserves semantic meaning by splitting on paragraph and sentence boundaries first, ensuring coherent chunks.

## Context Generation

The library can automatically generate contextual information for each chunk using OpenAI's models. This context helps improve retrieval accuracy in RAG applications by providing additional information about where each chunk fits within the larger document.

## Output Formats

### JSON Output
Structured output containing:
- Document metadata
- Individual chunks with content and metadata
- Context information (if enabled)
- Processing statistics

### Text Output
Simple text file with all chunks for easy review and debugging.

## CLI Usage

```bash
contextual-chunker [OPTIONS] INPUT

Arguments:
  INPUT                    Input file or directory path

Options:
  -o, --output-dir TEXT    Output directory (default: ./chunked_documents)
  -s, --chunk-size INT     Chunk size in characters (default: 1000)
  -p, --chunk-overlap INT  Chunk overlap in characters (default: 200)
  -t, --strategy CHOICE    Chunking strategy: recursive|semantic (default: recursive)
  --no-context            Disable context generation
  --context-model TEXT    OpenAI model for context (default: gpt-4o-mini)
  -j, --threads INT       Parallel threads (default: 5)
  -e, --extensions LIST   File extensions to process (default: .pdf .txt .md)
  --save-txt              Also save chunks to text file
  -r, --recursive         Process directories recursively
```

## Environment Variables

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Examples

### Basic PDF Processing
```python
from contextual_chunker import DocumentChunker, create_chunking_config

config = create_chunking_config(
    chunk_size=1000,
    save_contexts=False  # Disable context generation
)

chunker = DocumentChunker(config)
results = chunker.process_pdf_files(["research_paper.pdf"])
chunker.save_results(results)
```

### Advanced Configuration with Context
```python
config = create_chunking_config(
    openai_api_key="sk-...",
    chunk_size=1500,
    chunk_overlap=150,
    chunking_strategy="semantic",
    context_model="gpt-4",
    parallel_threads=8,
    save_contexts=True
)

chunker = DocumentChunker(config)
results = chunker.process_directory("./research_papers", recursive=True)
output_file = chunker.save_results(results)

# Also save as text file
from document_chunker import save_chunks_to_txt
save_chunks_to_txt(output_file, "chunks.txt")
```

## Requirements

- Python 3.8+
- OpenAI API key (for context generation)
- PyMuPDF or PyPDF2 (for PDF processing)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.