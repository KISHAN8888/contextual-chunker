
"""
Command Line Interface for Document Chunker
"""

import argparse
import sys
from pathlib import Path
from .chunker import DocumentChunker, create_chunking_config, save_chunks_to_txt


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Document Chunker - Split documents into chunks with optional context generation"
    )
    
    # Input options
    parser.add_argument(
        'input', 
        help='Input file or directory path'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='./chunked_documents',
        help='Output directory for results (default: ./chunked_documents)'
    )
    
    parser.add_argument(
        '--chunk-size', '-s',
        type=int,
        default=1000,
        help='Chunk size in characters (default: 1000)'
    )
    
    parser.add_argument(
        '--chunk-overlap', '-p',
        type=int,
        default=200,
        help='Chunk overlap in characters (default: 200)'
    )
    
    parser.add_argument(
        '--strategy', '-t',
        choices=['recursive', 'semantic'],
        default='recursive',
        help='Chunking strategy (default: recursive)'
    )
    
    parser.add_argument(
        '--no-context',
        action='store_true',
        help='Disable context generation'
    )
    
    parser.add_argument(
        '--context-model',
        default='gpt-4o-mini',
        help='OpenAI model for context generation (default: gpt-4o-mini)'
    )
    
    parser.add_argument(
        '--threads', '-j',
        type=int,
        default=5,
        help='Number of parallel threads for context generation (default: 5)'
    )
    
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=['.pdf', '.txt', '.md'],
        help='File extensions to process (default: .pdf .txt .md)'
    )
    
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Also save chunks to a text file'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        default=True,
        help='Process directories recursively (default: True)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Create configuration
    config = create_chunking_config(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.strategy,
        save_contexts=not args.no_context,
        context_model=args.context_model,
        parallel_threads=args.threads,
        output_dir=args.output_dir
    )
    
    try:
        # Initialize chunker
        chunker = DocumentChunker(config)
        
        # Process input
        if input_path.is_file():
            if input_path.suffix.lower() == '.pdf':
                results = chunker.process_pdf_files([str(input_path)])
            elif input_path.suffix.lower() in ['.txt', '.md']:
                results = chunker.process_text_files([str(input_path)])
            else:
                print(f"Error: Unsupported file type: {input_path.suffix}")
                sys.exit(1)
        else:
            # Process directory
            results = chunker.process_directory(
                str(input_path), 
                file_extensions=args.extensions,
                recursive=args.recursive
            )
        
        # Save results
        output_file = chunker.save_results(results)
        
        # Save to text file if requested
        if args.save_txt:
            txt_file = output_file.replace('.json', '.txt')
            save_chunks_to_txt(output_file, txt_file)
            print(f"Text file saved to: {txt_file}")
        
        print(f"\nProcessing complete!")
        print(f"Documents processed: {len(results['documents'])}")
        print(f"Total chunks created: {results['total_chunks']}")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()