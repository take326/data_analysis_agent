"""
Test script for Azure Document Intelligence Multi-Vector RAG implementation.

Usage:
    python3 test_azure_di_rag.py path/to/document.pdf
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.tools.azure_di_rag import process_pdf


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_azure_di_rag.py path/to/document.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Processing PDF: {pdf_path}")
    print("=" * 60)
    
    try:
        vectorstore, docstore = process_pdf(pdf_path)
        
        print("\n" + "=" * 60)
        print("✅ Processing completed successfully!")
        print(f"Vectorstore: {len(vectorstore.docstore._dict)} documents")
        print(f"Docstore: {len(docstore)} items")
        
        # Docstoreの内訳を表示
        types = {}
        for item in docstore.values():
            if isinstance(item, dict):
                item_type = item.get('type', 'unknown')
            else:
                item_type = 'text'
            types[item_type] = types.get(item_type, 0) + 1
        
        print("\nDocstore breakdown:")
        for item_type, count in types.items():
            print(f"  - {item_type}: {count}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
