from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ディレクトリ定義
USER_DATA_DIR = Path("user_data")
DOCUMENTS_DIR = USER_DATA_DIR / "documents"
DOCUMENTS_FAISS_DIR = USER_DATA_DIR / "documents_faiss"
MERGED_SUBDIR = "_merged"
MERGED_DIR = DOCUMENTS_FAISS_DIR / MERGED_SUBDIR

# ディレクトリ作成
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_FAISS_DIR.mkdir(parents=True, exist_ok=True)

def load_document_faiss() -> Optional[FAISS]:
    """
    文書用FAISSベクトルストアを読み込み（全PDFを統合）
    
    Returns:
        FAISSベクトルストア、存在しない場合はNone
    """
    load_dotenv()
    
    # 全てのPDFディレクトリを探す
    if not DOCUMENTS_FAISS_DIR.exists():
        return None

    # _merged があればそれを優先してロード（検索時の全ロード＆全マージを避ける）
    merged_index = MERGED_DIR / "index.faiss"
    if merged_index.exists():
        embeddings = OpenAIEmbeddings()
        try:
            merged_vectorstore = FAISS.load_local(
                str(MERGED_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"[Document Ops] Loaded merged vectorstore from: {MERGED_SUBDIR}")
            return merged_vectorstore
        except Exception as e:
            print(f"[Document Ops] Failed to load merged vectorstore: {e}")
            # 破損などの場合は従来の統合ロジックにフォールバック（初回だけ重い）
    
    # サブディレクトリを探す
    subdirs = [d for d in DOCUMENTS_FAISS_DIR.iterdir() if d.is_dir() and d.name != MERGED_SUBDIR]
    if not subdirs:
        return None
    
    embeddings = OpenAIEmbeddings()
    merged_vectorstore = None
    
    # 全てのベクトルストアを統合
    for subdir in subdirs:
        index_file = subdir / "index.faiss"
        if not index_file.exists():
            continue
        
        try:
            vectorstore = FAISS.load_local(
                str(subdir),
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            if merged_vectorstore is None:
                merged_vectorstore = vectorstore
            else:
                # 既存のベクトルストアにマージ
                merged_vectorstore.merge_from(vectorstore)
            
            print(f"[Document Ops] Loaded vectorstore from: {subdir.name}")
        except Exception as e:
            print(f"[Document Ops] Failed to load {subdir.name}: {e}")
            continue

    # 統合結果を _merged として永続化（次回検索は _merged をロードするだけになる）
    if merged_vectorstore is not None:
        try:
            MERGED_DIR.mkdir(parents=True, exist_ok=True)
            merged_vectorstore.save_local(str(MERGED_DIR))
            print(f"[Document Ops] Saved merged vectorstore to: {MERGED_SUBDIR}")
        except Exception as e:
            print(f"[Document Ops] Failed to save merged vectorstore: {e}")

    return merged_vectorstore
