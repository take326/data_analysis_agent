"""
Document search using Azure Document Intelligence based Multi-Vector RAG.

This module provides document search functionality with:
- Azure DI for PDF parsing
- Multi-Vector RAG (search with summaries, answer with original data)
- Support for tables (Markdown), figures (images + descriptions), and text
"""

from __future__ import annotations

import os
from typing import Optional
from pathlib import Path
import json
import base64

import cohere
from dotenv import load_dotenv
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .document_ops import load_document_faiss


BASE_DIR = Path(__file__).resolve().parents[3]  # data_analysis_agent/
USER_DATA_DIR = BASE_DIR / "user_data"
DOCS_FAISS_DIR = USER_DATA_DIR / "documents_faiss"


def _resolve_user_path(p: str | None) -> Path | None:
    """
    user_data配下などの相対パスを、プロジェクトルート基準で絶対パスに解決する。
    Streamlitのcwdゆらぎで Path(...).exists() が失敗するのを防ぐ。
    """
    if not p:
        return None
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (BASE_DIR / pp).resolve()


@traceable(name="search_documents_with_relevance")
def search_documents_with_relevance(query: str) -> Optional[list[dict]]:
    """
    文書検索（Azure DI Multi-Vector RAG）
    
    Args:
        query: 検索クエリ
        
    Returns:
        [
            {
                "filename": "document.pdf",
                "relevance_score": 0.85,
                "text": "...",
                "type": "table" | "figure" | "text",
                "page": 3,
                "image_path": "..." (figureの場合のみ)
            },
            ...
        ]
        または None
    """
    load_dotenv()
    
    # FAISSベクトルストアを読み込み
    vectorstore = load_document_faiss()
    if vectorstore is None:
        print("[Document Search] No FAISS index found")
        return None
    
    # FAISS検索（スコア付きで上位10件を取得してログ出力）
    scored = vectorstore.similarity_search_with_score(query, k=10)
    results = [doc for doc, _score in scored]
    print(f"[Document Search] Query: {query}")
    print(f"[Document Search] FAISS found {len(results)} candidates")

    print("[Document Search] FAISS top-10 candidates:")
    for i, (doc, score) in enumerate(scored, 1):
        doc_id = doc.metadata.get("doc_id")
        doc_type = doc.metadata.get("type", "text")
        page = doc.metadata.get("page")
        filename = doc.metadata.get("filename", "document.pdf")
        snippet = (doc.page_content or "").replace("\n", " ")[:120]
        try:
            score_str = f"{float(score):.4f}"
        except Exception:
            score_str = str(score)
        print(
            f"  [{i:02d}] score={score_str} {filename} p{page} {doc_type} id={doc_id} :: {snippet}"
        )
    
    if not results:
        print("[Document Search] No results from FAISS")
        return None
    
    # Multi-Vector RAG: 全てのDocstoreから元データを取得
    docstore_files = list(DOCS_FAISS_DIR.glob("*_docstore.json"))
    docstore = {}
    
    # 全てのdocstoreをマージ
    for docstore_file in docstore_files:
        try:
            with open(docstore_file, 'r', encoding='utf-8') as f:
                file_docstore = json.load(f)
                docstore.update(file_docstore)
            print(f"[Document Search] Loaded docstore: {docstore_file.name}")
        except Exception as e:
            print(f"[Document Search] Failed to load {docstore_file.name}: {e}")
            continue
    
    if docstore:
        print(f"[Document Search] Total docstore items: {len(docstore)}")
    
    # 検索結果に元データを追加
    enriched_results = []
    for doc in results:
        doc_id = doc.metadata.get("doc_id")
        doc_type = doc.metadata.get("type", "text")
        
        # Docstoreから元データを取得
        original_data = None
        if doc_id and doc_id in docstore:
            original_data = docstore[doc_id]
        
        enriched_results.append({
            "doc": doc,
            "doc_type": doc_type,
            "original_data": original_data
        })
    
    # Cohere re-ranking（APIキーがある場合のみ）
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if cohere_api_key:
        try:
            co = cohere.Client(cohere_api_key)
            
            # ドキュメントをテキストに変換
            docs_text = [r["doc"].page_content for r in enriched_results]
            
            # Re-ranking
            rerank_response = co.rerank(
                model="rerank-multilingual-v3.0",
                query=query,
                documents=docs_text,
                # FAISS候補（k=10）全体にスコア付けしてから、重複排除しつつ上位3件を返す
                top_n=len(docs_text)
            )
            
            print(f"[Document Search] Cohere re-ranking completed")
            if rerank_response.results:
                print(f"[Document Search] Top score: {rerank_response.results[0].relevance_score:.3f}")
            
            # 関連性スコアでフィルタリング
            if rerank_response.results:
                sources = []
                seen_doc_ids: set[str] = set()
                seen_texts: set[str] = set()

                def _norm_text(s: str) -> str:
                    # whitespaceゆらぎだけ潰す（"★★★" 等の完全一致を重複判定しやすくする）
                    return " ".join((s or "").split())

                # rerankスコア順に走査して、重複を飛ばしながら3件集める
                for r in rerank_response.results:
                    enriched = enriched_results[r.index]
                    doc = enriched["doc"]
                    
                    # 元データを使用
                    source_data = _prepare_source_data(enriched, doc)
                    source_data["relevance_score"] = r.relevance_score
                    doc_id = doc.metadata.get("doc_id")
                    text_key = _norm_text(source_data.get("text", ""))

                    if isinstance(doc_id, str) and doc_id:
                        if doc_id in seen_doc_ids:
                            continue
                        seen_doc_ids.add(doc_id)

                    if text_key:
                        if text_key in seen_texts:
                            continue
                        seen_texts.add(text_key)

                    sources.append(source_data)
                    if len(sources) >= 3:
                        break
                
                if sources:
                    print(f"[Document Search] Returning {len(sources)} sources")
                    for i, s in enumerate(sources):
                        print(f"  [{i+1}] {s['filename']} page {s.get('page')} ({s.get('type')}, score: {s['relevance_score']:.3f})")
                    return sources
                else:
                    print("[Document Search] No sources")
                    return None
            else:
                print("[Document Search] No re-ranking results")
                return None
        except Exception as e:
            print(f"Cohere re-ranking failed: {e}")
            # フォールバック不要: rerank失敗時は結果なし扱い
            return None
    else:
        # フォールバック不要: Cohereが無い場合は結果なし扱い
        print("[Document Search] COHERE_API_KEY not set; skipping document search")
        return None


def _prepare_source_data(enriched: dict, doc) -> dict:
    """ソースデータを準備"""
    doc_type = enriched["doc_type"]
    original_data = enriched["original_data"]
    
    source = {
        "filename": doc.metadata.get("filename", "document.pdf"),
        "type": doc_type,
        "page": doc.metadata.get("page")
    }
    
    if original_data:
        if doc_type == "table":
            # 表: 検索用はdescription（なければ旧markdown互換）、回答用に画像パスも渡す
            if isinstance(original_data, dict):
                source["text"] = (
                    original_data.get("description")
                    or original_data.get("markdown")
                    or doc.page_content
                )
                source["image_path"] = original_data.get("image_path")
            else:
                source["text"] = doc.page_content
        elif doc_type == "figure":
            # 図: 説明文を使用、画像パスも保存
            source["text"] = original_data.get("description", doc.page_content)
            source["image_path"] = original_data.get("image_path")
            source["caption"] = original_data.get("caption")
        else:
            # テキスト: 旧形式(str) / 新形式(dict: {"type":"text","text":...,"page":...}) の両対応
            if isinstance(original_data, dict) and "text" in original_data:
                source["text"] = original_data["text"]
            else:
                source["text"] = original_data if isinstance(original_data, str) else doc.page_content
    else:
        source["text"] = doc.page_content
    
    return source


@traceable(name="generate_answer_from_documents")
def generate_answer_from_documents(query: str, sources: list[dict]) -> str:
    """
    検索された文書を使ってLLMが回答を生成（画像対応）
    
    Args:
        query: ユーザーの質問
        sources: 検索結果のソース
        
    Returns:
        LLMが生成した回答
    """
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # 画像が含まれる場合はVision対応モデルを使用
    has_images = any(
        s.get("type") in ("figure", "table") and s.get("image_path") for s in sources
    )
    if has_images:
        model = "gpt-4o"  # Vision対応
    
    llm = ChatOpenAI(model=model, temperature=0)
    
    # コンテキスト構築
    context_parts = []
    image_contents = []
    
    for i, s in enumerate(sources, 1):
        source_type = s.get('type', 'text')
        filename = s.get('filename', 'Unknown')
        page = s.get('page', 'N/A')
        text = s.get('text', '')
        
        if source_type == 'table':
            context_parts.append(f"[Source {i}: {filename}, Page {page}, Type: Table]\\n{text}")

            # 表画像をbase64エンコードして追加（元データをVision LLMへ渡す）
            image_path = _resolve_user_path(s.get('image_path'))
            if image_path and image_path.exists():
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                })
        elif source_type == 'figure':
            caption = s.get('caption', '')
            context_parts.append(f"[Source {i}: {filename}, Page {page}, Type: Figure]\\nCaption: {caption}\\nDescription: {text}")
            
            # 画像をbase64エンコードして追加
            image_path = _resolve_user_path(s.get('image_path'))
            if image_path and image_path.exists():
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                })
        else:
            context_parts.append(f"[Source {i}: {filename}, Page {page}]\\n{text}")
    
    context = "\\n\\n".join(context_parts)
    
    prompt_text = f"""You are a helpful assistant that answers questions based on provided documents.

Question: {query}

Documents:
{context}

Instructions:
- Answer the question based ONLY on the information in the documents above
- For tables in Markdown format, parse them carefully to extract relevant data
- For figures, use both the description and the actual image (if provided)
- If the documents don't contain enough information to answer, say so
- Be concise and clear
- Provide specific numbers and data when available
- Cite which source(s) you used (e.g., "According to Source 1...")

Answer:"""
    
    # メッセージ構築（画像がある場合はマルチモーダル）
    if image_contents:
        message_content = [{"type": "text", "text": prompt_text}] + image_contents
        message = HumanMessage(content=message_content)
        return llm.invoke([message]).content
    else:
        return llm.invoke(prompt_text).content
