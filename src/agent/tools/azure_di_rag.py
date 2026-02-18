"""
Azure Document Intelligence based Multi-Vector RAG implementation.

This module implements the Multi-Vector Retriever pattern using Azure DI:
- Search: Use LLM-generated descriptions (optimized for vector similarity)
- Answer: Use original data (tables/figures as images for Vision LLM)
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import uuid
import os
import base64
from io import BytesIO
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import fitz  # PyMuPDF
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
import json


# ディレクトリ設定
USER_DATA_DIR = Path("user_data")
DOCUMENTS_DIR = USER_DATA_DIR / "documents"
FAISS_DIR = USER_DATA_DIR / "documents_faiss"

# ディレクトリ作成
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)


def get_azure_di_client() -> DocumentIntelligenceClient:
    """Azure Document Intelligence クライアントを取得"""
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    if not endpoint or not key:
        raise ValueError("Azure DI環境変数が設定されていません")
    
    return DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )


def describe_table_image_with_vision_llm(image_path: Path) -> str:
    """
    Vision LLMで「表」画像を検索用に説明（行名×列名→値の形式で列挙）
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    model = os.getenv("OPENAI_MODEL_VISION", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=0)

    prompt = """あなたは「表」の画像を見ています。表から読み取れる情報を、できるだけ多く日本語で列挙してください。

必須:
- 表の内容（セル値）を可能な限り多く列挙（省略しない）
  ※行名・列名と紐づく形で書ける範囲で書く（ただし特定の固定フォーマットは不要）

ルール:
- 数値・単位・記号（%, 円, 千円, 百万円, ★, 易/中/難 等）は画像の表記をそのまま正確に写す（推測禁止）
- 読めない箇所は「判読不能」と書く（推測しない）
- Markdown表を再生成しない（説明文として出す）
"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]
    )
    return llm.invoke([message]).content.strip()


def crop_image_from_pdf(pdf_path: Path, page_num: int, polygon: List[float]) -> Image.Image:
    """
    PDFから図を切り出し
    
    Args:
        pdf_path: PDFファイルのパス
        page_num: ページ番号（1-indexed）
        polygon: 座標情報 [x1, y1, x2, y2, ...]
        
    Returns:
        切り出した画像
    """
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_num - 1)  # 0-indexed
    
    # polygonから矩形を計算
    x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
    y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    
    # 座標をポイント単位に変換
    bbox_points = [x * 72 for x in bbox]
    rect = fitz.Rect(bbox_points)
    
    # 画像切り出し
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), clip=rect)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    doc.close()
    return img


def describe_image_with_vision_llm(image_path: Path, caption: Optional[str] = None) -> str:
    """
    Vision LLMで画像を説明
    
    Args:
        image_path: 画像ファイルのパス
        caption: キャプション（オプション）
        
    Returns:
        説明文
    """
    # 画像をbase64エンコード
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = f"""この図・グラフを詳しく説明してください。
{f"キャプション: {caption}" if caption else ""}

以下の情報を含めてください:
- 図の種類（棒グラフ、折れ線グラフ、フローチャートなど）
- 主要なデータポイントや傾向
- 軸ラベルや凡例の情報
- 重要な洞察
"""
    
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
    ])
    
    response = llm.invoke([message])
    return response.content


def process_pdf(pdf_path: Path) -> Tuple[FAISS, Dict]:
    """
    Azure DIとLLMを使ってPDFを処理
    
    Args:
        pdf_path: PDFファイルのパス
        
    Returns:
        (vectorstore, docstore)のタプル
    """
    print(f"PDFを解析中: {pdf_path}")
    
    # Azure DI クライアント
    client = get_azure_di_client()
    
    embeddings = OpenAIEmbeddings()

    documents: list[Document] = []
    docstore: Dict = {}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", " ", ""],
    )

    # テキストチャンク方針:
    # - 短すぎる段落(len<=10)は捨てる
    # - ページ末尾段落が句読点で終わっていない場合のみ、次ページ先頭段落に結合して索引化
    # - carry単体は索引化しない（重複防止）
    MIN_PAR_LEN = 10
    PUNCT_END = ("。", "！", "？", ".", "!", "?")
    carry_text: str | None = None

    # PDFをページ単位で処理（スライド/画像多めPDFの取りこぼし・サイズ制限対策）
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    print(f"PDF pages: {total_pages}")

    for page_idx in range(total_pages):
        actual_page_num = page_idx + 1
        print(f"\n--- Page {actual_page_num}/{total_pages} ---")

        writer = PdfWriter()
        writer.add_page(reader.pages[page_idx])
        buf = BytesIO()
        writer.write(buf)
        buf.seek(0)

        try:
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=buf,
                content_type="application/pdf",
            )
            result = poller.result()
        except Exception as e:
            print(f"❌ Page {actual_page_num} failed: {e}")
            continue

        # 表を処理
        tables = result.tables if result.tables else []
        print(f"表を処理中: {len(tables)}個")
        for i, table in enumerate(tables):
            print(f"  表 {i+1}/{len(tables)} を処理中...")
            table_id = str(uuid.uuid4())

            brs = getattr(table, "bounding_regions", None) or []
            if not brs or not getattr(brs[0], "polygon", None):
                print("  ⚠️ table polygon not found; skipping")
                continue

            polygon = brs[0].polygon

            image_path = DOCUMENTS_DIR / f"table_{table_id}.png"
            image = crop_image_from_pdf(pdf_path, actual_page_num, polygon)
            image.save(image_path)

            description = describe_table_image_with_vision_llm(image_path)

            # FAISS: 検索用descriptionのみ
            documents.append(
                Document(
                    page_content=description,
                    metadata={
                        "doc_id": table_id,
                        "type": "table",
                        "page": actual_page_num,
                        "filename": pdf_path.name,
                        "pdf_name": pdf_path.stem,
                    },
                )
            )
            # docstore: 回答用に画像も保持
            docstore[table_id] = {
                "type": "table",
                "description": description,
                "image_path": str(image_path),
                "page": actual_page_num,
                "filename": pdf_path.name,
            }

        # 図を処理
        figures = result.figures if result.figures else []
        print(f"図を処理中: {len(figures)}個")
        for i, figure in enumerate(figures):
            print(f"  図 {i+1}/{len(figures)} を処理中...")
            figure_id = str(uuid.uuid4())

            brs = getattr(figure, "bounding_regions", None) or []
            if not brs or not getattr(brs[0], "polygon", None):
                print("  ⚠️ figure polygon not found; skipping")
                continue

            polygon = brs[0].polygon
            caption = figure.caption.content if figure.caption else None

            image_path = DOCUMENTS_DIR / f"figure_{figure_id}.png"
            image = crop_image_from_pdf(pdf_path, actual_page_num, polygon)
            image.save(image_path)

            description = describe_image_with_vision_llm(image_path, caption)

            search_text = f"{caption}\n\n{description}" if caption else description
            documents.append(
                Document(
                    page_content=search_text,
                    metadata={"doc_id": figure_id, "type": "figure", "page": actual_page_num},
                )
            )

            docstore[figure_id] = {
                "type": "figure",
                "image_path": str(image_path),
                "description": description,
                "caption": caption,
                "page": actual_page_num,
                "filename": pdf_path.name,
            }

        # テキストを処理
        paragraphs = result.paragraphs if result.paragraphs else []
        print(f"テキストを処理中: {len(paragraphs)}個の段落")

        # 1) ページ内paragraphを収集（短すぎる断片は捨てる）
        page_texts: list[str] = []
        for paragraph in paragraphs:
            t = (getattr(paragraph, "content", "") or "").strip()
            if len(t) <= MIN_PAR_LEN:
                continue
            page_texts.append(t)

        # 2) carry を「次ページ先頭paragraph」にだけ結合（carryは単体で索引化しない）
        if carry_text and page_texts:
            page_texts[0] = carry_text.rstrip() + "\n" + page_texts[0]
            carry_text = None  # 使い切ったのでクリア

        # 3) 次ページがある場合のみ、このページ末尾paragraphを carry にするか判定（句読点で終わってない時だけ）
        if page_idx < (total_pages - 1) and page_texts:
            last = page_texts[-1].rstrip()
            if not last.endswith(PUNCT_END):
                carry_text = page_texts.pop()  # このページでは索引化しない（重複防止）

        # 4) 残りをparagraph単位で索引化（長文だけsplit）
        for text in page_texts:
            if len(text) <= 1000:
                chunk_id = str(uuid.uuid4())
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "doc_id": chunk_id,
                            "type": "text",
                            "page": actual_page_num,
                            "filename": pdf_path.name,
                            "pdf_name": pdf_path.stem,
                        },
                    )
                )
                docstore[chunk_id] = {
                    "type": "text",
                    "text": text,
                    "page": actual_page_num,
                    "filename": pdf_path.name,
                }
            else:
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "doc_id": chunk_id,
                                "type": "text",
                                "page": actual_page_num,
                                "filename": pdf_path.name,
                                "pdf_name": pdf_path.stem,
                            },
                        )
                    )
                    docstore[chunk_id] = {
                        "type": "text",
                        "text": chunk,
                        "page": actual_page_num,
                        "filename": pdf_path.name,
                    }
    
    # Vectorstore構築
    print("\nFAISS vectorstoreを構築中...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # 保存
    pdf_name = pdf_path.stem
    vectorstore.save_local(str(FAISS_DIR / pdf_name))
    save_docstore(docstore, pdf_name)

    # 追加: 統合済み _merged を更新（検索時の全ロード＆全マージを避ける）
    merged_dir = FAISS_DIR / "_merged"
    merged_index = merged_dir / "index.faiss"
    try:
        if merged_index.exists():
            merged_vectorstore = FAISS.load_local(
                str(merged_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            merged_vectorstore.merge_from(vectorstore)
        else:
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_vectorstore = vectorstore

        merged_vectorstore.save_local(str(merged_dir))
        print(f"✅ Updated merged FAISS: {merged_dir}")
    except Exception as e:
        # _merged 更新に失敗しても、PDF単体の保存は成功しているので致命ではない
        print(f"⚠️ Failed to update merged FAISS: {e}")
    
    print("✅ 処理完了")
    return vectorstore, docstore


def save_docstore(docstore: Dict, pdf_name: str):
    """Docstoreを保存"""
    docstore_path = FAISS_DIR / f"{pdf_name}_docstore.json"
    with open(docstore_path, 'w', encoding='utf-8') as f:
        json.dump(docstore, f, ensure_ascii=False, indent=2)
    print(f"Docstoreを保存: {docstore_path}")
