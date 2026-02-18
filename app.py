from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

import base64
import uuid

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import create_graph
from src.agent.models import ExecResult, ReasonDecision, ReportOutput


st.set_page_config(page_title="Data Analysis AI Agent", layout="wide")


def _init_session():
    if "app" not in st.session_state:
        st.session_state.app = create_graph()
    if "state" not in st.session_state:
        st.session_state.state = None
    # ChatGPT風に表示するチャット履歴（表示用。LLM用の state["messages"] とは分離）
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _reset():
    st.session_state.state = None
    st.session_state.chat_history = []


def _render_report(report: dict):
    ro = ReportOutput.model_validate(report)
    st.markdown("**📊 Report Summary**")
    st.markdown(ro.summary)

    if ro.table_markdown:
        st.markdown("**📋 Tables**")
        for t in ro.table_markdown:
            st.markdown(t)

    if ro.plot_png_base64:
        st.markdown("**📈 Plots**")
        for b64 in ro.plot_png_base64:
            st.image(base64.b64decode(b64))


_init_session()

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if st.button("Reset session"):
        _reset()
    st.caption("env: OPENAI_API_KEY / OPENAI_MODEL")
    
    # Phase 3: Saved Models
    st.divider()
    st.subheader("📊 Saved Models")
    from src.agent.tools.model_ops import list_saved_models
    models = list_saved_models()
    
    if models:
        for model in models[:5]:  # 最新5件
            st.markdown(f"**{model['model_name']}**")
            st.caption(f"Type: {model['model_type']}")
            score_label = "R²" if model['task_type'] == 'regression' else "Acc"
            st.caption(f"Test {score_label}: {model.get('test_score', 0):.3f}")
            st.caption(f"Created: {model['created_at'][:10]}")
    else:
        st.info("No models saved yet")

if "processing" not in st.session_state:
    st.session_state.processing = False

# タブ構成（CSVの有無に関わらず表示）
tab1, tab2, tab3 = st.tabs(["💬 Analysis", "🔮 Prediction", "📄 Document Search"])

# Tab 1: Analysis (Data Preview + Chat)
with tab1:
    if uploaded is None:
        st.info("📁 Upload a CSV file to start analysis.")
    else:
        df = pd.read_csv(uploaded)
        
        st.markdown("---")
        
        # メモリを読み込み
        from src.agent.memory.loader import load_memory
        memories = [m.model_dump() for m in load_memory()]
        
        if st.session_state.state is None:
            st.session_state.state = {
                "messages": [],
                "df": df,
                "uploaded_filename": uploaded.name,  # ファイル名を保存
                "memories": memories,
                "decision": None,
                "last_code": None,
                "last_exec": None,
                "report": None,
                "code_run_count": 0,  # 無限ループ防止用カウンター
            }
        else:
            # dfは常に最新アップロードを優先（単一CSV前提）
            st.session_state.state["df"] = df
            st.session_state.state["uploaded_filename"] = uploaded.name  # ファイル名を更新
            # メモリも毎回最新を読み込み
            st.session_state.state["memories"] = memories
        
        state = st.session_state.state
        
        # Data Preview（コンパクト）
        with st.expander(f"📊 Data Preview ({df.shape[0]} rows × {df.shape[1]} columns)", expanded=False):
            st.dataframe(df, use_container_width=True)
        
        # Chat履歴（スクロール可能なコンテナ）
        chat_container = st.container(height=500)
        with chat_container:
            # ChatGPT風: chat_history を時系列で表示（LLM用messagesとは分離）
            for e in st.session_state.get("chat_history", []):
                etype = e.get("type")
                if etype == "user":
                    with st.chat_message("user"):
                        st.write(e.get("text", ""))
                elif etype == "assistant":
                    with st.chat_message("assistant"):
                        st.write(e.get("text", ""))
                elif etype == "code":
                    with st.chat_message("assistant"):
                        with st.expander("📝 実行コード", expanded=False):
                            st.code(e.get("code", ""), language="python")
                elif etype == "report":
                    with st.chat_message("assistant"):
                        _render_report(e["report"])
            
             # レポート表示は chat_history 側に一本化（時系列の中に残す）
            
            # 処理中：スピナーを表示しながら実行
            if st.session_state.processing:
                with st.chat_message("assistant"):
                    with st.spinner("分析中..."):
                        prev_report = st.session_state.state.get("report")
                        prev_last_code = st.session_state.state.get("last_code")
                        prev_messages = list(st.session_state.state.get("messages", []))
                        agent_result = st.session_state.app.invoke(st.session_state.state)
                        st.session_state.state = agent_result
            
                        # run_code で生成されたコードを履歴に積む（同一内容の重複は避ける）
                        new_last_code = agent_result.get("last_code")
                        if new_last_code and new_last_code != prev_last_code:
                            last = st.session_state.chat_history[-1] if st.session_state.chat_history else None
                            if not (last and last.get("type") == "code" and last.get("code") == new_last_code):
                                st.session_state.chat_history.append({"type": "code", "code": new_last_code})
            
                        # ask_clarification 等で増えたAIMessageを chat_history に積む（report_summaryタグは除外）
                        new_messages = list(agent_result.get("messages", []))
                        if len(new_messages) > len(prev_messages):
                            for m in new_messages[len(prev_messages) :]:
                                if isinstance(m, AIMessage) and m.additional_kwargs.get("source") == "report_summary":
                                    continue
                                if isinstance(m, AIMessage):
                                    st.session_state.chat_history.append({"type": "assistant", "text": m.content})
            
                        new_report = agent_result.get("report")
                        if new_report and new_report != prev_report:
                            st.session_state.chat_history.append({"type": "report", "report": new_report})
                st.session_state.processing = False
                st.rerun()
        
        # チャット入力（コンテナの外 = 下部に固定）
        user_text = st.chat_input("分析内容を入力...")
        if user_text:
            # 表示用の履歴に積む（ChatGPT風）
            st.session_state.chat_history.append({"type": "user", "text": user_text})
            
            # Stateに保存
            st.session_state.state["messages"] = list(st.session_state.state["messages"]) + [
                HumanMessage(content=user_text)
            ]
            st.session_state.state["code_run_count"] = 0  # 新しいリクエストなのでカウンターリセット
            
            st.session_state.processing = True
            st.rerun()

# Tab 2: Prediction
with tab2:
    from src.agent.tools.model_ops import list_saved_models, load_model
    
    st.header("🔮 Model Prediction")
    
    models = list_saved_models()
    
    if not models:
        st.info("📭 No models available. Train a model first in the Analysis tab!")
    else:
        # モデル選択
        model_names = [m['model_name'] for m in models]
        selected_name = st.selectbox("Select Model", model_names)
        
        # 選択されたモデルのメタデータ
        selected_model = next(m for m in models if m['model_name'] == selected_name)
        
        # モデル情報表示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", selected_model['model_type'])
        with col2:
            st.metric("Task Type", selected_model['task_type'].capitalize())
        with col3:
            score_label = "Test R²" if selected_model['task_type'] == 'regression' else "Test Accuracy"
            st.metric(score_label, f"{selected_model.get('test_score', 0):.3f}")
        
        st.divider()
        
        # 動的フォーム生成
        st.subheader("Input Features")
        
        input_values = []
        cols = st.columns(2)
        categorical_features = selected_model.get('categorical_features', [])
        categorical_mappings = selected_model.get('categorical_mappings', {})
        
        for i, feature in enumerate(selected_model['feature_names']):
            with cols[i % 2]:
                if feature in categorical_features:
                    # カテゴリ変数: ドロップダウンで選択
                    mappings = categorical_mappings.get(feature, {})
                    if mappings:
                        # JSONは数値キーを文字列に変換するので、整数に戻す
                        mappings = {int(k): v for k, v in mappings.items()}
                        # ラベル名のリストを作成（数値順）
                        options = [mappings[k] for k in sorted(mappings.keys())]
                        selected_label = st.selectbox(
                            feature,
                            options=options,
                            index=0,
                            key=f"input_{feature}"
                        )
                        # ラベル名から数値に逆変換
                        label_to_value = {v: k for k, v in mappings.items()}
                        value = label_to_value[selected_label]
                    else:
                        # マッピングがない場合は数値入力にフォールバック
                        value = st.number_input(
                            feature,
                            min_value=0,
                            max_value=10,
                            value=0,
                            step=1,
                            key=f"input_{feature}",
                            help="Categorical feature (encoded as numbers)"
                        )
                else:
                    # 数値変数: 通常の数値入力
                    value = st.number_input(
                        feature,
                        value=0.0,
                        key=f"input_{feature}",
                        format="%.4f"
                    )
                input_values.append(value)
        
        st.divider()
        
        # 予測ボタン
        if st.button("🎯 Predict", type="primary", use_container_width=True):
            try:
                # モデル読み込み
                model, metadata = load_model(selected_model['model_id'])
                
                # 予測
                prediction = model.predict([input_values])
                
                # 結果表示
                predicted_value = prediction[0]
                if isinstance(predicted_value, float):
                    st.success(f"**{selected_model['target_name']}**: {predicted_value:.4f}")
                else:
                    st.success(f"**{selected_model['target_name']}**: {predicted_value}")
                
                # 詳細情報
                with st.expander("📊 Prediction Details"):
                    st.write("**Input Values:**")
                    for feature, value in zip(selected_model['feature_names'], input_values):
                        st.write(f"- {feature}: {value}")
                    st.write(f"**Model**: {selected_model['model_name']}")
                    st.write(f"**Model Type**: {selected_model['model_type']}")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")


# Tab 3: Document Search
with tab3:
    from src.agent.tools.document_search import search_documents_with_relevance
    from src.agent.tools.document_ops import DOCUMENTS_FAISS_DIR
    
    st.header("📄 Document Search")
    st.markdown("Upload and search through documents for relevant information")

    # rerun（検索/ボタン押下/入力）で同じアップロードPDFを再処理しないためのメモ
    if "indexed_doc_hashes" not in st.session_state:
        st.session_state.indexed_doc_hashes = set()
    
    # Document Upload Section
    st.subheader("📤 Upload Documents")
    uploaded_docs = st.file_uploader(
        "Upload documents to index",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDF (.pdf) files",
        key="doc_upload"
    )
    
    if uploaded_docs:
        from src.agent.tools.document_ops import DOCUMENTS_DIR
        from src.agent.tools.azure_di_rag import process_pdf
        import hashlib

        # インデックスが空なら、取り込み済みキャッシュもクリア（リセット時の整合用）
        if not any(DOCUMENTS_FAISS_DIR.glob("*/index.faiss")):
            st.session_state.indexed_doc_hashes.clear()
        
        for doc_file in uploaded_docs:
            # Streamlit rerun で同一ファイルが何度も処理されないように内容hashで判定
            data = doc_file.getvalue()
            content_hash = hashlib.sha256(data).hexdigest()
            cache_key = f"{doc_file.name}:{content_hash}"
            if cache_key in st.session_state.indexed_doc_hashes:
                continue

            # Save file
            save_path = DOCUMENTS_DIR / doc_file.name
            save_path.write_bytes(data)
            
            # Process with Azure DI for PDFs
            if save_path.suffix.lower() != ".pdf":
                st.error(f"❌ Unsupported file type: {save_path.suffix} (PDF only)")
                continue

            with st.spinner(f"Processing {doc_file.name} with Azure Document Intelligence..."):
                try:
                    # Azure DIで処理（ベクトルストアとdocstoreを自動保存）
                    vectorstore, docstore = process_pdf(save_path)
                    st.success(f"✅ {doc_file.name} processed successfully with Azure DI")
                    st.info(f"📊 Extracted: {len(docstore)} items (tables, figures, text)")
                    st.session_state.indexed_doc_hashes.add(cache_key)
                except Exception as e:
                    st.error(f"❌ Failed to process {doc_file.name}: {e}")

    
    st.markdown("---")
    
    # Document Search Section
    st.subheader("🔍 Search Documents")
    
    # Check if documents exist
    has_pdf_faiss_index = any(DOCUMENTS_FAISS_DIR.glob("*/index.faiss"))
    if not has_pdf_faiss_index:
        st.info("📭 No documents indexed yet. Upload documents above to enable search.")
    else:
        with st.form("doc_search_form", clear_on_submit=False):
            query = st.text_input(
                "Enter your search query:",
                key="doc_search_query",
                placeholder="e.g., What are the customer feedback themes?",
            )
            submitted = st.form_submit_button("🔍 Search")

        if submitted:
            if query:
                with st.spinner("Searching documents and generating answer..."):
                    from src.agent.tools.document_search import generate_answer_from_documents

                    sources = search_documents_with_relevance(query)

                    if sources:
                        # Generate answer using LLM
                        answer = generate_answer_from_documents(query, sources)

                        # Display answer
                        st.markdown("### 💬 Answer")
                        st.markdown(answer)

                        # Display sources
                        st.markdown("---")
                        st.markdown("### 📚 Sources")

                        for i, source in enumerate(sources, 1):
                            score = source.get("relevance_score", 0)
                            source_file = source.get("filename", "Unknown")
                            source_type = source.get("type", "text")
                            page = source.get("page", "N/A")
                            text = source.get("text", "")
                            image_path = source.get("image_path")
                            caption = source.get("caption")

                            # タイプに応じたアイコン
                            type_icon = {"table": "📊", "figure": "🖼️", "text": "📝"}.get(source_type, "📄")

                            with st.expander(
                                f"{type_icon} Source {i}: {source_file} (Page {page}, Type: {source_type}, Relevance: {score:.2f})"
                            ):
                                if source_type == "table":
                                    st.markdown("**Type:** Table (Markdown)")
                                    st.markdown(text)
                                elif source_type == "figure":
                                    from pathlib import Path

                                    st.markdown("**Type:** Figure")
                                    if caption:
                                        st.markdown(f"**Caption:** {caption}")
                                    if image_path and Path(image_path).exists():
                                        st.image(image_path)
                                    st.markdown(text)
                                else:
                                    st.markdown(text)
                    else:
                        st.info("No relevant documents found. Try a different query.")
            else:
                st.warning("Please enter a search query")
