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

    if ro.json:
        st.markdown("**🔧 JSON**")
        for j in ro.json:
            st.json(j)


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

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)

if "processing" not in st.session_state:
    st.session_state.processing = False

# メモリを読み込み
from src.agent.memory.loader import load_memory
memories = [m.model_dump() for m in load_memory()]

if st.session_state.state is None:
    st.session_state.state = {
        "messages": [],
        "df": df,
        "memories": memories,
        "decision": None,
        "last_code": None,
        "last_exec": None,
        "report": None,
    }
else:
    # dfは常に最新アップロードを優先（単一CSV前提）
    st.session_state.state["df"] = df
    # メモリも毎回最新を読み込み
    st.session_state.state["memories"] = memories

state = st.session_state.state

# タブ構成
tab1, tab2 = st.tabs(["� Analysis", " Prediction"])

# Tab 1: Analysis (Data Preview + Chat)
with tab1:
    # Data Preview（コンパクト）
    with st.expander(f"📊 Data Preview ({df.shape[0]} rows × {df.shape[1]} columns)", expanded=False):
        st.dataframe(df, use_container_width=True)
    
    # Chat
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

    # チャット入力
    user_text = st.chat_input("分析内容を入力...")
    if user_text:
        # 表示用の履歴に積む（ChatGPT風）
        st.session_state.chat_history.append({"type": "user", "text": user_text})
        st.session_state.state["messages"] = list(st.session_state.state["messages"]) + [
            HumanMessage(content=user_text)
        ]
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
                    # カテゴリ変数: 数値入力 + ヘルプテキスト
                    mappings = categorical_mappings.get(feature, {})
                    if mappings:
                        # JSONは数値キーを文字列に変換するので、整数に戻す
                        mappings = {int(k): v for k, v in mappings.items()}
                        help_text = ", ".join([f"{k}={v}" for k, v in sorted(mappings.items())])
                        max_val = max(mappings.keys())
                    else:
                        help_text = "Categorical feature (encoded as numbers)"
                        max_val = 10
                    
                    value = st.number_input(
                        feature,
                        min_value=0,
                        max_value=max_val,
                        value=0,
                        step=1,
                        key=f"input_{feature}",
                        help=help_text
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
                st.success(f"**{selected_model['target_name']}**: {prediction[0]:.4f}")
                
                # 詳細情報
                with st.expander("📊 Prediction Details"):
                    st.write("**Input Values:**")
                    for feature, value in zip(selected_model['feature_names'], input_values):
                        st.write(f"- {feature}: {value}")
                    st.write(f"**Model**: {selected_model['model_name']}")
                    st.write(f"**Model Type**: {selected_model['model_type']}")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")




