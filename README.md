# Data Analysis AI Agent (LangGraph + Streamlit)

`spec/data_analysis_agent/design.md` に基づいたプロトタイプ実装です。

## セットアップ（uv）

```bash
uv sync
cp env.example .env
```

環境変数 `OPENAI_API_KEY` を設定してください。

## 起動

```bash
uv run streamlit run app.py
```

## 使い方

1. Streamlit画面でCSVをアップロード
2. 依頼文を入力
3. エージェントが `reason -> run_code -> ... -> report` を実行
4. 不明確なら質問が返って止まるので、回答して再実行


