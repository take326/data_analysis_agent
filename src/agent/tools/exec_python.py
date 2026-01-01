from __future__ import annotations

import ast
import base64
import contextlib
import io
import time
from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt

from ..models import ExecPythonInput, ExecPythonOutput, ExecResult
from .model_ops import save_model_automatically


@dataclass(frozen=True)
class GuardViolation(Exception):
    message: str


_ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
}

_BANNED_CALLS = {"open", "__import__", "eval", "exec", "compile", "input"}


def _guard_code(code: str) -> None:
    """
    簡易ガード（プロトタイプ）。
    - 同一プロセスexecは完全にサンドボックス化できないため、危険度の高いものを雑に拒否する。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise GuardViolation(f"SyntaxError: {e}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top not in _ALLOWED_IMPORTS:
                    raise GuardViolation(f"Import '{top}' is not allowed.")
        if isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".", 1)[0]
                if top not in _ALLOWED_IMPORTS:
                    raise GuardViolation(f"ImportFrom '{top}' is not allowed.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALLS:
            raise GuardViolation(f"Call '{node.func.id}()' is not allowed.")


def _cap_output(stdout: str, stderr: str, max_chars: int) -> tuple[str, str]:
    combined = stdout + stderr
    if len(combined) <= max_chars:
        return stdout, stderr
    keep = max_chars // 2
    return stdout[:keep], stderr[:keep]


def _capture_matplotlib_figures(max_figs: int = 5) -> list[str]:
    """
    現在のmatplotlib figureをPNG(base64)化して返す。
    """
    out: list[str] = []
    fignums = list(plt.get_fignums())[:max_figs]
    for num in fignums:
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        out.append(base64.b64encode(buf.read()).decode("ascii"))
    # 後片付け（Streamlitでメモリリークしやすい）
    plt.close("all")
    return out


def exec_python(inp: ExecPythonInput) -> ExecPythonOutput:
    """
    Pythonコードを同一プロセスでexecし、stdout/stderrと簡易成果物を返す。

    制約:
    - 厳密なタイムアウト強制停止は困難（別プロセスでないため）。
    - プロトタイプとして「出力上限」「簡易ガード」「図の自動収集」を行う。
    """
    try:
        _guard_code(inp.code)
    except GuardViolation as v:
        return ExecPythonOutput(
            result=ExecResult(ok=False, stderr=v.message, error_type="PolicyError", error_message=v.message)
        )

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    start = time.time()
    g: dict[str, Any] = {
        "__builtins__": __builtins__,
        **inp.context,
    }
    l: dict[str, Any] = {}

    ok = True
    err_type: Optional[str] = None
    err_msg: Optional[str] = None

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(inp.code, g, l)
    except Exception as e:
        ok = False
        err_type = type(e).__name__
        err_msg = str(e)

    elapsed = time.time() - start
    stdout = stdout_buf.getvalue()
    stderr = stderr_buf.getvalue()

    # 「簡易」タイムアウト（実際には止められないので、超過を失敗として扱うだけ）
    if elapsed > inp.timeout_sec:
        ok = False
        err_type = err_type or "TimeoutError"
        err_msg = err_msg or f"Exceeded {inp.timeout_sec}s"
        stderr = (stderr + "\n" if stderr else "") + f"[timeout] elapsed={elapsed:.2f}s"

    stdout, stderr = _cap_output(stdout, stderr, inp.max_output_chars)

    # 成果物プロトコル（任意）: TABLE_MARKDOWN / JSON_OUT
    table_markdown: list[str] = []
    if isinstance(l.get("TABLE_MARKDOWN"), list):
        table_markdown = [x for x in l["TABLE_MARKDOWN"] if isinstance(x, str)]

    json_out: list[Any] = []
    if isinstance(l.get("JSON_OUT"), list):
        json_out = l["JSON_OUT"]

    # matplotlib図は自動収集（コードがpltで描いていれば拾える）
    plot_png_base64 = _capture_matplotlib_figures()
    
    # MODEL変数をチェックして自動保存
    saved_model_id: Optional[str] = None
    if l.get("MODEL") is not None:  # ローカル変数をチェック
        try:
            model = l["MODEL"]
            metadata = l.get("MODEL_METADATA", {})
            saved_model_id = save_model_automatically(model, metadata)
            # 保存成功メッセージを追加
            stdout = (stdout + "\n" if stdout else "") + f"✅ Model saved: {saved_model_id}"
        except Exception as e:
            # 保存失敗時はエラーメッセージを追加（実行自体は成功扱い）
            stderr = (stderr + "\n" if stderr else "") + f"⚠️ Model save failed: {e}"

    return ExecPythonOutput(
        result=ExecResult(
            ok=ok,
            stdout=stdout,
            stderr=stderr,
            plot_png_base64=plot_png_base64,
            table_markdown=table_markdown,
            json=json_out,
            error_type=err_type,
            error_message=err_msg,
            saved_model_id=saved_model_id,
        )
    )


