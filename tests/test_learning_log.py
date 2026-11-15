"""学習ログが承認前や拒否時に保存されないことを確認する。"""

import json

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from app import tools
from app.agent import agent, run_with_approval


def request_save(messages, info):
    if any(
        isinstance(part, ToolReturnPart) and part.tool_name == "save_learning_log"
        for message in messages
        for part in message.parts
    ):
        return ModelResponse(parts=[TextPart("処理を終了しました。")])
    return ModelResponse(parts=[ToolCallPart(
        "save_learning_log",
        {"lecture": 3, "content": "ツールの説明文で検索の仕方が変わった。"},
        tool_call_id="save-1",
    )])


@pytest.mark.parametrize("answer, expected_files", [("y", 1), ("n", 0), ("", 0), ("yes?", 0)])
def test_cli_saves_only_after_explicit_approval(tmp_path, monkeypatch, answer, expected_files):
    folder = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", folder)

    def reply(prompt):
        assert not folder.exists(), "承認前にログを保存している"
        return answer

    monkeypatch.setattr("builtins.input", reply)
    result = run_with_approval(agent, "第3回の学習内容を保存して", FunctionModel(request_save))
    paths = list(folder.glob("*.json"))
    assert len(paths) == expected_files
    assert isinstance(result.output, str)
    assert result.usage.requests == 2
    returns = [
        part for message in result.all_messages() for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "save_learning_log"
    ]
    assert len(returns) == 1
    if expected_files:
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        assert data["lecture"] == 3
        assert data["content"] == "ツールの説明文で検索の仕方が変わった。"
        assert data["saved_at"]
    else:
        assert not folder.exists()
