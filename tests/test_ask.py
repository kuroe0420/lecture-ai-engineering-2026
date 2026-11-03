"""ネットワークを使わず、実際のSDKによる履歴の組み立てと計測を検証する。"""

import json

from google import genai
from google.genai import types
import pytest

from app.ask import Conversation, save_record
from app.experiment import main as experiment_main, write_report


def test_sdk_sends_history_and_new_conversation_starts_empty(monkeypatch):
    requests = []
    replies = iter(["猫です", "さっきの猫です", "覚えていません"])

    def generate(**kwargs):
        requests.append(list(kwargs["contents"]))
        return types.GenerateContentResponse(
            candidates=[types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=next(replies))]),
                finish_reason="STOP",
            )],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10, candidates_token_count=20,
                thoughts_token_count=5, total_token_count=35,
            ),
        )

    with genai.Client(api_key="test-not-a-real-key") as client:
        monkeypatch.setattr(client.models, "generate_content", generate)
        chat = Conversation(client)
        first = chat.send("好きな動物は猫です")
        chat.send("さっきの動物は？")
        Conversation(client).send("さっきの動物は？")

    assert [content.role for content in requests[1]] == ["user", "model", "user"]
    assert requests[1][0].parts[0].text == "好きな動物は猫です"
    assert requests[1][1].parts[0].text == "猫です"
    assert len(requests[2]) == 1
    assert first.input_tokens == 10 and first.output_tokens == 20
    assert first.thinking_tokens == 5 and first.total_tokens == 35
    assert first.elapsed_seconds >= 0 and first.finish_reason == "STOP"


def test_utf8_records_and_report_exclude_errors_from_average(tmp_path):
    path = tmp_path / "results.jsonl"
    good = dict(model="test-model", question="用語とは？", temperature=1.0,
                status="ok", repeat=1, elapsed_seconds=2.0, input_tokens=10,
                output_tokens=None, thinking_tokens=0, text="日本語の回答")
    failed = {**good, "status": "error", "repeat": 2, "elapsed_seconds": 100,
              "text": "", "error": "レート制限"}
    save_record(path, good)
    save_record(path, failed)
    assert [json.loads(line)["status"] for line in path.read_text(encoding="utf-8").splitlines()] == ["ok", "error"]
    write_report(path.with_suffix(".md"), [good, failed])
    text = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "1/2 | 2.00 | 10.00 | — | 0.00" in text
    assert "日本語の回答" in text and "レート制限" in text


@pytest.mark.parametrize("reason,text,status", [
    ("STOP", "回答", "ok"),
    ("MAX_TOKENS", "途中の回答", "truncated"),
    ("SAFETY", None, "empty"),
])
def test_missing_usage_and_incomplete_answers(monkeypatch, reason, text, status):
    with genai.Client(api_key="test-not-a-real-key") as client:
        response = types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text=text)] if text else []),
            finish_reason=reason,
        )])
        monkeypatch.setattr(client.models, "generate_content", lambda **kwargs: response)
        result = Conversation(client).send("質問")
    assert result.status == status
    assert result.input_tokens is None and result.output_tokens is None


def test_rate_limit_records_completed_work_and_stops(monkeypatch, tmp_path):
    from google.genai import errors

    calls = []

    def limited(self, question):
        calls.append(question)
        raise errors.ClientError(429, {"error": {"message": "quota"}})

    path = tmp_path / "limited.jsonl"
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(Conversation, "send", limited)
    monkeypatch.setattr("sys.argv", ["experiment", "--interval", "0", "--output", str(path)])
    with pytest.raises(SystemExit) as result:
        experiment_main()
    assert result.value.code == 1
    assert len(calls) == 1
    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["error_code"] == 429
    assert "0/1" in path.with_suffix(".md").read_text(encoding="utf-8")
