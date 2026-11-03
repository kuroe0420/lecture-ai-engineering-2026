"""第1回：単発質問、会話履歴、応答時間・トークン数の計測。

uv run python -m app.ask "LoRAとは何ですか？"
uv run python -m app.ask --chat --output lectures/01/results/chat.jsonl
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass
class Measurement:
    timestamp: str
    model: str
    question: str
    temperature: float
    max_output_tokens: int
    elapsed_seconds: float
    text: str
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None
    status: str = "ok"


class Conversation:
    """SDKのChatが過去の質問と回答を次の要求に含める。"""

    def __init__(self, client, model=MODEL, temperature=1.0, max_output_tokens=2048):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.chat = client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=max_output_tokens,
            ),
        )

    def send(self, question: str) -> Measurement:
        started = perf_counter()
        response = self.chat.send_message(question)
        elapsed = perf_counter() - started
        usage = response.usage_metadata
        text = response.text or ""
        candidates = response.candidates or []
        reason = candidates[0].finish_reason if candidates else None
        return Measurement(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=self.model, question=question, temperature=self.temperature,
            max_output_tokens=self.max_output_tokens, elapsed_seconds=elapsed,
            text=text,
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            thinking_tokens=usage.thoughts_token_count if usage else None,
            total_tokens=usage.total_token_count if usage else None,
            finish_reason=str(reason.value) if reason else None,
            status=("truncated" if reason == types.FinishReason.MAX_TOKENS else "ok") if text else "empty",
        )


def save_record(path: Path, record: dict) -> None:
    """質問・回答・計測値だけをUTF-8で追記する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def show_result(result: Measurement) -> None:
    print(result.text or "[回答文なし：終了理由を確認してください]")
    print(
        f"\n[計測] {result.elapsed_seconds:.3f}秒 / "
        f"入力={result.input_tokens} / 出力={result.output_tokens} / "
        f"思考={result.thinking_tokens} / 合計={result.total_tokens} / "
        f"終了理由={result.finish_reason}"
    )
    if result.finish_reason == "MAX_TOKENS":
        print("[注意] トークン上限で終了しました。回答が途中で切れている可能性があります。")


def ask(question: str) -> str:
    """元の単発質問用の関数。回答を返し、トークン数も表示する。"""
    with genai.Client(api_key=get_api_key()) as client:
        result = Conversation(client).send(question)
    print(f"[usage] 入力={result.input_tokens} / 出力={result.output_tokens}")
    return result.text


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMに質問し、回答と計測値を表示する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path, help="質問・回答・計測値をJSONLに追記する")
    args = parser.parse_args()

    if not 0 <= args.temperature <= 2 or args.max_output_tokens < 1:
        parser.error("temperatureは0〜2、max-output-tokensは1以上にしてください")
    print(f"[model] {MODEL}")
    if args.chat:
        print("会話モード：/exitで終了、/resetで履歴を消去。履歴は終了時に失われます。")
    session_id = datetime.now(timezone.utc).isoformat()
    try:
        with genai.Client(api_key=get_api_key()) as client:
            conversation = Conversation(client, MODEL, args.temperature, args.max_output_tokens)
            question = args.question if args.chat else (args.question or DEFAULT_QUESTION)
            turn = 0
            while True:
                if question is None:
                    question = input("\nあなた> ").strip()
                if args.chat and question == "/exit":
                    break
                if args.chat and question == "/reset":
                    conversation = Conversation(client, MODEL, args.temperature, args.max_output_tokens)
                    session_id = datetime.now(timezone.utc).isoformat()
                    turn = 0
                    print("会話履歴を消去しました。")
                elif question:
                    print(f"[質問] {question}\n")
                    result = conversation.send(question)
                    turn += 1
                    show_result(result)
                    if args.output:
                        save_record(args.output, {
                            **asdict(result), "session_id": session_id, "turn": turn,
                        })
                if not args.chat:
                    break
                question = None
    except errors.APIError as error:
        print(api_error_message(error.code))
        raise SystemExit(1)
    except (EOFError, KeyboardInterrupt):
        print("\n終了しました。保存済みの記録は残ります。")


if __name__ == "__main__":
    main()
