"""第1回の基礎・発展：各条件を独立した会話で3回ずつ実測する。"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter, sleep

from google import genai
from google.genai import errors

from app.ask import Conversation, DEFAULT_QUESTION, save_record, show_result
from app.config import MODEL, api_error_message, get_api_key


def write_report(path: Path, records: list[dict]) -> None:
    """成功分の平均と各回答を保存する。欠測値を0として扱わない。"""
    lines = [
        "# 第1回 実験結果", "",
        "各リクエストは新しい会話です。平均はokのみ（途中終了・回答文なし・APIエラーを除外）。"
        "経過時間はSDK呼び出し開始から応答受信までで、"
        "通信・SDK内の再試行を含み、モデル内部の推論時間だけではありません。",
        "入力・出力・思考はAPIの返却値です。未取得は欠測で、0ではありません。"
        "合計は思考等を含むため、入力＋出力と一致しない場合があります。", "",
        "| モデル | 質問 | temperature | 成功/試行 | 平均秒 | 平均入力 | 平均出力 | 平均思考 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    groups = dict.fromkeys((r["model"], r["question"], r["temperature"]) for r in records)
    for model, question, temperature in groups:
        attempted = [r for r in records if (r["model"], r["question"], r["temperature"]) == (model, question, temperature)]
        good = [r for r in attempted if r["status"] == "ok"]

        def average(field):
            values = [r[field] for r in good if r.get(field) is not None]
            return f"{mean(values):.2f}" if values else "—"

        escaped = question.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {model} | {escaped} | {temperature} | {len(good)}/{len(attempted)} | "
            f"{average('elapsed_seconds')} | {average('input_tokens')} | "
            f"{average('output_tokens')} | {average('thinking_tokens')} |"
        )
    lines.extend(["", "## 各回の回答と失敗", ""])
    for row in records:
        lines.extend([
            f"### {row['model']} / 試行{row['repeat']} / {row['status']}", "",
            f"質問：{row['question']}", "",
            f"temperature={row['temperature']} / {row['elapsed_seconds']:.3f}秒 / "
            f"終了理由={row.get('finish_reason', '—')}", "",
            row.get("text") or row.get("error", "回答文なし"), "",
        ])
    lines.extend([
        "## 補足", "",
        "比較・考察は `lectures/01/answer.md` に記載。",
        "各条件3回の小規模比較であり、速度・精度の一般的な優劣を断定しない。", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=[MODEL, "gemini-3.1-flash-lite"])
    parser.add_argument("--questions", nargs="+", default=["LoRAとは何ですか？", DEFAULT_QUESTION])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--interval", type=float, default=20, help="呼び出し間の待機秒数。上限回避を保証しない")
    parser.add_argument("--output", type=Path, help="JSONL保存先。省略時は日時付きの新規ファイル")
    args = parser.parse_args()
    if args.repeats < 1 or args.interval < 0 or args.max_output_tokens < 1 or not 0 <= args.temperature <= 2:
        parser.error("回数・出力上限は1以上、待機秒数は0以上、temperatureは0〜2です")
    output = args.output or Path("lectures/01/results") / f"experiment-{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
    if output.suffix.lower() != ".jsonl":
        parser.error("保存先の拡張子は.jsonlにしてください（.mdの集計を別途作成します）")
    if output.exists() or output.with_suffix(".md").exists():
        parser.error("保存先が既にあります。別名を指定してください")
    total = len(args.models) * len(args.questions) * args.repeats
    print(f"質問を最大{total}件送信します（SDK内の自動再試行を除く）。temperature={args.temperature}、待機={args.interval}秒")
    print(f"保存先：{output} と {output.with_suffix('.md')}\nAPIエラーでは記録して停止します。")
    records = []
    try:
        with genai.Client(api_key=get_api_key()) as client:
            for index, (model, question, repeat) in enumerate(product(args.models, args.questions, range(1, args.repeats + 1))):
                if index:
                    sleep(args.interval)
                print(f"\n[{index + 1}/{total}] {model} / {question} / 試行{repeat}", flush=True)
                started = perf_counter()
                try:
                    # 毎回Chatを作り直し、過去の回答が比較に混ざるのを防ぐ。
                    result = Conversation(client, model, args.temperature, args.max_output_tokens).send(question)
                    row = {**asdict(result), "repeat": repeat}
                    show_result(result)
                except errors.APIError as error:
                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": model, "question": question, "repeat": repeat,
                        "temperature": args.temperature, "max_output_tokens": args.max_output_tokens,
                        "elapsed_seconds": perf_counter() - started,
                        "status": "error", "error_code": error.code,
                        "error": api_error_message(error.code),
                    }
                records.append(row)
                save_record(output, row)
                write_report(output.with_suffix(".md"), records)
                if row["status"] == "error":
                    print(row["error"])
                    raise SystemExit(1)
    except KeyboardInterrupt:
        print("\n中断しました。完了した試行は保存されています。")
        raise SystemExit(130)
    print(f"\n実験完了：{len(records)}件。比較・考察の記入先：lectures/01/answer.md")


if __name__ == "__main__":
    main()
